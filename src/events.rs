//! Tasks implementing repeating async events.
//!
//! # Tasks
//! - ``solar_event_task``: temporarily disables kotekan RFI flagging around
//!   solar transit

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use chrono::{DateTime, Utc};
use sunrise::{Coordinates, SolarDay, SolarEvent};

use reqwest::Client;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};

use eyre::{OptionExt, Report, WrapErr, eyre};

use serde_json::json;

use crate::config::SharedAppConfig;
use crate::metrics::SharedMetrics;

/// Get the seconds until a future unix time.
///
/// Useful for triggering fixed-duration `sleep` calls.
///
/// Returns `None` if `unix_time` is in the past.
fn seconds_until(unix_time: i64) -> Option<Duration> {
    let now_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()?
        .as_secs()
        .cast_signed();

    let delta_secs = unix_time - now_unix;

    if delta_secs < 0 {
        return None;
    }

    Some(Duration::from_secs(delta_secs.cast_unsigned()))
}

/// Compute solar noon for `days_offset` days in the future, relative to now.
///
/// Return `None` if the computation failed for some reason.
fn solar_noon(coord: Coordinates, altitude: f64, days_offset: i64) -> Option<DateTime<Utc>> {
    let mut requested_time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()?
        .as_secs()
        .cast_signed();

    requested_time += days_offset * 86_400;

    // Sort out the sunrise, sunset, and noon
    let date = DateTime::<Utc>::from_timestamp(requested_time, 0)?.date_naive();

    let day = SolarDay::new(coord, date).with_altitude(altitude);

    let rise = day.event_time(SolarEvent::Sunrise)?.timestamp();
    let set = day.event_time(SolarEvent::Sunset)?.timestamp();

    // Return solar noon as the halfway point between sunrise and sunset
    DateTime::<Utc>::from_timestamp(i64::midpoint(rise, set), 0)
}

/// Error handling for event tasks.
#[derive(Debug, thiserror::Error)]
pub enum PostError {
    /// Network/connection failure
    #[error("client request failed")]
    Request(#[from] Report),
    /// Bad status
    #[error("bad status {status}: {body}")]
    BadStatus { status: u16, body: String },
}

/// Send an enable/disable event to the zeroing endpoint.
async fn post_event(
    client: &Client,
    headers: &HeaderMap,
    endpoint: &str,
    target: &str,
    enable: bool,
) -> Result<reqwest::StatusCode, PostError> {
    let payload = json!({target: enable});

    let result = client
        .post(endpoint)
        .headers(headers.clone())
        .json(&payload)
        .send()
        .await
        .wrap_err("failed to send event signal to endpoint")?;

    let status = result.status();

    if !status.is_success() {
        return Err(PostError::BadStatus {
            status: status.as_u16(),
            body: status
                .canonical_reason()
                .map_or("null".into(), std::string::ToString::to_string),
        });
    }

    Ok(status)
}

/// Task to run solar noon zeroing.
///
/// On startup:
/// 1. Create the `reqwest::Client` and headers
/// 2. Compute the next solar noon
///
/// On each iteration:
/// 1. Sleep until 1/2 of the total downtime before solar noon.
/// 2. Sends the zeroing `on` command.
/// 3. Sleep until the end of the total downtime.
/// 4. Sends the zeroing `off` command.
/// 5. Compute the next solar noon.
///
/// Intended to be run with [`tokio::spawn`]. Since this is an async
/// task, it will almost never consume resources.
///
/// # Panics
/// Panics if the solar noon estimation fails.
pub async fn solar_event_task(metrics: SharedMetrics, config: SharedAppConfig) -> eyre::Result<()> {
    let (Some(telescope), Some(zeroing)) = (&config.telescope, &config.zeroing) else {
        tracing::info!("solar event config not set - task going into permanent idle");
        // Won't wake, so no CPU consumed
        std::future::pending::<()>().await;
        unreachable!();
    };

    tracing::debug!(
        "Running solar task for telescope\n{:#?} with endpoint parameters\n{:#?}",
        telescope,
        zeroing
    );
    // Construct the addresses only once
    let first_stage_addr = format!("https://{}/{}", &zeroing.hostname, &zeroing.first_stage);
    let second_stage_addr = format!("https://{}/{}", &zeroing.hostname, &zeroing.second_stage);

    // Create a new requests client
    let client = Client::new();
    // Contruct headers
    let mut headers = HeaderMap::new();
    headers.insert(
        CONTENT_TYPE,
        HeaderValue::from_static("application/json; charset=UTF-8"),
    );

    // Create a fixed coordinate object
    let coords = Coordinates::new(telescope.latitude, telescope.longitude).ok_or_else(|| {
        eyre!(
            "invalid coordinates: {:#?}, {:#?}",
            telescope.latitude,
            telescope.longitude
        )
    })?;

    // One-time calculation of the next solar noon
    let mut next_noon =
        solar_noon(coords, telescope.altitude, 0).ok_or_eyre("failed to compute solar noon.")?;

    loop {
        #[allow(
            clippy::integer_division,
            reason = "integer division downcasting is desired behaviour"
        )]
        let next_event_start = next_noon.timestamp() - zeroing.downtime.cast_signed() / 2;
        let next_event_end = next_event_start + zeroing.downtime.cast_signed();

        tracing::info!("Next solar noon window at {next_noon}");

        // Sleep until the next zeroing disable event
        if let Some(t) = seconds_until(next_event_start) {
            tokio::time::sleep(t).await;
            // Send the second-stage event first
            match post_event(
                &client,
                &headers,
                &second_stage_addr,
                &zeroing.target,
                false,
            )
            .await
            {
                Ok(_) => metrics.rfi_zeroing.set_second(false),
                Err(PostError::BadStatus { status, body }) => {
                    tracing::warn!(%status, %body);
                }
                Err(PostError::Request(report)) => {
                    tracing::error!(error = ?report);
                }
            }
            match post_event(&client, &headers, &first_stage_addr, &zeroing.target, true).await {
                Ok(_) => metrics.rfi_zeroing.set_first(false),
                Err(PostError::BadStatus { status, body }) => {
                    tracing::warn!(%status, %body);
                }
                Err(PostError::Request(report)) => {
                    tracing::error!(error = ?report);
                }
            }
        }

        // Sleep until the next enable event
        if let Some(t) = seconds_until(next_event_end) {
            tokio::time::sleep(t).await;
            // Send the first-stage event first
            match post_event(&client, &headers, &first_stage_addr, &zeroing.target, true).await {
                Ok(_) => metrics.rfi_zeroing.set_first(true),
                Err(PostError::BadStatus { status, body }) => {
                    tracing::warn!(%status, %body);
                }
                Err(PostError::Request(report)) => {
                    tracing::error!(error = ?report);
                }
            }
            match post_event(&client, &headers, &second_stage_addr, &zeroing.target, true).await {
                Ok(_) => metrics.rfi_zeroing.set_second(true),
                Err(PostError::BadStatus { status, body }) => {
                    tracing::warn!(%status, %body);
                }
                Err(PostError::Request(report)) => {
                    tracing::error!(error = ?report);
                }
            }
        } else {
            tracing::debug!("Solar noon event time has already passed. Skipping...");
        }

        // Get the next solar noon window. Have to use this extra variable assignment
        // because rust doesn't let us use attributes on expressions
        let try_next_noon =
            solar_noon(coords, telescope.altitude, 1).ok_or_eyre("failed to compute solar noon")?;
        next_noon = try_next_noon;
    }
}
