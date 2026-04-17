//! RFI zeroing around solar noon

use std::time::{SystemTime, UNIX_EPOCH};

use chrono::{DateTime, Utc};
use sunrise::{Coordinates, SolarDay, SolarEvent};
use tokio::time::{Instant, sleep_until};

use reqwest::Client;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};

use serde_json::json;

use crate::config::SharedAppConfig;
use crate::metrics::SharedMetrics;

/// Convert a Unix timestamp to a [`tokio::time::Instant`].
///
/// Useful for triggering fixed-duration `sleep_until` calls.
///
/// Returns `None` if `unix_time` is in the past.
fn unix_to_instant(unix_time: i64) -> Option<Instant> {
    let now_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()?
        .as_secs()
        .cast_signed();

    let delta_secs = unix_time - now_unix;

    if delta_secs < 0 {
        return None;
    }

    Some(Instant::now() + std::time::Duration::from_secs(delta_secs.cast_unsigned()))
}

/// Compute solar noon for `days_offset` days from today
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

/// Send an enable/disable event to the zeroing endpoint
async fn post_event(
    client: &Client,
    headers: &HeaderMap,
    endpoint: &str,
    target: &str,
    enable: bool,
) -> Result<(), String> {
    let payload = json!({target: enable});

    let result = client
        .post(endpoint)
        .headers(headers.clone())
        .json(&payload)
        .send()
        .await
        .map_err(|e| format!("failed to send signal to endpoint {endpoint}: {e}"))?;

    if !result.status().is_success() {
        return Err(format!(
            "Failed to send {enable} to {endpoint}: {}",
            result.status()
        ));
    }

    Ok(())
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
pub async fn solar_event_task(metrics: SharedMetrics, config: SharedAppConfig) {
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
    #[allow(clippy::unwrap_used, reason = "panic on fail is desired behaviour")]
    let coords: Coordinates = Coordinates::new(telescope.latitude, telescope.longitude).unwrap();

    // One-time calculation of the next solar noon
    #[allow(clippy::unwrap_used, reason = "panic on fail is desired behaviour")]
    let mut next_noon = solar_noon(coords, telescope.altitude, 0).unwrap();

    loop {
        #[allow(
            clippy::integer_division,
            reason = "integer division downcasting is desired behaviour"
        )]
        let next_event_start = next_noon.timestamp() - zeroing.downtime.cast_signed() / 2;
        let next_event_end = next_event_start + zeroing.downtime.cast_signed();

        tracing::info!("Next solar noon window at {next_noon}");

        // Sleep until the next zeroing disable event
        if let Some(t) = unix_to_instant(next_event_start) {
            sleep_until(t).await;
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
                Ok(()) => metrics.rfi_zeroing.set_second(false),
                Err(e) => tracing::warn!("{e}"),
            }
            match post_event(&client, &headers, &first_stage_addr, &zeroing.target, true).await {
                Ok(()) => metrics.rfi_zeroing.set_first(false),
                Err(e) => tracing::warn!("{e}"),
            }
        }

        // Sleep until the next enable event
        if let Some(t) = unix_to_instant(next_event_end) {
            sleep_until(t).await;
            // Send the first-stage event first
            match post_event(&client, &headers, &first_stage_addr, &zeroing.target, true).await {
                Ok(()) => metrics.rfi_zeroing.set_first(true),
                Err(e) => tracing::warn!("{e}"),
            }
            match post_event(&client, &headers, &second_stage_addr, &zeroing.target, true).await {
                Ok(()) => metrics.rfi_zeroing.set_second(true),
                Err(e) => tracing::warn!("{e}"),
            }
        } else {
            tracing::debug!("Solar post-noon event time has already passed. Skipping...");
        }

        // Get the next solar noon window. Have to use this extra variable assignment
        // because rust doesn't let us use attributes on expressions
        #[allow(clippy::unwrap_used, reason = "panic on fail is desired behaviour")]
        let try_next_noon = solar_noon(coords, telescope.altitude, 1).unwrap();
        next_noon = try_next_noon;
    }
}
