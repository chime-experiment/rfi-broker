//! Task to disable RFI zeroing during solar transit.

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use chrono::{DateTime, TimeDelta, Utc};
use sunrise::{Coordinates, SolarDay, SolarEvent};

use reqwest::Client;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};

use eyre::{OptionExt, WrapErr, bail, eyre};

use serde_json::{Value, json};

use crate::config::AppConfig;
use crate::state::Metrics;

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

/// Send an enable/disable event to the zeroing endpoint.
async fn post_event(
    client: &Client,
    headers: &HeaderMap,
    endpoint: &str,
    payload: &Value,
) -> eyre::Result<reqwest::StatusCode> {
    let result = client
        .post(endpoint)
        .headers(headers.clone())
        .json(&payload)
        .send()
        .await
        .wrap_err("failed to send event signal to endpoint")?;

    let status = result.status();

    if !status.is_success() {
        bail!(
            "Bad status: {status}, reason: {}",
            status
                .canonical_reason()
                .map_or("null".into(), std::string::ToString::to_string)
        );
    }
    tracing::debug!("sent {payload} to endpoint: {endpoint}");

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
/// task, it will almost never consume resources. If no config is provided,
/// that task runs as an indefinitely-pending future which will never
/// resolve or consume resources.
///
/// # Errors
/// Errors if computing solar noon fails, or the telescope coordinates
/// are invalid.
#[allow(clippy::too_many_lines, reason = "lots of debug lines")]
pub async fn solar_event_task(
    metrics: Arc<Metrics>,
    config: Option<Arc<AppConfig>>,
) -> eyre::Result<()> {
    // If either `config` is None, or either of the required config entries
    // is None, this task is enter a permanent pending state.
    let Some((telescope, zeroing)) = config
        .as_ref()
        .and_then(|c| Some((c.telescope.as_ref()?, c.zeroing.as_ref()?)))
    else {
        tracing::info!("solar event config not set - task going into idle state");
        // Won't wake, so no CPU consumed
        std::future::pending::<()>().await;
        unreachable!();
    };

    tracing::info!(
        telescope = ?telescope,
        zeroing = ?zeroing,
        "solar RFI zeroing task started",
    );
    // Construct the addresses
    let first_stage_addr = format!("{}/{}", zeroing.url, zeroing.first_stage_endpoint);
    let second_stage_addr = format!("{}/{}", zeroing.url, zeroing.second_stage_endpoint);

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

    // One-time calculation of solar noon for the current day. This could be in the past
    let mut next_noon =
        solar_noon(coords, telescope.altitude, 0).ok_or_eyre("failed to compute solar noon.")?;

    loop {
        #[allow(
            clippy::integer_division,
            reason = "integer division downcasting is desired behaviour"
        )]
        let next_event_start = next_noon - TimeDelta::seconds(zeroing.downtime.cast_signed() / 2);
        let next_event_end = next_event_start + TimeDelta::seconds(zeroing.downtime.cast_signed());

        tracing::info!("next solar noon window at {next_noon}");
        // targets want times in nanoseconds
        let event_start_ns = next_event_start.timestamp_nanos_opt().unwrap_or(0);
        // for an event in the past, just send the current time. This should only
        // occur at startup
        let event_end_ns = next_event_end
            .max(DateTime::<Utc>::from(SystemTime::now()))
            .timestamp_nanos_opt()
            .unwrap_or(0);

        let disable_payload =
            json!({(&zeroing.toggle_value): false, "valid_at_time_ns": event_start_ns});
        let enable_payload =
            json!({(&zeroing.toggle_value): true, "valid_at_time_ns": event_end_ns});

        // Sleep until the next zeroing disable event
        if let Some(t) = seconds_until(next_event_start.timestamp()) {
            tokio::time::sleep(t).await;
            // Send the second-stage event first
            tracing::debug!(
                addr = ?second_stage_addr,
                payload = ?disable_payload,
                "sending second-stage `disable` event..."
            );
            post_event(&client, &headers, &second_stage_addr, &disable_payload)
                .await
                .inspect(|_| metrics.rfi_zeroing.set_second(false))
                .inspect_err(
                    |err| tracing::warn!(error = ?err, "failed to disable second-stage zeroing"),
                )
                .ok();

            tracing::debug!(
                addr = ?first_stage_addr,
                payload = ?disable_payload,
                "second first-stage `disable` event..."
            );
            post_event(&client, &headers, &first_stage_addr, &disable_payload)
                .await
                .inspect(|_| metrics.rfi_zeroing.set_first(false))
                .inspect_err(
                    |err| tracing::warn!(error = ?err, "failed to disable first-stage zeroing"),
                )
                .ok();
        }

        // Sleep until the next enable event. If the even has passed, we still want
        // to make sure that zeroing is enabled outside of the transit window. The
        // `else` case here should only be accessible on the first pass of this loop.
        if let Some(t) = seconds_until(next_event_end.timestamp()) {
            tokio::time::sleep(t).await;
        } else {
            tracing::info!(
                "solar noon event time has already passed, but we'll ensure \
                that zeroing is enabled anyway."
            );
        }

        // Send the first-stage event first
        tracing::debug!(
            addr = ?first_stage_addr,
            payload = ?enable_payload,
            "sending first-stage `enable` event..."
        );
        post_event(&client, &headers, &first_stage_addr, &enable_payload)
            .await
            .inspect(|_| metrics.rfi_zeroing.set_first(true))
            .inspect_err(|err| tracing::warn!(error = ?err, "failed to enable first-stage zeroing"))
            .ok();

        tracing::debug!(
            addr = ?second_stage_addr,
            payload = ?enable_payload,
            "second second-stage `enable` event..."
        );
        post_event(&client, &headers, &second_stage_addr, &enable_payload)
            .await
            .inspect(|_| metrics.rfi_zeroing.set_second(true))
            .inspect_err(
                |err| tracing::warn!(error = ?err, "failed to enable second-stage zeroing"),
            )
            .ok();

        // Get the next solar noon window
        next_noon =
            solar_noon(coords, telescope.altitude, 1).ok_or_eyre("failed to compute solar noon")?;
    }
}
