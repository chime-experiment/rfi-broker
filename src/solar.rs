//! RFI zeroing around solar noon

use std::time::{SystemTime, UNIX_EPOCH};

use chrono::{DateTime, Utc};
use sunrise::{Coordinates, SolarDay, SolarEvent};
use tokio::time::{Instant, sleep_until};

use reqwest::Client;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
use serde_json::json;

use crate::metrics::SharedMetrics;

/// # Configuration
// NB: these values are hard-coded for CHIME. They should be moved to
// a configuration file somehow.
const LATITUDE: f64 = 49.320_709_219_4;
const LONGITUDE: f64 = -119.623_677_431_0;
const ALTITUDE: f64 = 555.372;

const DOWNTIME_S: i64 = 3600; // 1 hour

const FIRST_STAGE_ENDPOINT: &str = "http://csBfs:54323/rfi-zeroing-toggle-first-stage";
const SECOND_STAGE_ENDPOINT: &str = "http://csBfs:54323/rfi-zeroing-toggle-second-stage";
const TARGET: &str = "rfi-zeroing";

/// Convert a Unix timestamp to a [`tokio::time::Instant`].
///
/// Useful for triggering fixed-duration `sleep_until` calls.
///
/// Returns `None` if `unix_time` is in the past.
fn unix_to_instant(unix_time: i64) -> Option<Instant> {
    let now_unix = SystemTime::now().duration_since(UNIX_EPOCH).ok()?;

    let delta_secs = unix_time - now_unix.as_secs().cast_signed();

    if delta_secs < 0 {
        return None;
    }

    Some(Instant::now() + std::time::Duration::from_secs(delta_secs.cast_unsigned()))
}

/// Compute solar noon for `days_offset` days from today
fn solar_noon(coord: Coordinates, days_offset: i64) -> Option<DateTime<Utc>> {
    let mut unix_time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()?
        .as_secs()
        .cast_signed();
    unix_time += days_offset * 86_400;

    // Sort out the sunrise, sunset, and noon
    let date = DateTime::<Utc>::from_timestamp(unix_time, 0)?.date_naive();

    let day = SolarDay::new(coord, date).with_altitude(ALTITUDE);

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
pub async fn solar_event_task(metrics: SharedMetrics) {
    // Create a new requests client
    let client = Client::new();
    // Contruct headers
    let mut headers = HeaderMap::new();
    headers.insert(
        CONTENT_TYPE,
        HeaderValue::from_static("application/json; charset=UTF-8"),
    );

    // Create a fixed coordinate object
    let coords: Coordinates = Coordinates::new(LATITUDE, LONGITUDE).unwrap();

    // One-time calculation of the next solar noon
    let mut noon_delta = solar_noon(coords, 0).unwrap().timestamp();

    tracing::debug!("Next solar noon in {noon_delta} seconds.");

    loop {
        let next_event_start = noon_delta - DOWNTIME_S / 2;
        let next_event_end = next_event_start + DOWNTIME_S;

        // Sleep until the next zeroing disable event
        if let Some(t) = unix_to_instant(next_event_start) {
            sleep_until(t).await;
            // Send the second-stage event first
            match post_event(&client, &headers, SECOND_STAGE_ENDPOINT, TARGET, false).await {
                Ok(()) => metrics.rfi_zeroing.set_second(false),
                Err(e) => tracing::warn!("{e}"),
            }
            match post_event(&client, &headers, FIRST_STAGE_ENDPOINT, TARGET, true).await {
                Ok(()) => metrics.rfi_zeroing.set_first(false),
                Err(e) => tracing::warn!("{e}"),
            }
        }

        // Sleep until the next enable event
        if let Some(t) = unix_to_instant(next_event_end) {
            sleep_until(t).await;
            // Send the first-stage event first
            match post_event(&client, &headers, FIRST_STAGE_ENDPOINT, TARGET, true).await {
                Ok(()) => metrics.rfi_zeroing.set_first(true),
                Err(e) => tracing::warn!("{e}"),
            }
            match post_event(&client, &headers, SECOND_STAGE_ENDPOINT, TARGET, true).await {
                Ok(()) => metrics.rfi_zeroing.set_second(true),
                Err(e) => tracing::warn!("{e}"),
            }
        } else {
            tracing::debug!("Solar post-noon event time has already passed. Skipping...");
        }

        // Get the next solar noon window
        noon_delta = solar_noon(coords, 1).unwrap().timestamp();

        tracing::debug!("Next solar noon in {noon_delta} seconds.");
    }
}
