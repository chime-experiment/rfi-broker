//! Axum endpoints and associated functions.
#[cfg(debug_assertions)]
use std::fmt::Write;

use axum::{Json, extract::State, http::StatusCode, response::IntoResponse};

use eyre::{OptionExt, bail};

use ndarray::{ArrayD, Axis};

use crate::datastate::SharedDataState;
use crate::metrics::SharedMetrics;
use crate::stats;

/// `GET /meta` - snapshot of state metadata.
///
/// Returns `500` if serialisation fails.
pub async fn metadata(State(state): State<SharedDataState>) -> impl IntoResponse {
    let Some(meta) = state.metadata.get() else {
        return Err::<_, (StatusCode, String)>((
            StatusCode::NO_CONTENT,
            "metadata not available".into(),
        ));
    };

    let meta: serde_json::Value = serde_json::to_value(*meta.lock())
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok::<_, (StatusCode, String)>(Json(meta))
}

/// Return an error as an ``INTERNAL_SERVER_ERROR``.
#[allow(
    clippy::needless_pass_by_value,
    reason = "error will always be consumed"
)]
fn handler_err(e: impl ToString) -> (StatusCode, String) {
    (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
}

/// `GET /data` — snapshot most recent frame in all dataset ring buffers.
///
/// Only exists in debug builds
#[cfg(debug_assertions)]
pub async fn data(State(state): State<SharedDataState>) -> Result<String, (StatusCode, String)> {
    let mut out = String::new();

    // Dump all the current buffers
    if let Some(frac_flagged) = state.frac_flagged.get() {
        writeln!(out, "-- frac_flagged --").map_err(handler_err)?;
        writeln!(out, "  frame_count : {:?}", frac_flagged.len()).map_err(handler_err)?;
        writeln!(out, "  frames_in_queue : {:?}", frac_flagged.queue_len()).map_err(handler_err)?;
        writeln!(out, "  frame_shape : {:?}", frac_flagged.shape()).map_err(handler_err)?;

        if let Some(frame) = frac_flagged.last() {
            writeln!(out, "{:#?}", frame.array).map_err(handler_err)?;
            writeln!(out, "{:?}", frame.mask).map_err(handler_err)?;
        }
        writeln!(out).map_err(handler_err)?; // blank line
    }

    if let Some(sktilde_avg) = state.sktilde_avg.get() {
        writeln!(out, "-- sktilde_avg --").map_err(handler_err)?;
        writeln!(out, "  frame_count : {:?}", sktilde_avg.len()).map_err(handler_err)?;
        writeln!(out, "  frames_in_queue : {:?}", sktilde_avg.queue_len()).map_err(handler_err)?;
        writeln!(out, "  frame_shape : {:?}", sktilde_avg.shape()).map_err(handler_err)?;

        if let Some(frame) = sktilde_avg.last() {
            writeln!(out, "{:#?}", frame.array).map_err(handler_err)?;
            writeln!(out, "{:?}", frame.mask).map_err(handler_err)?;
        }
        writeln!(out).map_err(handler_err)?;
    }

    if let Some(bad_feed_counts) = state.bad_feed_counts.get() {
        writeln!(out, "-- bad_feed_counts --").map_err(handler_err)?;
        writeln!(out, "  frame_count : {:?}", bad_feed_counts.len()).map_err(handler_err)?;
        writeln!(out, "  frames_in_queue : {:?}", bad_feed_counts.queue_len())
            .map_err(handler_err)?;
        writeln!(out, "  frame_shape : {:?}", bad_feed_counts.shape()).map_err(handler_err)?;

        if let Some(frame) = bad_feed_counts.last() {
            let favg = frame.array.sum_axis(Axis(1));
            writeln!(out, "{favg:#?}").map_err(handler_err)?;
        }
        writeln!(out).map_err(handler_err)?;
    }

    Ok(out)
}

/// `GET /metrics` - dumps the current prometheus metrics.
///
/// Returns `500` if serialisation fails.
pub async fn metrics(
    State(metrics): State<SharedMetrics>,
) -> Result<impl IntoResponse, impl IntoResponse> {
    let mut result = serde_json::Map::new();

    // Track packets received and rejected
    let rejected_samples: u64 = metrics.packet_loss.lost();
    let total_samples: u64 = metrics.packet_loss.total();
    result.insert(
        "rejected_sample_count".into(),
        serde_json::to_value(rejected_samples).map_err(handler_err)?,
    );
    result.insert(
        "total_sample_count".into(),
        serde_json::to_value(total_samples).map_err(handler_err)?,
    );

    let first_stage: bool = metrics.rfi_zeroing.first();
    let second_stage: bool = metrics.rfi_zeroing.second();
    result.insert(
        "first_stage_enabled".into(),
        serde_json::to_value(first_stage).map_err(handler_err)?,
    );
    result.insert(
        "second_stage_enabled".into(),
        serde_json::to_value(second_stage).map_err(handler_err)?,
    );

    Ok::<_, (StatusCode, String)>(Json(result))
}

/// `GET /` - dumps the result of `bad_input_likelihood`.
///
/// Can return any error which occurs while computing the metric.
///
/// Required for external compatibility.
pub async fn dump_bad_input_likelihood(
    State(state): State<SharedDataState>,
) -> Result<String, (StatusCode, String)> {
    let metric = compute_bad_input_likelihood(&state).map_err(handler_err)?;

    let metric_fmt = metric
        .iter()
        .map(|x| format!("{x:.2}"))
        .collect::<Vec<_>>()
        .join(", ");

    Ok(format!("rfi_bad_input_mask = [{metric_fmt}]\n"))
}

/// `GET /inputs` - likelihood that any given input is corrupted.
///
/// Returns `500` if any error occurs when computing the metric.
pub async fn get_bad_input_likelihood(
    State(state): State<SharedDataState>,
) -> Result<impl IntoResponse, impl IntoResponse> {
    let metric = compute_bad_input_likelihood(&state);

    match metric {
        Ok(metric) => {
            // Package the result with its name and serialize
            let mut result = serde_json::Map::new();

            result.insert(
                "bad_input_likelihood".into(),
                serde_json::to_value(&metric).map_err(handler_err)?,
            );

            Ok::<_, (StatusCode, String)>(Json(result))
        }
        Err(e) => Err(handler_err(e)),
    }
}

/// Compute the likelihood that an input is bad, based on the `bad_feed_counts`
/// dataset in the shared state.
///
/// The likelihood metric is computed by averaging the number of "bad" feeds
/// computed by kotekan over time, then taking a median over frequency to
/// produce a single likelihood value per element. The final likelihood
/// is derived as a percentage and normalized by the number of kotekan
/// frames provided by each packet.
fn compute_bad_input_likelihood(state: &SharedDataState) -> eyre::Result<ArrayD<f32>> {
    // Grab the buffer if it exists
    let Some(buf) = &state.bad_feed_counts.get() else {
        bail!("data buffer not initialized");
    };

    let Some(arr) = &buf.stack_array(0) else {
        bail!("data buffer is empty");
    };

    let Some(mask) = &buf.stack_mask() else {
        bail!("data buffer incorrectly formatted");
    };

    // Compute the per-feed likelihood metric. This is guaranteed
    // to succeed because call to `&buf.stack` above would have
    // failed if the array was empty
    let (mean, norm) = stats::masked_mean_0th_axis(arr, mask);
    // Remove any frequencies which are entirely zero - these were
    // never received and shouldn't be included in the median
    let indices: Vec<usize> = norm
        .iter()
        .enumerate()
        .filter(|(_, v)| v.abs() > f32::EPSILON)
        .map(|(i, _)| i)
        .collect();

    let mean_reduced = mean.select(Axis(0), &indices);
    let mut median = stats::median_axis(&mean_reduced.view(), Axis(0));

    // Convert to a percentage and normalize by the number of frames per packet
    let meta = state
        .metadata
        .get()
        .ok_or_eyre("metadata is not accessible")?;
    // NB: this is what was done before, but unclear as to why
    #[allow(
        clippy::cast_precision_loss,
        reason = "values too small for precision loss"
    )]
    let norm = 100.0 / meta.lock().frames_per_packet as f32;
    median *= norm;

    Ok(median)
}
