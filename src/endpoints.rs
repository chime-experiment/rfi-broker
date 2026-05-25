//! Axum endpoints and associated functions.
#[cfg(debug_assertions)]
use {
    axum::extract::Query,
    ndarray::Axis,
    std::{fmt::Write, path::Path},
};

use axum::{Json, extract::State, http::StatusCode, response::IntoResponse};

#[cfg(debug_assertions)]
use {ndarray_npy::write_npy, serde::Deserialize};

use crate::state::AppState;

/// Return an error as an ``INTERNAL_SERVER_ERROR``.
#[allow(
    clippy::needless_pass_by_value,
    reason = "error will always be consumed"
)]
fn handler_err(e: impl ToString) -> (StatusCode, String) {
    (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
}

/// `GET /meta` - snapshot of state metadata.
///
/// Returns `500` if serialisation fails.
pub async fn metadata(State(state): State<AppState>) -> impl IntoResponse {
    let Some(meta) = state.buffers.metadata.get() else {
        return Err::<_, (StatusCode, String)>((
            StatusCode::NO_CONTENT,
            "metadata not available".into(),
        ));
    };

    let meta: serde_json::Value = serde_json::to_value(*meta.lock()).map_err(handler_err)?;

    Ok::<_, (StatusCode, String)>(Json(meta))
}

/// `GET /metrics` - dumps the current prometheus metrics.
///
/// Returns `500` if serialisation fails.
pub async fn metrics(
    State(state): State<AppState>,
) -> Result<impl IntoResponse, impl IntoResponse> {
    let mut result = serde_json::Map::new();

    // Track packets received and rejected
    let rejected_samples: u64 = state.metrics.packet_loss.lost();
    let total_samples: u64 = state.metrics.packet_loss.total();
    result.insert(
        "rejected_sample_count".into(),
        serde_json::to_value(rejected_samples).map_err(handler_err)?,
    );
    result.insert(
        "total_sample_count".into(),
        serde_json::to_value(total_samples).map_err(handler_err)?,
    );

    let first_stage: bool = state.metrics.rfi_zeroing.first();
    let second_stage: bool = state.metrics.rfi_zeroing.second();
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
    State(state): State<AppState>,
) -> Result<String, (StatusCode, String)> {
    let Some(metric) = state.metrics.bad_input_likelihood.value() else {
        return Err(handler_err("data buffer not initialized"));
    };

    let metric_fmt = metric
        .iter()
        // the caller for this endpoint assumes a percentage value,
        // while the likelihood is computed from 0.0 to 1.0.
        .map(|x| format!("{:.2}", 100.0 * x))
        .collect::<Vec<_>>()
        .join(", ");

    Ok(format!("rfi_bad_input_mask = [{metric_fmt}]\n"))
}

/// `GET /inputs` - likelihood that any given input is corrupted.
///
/// Returns `500` if any error occurs when computing the metric.
pub async fn get_bad_input_likelihood(
    State(state): State<AppState>,
) -> Result<impl IntoResponse, impl IntoResponse> {
    let Some(metric) = state.metrics.bad_input_likelihood.value() else {
        return Err(handler_err("data buffer not initialized"));
    };

    // Package the result with its name and serialize
    let mut result = serde_json::Map::new();

    result.insert(
        "bad_input_likelihood".into(),
        serde_json::to_value(&metric).map_err(handler_err)?,
    );

    Ok::<_, (StatusCode, String)>(Json(result))
}

/// `GET /last-frame` — snapshot most recent frame in all dataset ring buffers.
///
/// Only exists in debug builds
#[cfg(debug_assertions)]
pub async fn last_frame(State(state): State<AppState>) -> Result<String, (StatusCode, String)> {
    let mut out = String::new();

    // Dump all the current buffers
    if let Some(frac_flagged) = state.buffers.frac_flagged.get() {
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

    if let Some(sktilde_avg) = state.buffers.sktilde_avg.get() {
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

    if let Some(bad_feed_counts) = state.buffers.bad_feed_counts.get() {
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

/// `GET /write-data` - dump buffers into a set of .npy files.
///
/// Only available in debug builds.
#[cfg(debug_assertions)]
#[derive(Deserialize)]
pub struct DumpParams {
    path: String,
}

#[cfg(debug_assertions)]
pub async fn write_buffers(
    Query(params): Query<DumpParams>,
    State(state): State<AppState>,
) -> Result<impl IntoResponse, impl IntoResponse> {
    // validate the provided path
    let path = Path::new(&params.path);

    if !path.is_absolute() {
        return Err(handler_err(format!(
            "path must be absolute - got {}",
            path.display()
        )));
    }

    if !path.is_dir() {
        return Err(handler_err(format!(
            "path must be a directory - got {}",
            path.display()
        )));
    }

    if !path.exists() {
        return Err(handler_err(format!(
            "path must exist - got {}",
            path.display()
        )));
    }
    if let Some(sktilde_avg) = state.buffers.sktilde_avg.get()
        && let Some(arr) = sktilde_avg.stack_array(0)
        && let Some(mask) = sktilde_avg.stack_mask()
    {
        let mut path = params.path.clone();
        path.push_str("/sktilde_avg.npy");
        write_npy(path, &arr).map_err(handler_err)?;
        // write the mask out as well
        let mut path = params.path.clone();
        path.push_str("/sktilde_avg_mask.npy");
        write_npy(path, &mask).map_err(handler_err)?;
    }

    if let Some(bad_feed_counts) = state.buffers.bad_feed_counts.get()
        && let Some(arr) = bad_feed_counts.stack_array(0)
        && let Some(mask) = bad_feed_counts.stack_mask()
    {
        let mut path = params.path.clone();
        path.push_str("/bad_feed_counts.npy");
        write_npy(path, &arr).map_err(handler_err)?;
        // write the mask
        let mut path = params.path.clone();
        path.push_str("/bad_feed_counts_mask.npy");
        write_npy(path, &mask).map_err(handler_err)?;
    }

    Ok::<_, (StatusCode, String)>((StatusCode::OK, params.path.clone()))
}
