//! Axum endpoints and associated functions.
#[cfg(debug_assertions)]
use {
    crate::buffer::{stack_buffer_array, stack_buffer_mask},
    ndarray::Axis,
    ndarray_npy::write_npy,
    serde::Deserialize,
    std::{fmt::Write, path::Path},
    tokio::time::Duration,
};

use axum::{Json, extract::State, http::StatusCode, response::IntoResponse};
use prometheus_client::encoding::text::encode;

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

/// `GET /human-metrics` - dumps metrics in a human-readable way.
///
/// Returns `500` if serialisation fails.
pub async fn human_metrics(
    State(state): State<AppState>,
) -> Result<impl IntoResponse, impl IntoResponse> {
    let mut result = serde_json::Map::new();

    // Track packets received and rejected
    let rejected_samples: u64 = state.metrics.packet_loss.lost.get();
    let total_samples: u64 = state.metrics.packet_loss.total.get();
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

/// `GET /metrics` - update and serialize Prometheus metrics.
pub async fn metrics(
    State(state): State<AppState>,
) -> Result<impl IntoResponse, impl IntoResponse> {
    // update the prometheus representation of the bad input likelihood
    if let Some(likelihood) = state.computed.bad_input_likelihood.value() {
        state
            .metrics
            .bad_input_likelihood
            .update_from_slice(&likelihood)
            .map_err(handler_err)?;
    }

    if let Some(sktilde_avg) = state.buffers.sktilde_avg.get()
        && let Some(frame) = sktilde_avg.last_frame()
        && let Some(arr) = frame.array.as_slice()
    {
        state
            .metrics
            .sktilde_avg
            .update_from_slice(arr)
            .map_err(handler_err)?;
    }

    if let Some(frac_flagged) = state.buffers.frac_flagged.get()
        && let Some(frame) = frac_flagged.last_frame()
        && let Some(arr) = frame.array.as_slice()
    {
        state
            .metrics
            .frac_flagged
            .update_from_slice(arr)
            .map_err(handler_err)?;
    }

    let mut body = String::new();
    encode(&mut body, state.metrics.registry())
        .map(|()| Ok::<_, (StatusCode, String)>((StatusCode::OK, body)))
        .map_err(handler_err)?
}

/// `GET /` - dumps the result of `bad_input_likelihood`.
///
/// Can return any error which occurs while computing the metric.
///
/// Required for external compatibility.
pub async fn dump_bad_input_likelihood(
    State(state): State<AppState>,
) -> Result<String, (StatusCode, String)> {
    let Some(metric) = state.computed.bad_input_likelihood.value() else {
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
    let Some(metric) = state.computed.bad_input_likelihood.value() else {
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

#[cfg(debug_assertions)]
/// `GET /last-frame` — snapshot most recent frame in all dataset buffers.
pub async fn last_frame(State(state): State<AppState>) -> Result<String, (StatusCode, String)> {
    let mut out = String::new();

    // Dump all the current buffers
    if let Some(frac_flagged) = state.buffers.frac_flagged.get() {
        writeln!(out, "-- frac_flagged --").map_err(handler_err)?;
        writeln!(out, "  frame_shape : {:?}", frac_flagged.shape()).map_err(handler_err)?;

        if let Some(frame) = frac_flagged.last_frame() {
            writeln!(out, "{:#?}", frame.array).map_err(handler_err)?;
            writeln!(out, "{:?}", frame.mask).map_err(handler_err)?;
        }
        writeln!(out).map_err(handler_err)?; // blank line
    }

    if let Some(sktilde_avg) = state.buffers.sktilde_avg.get() {
        writeln!(out, "-- sktilde_avg --").map_err(handler_err)?;
        writeln!(out, "  frame_shape : {:?}", sktilde_avg.shape()).map_err(handler_err)?;

        if let Some(frame) = sktilde_avg.last_frame() {
            writeln!(out, "{:#?}", frame.array).map_err(handler_err)?;
            writeln!(out, "{:?}", frame.mask).map_err(handler_err)?;
        }
        writeln!(out).map_err(handler_err)?;
    }

    if let Some(skbar_avg) = state.buffers.skbar_avg.get() {
        writeln!(out, "-- skbar_avg --").map_err(handler_err)?;
        writeln!(out, "  frame_shape : {:?}", skbar_avg.shape()).map_err(handler_err)?;

        if let Some(frame) = skbar_avg.last_frame() {
            let favg = frame.array.sum_axis(Axis(1));
            writeln!(out, "{favg:#?}").map_err(handler_err)?;
        }
        writeln!(out).map_err(handler_err)?;
    }

    Ok(out)
}

#[cfg(debug_assertions)]
/// Query arguments for `write_buffers`.
#[derive(Deserialize)]
pub struct DumpParams {
    path: String,
    nsamples: usize,
}

#[cfg(debug_assertions)]
/// `POST /write-buffers` - dump buffers into a set of .npy files.
pub async fn write_buffers(
    State(state): State<AppState>,
    Json(params): Json<DumpParams>,
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

    let n = params.nsamples;
    // large timeout in case there's any network latency
    let timeout = Duration::from_secs(2);

    if let Some(sktilde_avg) = state.buffers.sktilde_avg.get()
        && let Some(skbar_avg) = state.buffers.skbar_avg.get()
    {
        let (sktilde_vec, skbar_vec) = tokio::join!(
            sktilde_avg.accumulate(n, timeout),
            skbar_avg.accumulate(n, timeout)
        );

        // unpack and propagate errors
        let sktilde_vec = sktilde_vec.map_err(handler_err)?;
        let skbar_vec = skbar_vec.map_err(handler_err)?;

        // stack into array and mask and write out
        if let Some(arr) = stack_buffer_array(&sktilde_vec, 0)
            && let Some(mask) = stack_buffer_mask(&sktilde_vec)
        {
            let mut path = params.path.clone();
            path.push_str("/sktilde_avg.npy");
            write_npy(path, &arr).map_err(handler_err)?;
            // write the mask out as well
            let mut path = params.path.clone();
            path.push_str("/sktilde_avg_mask.npy");
            write_npy(path, &mask).map_err(handler_err)?;
        }

        if let Some(arr) = stack_buffer_array(&skbar_vec, 0)
            && let Some(mask) = stack_buffer_mask(&skbar_vec)
        {
            let mut path = params.path.clone();
            path.push_str("/skbar_avg.npy");
            write_npy(path, &arr).map_err(handler_err)?;
            // write the mask
            let mut path = params.path.clone();
            path.push_str("/skbar_avg_mask.npy");
            write_npy(path, &mask).map_err(handler_err)?;
        }
    }

    Ok::<_, (StatusCode, String)>((StatusCode::OK, params.path.clone()))
}
