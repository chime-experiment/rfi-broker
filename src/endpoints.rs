//! Axum endpoints and associated functions.
use std::fmt::Write;

use axum::{Json, extract::State, http::StatusCode, response::IntoResponse};

use eyre::{OptionExt, bail};

use ndarray::{Array, Array2, ArrayD, ArrayView, Axis, Dimension, RemoveAxis};

use crate::datastate::SharedDataState;
use crate::metrics::SharedMetrics;

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
            writeln!(out, "{frame:#?}").map_err(handler_err)?;
        }
        writeln!(out).map_err(handler_err)?; // blank line
    }

    if let Some(sktilde_avg) = state.sktilde_avg.get() {
        writeln!(out, "-- sktilde_avg --").map_err(handler_err)?;
        writeln!(out, "  frame_count : {:?}", sktilde_avg.len()).map_err(handler_err)?;
        writeln!(out, "  frames_in_queue : {:?}", sktilde_avg.queue_len()).map_err(handler_err)?;
        writeln!(out, "  frame_shape : {:?}", sktilde_avg.shape()).map_err(handler_err)?;

        if let Some(frame) = sktilde_avg.last() {
            writeln!(out, "{frame:#?}").map_err(handler_err)?;
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
            writeln!(out, "{frame:#?}").map_err(handler_err)?;
        }
        writeln!(out).map_err(handler_err)?;
    }

    Ok(out)
}

/// `GET /metrics` - dumps the current prometheus metrics.
///
/// Returns `500` if serialisation fails.
pub async fn metrics(State(m): State<SharedMetrics>) -> impl IntoResponse {
    let metrics = m.serialize().map_err(handler_err)?;

    Ok::<_, (StatusCode, String)>(Json(metrics))
}

/// `GET /` - dumps the result of `bad_input_likelihood`.
///
/// Can return any error which occurs while computing the metric.
///
/// Required for external compatibility.
pub async fn dump_bad_input_likelihood(State(state): State<SharedDataState>) -> String {
    let metric = compute_bad_input_likelihood(&state);

    match metric {
        Ok(metric) => format!("rfi_bad_input_mask = {metric}"),
        Err(e) => e.to_string(),
    }
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
fn compute_bad_input_likelihood(state: &SharedDataState) -> eyre::Result<ArrayD<f64>> {
    // Grab the buffer if it exists
    let Some(buf) = &state.bad_feed_counts.get() else {
        bail!("data buffer not initialized");
    };

    let Some(arr) = &buf.stack_array(None) else {
        bail!("data buffer is empty");
    };

    let Some(mask) = &buf.stack_mask() else {
        bail!("data buffer incorrectly formatted");
    };

    // Compute the per-feed likelihood metric. This is guaranteed
    // to succeed because call to `&buf.stack` above would have
    // failed if the array was empty
    let mean_val = masked_sum_mean(arr, mask);
    let mut median = median_axis(&mean_val.view(), Axis(0));

    // Convert to a percentage and normalize by the number of frames per packet
    let meta = state
        .metadata
        .get()
        .ok_or_eyre("metadata is not accessible")?;
    // NB: this is what was done before, but unclear as to why
    let norm = 100.0 / f64::from(meta.lock().frames_per_packet);
    median *= norm;

    Ok(median)
}

/// Compute a masked mean over the 0th axis of an [`ArrayD`].
///
/// The mask must be two-dimensional, with axes matching the first and
/// last axes of `arr`.
fn masked_sum_mean(arr: &ArrayD<u8>, mask: &Array2<u8>) -> ArrayD<f32> {
    // Max buffer length is 64, so can guarantee that accumulating
    // to u16 will not overflow
    let sum: ArrayD<u16> =
        // `as_standard_layout` is zero-copy if `arr` is c-contiguous, but
        // provides guarantees to the compiler about axis contiguousness
        arr.as_standard_layout()
            .fold_axis(Axis(arr.ndim() - 1), 0u16, |&acc, &x| acc + u16::from(x));
    // compute the norm directly as the float reciprocal
    let norm: Array2<f32> = mask
        .as_standard_layout()
        .mapv(u16::from)
        .sum_axis(Axis(1))
        .mapv(|x| {
            // Only invert if norm is non-zero. In theory, LLVM should
            // convert this to a branchless instruction
            if x == 0 {
                0.0f32
            } else {
                1.0f32 / f32::from(x)
            }
        })
        .insert_axis(Axis(1));

    sum.mapv(f32::from) * &norm
}

/// Compute the median of an array across an arbitrary axis.
fn median_axis<D>(arr: &ArrayView<f32, D>, axis: Axis) -> Array<f64, D::Smaller>
where
    D: Dimension + RemoveAxis,
{
    arr.map_axis(axis, |lane| {
        let mut v: Vec<f64> = lane.iter().map(|&x| f64::from(x)).collect();
        #[allow(
            clippy::integer_division,
            reason = "integer truncation is the desired behaviour"
        )]
        // Ensures there are at least 2 elements - if not, return early
        let mid: usize = v.len() / 2;
        if mid == 0 {
            return 0_f64;
        }

        v.select_nth_unstable_by(mid, f64::total_cmp);
        let upper = *v.get(mid).unwrap_or(&0_f64);

        if v.len().is_multiple_of(2) {
            v.select_nth_unstable_by(mid - 1, f64::total_cmp);
            f64::midpoint(*v.get(mid - 1).unwrap_or(&0_f64), upper)
        } else {
            upper
        }
    })
}
