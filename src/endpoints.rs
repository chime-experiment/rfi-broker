//! Axum handlers to expose data.

use axum::{Json, extract::State, http::StatusCode, response::IntoResponse};
use serde_json::json;

use ndarray::{Array, Array2, ArrayD, ArrayView, ArrayViewD, Axis, Dimension, RemoveAxis};

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

/// `GET /data` — snapshot of all dataset ring buffers.
///
/// Returns `500` if serialisation of any frame fails.
pub async fn data(State(state): State<SharedDataState>) -> impl IntoResponse {
    let mut result = serde_json::Map::new();

    // Dump all the current buffers
    if let Some(frac_flagged) = state.frac_flagged.get() {
        let len = frac_flagged.len();
        let frac_flagged = frac_flagged
            .serialize()
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

        result.insert(
            "frac_flagged".into(),
            json!({"frame_count": len, "frames": frac_flagged}),
        );
    }

    if let Some(sktilde_avg) = state.sktilde_avg.get() {
        let len = sktilde_avg.len();
        let sktilde_avg = sktilde_avg
            .serialize()
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

        result.insert(
            "sktilde_avg".into(),
            json!({"frame_count": len, "frames": sktilde_avg}),
        );
    }

    if let Some(bad_feed_counts) = state.bad_feed_counts.get() {
        let len = bad_feed_counts.len();
        let bad_feed_counts = bad_feed_counts
            .serialize()
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

        result.insert(
            "bad_feed_counts".into(),
            json!({"frame_count": len, "frames": bad_feed_counts}),
        );
    }

    Ok::<_, (StatusCode, String)>(Json(result))
}

/// `GET /metrics` - dumps the current prometheus metrics.
pub async fn metrics(State(m): State<SharedMetrics>) -> impl IntoResponse {
    let metrics = m
        .serialize()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok::<_, (StatusCode, String)>(Json(metrics))
}

/// `GET /` - dumps the result of `bad_input_likelihood`.
///
/// Required for external compatibility.
pub async fn dump_bad_input_likelihood(State(state): State<SharedDataState>) -> String {
    let metric = compute_bad_input_likelihood(&state);

    match metric {
        Ok(metric) => format!("rfi_bad_input_mask = {metric}"),
        Err(e) => e,
    }
}

/// `GET /inputs` - likelihood that any given input is corrupted.
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
                serde_json::to_value(&metric)
                    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?,
            );

            Ok::<_, (StatusCode, String)>(Json(result))
        }
        Err(e) => Err::<_, (StatusCode, String)>((StatusCode::INTERNAL_SERVER_ERROR, e)),
    }
}

/// Compute the likelihood that an input is bad, based on a [freq, input, time]
/// array.
fn compute_bad_input_likelihood(state: &SharedDataState) -> Result<ArrayD<f64>, String> {
    // Grab the buffer if it exists
    let Some(buf) = &state.bad_feed_counts.get() else {
        return Err("data buffer not initialized".into());
    };

    let Some(arr) = &buf.stack_array(None) else {
        return Err("data buffer is empty".into());
    };

    let Some(mask) = &buf.stack_mask() else {
        return Err("data buffer incorrectly formatted".into());
    };

    // Compute the per-feed likelihood metric. This is guaranteed
    // to succeed because call to `&buf.stack` above would have
    // failed if the array was empty
    let mean_val = masked_mean_first_axis(arr, mask)?;
    let mut median: ArrayD<f64> = median_axis(&mean_val.view(), Axis(0));

    // Convert to a percentage and normalize by the number of frames per packet
    let meta = state.metadata.get().ok_or("metadata is not accessible")?;
    // NB: this is what was done before, but unclear as to why
    let norm = 100.0 / f64::from(meta.lock().frames_per_packet);
    median *= norm;

    Ok(median)
}

fn masked_mean_first_axis(arr: &ArrayD<u8>, mask: &Array2<u8>) -> Result<ArrayD<u8>, String> {
    let sum: ArrayD<u8> = arr.sum_axis(Axis(arr.ndim()));
    let norm = mask.sum_axis(Axis(1));

    let mean_val: Vec<ArrayD<u8>> = sum
        .axis_iter(Axis(0))
        .zip(norm.iter())
        .filter(|(_, val)| **val > 0_u8)
        .map(|(slice, &val)| &slice / val)
        .collect();

    // Convert to views and stack
    let mean_val: Vec<ArrayViewD<u8>> = mean_val.iter().map(|x| x.view()).collect();

    ndarray::stack(Axis(0), &mean_val).map_err(|e| e.to_string())
}

fn median_axis<D>(arr: &ArrayView<u8, D>, axis: Axis) -> Array<f64, D::Smaller>
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
