//! Axum request handlers.
//!
//! Each function corresponds to one named endpoint and can be selectively
//! registered via [`crate::config::Config`].
//!
//! Handlers that read UDP data receive a clone of the [`SharedRingBuffer`] via
//! Axum's [`State`] extractor. Locks are held only for the snapshot copy, so
//! contention with the UDP writer is minimal.

use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use serde_json::json;

use ndarray::{ArrayD, ArrayView1, Axis};

use crate::datastate::SharedDataState;

/// `GET /data` — snapshot of all dataset ring buffers.
///
/// Returns `500` if serialisation of any frame fails.
pub async fn data(State(state): State<SharedDataState>) -> impl IntoResponse {
    let mut result = serde_json::Map::new();

    // Dump the metadata
    let meta = serde_json::to_value(*state.metadata.lock().unwrap())
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    result.insert("metadata".into(), meta);

    // Dump all the current buffers
    let frac_flagged = state
        .frac_flagged
        .serialize()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let sktilde_avg = state
        .sktilde_avg
        .serialize()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let bad_feed_counts = state
        .bad_feed_counts
        .serialize()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    result.insert(
        "frac_flagged".into(),
        json!({"frame_count": frac_flagged.len(), "frames": frac_flagged}),
    );
    result.insert(
        "sktilde_avg".into(),
        json!({"frame_count": sktilde_avg.len(), "frames": sktilde_avg}),
    );
    result.insert(
        "bad_feed_counts".into(),
        json!({"frame_count": bad_feed_counts.len(), "frames": bad_feed_counts}),
    );

    Ok::<_, (StatusCode, String)>(Json(result))
}

/// `GET /` - dumps the result of `bad_input_likelihood`.
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
                serde_json::to_value(&metric).unwrap(),
            );

            Ok::<_, (StatusCode, String)>(Json(result))
        }
        Err(e) => {
            Err::<_, (StatusCode, String)>((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
        }
    }
}

/// Compute the likelihood that an input is bad, based on a [freq, input, time]
/// array.
fn compute_bad_input_likelihood(
    state: &SharedDataState,
) -> Result<ArrayD<f64>, Box<dyn std::error::Error>> {
    // Grab the buffer if it exists
    let buf = &state.bad_feed_counts;

    // Stack the buffer over a trailing axis and grab the underlying [`ArrayD`]
    let Some(frame_shape): Option<&Vec<usize>> = buf.frame_shape() else {
        return Err("data buffer is not initialized".into());
    };
    let ax: usize = frame_shape.len();

    let Some(arr): &Option<ArrayD<u8>> = &buf.stack(ax) else {
        return Err("data buffer is empty".into());
    };

    // Compute the per-feed likelihood metric. This is guaranteed
    // to succeed because call to `&buf.stack` above would have
    // failed if the array was empty
    let mean_val: ArrayD<u8> = arr.mean_axis(Axis(ax)).unwrap();
    let median: Vec<f64> = mean_val
        .lanes(Axis(0))
        .into_iter()
        .map(median_of_row)
        .collect();

    // Array shape minus the first dimension
    let shape: Vec<usize> = mean_val.shape()[1..].to_vec();

    let metric: ArrayD<f64> = ArrayD::<f64>::from_shape_vec(shape, median)?;

    Ok(metric)
}

/// Helper utility to get the median of a 1D ``ArrayView``
fn median_of_row(row: ArrayView1<u8>) -> f64 {
    let mut v: Vec<u8> = row.to_vec();
    v.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let n = v.len();

    if n % 2 == 1 {
        f64::from(v[n / 2])
    } else {
        f64::from(v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}
