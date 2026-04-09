//! Axum handlers to expose data.

use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use serde_json::json;

use ndarray::{Array, ArrayD, ArrayView, Axis, Dimension, RemoveAxis};

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

    // TODO: get this working somehow
    // Add the expected node count from [`Metrics`], if it exists
    // if let Some(node_count) = metrics.expected_node_count {
    //     meta["expected_node_count"] = json!(node_count);
    // }

    Ok::<_, (StatusCode, String)>(Json(meta))
}

/// `GET /data` — snapshot of all dataset ring buffers.
///
/// Returns `500` if serialisation of any frame fails.
pub async fn data(State(state): State<SharedDataState>) -> impl IntoResponse {
    let mut result = serde_json::Map::new();

    // Dump all the current buffers
    if let Some(frac_flagged) = state.frac_flagged.get() {
        let frac_flagged = frac_flagged
            .serialize()
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

        result.insert(
            "frac_flagged".into(),
            json!({"frame_count": frac_flagged.len(), "frames": frac_flagged}),
        );
    }

    if let Some(sktilde_avg) = state.sktilde_avg.get() {
        let sktilde_avg = sktilde_avg
            .serialize()
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

        result.insert(
            "sktilde_avg".into(),
            json!({"frame_count": sktilde_avg.len(), "frames": sktilde_avg}),
        );
    }

    if let Some(bad_feed_counts) = state.bad_feed_counts.get() {
        let bad_feed_counts = bad_feed_counts
            .serialize()
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

        result.insert(
            "bad_feed_counts".into(),
            json!({"frame_count": bad_feed_counts.len(), "frames": bad_feed_counts}),
        );
    }

    Ok::<_, (StatusCode, String)>(Json(result))
}

/// `GET /metrics` - dumps the current prometheus metrics.
pub async fn metrics(State(m): State<SharedMetrics>) -> impl IntoResponse {
    Ok::<_, (StatusCode, String)>(Json(m.render()))
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

    let Some(arr) = &buf.stack(None) else {
        return Err("data buffer is empty".into());
    };

    // Compute the per-feed likelihood metric. This is guaranteed
    // to succeed because call to `&buf.stack` above would have
    // failed if the array was empty
    let mean_val: ArrayD<u8> = arr.mean_axis(Axis(arr.ndim())).unwrap();

    // let median: ArrayD<u8> = mean_val.quantile_axis_mut(Axis(0), 0.5, &Linear)?;
    let mut median: ArrayD<f64> = median_axis(&mean_val.view(), Axis(0));

    // Convert to a percentage and normalize by the number of frames per packet
    // NB: this is what was done before, but unclear as to why
    // #[allow(clippy::cast_precision_loss)]
    let norm = 100.0 / f64::from(state.metadata.get().unwrap().lock().frames_per_packet);
    median *= norm;

    Ok(median)
}

fn median_axis<D>(arr: &ArrayView<u8, D>, axis: Axis) -> Array<f64, D::Smaller>
where
    D: Dimension + RemoveAxis,
{
    arr.map_axis(axis, |lane| {
        let mut v: Vec<f64> = lane.iter().map(|&x| f64::from(x)).collect();
        let mid = v.len() / 2;

        v.select_nth_unstable_by(mid, f64::total_cmp);
        let upper = v[mid];

        if v.len().is_multiple_of(2) {
            v.select_nth_unstable_by(mid - 1, f64::total_cmp);
            f64::midpoint(v[mid - 1], upper)
        } else {
            upper
        }
    })
}
