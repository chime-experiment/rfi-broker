//! Axum request handlers.
//!
//! Each function corresponds to one named endpoint and can be selectively
//! registered via [`crate::config::Config`].
//!
//! Handlers that read UDP data receive a clone of the [`SharedRingBuffer`] via
//! Axum's [`State`] extractor. Locks are held only for the snapshot copy, so
//! contention with the UDP writer is minimal.

use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use ndarray::Axis;
use serde_json::{json, Value};

use crate::datastate::{SharedDataState, TypedBuffer};

///// `GET /data` — snapshot of all dataset ring buffers.
///
/// Returns a JSON object keyed by dataset name, each containing `dim_names`
/// and a list of frames. Returns `500` if serialisation of any frame fails.
pub async fn data(State(store): State<SharedDataState>) -> impl IntoResponse {
    let mut result = serde_json::Map::new();

    for (name, buf) in &store.buffers {
        let frames = buf
            .serialize()
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

        result.insert(
            name.clone(),
            json!({
                "dim_names": buf.dims(),
                "frame_count": frames.len(),
                "frames": frames,
            }),
        );
    }

    Ok::<_, (StatusCode, String)>(Json(Value::Object(result)))
}

/// `GET /mean` — per-element mean of the three `f32` datasets.
///
/// Returns the mean frame for datasets `a`, `b`, and `c`. Dataset `d` (u8)
/// is excluded as mean is not defined for integer arrays.
/// Returns `null` for any dataset whose buffer is empty.
// NB: this is just an example to use when coming up with some of the
// more complicated ones
#[allow(unused)] // Get rid of annoying warnings since this isn't permanent
pub async fn mean(State(store): State<SharedDataState>) -> Json<Value> {
    let mut result = serde_json::Map::new();

    for (name, buf) in &store.buffers {
        if let TypedBuffer::F32(rb) = buf {
            // Stack over the last axis
            let ax: usize = *&rb.dims.len() - 1;
            let mean_val = &rb
                .stack(ax as i64)
                .unwrap()
                .mean_axis(Axis(ax))
                .and_then(|arr| serde_json::to_value(&arr).ok());

            result.insert(
                name.clone(),
                json!({
                    "dim_names": buf.dims(),
                    "mean": mean_val,
                }),
            );
        }
    }

    Json(Value::Object(result))
}
