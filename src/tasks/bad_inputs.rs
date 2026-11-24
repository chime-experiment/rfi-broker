//! Task to update the `bad_input_likelihood` metric

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::broadcast::error::RecvError;

use ndarray::Ix2;

use eyre::bail;

use crate::state::{Buffers, Computed};
use rfi_broker::core::stats::sk_fisher_chi2;

/// Task to update the `bad_input_likelihood` metric every time a
/// new frame is generated.
pub async fn bad_input_task(buffers: Arc<Buffers>, computed: Arc<Computed>) -> eyre::Result<()> {
    // Subscribe to the correct state buffer, waiting until some data exists
    let buf = loop {
        if let Some(buf) = buffers.skbar_avg.get() {
            break buf;
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    };
    let mut event_rx = buf.subscribe();

    // Wait for new data frames to arrive and handle accordingly
    loop {
        let frame = match event_rx.recv().await {
            Ok(frame) => frame,
            Err(RecvError::Lagged(skipped)) => {
                tracing::debug!("receive buffer overflowed = {skipped} frames were missed");
                continue;
            }
            Err(RecvError::Closed) => {
                bail!("frame sender connection closed!");
            }
        };

        // Convert the dynamically-sized array into a 2D array, which should be
        // guaranteed to succeed
        let Some(arr) = frame.array
            .view()
            .into_dimensionality::<Ix2>()
            .inspect_err(|err| tracing::error!(error = ?err, "failed to cast expected 2D array into 2 dimensions: {err}"))
            .ok()
        else {
            continue
        };
        // sort out how many frequencies were received
        let k: u16 = frame.mask.iter().map(|&b| u16::from(b)).sum();

        // compute the likelihood that a given feed is bad
        let update_val = sk_fisher_chi2::<f32>(&arr, f32::from(k));

        // Update the exponentially-weighted moving average
        if let Some(sl) = update_val.as_slice() {
            computed
                .bad_input_likelihood
                .update(sl)
                .inspect_err(
                    |err| tracing::warn!(error = ?err, "failed to update bad input metric: {err}"),
                )
                .ok();
        } else {
            tracing::info!("got an empty result");
        }
    }
}
