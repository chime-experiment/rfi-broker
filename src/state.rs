//! Implements a global [`AppState`] and member states, including
//! - [`Computed`] - computed/derived quantities
//! - [`Metrics`] - application Prometheus metrics
//! - [`Buffers`] - ringbuffers for each incoming dataset

use core::sync::atomic::AtomicU64;
use std::sync::{Arc, OnceLock};

use eyre::OptionExt;
use parking_lot::Mutex;

use axum::extract::FromRef;

use prometheus_client::registry::Registry;

use crate::metrics;
use crate::packet::{Body, Header, Packet, packet_types};
use crate::ringbuffer::RingBuffer;
use crate::stats;

/// Bad input likelihood loookback num samples
const BAD_INPUT_LIKELIHOOD_LOOKBACK: u16 = 64;

/// Store for computed quantities.
#[derive(Default)]
pub struct Computed {
    /// Current likelihood that a given input is bad
    pub bad_input_likelihood: stats::MovingAverage<BAD_INPUT_LIKELIHOOD_LOOKBACK>,
}

/// Store for application metrics.
///
/// Intended to be wrapped in a [`std::sync::Arc`] to be shared
/// throughout async tasks.
pub struct Metrics {
    /// Prometheus metrics registry
    registry: Registry,
    /// Packet lost count tracker
    pub packet_loss: metrics::SampleLossTracker,
    /// Current state of RFI zeroing, according to this broker
    pub rfi_zeroing: metrics::RFIZeroingTracker,
    /// Family of gauges storing the bad input likelihood. This stores the
    /// metric as a Prometheus family, and should only copy values from
    /// the actual computed metric
    pub bad_input_likelihood: metrics::LazyGaugeFamily<f64, AtomicU64>,
}

impl Default for Metrics {
    fn default() -> Self {
        // Initialize members
        let packet_loss = metrics::SampleLossTracker::default();
        let rfi_zeroing = metrics::RFIZeroingTracker::default();
        let bad_input_likelihood = metrics::LazyGaugeFamily::<f64, AtomicU64>::default();
        let mut registry = Registry::default();

        // populate registry
        registry.register(
            "rfireceiver_packets_received_total",
            "Total packets received",
            packet_loss.total.clone(),
        );
        registry.register(
            "rfireceiver_packets_dropped_total",
            "Total packets dropped",
            packet_loss.lost.clone(),
        );
        registry.register(
            "rfireceiver_rfi_zeroing_first_stage_enabled",
            "Whether or not the receiver thinks the first stage excision is enabled",
            rfi_zeroing.first_stage.clone(),
        );
        registry.register(
            "rfireceiver_rfi_zeroing_second_stage_enabled",
            "Whether or not the receiver thinks the second stage excision is enabled",
            rfi_zeroing.second_stage.clone(),
        );
        registry.register(
            "rfireceiver_bad_input_likelihood",
            "Per-element likelihood that a given feed is bad",
            bad_input_likelihood.values.clone(),
        );

        Self {
            registry,
            packet_loss,
            rfi_zeroing,
            bad_input_likelihood,
        }
    }
}

impl Metrics {
    pub const fn registry(&self) -> &Registry {
        &self.registry
    }
}

/// Store for application data buffers.
///
/// Datasets are application specific, and matches those expected
/// in [`crate::packet::Packet`].
///
/// Buffers are created lazily - only instantiated when a packet is
/// received and parsed. This is implemented via a ``OnceLock``.
///
/// Each buffer is independently typed and locked, so reads/writes on one
/// dataset never blocks another.
#[derive(Default, Debug)]
pub struct Buffers {
    /// Fixed instance of the packet header, whose values should
    /// be set by the first valid packet
    pub metadata: OnceLock<Mutex<Header>>,
    /// Ringbuffers holding associated datasets from the
    /// packet body. Implements `Default`.
    pub frac_flagged: OnceLock<RingBuffer<packet_types::FracFlaggedType>>,
    pub sktilde_avg: OnceLock<RingBuffer<packet_types::SkTildeType>>,
    pub bad_feed_counts: OnceLock<RingBuffer<packet_types::BadFeedType>>,
}

impl Buffers {
    /// Push a packet to the state, initializing on first push.
    pub fn push(&self, packet: Packet) -> eyre::Result<u64> {
        let body: Body = packet.body;
        let header: Header = packet.header;

        // Check that the metadata is as-expected and initialize otherwise
        self.metadata
            .get_or_init(|| Mutex::new(header))
            .lock()
            .check_expected_equal(&packet.header)?;

        // Convert the frequency indices into the expected type
        let indices: Vec<usize> = body.freq_ids.iter().map(|&x| x as usize).collect();
        let id = header.seq_num.cast_unsigned();
        let axis: usize = 0;

        // Push to each ringbuffer, initializing if this is the first push
        self.frac_flagged
            .get_or_init(|| {
                RingBuffer::<packet_types::FracFlaggedType>::new(vec![
                    header.num_total_freq as usize,
                ])
            })
            .push_vec(body.frac_flagged, id, &indices, axis)?;

        self.sktilde_avg
            .get_or_init(|| {
                RingBuffer::<packet_types::SkTildeType>::new(vec![header.num_total_freq as usize])
            })
            .push_vec(body.sktilde_avg, id, &indices, axis)?;

        self.bad_feed_counts
            .get_or_init(|| {
                RingBuffer::<packet_types::BadFeedType>::new(vec![
                    header.num_total_freq as usize,
                    header.num_elements as usize,
                ])
            })
            .push_vec(body.bad_feed_counts, id, &indices, axis)?;

        // Update the metadata since we got here
        let meta = self
            .metadata
            .get()
            .ok_or_eyre("unexpected failure accessing existing metadata")?;
        *meta.lock() = header;

        Ok(id)
    }

    /// Flush all buffers - that is, push all partial frames to
    /// the buffer.
    pub fn flush(&self) -> usize {
        let mut nflushed = 0;

        if let Some(buf) = self.frac_flagged.get() {
            nflushed += buf.flush();
        }
        if let Some(buf) = self.sktilde_avg.get() {
            nflushed += buf.flush();
        }
        if let Some(buf) = self.bad_feed_counts.get() {
            nflushed += buf.flush();
        }

        nflushed
    }

    /// Clear all buffers - that is, remove all frames from
    /// the buffer.
    pub fn clear(&self) -> usize {
        let mut ncleared = 0;

        if let Some(buf) = self.frac_flagged.get() {
            ncleared += buf.clear();
        }
        if let Some(buf) = self.sktilde_avg.get() {
            ncleared += buf.clear();
        }
        if let Some(buf) = self.bad_feed_counts.get() {
            ncleared += buf.clear();
        }

        ncleared
    }
}

/// Shared application state.
///
/// Intended to be wrapped with an [`Arc`] for use with
/// [`tokio`] and [`axum`].
#[derive(Default, Clone, FromRef)]
pub struct AppState {
    /// Application metrics
    pub metrics: Arc<Metrics>,
    /// Computed quantities,
    pub computed: Arc<Computed>,
    /// Application buffers
    pub buffers: Arc<Buffers>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packet::tests::make_packets;

    /// Test that packets are successfully parsed and pushed into
    /// the corresponding [`RingBuffer`]s.
    #[test]
    fn test_push_packets() -> Result<(), Box<dyn std::error::Error>> {
        let state = Buffers::default();

        // Produce and push a couple of packets
        let packets = make_packets(4, 2)?;
        assert_eq!(packets.len(), 2);

        for packet in &packets {
            state.push(packet.clone())?;
        }

        // There should now be some data in the various buffers
        assert!(state.metadata.get().is_some());
        assert!(state.frac_flagged.get().is_some());
        assert!(state.sktilde_avg.get().is_some());
        assert!(state.bad_feed_counts.get().is_some());

        // Check that the metadata is what we expect
        let meta = state.metadata.get().ok_or("error getting metadata")?.lock();
        let header = packets
            .get(1)
            .ok_or("packets not successfully constructed")?
            .header;
        meta.check_expected_equal(&header)?;

        // Check that specific metadata values are correct
        assert_eq!(meta.num_total_freq, 4);
        assert_eq!(meta.num_local_freq, 2);
        assert_eq!(meta.num_elements, 10);

        // Check that each buffer has been initialized
        let frac_flagged = state
            .frac_flagged
            .get()
            .ok_or("error getting `frac_flagged`")?;

        let sktilde_avg = state
            .sktilde_avg
            .get()
            .ok_or("error getting `sktilde_avg`")?;

        let bad_feed_counts = state
            .bad_feed_counts
            .get()
            .ok_or("error getting `bad_feed_counts`")?;

        // Check that each buffer has the correct shape
        assert_eq!(*frac_flagged.shape(), [4]);
        assert_eq!(*sktilde_avg.shape(), [4]);
        assert_eq!(*bad_feed_counts.shape(), [4, 10]);

        // Check that a complete frame has been pushed to each buffer
        assert_eq!(frac_flagged.len(), 1);
        assert_eq!(frac_flagged.queue_len(), 0);

        assert_eq!(sktilde_avg.len(), 1);
        assert_eq!(sktilde_avg.queue_len(), 0);

        assert_eq!(bad_feed_counts.len(), 1);
        assert_eq!(bad_feed_counts.queue_len(), 0);

        Ok(())
    }
}
