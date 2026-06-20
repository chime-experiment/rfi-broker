//! Implements a global [`AppState`] and member states, including
//! - [`Computed`] - computed/derived quantities
//! - [`Metrics`] - application Prometheus metrics
//! - [`Buffers`] - ringbuffers for each incoming dataset

use core::sync::atomic::{AtomicU32, AtomicU64};
use std::sync::{Arc, OnceLock};

use eyre::OptionExt;
use parking_lot::Mutex;

use axum::extract::FromRef;

use prometheus_client::registry::Registry;

use crate::buffer::Buffer;
use crate::metrics;
use crate::packet::{Body, Header, Packet, packet_types};
use crate::stats;

/// Bad input likelihood lookback num samples
const BAD_INPUT_LIKELIHOOD_LOOKBACK: u16 = 32;
/// Maximum number of array frames retained in the ring buffer
const TX_BUFFER_CAPACITY: usize = 32;

/// Store for computed quantities.
///
/// Intended to be wrapped in a [`std::sync::Arc`] to be shared
/// throughout async tasks.
#[derive(Default)]
pub struct Computed {
    /// Current likelihood that a given input is bad
    pub bad_input_likelihood: stats::MovingAverage<BAD_INPUT_LIKELIHOOD_LOOKBACK>,
}

/// Store for prometheus metrics.
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
    /// Family of gauges storing fraction of flagged samples
    pub frac_flagged: metrics::LazyGaugeFamily<f32, AtomicU32>,
    /// Family of gauges storing average SK
    pub sktilde_avg: metrics::LazyGaugeFamily<f32, AtomicU32>,
}

impl Default for Metrics {
    fn default() -> Self {
        // Initialize members
        let packet_loss = metrics::SampleLossTracker::default();
        let rfi_zeroing = metrics::RFIZeroingTracker::default();
        let bad_input_likelihood = metrics::LazyGaugeFamily::<f64, AtomicU64>::new("feed_index");
        let frac_flagged = metrics::LazyGaugeFamily::<f32, AtomicU32>::new("freq_id");
        let sktilde_avg = metrics::LazyGaugeFamily::<f32, AtomicU32>::new("freq_id");
        let mut registry = Registry::default();

        // populate registry
        registry.register(
            "rfibroker_packets_received_total",
            "Total packets received",
            packet_loss.total.clone(),
        );
        registry.register(
            "rfibroker_packets_dropped_total",
            "Total packets dropped",
            packet_loss.lost.clone(),
        );
        registry.register(
            "rfibroker_rfi_zeroing_first_stage_enabled",
            "Whether or not the broker thinks the first stage excision is enabled",
            rfi_zeroing.first_stage.clone(),
        );
        registry.register(
            "rfibroker_rfi_zeroing_second_stage_enabled",
            "Whether or not the broker thinks the second stage excision is enabled",
            rfi_zeroing.second_stage.clone(),
        );
        registry.register(
            "rfibroker_bad_input_likelihood",
            "Per-element likelihood that a given feed is bad",
            bad_input_likelihood.values.clone(),
        );
        registry.register(
            "rfibroker_frac_flagged",
            "Fraction of flagged samples per frame for each frequency",
            frac_flagged.values.clone(),
        );
        registry.register(
            "rfibroker_sktilde_avg",
            "Average SK per frame for each frequency",
            sktilde_avg.values.clone(),
        );

        Self {
            registry,
            packet_loss,
            rfi_zeroing,
            bad_input_likelihood,
            frac_flagged,
            sktilde_avg,
        }
    }
}

impl Metrics {
    /// registry is not directly accessible
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
    pub frac_flagged: OnceLock<Buffer<packet_types::FracFlaggedType, TX_BUFFER_CAPACITY>>,
    pub sktilde_avg: OnceLock<Buffer<packet_types::SkType, TX_BUFFER_CAPACITY>>,
    pub skbar_avg: OnceLock<Buffer<packet_types::SkType, TX_BUFFER_CAPACITY>>,
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
                Buffer::<packet_types::FracFlaggedType, TX_BUFFER_CAPACITY>::new(vec![
                    header.num_total_freq as usize,
                ])
            })
            .push_vec(body.frac_flagged, id, &indices, axis)?;

        self.sktilde_avg
            .get_or_init(|| {
                Buffer::<packet_types::SkType, TX_BUFFER_CAPACITY>::new(vec![
                    header.num_total_freq as usize,
                ])
            })
            .push_vec(body.sktilde_avg, id, &indices, axis)?;

        self.skbar_avg
            .get_or_init(|| {
                Buffer::<packet_types::SkType, TX_BUFFER_CAPACITY>::new(vec![
                    header.num_total_freq as usize,
                    header.num_elements as usize,
                ])
            })
            .push_vec(body.skbar_avg, id, &indices, axis)?;

        // Update the metadata since we got here
        let meta = self
            .metadata
            .get()
            .ok_or_eyre("unexpected failure accessing existing metadata")?;
        *meta.lock() = header;

        Ok(id)
    }

    /// Flush all buffers - that is, push all partial frames.
    pub fn flush(&self) -> usize {
        let mut nflushed = 0;

        if let Some(buf) = self.frac_flagged.get() {
            nflushed += buf.flush();
        }
        if let Some(buf) = self.sktilde_avg.get() {
            nflushed += buf.flush();
        }
        if let Some(buf) = self.skbar_avg.get() {
            nflushed += buf.flush();
        }

        nflushed
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

    /// Test that are successfully initialized when the first packet
    /// is pushed.
    #[test]
    fn test_init_buffers() -> Result<(), Box<dyn std::error::Error>> {
        let state = Buffers::default();

        let packets = make_packets(4, 2)?;
        assert_eq!(packets.len(), 2);

        for packet in &packets {
            state.push(packet.clone())?;
        }
        // There should now be some data in the various buffers
        assert!(state.metadata.get().is_some());
        assert!(state.frac_flagged.get().is_some());
        assert!(state.sktilde_avg.get().is_some());
        assert!(state.skbar_avg.get().is_some());

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

        let skbar_avg = state.skbar_avg.get().ok_or("error getting `skbar_avg`")?;

        // Check that each buffer has the correct shape
        assert_eq!(*frac_flagged.shape(), [4]);
        assert_eq!(*sktilde_avg.shape(), [4]);
        assert_eq!(*skbar_avg.shape(), [4, 10]);

        Ok(())
    }

    /// Test that packets are successfully parsed and pushed into
    /// the corresponding [`Buffer`]s.
    #[test]
    fn test_push_packets() -> Result<(), Box<dyn std::error::Error>> {
        let state = Arc::new(Buffers::default());

        // Push some packets to the buffer to initialize
        let packets = make_packets(4, 2)?;
        for packet in &packets {
            state.push(packet.clone())?;
        }
        state.flush();

        // Check that each buffer has been initialized
        let frac_flagged = state
            .frac_flagged
            .get()
            .ok_or("error getting `frac_flagged`")?;
        let sktilde_avg = state
            .sktilde_avg
            .get()
            .ok_or("error getting `sktilde_avg`")?;
        let skbar_avg = state.skbar_avg.get().ok_or("error getting `skbar_avg`")?;

        // Check that a complete frame has been pushed to each buffer
        assert!(
            frac_flagged.last_frame().is_some(),
            "`frac_flagged` frame was not pushed"
        );
        assert!(
            sktilde_avg.last_frame().is_some(),
            "`sktilde_avg` frame was not pushed"
        );
        assert!(
            skbar_avg.last_frame().is_some(),
            "`skbar_avg` frame was not pushed"
        );

        Ok(())
    }
}
