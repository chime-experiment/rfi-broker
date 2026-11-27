//! Application metrics and underlying implementations, including internal
//! and Prometheus metrics.

use core::sync::atomic::{AtomicU32, AtomicU64};

use prometheus_client::registry::Registry;

use super::metric_types::{LazyGaugeFamily, RFIZeroingTracker, SampleLossTracker};
use rfi_broker::core::stats;

/// Bad input likelihood lookback num samples
const BAD_INPUT_LIKELIHOOD_LOOKBACK: u16 = 384;

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
    pub packet_loss: SampleLossTracker,
    /// Current state of RFI zeroing, according to this broker
    pub rfi_zeroing: RFIZeroingTracker,
    /// Family of gauges storing the bad input likelihood. This stores the
    /// metric as a Prometheus family, and should only copy values from
    /// the actual computed metric
    pub bad_input_likelihood: LazyGaugeFamily<f64, AtomicU64>,
    /// Family of gauges storing fraction of flagged samples
    pub frac_flagged: LazyGaugeFamily<f32, AtomicU32>,
    /// Family of gauges storing average SK
    pub sktilde_avg: LazyGaugeFamily<f32, AtomicU32>,
}

impl Default for Metrics {
    fn default() -> Self {
        // Initialize members
        let packet_loss = SampleLossTracker::default();
        let rfi_zeroing = RFIZeroingTracker::default();
        let bad_input_likelihood = LazyGaugeFamily::<f64, AtomicU64>::new("feed_index");
        let frac_flagged = LazyGaugeFamily::<f32, AtomicU32>::new("freq_id");
        let sktilde_avg = LazyGaugeFamily::<f32, AtomicU32>::new("freq_id");
        let mut registry = Registry::default();

        // populate registry
        registry.register(
            "rfibroker_packets_received",
            "Total packets received",
            packet_loss.total.clone(),
        );
        registry.register(
            "rfibroker_packets_dropped",
            "Total packets dropped",
            packet_loss.lost.clone(),
        );
        registry.register(
            "rfibroker_packets_skipped",
            "Total packets skipped",
            packet_loss.skipped.clone(),
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
