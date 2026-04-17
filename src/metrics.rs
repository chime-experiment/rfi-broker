//! Application metrics state, including internal and Prometheus metrics.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use prometheus::{Gauge, GaugeVec, Opts, Registry, TextEncoder};

use crate::datastate::SharedDataState;

/// Tracker for a sample loss count/fraction.
// NB: it would be good for this to be a rolling metric
// or something, instead of looking at the entire duration.
#[derive(Default)]
pub struct SampleLossTracker {
    total: Arc<AtomicU64>,
    lost: Arc<AtomicU64>,
}

impl SampleLossTracker {
    /// Record a received sample.
    pub fn inc_recv(&self) {
        self.total.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a lost sample.
    ///
    /// Increments the total number of samples internally.
    pub fn inc_lost(&self) {
        // Increment both counters
        self.inc_recv();
        self.lost.fetch_add(1, Ordering::Relaxed);
    }

    #[allow(
        clippy::cast_precision_loss,
        reason = "expected value range is below value for truncation"
    )]
    /// Compute the fraction of lost samples for the entire duration
    /// during which this metric has been recorded.
    pub fn frac_lost(&self) -> f64 {
        let total = self.total.load(Ordering::Relaxed) as f64;
        let lost = self.lost.load(Ordering::Relaxed) as f64;

        if total < f64::EPSILON {
            0.0
        } else {
            lost / total
        }
    }
}

/// Tracker for first/second-stage RFI zeroing.
#[derive(Default)]
pub struct RFIZeroingTracker {
    first_stage: Arc<AtomicBool>,
    second_stage: Arc<AtomicBool>,
}

impl RFIZeroingTracker {
    /// Record the state of first-stage flagging.
    pub fn set_first(&self, value: bool) {
        self.first_stage.store(value, Ordering::Relaxed);
    }

    /// Record the state of second-stage flagging.
    pub fn set_second(&self, value: bool) {
        self.second_stage.store(value, Ordering::Relaxed);
    }
}

/// Shared application state for metrics.
///
/// Intended to be wrapped in a [`std::sync::Arc`] to be shared
/// throughout async tasks.
pub struct Metrics {
    /// # Prometheus metrics
    /// Prometheus registry
    registry: Registry,
    /// Fraction of flagged samples per frequency
    frac_flagged_prom: GaugeVec,
    /// Average SK value per frequency
    sktilde_prom: GaugeVec,
    /// Dropped packet fraction
    packet_loss_prom: Gauge,
    /// # Externally-visible metrics handlers
    /// Packet lost count tracker
    pub packet_loss: SampleLossTracker,
    /// Current state of RFI zeroing, according to
    /// this broker
    pub rfi_zeroing: RFIZeroingTracker,
}

/// Alias for shared metrics type.
pub type SharedMetrics = Arc<Metrics>;

impl Default for Metrics {
    /// Creates and registers all metrics into a [`Register`]
    #[allow(clippy::unwrap_used, reason = "unwrap is guaranteed to succeed")]
    fn default() -> Self {
        let registry = Registry::new();

        let frac_flagged_prom = GaugeVec::new(
            Opts::new(
                "rfi_receiver_kotekan_first_stage_frac_flagged",
                "Fraction of RFI samples flagged by kotekan in the first-stage excision.",
            ),
            &["freq_index"],
        )
        .unwrap();

        let sktilde_prom = GaugeVec::new(
            Opts::new(
                "rfi_receiver_kotekan_sktilde_avg",
                "Feed-averaged Spectral Kurtosis integrated over ~1.2 seconds.",
            ),
            &["freq_index"],
        )
        .unwrap();

        let packet_loss_prom = Gauge::new(
            "rfi_receiver_lost_packets_frac",
            "Fraction of dropped or mishandled packets, not including those dropped at the OS level."
        ).unwrap();

        registry
            .register(Box::new(frac_flagged_prom.clone()))
            .unwrap();
        registry.register(Box::new(sktilde_prom.clone())).unwrap();
        registry
            .register(Box::new(packet_loss_prom.clone()))
            .unwrap();

        Self {
            registry,
            frac_flagged_prom,
            sktilde_prom,
            packet_loss_prom,
            packet_loss: SampleLossTracker::default(),
            rfi_zeroing: RFIZeroingTracker::default(),
        }
    }
}

impl Metrics {
    /// Render prometheus metrics
    pub fn serialize(&self) -> Result<String, prometheus::Error> {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();

        encoder.encode_to_string(&metric_families)
    }
}

/// Update metrics based on a [`SharedDataState`].
///
/// Only metrics which are trivial to compute should be updated here.
///
/// This is independent of the trigger mechanism - it should be
/// called from an async function run using ``tokio::spawn``.
pub fn update_metrics(metrics: &SharedMetrics, state: &SharedDataState) {
    // Use just the most recent frame
    if let Some(frac_flagged) = state.frac_flagged.get()
        && let Some(frame) = frac_flagged.last()
    {
        // Iterate frequencies and update for each
        for (label, val) in frame
            .array
            .iter()
            .enumerate()
            .map(|(i, val)| (i.to_string(), val))
        {
            metrics
                .frac_flagged_prom
                .with_label_values(&[&label])
                .set(f64::from(*val));
        }
    }

    if let Some(sktilde) = state.sktilde_avg.get()
        && let Some(frame) = sktilde.last()
    {
        for (label, val) in frame
            .array
            .iter()
            .enumerate()
            .map(|(i, val)| (i.to_string(), val))
        {
            metrics
                .sktilde_prom
                .with_label_values(&[&label])
                .set(f64::from(*val));
        }
    }

    metrics
        .packet_loss_prom
        .set(metrics.packet_loss.frac_lost());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// Test that the [`SampleLossTracker`] produces the expected result.
    fn test_sample_loss() {
        let tracker = SampleLossTracker::default();

        for _ in 0..8 {
            tracker.inc_recv();
        }

        for _ in 0..2 {
            tracker.inc_lost();
        }

        let frac = tracker.frac_lost();

        // Check that the fraction is lost is as expected,
        // within a tolerance
        assert!(
            (frac - 0.2).abs() < 1.0e-6,
            "`frac_lost`={frac} is not within tolerance `1.0e-6` of expectation=`0.2`"
        );
    }
}
