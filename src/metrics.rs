//! Prometheus metrics.

use std::sync::Arc;

use prometheus::{GaugeVec, Opts, Registry, TextEncoder};

use crate::datastate::SharedDataState;

/// Metrics store
pub struct Metrics {
    /// # Exposed metrics
    /// Prometheus registry
    pub registry: Registry,
    /// Fraction of flagged samples per frequency
    pub frac_flagged: GaugeVec,
    /// Average SK value per frequency
    pub sktilde: GaugeVec,
}

/// Alias for shared metrics type
pub type SharedMetrics = Arc<Metrics>;

impl Metrics {
    /// Creates and registers all metrics into a [`Register`]
    pub fn new() -> Self {
        let registry = Registry::new();

        let frac_flagged = GaugeVec::new(
            Opts::new(
                "rfi_receiver_kotekan_first_stage_frac_flagged",
                "Fraction of RFI samples flagged by kotekan in the first-stage excision.",
            ),
            &["freq_index"],
        )
        .unwrap();

        let sktilde = GaugeVec::new(
            Opts::new(
                "rfi_receiver_kotekan_sktilde_avg",
                "Feed-averaged Spectral Kurtosis integrated over ~1.2 seconds.",
            ),
            &["freq_index"],
        )
        .unwrap();

        registry.register(Box::new(frac_flagged.clone())).unwrap();
        registry.register(Box::new(sktilde.clone())).unwrap();

        Self {
            registry,
            frac_flagged,
            sktilde,
        }
    }

    /// Render all metrics
    pub fn render(&self) -> String {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        encoder
            .encode_to_string(&metric_families)
            .unwrap_or_else(|e| format!("error encoding metrics: {e}\n"))
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Update computationally-simple metrics based on a [`SharedDataState`].
///
/// Only metrics which are trivial to compute should be updated here.
///
/// This is independent of the trigger mechanism - it should be
/// called from an async function run using ``tokio::spawn``.
pub fn update_basic_metrics(metrics: &SharedMetrics, state: &SharedDataState) {
    // Use just the most recent frame
    if let Some(frac_flagged) = state.frac_flagged.last() {
        // Iterate frequencies and update for each
        for (ii, val) in frac_flagged.array.iter().enumerate() {
            let label: String = ii.to_string();
            metrics
                .frac_flagged
                .with_label_values(&[&label])
                .set(f64::from(*val));
        }
    }

    if let Some(sktilde) = state.sktilde_avg.last() {
        for (ii, val) in sktilde.array.iter().enumerate() {
            let label: String = ii.to_string();
            metrics
                .sktilde
                .with_label_values(&[&label])
                .set(f64::from(*val));
        }
    }
}

/// Update all metrics based on a [`SharedDataState`].
///
/// This is independent of the trigger mechanism.
#[allow(dead_code)]
pub fn update_extra_metrics(metrics: &SharedMetrics, state: &SharedDataState) {
    // NB: any other metric updates can go here. This is intended to decouple
    // the trivial updates, which can happen frequently, from anything more
    // complicated (as-needed).
    // NB: for example, if we wanted to expose the bad_input_likelihood as a metric,
    // we would probably want it to be computed at a lower cadence.
    update_basic_metrics(metrics, state);
}
