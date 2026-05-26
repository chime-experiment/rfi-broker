//! Application metrics and underlying implementations, including internal
//! and Prometheus metrics.
use std::sync::OnceLock;

use eyre::bail;

use prometheus_client::{
    encoding::EncodeLabelSet,
    metrics::{
        counter::Counter,
        family::Family,
        gauge::{Atomic, Gauge},
    },
};

/// Tracker for a sample loss count/fraction.
#[derive(Default)]
pub struct SampleLossTracker {
    pub total: Counter,
    pub lost: Counter,
}

impl SampleLossTracker {
    /// Record a received sample.
    pub fn inc_recv(&self) {
        self.total.inc();
    }

    /// Record a lost sample.
    ///
    /// Increments the total number of samples internally.
    pub fn inc_lost(&self) {
        // Increment both counters
        self.inc_recv();
        self.lost.inc();
    }
}

/// Tracker for first/second-stage RFI zeroing.
#[derive(Default)]
pub struct RFIZeroingTracker {
    pub first_stage: Gauge,
    pub second_stage: Gauge,
}

impl RFIZeroingTracker {
    /// Record the state of first-stage flagging.
    pub fn set_first(&self, value: bool) {
        self.first_stage.set(i64::from(value));
    }

    /// Record the state of second-stage flagging.
    pub fn set_second(&self, value: bool) {
        self.second_stage.set(i64::from(value));
    }

    /// Get the value of the first stage status
    pub fn first(&self) -> bool {
        self.first_stage.get() != 0
    }

    /// Get the value of the second stage status
    pub fn second(&self) -> bool {
        self.second_stage.get() != 0
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct IndexLabel {
    pub index: usize,
}

/// Lazy tracker for a family of Gauges.
#[derive(Debug, Default)]
pub struct LazyGaugeFamily<T, A>
where
    Gauge<T, A>: Clone,
{
    pub values: Family<IndexLabel, Gauge<T, A>>,
    handles: OnceLock<Vec<Gauge<T, A>>>,
}

impl<T, A> LazyGaugeFamily<T, A>
where
    T: Copy,
    A: Atomic<T>,
    Gauge<T, A>: Clone,
{
    pub fn sync_from_slice(&self, values: &[T]) -> eyre::Result<()> {
        let handles = self.handles.get_or_init(|| {
            (0..values.len())
                .map(|index| self.values.get_or_create(&IndexLabel { index }).clone())
                .collect()
        });

        if handles.len() != values.len() {
            bail!(
                "received unexpected number of values: {} != {}",
                values.len(),
                handles.len()
            );
        }

        for (gauge, value) in handles.iter().zip(values.iter().copied()) {
            gauge.set(value);
        }

        Ok(())
    }
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

        #[allow(
            clippy::cast_precision_loss,
            reason = "values are too small for precision loss"
        )]
        let frac = tracker.lost.get() as f64 / tracker.total.get() as f64;

        // Check that the fraction is lost is as expected,
        // within a tolerance
        assert!(
            (frac - 0.2).abs() < 1.0e-6,
            "`frac_lost`={frac} is not within tolerance `1.0e-6` of expectation=`0.2`"
        );
    }
}
