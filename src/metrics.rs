//! Application metrics state, including internal and Prometheus metrics.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Tracker for a sample loss count/fraction.
// NB: it would be good for this to be a rolling metric
// or something, instead of looking at the entire duration.
#[derive(Default)]
pub struct SampleLossTracker {
    total: AtomicU64,
    lost: AtomicU64,
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

    /// Get the lost sample count
    pub fn lost(&self) -> u64 {
        self.lost.load(Ordering::Relaxed)
    }

    /// Get the total sample count
    pub fn total(&self) -> u64 {
        self.total.load(Ordering::Relaxed)
    }
}

/// Tracker for first/second-stage RFI zeroing.
#[derive(Default)]
pub struct RFIZeroingTracker {
    first_stage: AtomicBool,
    second_stage: AtomicBool,
}

impl RFIZeroingTracker {
    /// Record the state of first-stage flagging.
    pub fn set_first(&self, value: bool) {
        self.first_stage.store(value, Ordering::Release);
    }

    /// Record the state of second-stage flagging.
    pub fn set_second(&self, value: bool) {
        self.second_stage.store(value, Ordering::Release);
    }

    /// Get the value of the first stage status
    pub fn first(&self) -> bool {
        self.first_stage.load(Ordering::Relaxed)
    }

    /// Get the value of the second stage status
    pub fn second(&self) -> bool {
        self.second_stage.load(Ordering::Relaxed)
    }
}

/// Shared application state for metrics.
///
/// Intended to be wrapped in a [`std::sync::Arc`] to be shared
/// throughout async tasks.
#[derive(Default)]
pub struct Metrics {
    /// Packet lost count tracker
    pub packet_loss: SampleLossTracker,
    /// Current state of RFI zeroing, according to
    /// this broker
    pub rfi_zeroing: RFIZeroingTracker,
}

/// Alias for shared metrics type.
pub type SharedMetrics = Arc<Metrics>;

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

        let frac = tracker.lost() as f64 / tracker.total() as f64;

        // Check that the fraction is lost is as expected,
        // within a tolerance
        assert!(
            (frac - 0.2).abs() < 1.0e-6,
            "`frac_lost`={frac} is not within tolerance `1.0e-6` of expectation=`0.2`"
        );
    }
}
