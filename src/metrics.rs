//! Application metrics state, including internal and Prometheus metrics.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use parking_lot::Mutex;

use eyre::ensure;

/// Bad input likelihood loookback num samples
const BAD_INPUT_LIKELIHOOD_LOOKBACK: u16 = 256;

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

/// Implementation of an exponentially-weighted moving
/// average for independent values in a Vec.
pub struct Ewma<const N: u16> {
    alpha: f32,
    ialpha: f32,
    value: Mutex<Option<Vec<f32>>>,
}

impl<const N: u16> Default for Ewma<N> {
    fn default() -> Self {
        let alpha = 2f32 / (f32::from(N) + 1.0);
        Self {
            alpha,
            ialpha: 1.0 - alpha,
            value: Mutex::new(None),
        }
    }
}

impl<const N: u16> Ewma<N> {
    /// Return the current value
    pub fn value(&self) -> Option<Vec<f32>> {
        self.value.lock().clone()
    }

    /// Update the current value.
    ///
    /// If this is the first sample, the value will be
    /// equal to this sample.
    pub fn update(&self, sample: &[f32]) -> eyre::Result<()> {
        let mut guard = self.value.lock();

        if let Some(value) = guard.as_mut() {
            ensure!(
                value.len() == sample.len(),
                "length mismatch: expected {} got {}",
                value.len(),
                sample.len()
            );
            value
                .iter_mut()
                .zip(sample.iter())
                .for_each(|(v, s)| *v = self.ialpha.mul_add(*v, self.alpha * *s));
        } else {
            *guard = Some(sample.to_vec());
        }

        Ok(())
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
    /// Current likelihood that a given input is bad
    pub bad_input_likelihood: Ewma<BAD_INPUT_LIKELIHOOD_LOOKBACK>,
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

        #[allow(
            clippy::cast_precision_loss,
            reason = "values are too small for precision loss"
        )]
        let frac = tracker.lost() as f64 / tracker.total() as f64;

        // Check that the fraction is lost is as expected,
        // within a tolerance
        assert!(
            (frac - 0.2).abs() < 1.0e-6,
            "`frac_lost`={frac} is not within tolerance `1.0e-6` of expectation=`0.2`"
        );
    }
}
