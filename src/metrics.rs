//! Application metrics and underlying implementations, including internal
//! and Prometheus metrics.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use parking_lot::Mutex;

use eyre::ensure;

/// Tracker for a sample loss count/fraction.
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
pub struct MovingAverage<const N: u16> {
    alpha: f64,
    ialpha: f64,
    value: Mutex<Option<Vec<f64>>>,
}

impl<const N: u16> Default for MovingAverage<N> {
    fn default() -> Self {
        let alpha = 2f64 / (f64::from(N) + 1.0);
        Self {
            alpha,
            ialpha: 1.0 - alpha,
            value: Mutex::new(None),
        }
    }
}

impl<const N: u16> MovingAverage<N> {
    /// Return the current value
    pub fn value(&self) -> Option<Vec<f64>> {
        self.value.lock().clone()
    }

    /// Update the current value.
    ///
    /// If this is the first sample, the value will be
    /// equal to this sample.
    pub fn update(&self, sample: &[f64]) -> eyre::Result<()> {
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
