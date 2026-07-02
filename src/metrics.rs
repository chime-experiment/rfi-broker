//! Application metrics and underlying implementations, including internal
//! and Prometheus metrics.
use std::fmt::Write;
use std::sync::OnceLock;

use eyre::bail;
use num_traits::Float;
use prometheus_client::{
    encoding::{EncodeLabelSet, LabelSetEncoder},
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

/// Single metric label for a [`Family`] of metric types.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct IndexLabel {
    name: &'static str,
    index: usize,
}

impl EncodeLabelSet for IndexLabel {
    /// Custom encoder to handle label name and index values.
    fn encode(&self, encoder: &mut LabelSetEncoder<'_>) -> Result<(), std::fmt::Error> {
        let mut label = encoder.encode_label();

        let mut key = label.encode_label_key()?;
        key.write_str(self.name)?;

        let mut value = key.encode_label_value()?;
        value.write_str(&self.index.to_string())?;

        value.finish()
    }
}

/// Family of Gauges which are initialized lazily.
///
/// Tracks a family of Gauges with a single label name and label
/// values corresponding to index of each Gauge value in a provided
/// slice. Handles are not actually created until the first update
/// slice is received.
#[derive(Debug, Default)]
pub struct LazyGaugeFamily<T, A>
where
    T: Float,
    Gauge<T, A>: Clone,
{
    label_name: &'static str,
    pub values: Family<IndexLabel, Gauge<T, A>>,
    handles: OnceLock<Vec<Gauge<T, A>>>,
}

impl<T, A> LazyGaugeFamily<T, A>
where
    T: Float,
    Gauge<T, A>: Clone,
    Family<IndexLabel, Gauge<T, A>>: Default,
{
    pub fn new(label_name: &'static str) -> Self {
        Self {
            label_name,
            values: Family::<IndexLabel, Gauge<T, A>>::default(),
            handles: OnceLock::new(),
        }
    }
}

impl<T, A> LazyGaugeFamily<T, A>
where
    T: Float + Copy,
    A: Atomic<T>,
    Gauge<T, A>: Clone,
{
    /// Update the value of each Gauge based on values contained
    /// in a slice.
    ///
    /// Gauges are created the first time this is called, and all
    /// subsequent calls must provide a slice with the same length
    /// as the first call.
    ///
    /// Uses a boolean mask to determine if the update value is valid.
    /// Invalid values are set to `f64::NAN`.
    pub fn update_from_slice(&self, values: &[T], mask: Option<&[bool]>) -> eyre::Result<()> {
        let name = self.label_name;

        let handles = self.handles.get_or_init(|| {
            (0..values.len())
                .map(|index| {
                    self.values
                        .get_or_create(&IndexLabel { name, index })
                        .clone()
                })
                .collect()
        });

        if handles.len() != values.len() {
            bail!(
                "received unexpected number of values: {} != {}",
                values.len(),
                handles.len()
            );
        }

        // Update gauge values. This is the only computation that happens
        // beyond the first call
        match mask {
            Some(mask) => {
                // validate slice lengths
                if values.len() != mask.len() {
                    bail!(
                        "values and mask slices have different lengths: {} != {}",
                        values.len(),
                        mask.len()
                    );
                }
                // write NaN for values without a real sample
                for ((gauge, value), valid) in
                    handles.iter().zip(values.iter().copied()).zip(mask.iter())
                {
                    let v: T = if *valid { value } else { T::nan() };
                    gauge.set(v);
                }
            }
            None => {
                for (gauge, value) in handles.iter().zip(values.iter().copied()) {
                    gauge.set(value);
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use core::sync::atomic::AtomicU32;

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

    #[test]
    /// Test that `LazyGaugeFamily` metrics update correctly.
    fn test_update_from_slice() -> Result<(), Box<dyn std::error::Error>> {
        let gauge_family = LazyGaugeFamily::<f32, AtomicU32>::new("id");

        let values: Vec<f32> = vec![3.3, 1.2, 7.9];
        let mask: Vec<bool> = vec![true, false, true];

        // push the unmasked values and check
        gauge_family.update_from_slice(&values, None)?;
        for (i, v) in values.iter().enumerate() {
            let label = IndexLabel {
                name: "id",
                index: i,
            };
            if let Some(gauge) = gauge_family.values.get(&label) {
                assert_abs_diff_eq!(*v, gauge.get(), epsilon = 1e-3);
            } else {
                return Err(format!("failed to get gauge with index {i}").into());
            }
        }

        // repeat with a mask
        // push the unmasked values and check
        gauge_family.update_from_slice(&values, Some(&mask))?;
        for ((i, v), m) in values.iter().enumerate().zip(mask.iter()) {
            let label = IndexLabel {
                name: "id",
                index: i,
            };
            if let Some(gauge) = gauge_family.values.get(&label) {
                if *m {
                    assert_abs_diff_eq!(*v, gauge.get(), epsilon = 1e-3);
                } else {
                    assert!(gauge.get().is_nan(), "expected NaN, got {:?}", gauge.get());
                }
            } else {
                return Err(format!("failed to get gauge with index {i}").into());
            }
        }

        Ok(())
    }
}
