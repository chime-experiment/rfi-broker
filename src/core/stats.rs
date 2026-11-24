//! Outlier statistics computed on ``ndarray::ArrayD``s.
#![allow(
    dead_code,
    reason = "generic implementations which might be used in the future"
)]

use parking_lot::Mutex;

use ndarray::{Array1, Array2, ArrayD, ArrayView2, Axis, Zip};
use num_traits::AsPrimitive;

use eyre::ensure;

use super::simd_stats;

/// Computes the chi2 CDF of a fisher sum of p-values of gaussian-distributed
/// spectral kurtosis data.
///
/// A p-value is computed for each element in the 2D input, and is then summed
/// across the 0th axis according to Fisher's method. Under assumed conditions,
/// the fisher sum has a chi-squared distribution, the CDF of which is returned
/// as a likelihood metric.
///
/// # Panics
/// Panics if the input array is not c-contiguous
pub fn sk_fisher_chi2<T>(arr: &ArrayView2<T>, k: T) -> Array1<f64>
where
    T: AsPrimitive<f64>,
{
    // normal distribution parameters
    const NORM_COEFF: f64 = 1.0 / std::f64::consts::SQRT_2;

    // array parameters
    #[allow(
        clippy::cast_precision_loss,
        reason = "value too small for precision loss"
    )]
    let (nc, n) = { (arr.ncols(), arr.ncols() as f64) };

    if nc == 0 {
        return Array1::<f64>::zeros(0);
    }

    // NB: this could in theory result in a full copy. For the purposes
    // of this application, the input is already c-contiguous
    let std_arr = arr.as_standard_layout();
    #[allow(
        clippy::unwrap_used,
        reason = "call to `as_standard_layout` ensures that this will succeed"
    )]
    let datasl = std_arr.as_slice().unwrap();

    // single-pass over rows to compute p-values and fisher sum
    let (mut metric, _, _) = datasl.chunks_exact(nc).fold(
        (
            vec![0.0f64; nc],
            Vec::<f64>::with_capacity(nc),
            Vec::<f64>::with_capacity(nc),
        ),
        |(mut acc, mut vals, mut scratch), row| {
            vals.clear();
            vals.extend(row.iter().map(|v| v.as_()));

            // median and stddev of this row
            let mu: f64 = median_scratch(&vals, &mut scratch);

            let std_coeff: f64 = {
                let sum_squared = simd_stats::sum_sq_dev_simd(&vals, mu);
                let std = (sum_squared / n).sqrt();

                NORM_COEFF / std
            };

            if std_coeff.is_finite() && std_coeff > 0.0 {
                simd_stats::accumulate_ln_p(&mut acc, &vals, mu, std_coeff);
            }

            (acc, vals, scratch)
        },
    );

    // chi2 of the fisher metric
    // df = 2k, so a = df/2 = k
    let k: f64 = k.as_();

    for a in &mut metric {
        // `checked_gamma_lr` errors if `x` is either 0 or infinite, so
        // x <= 0 -> g = 0.0 and x == inf -> g = 1.0
        // Fisher sum is multiplied by -2.0, which isn't done earlier, but the
        // input to the lower gamma func is divided by 2.0 so just use
        // the negative of the input
        *a = simd_stats::gamma_lr_wh(k, -*a);
    }

    Array1::<f64>::from(metric)
}

/// Compute a masked mean over the 0th axis of an [`ArrayD`].
///
/// The mask must be two-dimensional, with axes matching the first and
/// last axes of `arr`.
fn masked_mean(arr: &ArrayD<u8>, mask: &Array2<u8>, axis: usize) -> ArrayD<f32> {
    // Max buffer length is 64, so can guarantee that accumulating
    // to u16 will not overflow
    let mut mean: ArrayD<f32> = arr.fold_axis(Axis(axis), 0f32, |&acc, &x| acc + f32::from(x));
    // compute the norm directly as the float reciprocal
    let norm: Array1<f32> = mask.mapv(u16::from).sum_axis(Axis(axis)).mapv(|x| {
        // Only invert if norm is non-zero. In theory, LLVM should
        // convert this to a branchless instruction
        if x == 0 {
            0.0f32
        } else {
            1.0f32 / f32::from(x)
        }
    });

    // Normalize in-place to get masked mean. Reduced over the stacked axis,
    // now map over the freq axis (which is now axis 0)
    Zip::from(mean.axis_iter_mut(Axis(axis)))
        .and(&norm)
        .for_each(|mut lane, &w| {
            lane *= w;
        });

    mean
}

/// Compute the median of a slice.
#[must_use]
pub fn median(x: &[f64]) -> f64 {
    // need to copy because partial sort modifies in-place
    let mut v: Vec<f64> = Vec::with_capacity(x.len());

    median_scratch(x, &mut v)
}

/// Compute the median of a slice, using an external
/// scratch buffer.
fn median_scratch(x: &[f64], v: &mut Vec<f64>) -> f64 {
    v.clear();
    v.extend_from_slice(x);

    // handle special cases
    let len: usize = v.len();
    if len == 0 {
        return 0.0_f64;
    }
    if len == 1 {
        return *v.first().unwrap_or(&0.0_f64);
    }

    // Ensures there are at least 2 elements - if not, return early
    let mid: usize = len >> 1; // integer truncation intended

    v.select_nth_unstable_by(mid, f64::total_cmp);
    let upper = *v.get(mid).unwrap_or(&0.0_f64);

    if v.len().is_multiple_of(2) {
        v.select_nth_unstable_by(mid - 1, f64::total_cmp);
        f64::midpoint(*v.get(mid - 1).unwrap_or(&0.0_f64), upper)
    } else {
        upper
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
    use approx::assert_abs_diff_eq;
    use ndarray::s;

    /// Test that ``masked_mean_last_axis`` produces expectation.
    #[test]
    fn test_masked_mean_last_axis() {
        let mut data = ArrayD::<u8>::ones(ndarray::IxDyn(&[12, 3, 5]));
        let mut mask = Array2::<u8>::ones([12, 3]);

        // check that the unmasked mean is as-expected
        let mean_val = masked_mean(&data, &mask, 0);
        let expected = ArrayD::<f32>::ones(mean_val.raw_dim());
        // in the unmasked case, these should be the same
        let expected_mean_axis = data.mapv(f32::from).mean_axis(Axis(0)).unwrap();

        assert_abs_diff_eq!(mean_val, expected, epsilon = 1e-8);
        assert_abs_diff_eq!(mean_val, expected_mean_axis, epsilon = 1e-8);

        // zero out part of the mask and make sure the new mean is correct
        mask.slice_mut(s![3..5, ..]).fill(0u8);
        data.slice_mut(s![3..5, .., ..]).fill(0u8);

        let mean_val = masked_mean(&data, &mask, 0);
        // The unmasked mean should be smaller than the masked mean by a
        // factor of 1/6
        let expected_mean_axis = data.mapv(f32::from).mean_axis(Axis(0)).unwrap() + (1.0_f32 / 6.0);

        // masked mean should still be 1.0
        assert_abs_diff_eq!(mean_val, expected, epsilon = 1e-8);
        assert_abs_diff_eq!(mean_val, expected_mean_axis, epsilon = 1e-8);
    }

    /// Test that ``median`` produces expectation for odd-length array.
    #[test]
    fn test_median_axis_odd() {
        // Test the odd-length case first
        #[allow(
            clippy::cast_precision_loss,
            reason = "values too small for precision loss"
        )]
        let data = Array1::from_shape_fn(ndarray::Dim(11), |idx| idx as f64);
        let medval = median(data.as_slice().unwrap());

        assert_abs_diff_eq!(medval, 5.0_f64, epsilon = 1e-8);
    }

    /// Test that ``median`` produces expectation for even-length array.
    #[test]
    fn test_median_axis_even() {
        // Test the odd-length case first
        #[allow(
            clippy::cast_precision_loss,
            reason = "values too small for precision loss"
        )]
        let data = Array1::from_shape_fn(ndarray::Dim(10), |idx| idx as f64);
        let medval = median(data.as_slice().unwrap());

        assert_abs_diff_eq!(medval, 4.5_f64, epsilon = 1e-8);
    }
}
