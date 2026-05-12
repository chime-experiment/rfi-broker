//! Outlier statistics computed on ``ndarray::ArrayD``s.
#![allow(
    dead_code,
    reason = "generic implementations which might be used in the future"
)]

use ndarray::{
    Array, Array1, Array2, ArrayD, ArrayView, ArrayView2, Axis, Dimension, RemoveAxis, Zip,
};
use statrs::distribution::{Beta, ContinuousCDF, DiscreteCDF, Normal, Poisson};

use eyre::WrapErr;

/// Compute the likelihood that an input is bad, based on the `bad_feed_counts`
/// dataset in the shared state.
///
/// Counts are first summed over frequencies to produce a per-element trial
/// success count (maximum count is equivalent to the number of frequencies
/// times the number of trials per frame). The likelihood of a feed being an
/// outlier (i.e., bad) is the result of the cdf of a Poisson distribution for
/// the number of n-sigma outliers for that feed, fed into a Beta distribution
/// which acts as a ramp function to suppress the "badness" likelihood of a
/// handful of successes (because p is small and n is large, a dozen or so
/// excursions results in a likelihood of ~0.5, which may not be representative
/// of the metric that we want to produce).
///
/// The poisson distribution is used instead of a binomial test because n is large
/// and p is small, and the poisson is slightly more computationally efficient
/// to compute.
pub fn sum_poissonbeta_greater(
    arr: &ArrayView2<u8>,
    sigma: f64,
    n: u32,
    alpha: f64,
    beta: f64,
) -> eyre::Result<Array1<f64>> {
    // Approximately convert sigma to p, then compute lambda from n*p
    let ndist = Normal::new(0.0, 1.0).wrap_err("failed to construct normal distribution")?;
    let lambda = ndist.sf(sigma) * f64::from(n);
    // Create a poisson distribution and map the test across
    // each element. Extremely cheap to initialize
    let dist = Poisson::new(lambda).wrap_err("failed to construct new poisson distribution")?;
    let bdist = Beta::new(alpha, beta).wrap_err("failed to construct new beta distribution")?;

    // sum across frequencies
    let counts: Array1<u64> = arr.fold_axis(Axis(0), 0u64, |&acc, &x| acc + u64::from(x));

    // computes per-element beta_CDF(poisson_CDF(k - 1))
    Ok(counts.mapv(|k: u64| {
        if k == 0 {
            0.0
        } else {
            bdist.cdf(dist.cdf(k - 1))
        }
    }))
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

/// Compute the median of an array across an arbitrary axis.
fn median_axis<D>(arr: &ArrayView<f32, D>, axis: Axis) -> Array<f32, D::Smaller>
where
    D: Dimension + RemoveAxis,
{
    arr.map_axis(axis, |lane| {
        let mut v: Vec<f32> = lane.iter().copied().collect();
        #[allow(
            clippy::integer_division,
            reason = "integer truncation is the desired behaviour"
        )]
        // Ensures there are at least 2 elements - if not, return early
        let mid: usize = v.len() / 2;
        if mid == 0 {
            return 0_f32;
        }

        v.select_nth_unstable_by(mid, f32::total_cmp);
        let upper = *v.get(mid).unwrap_or(&0_f32);

        if v.len().is_multiple_of(2) {
            v.select_nth_unstable_by(mid - 1, f32::total_cmp);
            f32::midpoint(*v.get(mid - 1).unwrap_or(&0_f32), upper)
        } else {
            upper
        }
    })
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

    /// Test that ``median_axis`` produces expectation for odd-length array.
    #[test]
    fn test_median_axis_odd() {
        // Test the odd-length case first
        #[allow(
            clippy::cast_precision_loss,
            reason = "values too small for precision loss"
        )]
        // Use multiple rows to ensure that axis mapping is correct
        let data = ArrayD::from_shape_fn(ndarray::IxDyn(&[3, 13, 11]), |idx| idx[2] as f32);
        let medval = median_axis(&data.view(), Axis(2));

        let expected = ArrayD::<f32>::from_elem(medval.raw_dim(), 5.0_f32);

        assert_abs_diff_eq!(medval, expected, epsilon = 1e-8);
    }

    /// Test that ``median_axis`` produces expectation for even-length array.
    #[test]
    fn test_median_axis_even() {
        // Test the odd-length case first
        #[allow(
            clippy::cast_precision_loss,
            reason = "values too small for precision loss"
        )]
        // Use multiple rows to ensure that axis mapping is correct
        let data = ArrayD::from_shape_fn(ndarray::IxDyn(&[3, 13, 10]), |idx| idx[2] as f32);
        let medval = median_axis(&data.view(), Axis(2));

        let expected = ArrayD::<f32>::from_elem(medval.raw_dim(), 4.5_f32);

        assert_abs_diff_eq!(medval, expected, epsilon = 1e-8);
    }
}
