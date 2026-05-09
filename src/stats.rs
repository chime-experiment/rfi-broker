//! Outlier statistics computed on ``ndarray::ArrayD``s.
use ndarray::{Array, Array1, Array2, ArrayD, ArrayView, Axis, Dimension, RemoveAxis, Zip};

use eyre::bail;

// TODO: make a tokio task to monitor this metric and update at some cadence

/// Compute the likelihood that an input is bad, based on the `bad_feed_counts`
/// dataset in the shared state.
///
/// The likelihood metric is computed by averaging the number of "bad" feeds
/// computed by kotekan over time, then taking a median over frequency to
/// produce a single likelihood value per element. The final likelihood
/// is derived as a percentage and normalized by the number of kotekan
/// frames provided by each packet.
pub fn compute_bad_input_likelihood(
    arr: &ArrayD<u8>,
    mask: &Array2<u8>,
) -> eyre::Result<ArrayD<f32>> {
    // Confirm the array dimension
    if arr.ndim() != 3 {
        bail!("expected array with dimension 3, got {:#}", arr.ndim());
    }

    // Compute the per-feed likelihood metric. This is guaranteed
    // to succeed because call to `&buf.stack` above would have
    // failed if the array was empty
    let (mean, norm) = masked_mean_0th_axis(arr, mask);
    // Remove any frequencies which are entirely zero - these were
    // never received and shouldn't be included in the median
    let indices: Vec<usize> = norm
        .iter()
        .enumerate()
        .filter(|(_, v)| v.abs() > f32::EPSILON)
        .map(|(i, _)| i)
        .collect();

    let mean_reduced = mean.select(Axis(0), &indices);
    let mut median = median_axis(&mean_reduced.view(), Axis(0));

    // Convert to a percentage and normalize by the number of frames per packet
    // let meta = state
    //     .metadata
    //     .get()
    //     .ok_or_eyre("metadata is not accessible")?;
    // // NB: this is what was done before, but unclear as to why
    // #[allow(
    //     clippy::cast_precision_loss,
    //     reason = "values too small for precision loss"
    // )]
    let norm = 100.0 / 10.0; // meta.lock().frames_per_packet as f32;
    median *= norm;

    Ok(median)
}

/// Compute a masked mean over the 0th axis of an [`ArrayD`].
///
/// The mask must be two-dimensional, with axes matching the first and
/// last axes of `arr`.
fn masked_mean_0th_axis(arr: &ArrayD<u8>, mask: &Array2<u8>) -> (ArrayD<f32>, Array1<f32>) {
    // Max buffer length is 64, so can guarantee that accumulating
    // to u16 will not overflow
    let mut mean: ArrayD<f32> = arr.fold_axis(Axis(0), 0f32, |&acc, &x| acc + f32::from(x));
    // compute the norm directly as the float reciprocal
    let norm: Array1<f32> = mask.mapv(u16::from).sum_axis(Axis(0)).mapv(|x| {
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
    Zip::from(mean.axis_iter_mut(Axis(0)))
        .and(&norm)
        .for_each(|mut lane, &w| {
            lane *= w;
        });

    (mean, norm)
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
        let (mean_val, _) = masked_mean_0th_axis(&data, &mask);
        let expected = ArrayD::<f32>::ones(mean_val.raw_dim());
        // in the unmasked case, these should be the same
        let expected_mean_axis = data.mapv(f32::from).mean_axis(Axis(0)).unwrap();

        assert_abs_diff_eq!(mean_val, expected, epsilon = 1e-8);
        assert_abs_diff_eq!(mean_val, expected_mean_axis, epsilon = 1e-8);

        // zero out part of the mask and make sure the new mean is correct
        mask.slice_mut(s![3..5, ..]).fill(0u8);
        data.slice_mut(s![3..5, .., ..]).fill(0u8);

        let (mean_val, _) = masked_mean_0th_axis(&data, &mask);
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
