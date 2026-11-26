//! Common data generation for benchmarks.
#![allow(
    clippy::unwrap_used,
    reason = "unwrap allowed by benchmark construction"
)]

use ndarray::Array2;

/// Deterministic, approximately Gaussian (mean 1.0, sd 0.1) data of shape
/// (rows, cols), like spectral kurtosis values. Box-Muller on an LCG, so no
/// `rand` dependency and identical input on every run.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::missing_panics_doc,
    reason = "ok by construction"
)]
#[must_use]
pub fn generate_input_2d(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
    let mut state = seed;
    let mut next_uniform = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // (0, 1], avoids ln(0)
        (((state >> 11) as f64) + 1.0) / ((1u64 << 53) as f64)
    };

    let mut data = Vec::with_capacity(rows * cols);

    while data.len() < rows * cols {
        let (u1, u2) = (next_uniform(), next_uniform());
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;

        data.push((0.1 * r).mul_add(theta.cos(), 1.0));

        if data.len() < rows * cols {
            data.push((0.1 * r).mul_add(theta.sin(), 1.0));
        }
    }

    Array2::from_shape_vec((rows, cols), data).unwrap()
}
