//! SIMD implementations for the [`crate::core::stats`] module.
use std::f64::consts::FRAC_1_SQRT_2;
use wide::f64x4;

/// Constants for a fast erfc estimation
const P_FLOOR: f64 = 1.0e-50;
const C: [f64; 10] = [
    0.170_872_77,
    -0.822_152_23,
    1.488_515_87,
    -1.135_203_98,
    0.278_868_07,
    -0.186_288_06,
    0.096_784_18,
    0.374_091_96,
    1.000_023_68,
    -1.265_512_23,
];

/// erfc(z) for z >= 0; fractional error ~1.2e-7 (Numerical Recipes form)
#[inline]
fn erfc_v(z: f64x4) -> f64x4 {
    let one = f64x4::splat(1.0);
    let t = one / z.mul_add(f64x4::splat(0.5), one);
    let mut p = f64x4::splat(C[0]);

    for &c in &C[1..] {
        p = p.mul_add(t, f64x4::splat(c));
    }

    t * (p - z * z).exp()
}

/// Serial `erfc`
#[inline]
fn erfc_s(z: f64) -> f64 {
    let t = 1.0 / 0.5f64.mul_add(z, 1.0);
    let p = C.iter().fold(0.0f64, |acc, &c| acc.mul_add(t, c));

    t * z.mul_add(-z, p).exp()
}

/// Compute the Fisher metric of a statistic summed over rows.
///
/// # Notes
/// - Elementwise p = 2.0 * SF(|x - mu|, 0, std)
/// - Fisher metric is -2.0 * sum over rows of ln(p), but this function
///   omits the -2.0 multiplication.
pub fn accumulate_ln_p(acc: &mut [f64], vals: &[f64], mu: f64, coeff: f64) {
    // p-score takes the absolute-value of the centred `val` and passes it
    // to the SF of a normal distribution with standard deviation `std`.
    // these two operation can be combined in a slightly simplified way:
    // z = |x - mu| / (std * sqrt(2))
    // SF = 0.5 * erfc(z)
    // p = max(2.0 * SF, epsilon)
    let (m, cf) = (f64x4::splat(mu), f64x4::splat(coeff));
    let floor = f64x4::splat(P_FLOOR);

    let (a_chunks, a_tail) = acc.as_chunks_mut::<4>();
    let (v_chunks, v_tail) = vals.as_chunks::<4>();

    for (a, v) in a_chunks.iter_mut().zip(v_chunks) {
        let z = (f64x4::from(*v) - m).abs() * cf;
        let p = erfc_v(z).max(floor);
        // Fisher sum log(p)
        *a = (f64x4::from(*a) + p.ln()).to_array();
    }
    for (a, &v) in a_tail.iter_mut().zip(v_tail) {
        *a += erfc_s((v - mu).abs() * coeff).max(P_FLOOR).ln();
    }
}

/// SIMD sum of (x - mu)^2 over a slice.
#[must_use]
pub fn sum_sq_dev_simd(vals: &[f64], mu: f64) -> f64 {
    let m = f64x4::splat(mu);

    let (chunks, tail) = vals.as_chunks::<4>();

    let v = chunks.iter().fold(f64x4::ZERO, |s, c| {
        let d = f64x4::from(*c) - m;
        d.mul_add(d, s)
    });

    // accumulate the tail and chunk slots
    tail.iter().fold(v.reduce_add(), |s, &x| {
        let d = x - mu;
        d.mul_add(d, s)
    })
}

/// Wilson–Hilferty approximation of the regularized lower incomplete gamma P(k, s).
///
/// `s` = -(sum of ln p) = half the Fisher statistic; chi2 df = 2k.
///   (s/k)^(1/3) ~ Normal(1 - 1/(9k), 1/(9k))
/// Bounds: always returns a value in [0.0, 1.0].
///
/// # Note
/// This isn't SIMD-vectorized at the moment becauase of branching.
#[inline]
#[must_use]
pub fn gamma_lr_wh(k: f64, s: f64) -> f64 {
    let fallback = if s.is_finite() { 0.0 } else { 1.0 };

    if !s.is_finite() {
        return 1.0; // +inf (or NaN) -> 1.0
    }
    if s <= 0.0 {
        return 0.0; // x <= 0 -> 0.0
    }
    let c = 1.0 / (9.0 * k);
    let w = ((s / k).cbrt() - (1.0 - c)) / c.sqrt();

    // upper-tail probability  at |w|; erfc argument always positive
    let tail = 0.5 * erfc_s(w.abs() * FRAC_1_SQRT_2);
    let cdf = if w < 0.0 { tail } else { 1.0 - tail };

    if cdf.is_nan() {
        fallback
    } else {
        cdf.clamp(0.0, 1.0)
    }
}

#[cfg(test)]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::indexing_slicing,
    reason = "values validated for tests"
)]
mod tests {
    use super::*;
    use statrs::function::{erf, gamma};

    const LN_FLOOR: f64 = -115.129_254_649_702_29; // ln(1e-50)
    // erfc approx has ~1.2e-7 fractional error; allow headroom for wide's ln/exp
    const LN_TOL: f64 = 1.0e-6;

    // deterministic pseudo-random values in [-scale, scale], no rand dependency
    fn lcg_vec(n: usize, seed: u64, scale: f64) -> Vec<f64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let u = ((s >> 11) as f64) / ((1u64 << 53) as f64); // [0, 1)

                2.0f64.mul_add(u, -1.0) * scale
            })
            .collect()
    }

    fn ref_ln_p(z: f64) -> f64 {
        erf::erfc(z).max(1.0e-50).ln()
    }

    fn assert_close(a: f64, b: f64, abs: f64, rel: f64, ctx: &str) {
        let tol = abs.max(rel * b.abs());
        assert!(
            (a - b).abs() <= tol,
            "{ctx}: got {a}, expected {b}, tol {tol}"
        );
    }

    // ---------- erfc kernels (keep the pair you kept) ----------

    #[test]
    fn erfc_v_matches_reference() {
        let zs: Vec<f64> = (0..400).map(|i| f64::from(i) * 0.025).collect(); // 0..10

        let (chunks, _) = zs.as_chunks::<4>();

        for c in chunks {
            let got = erfc_v(f64x4::from([c[0], c[1], c[2], c[3]])).to_array();
            for (g, &z) in got.iter().zip(c) {
                // fractional error ~1.2e-7, plus margin; tiny values may underflow to 0
                assert_close(*g, erf::erfc(z), 1e-300, 5.0e-7, &format!("erfc_v z={z}"));
            }
        }
    }

    #[test]
    fn erfc_s_matches_vector() {
        for i in 0..200 {
            let z = f64::from(i) * 0.05;
            let v = erfc_v(f64x4::splat(z)).to_array()[0];
            assert_close(erfc_s(z), v, 1e-300, 1.0e-12, &format!("erfc_s vs v z={z}"));
        }
    }

    #[test]
    fn erfc_at_zero_is_one() {
        assert_close(erfc_s(0.0), 1.0, 1e-6, 0.0, "erfc(0)");
    }

    // ---------- sum_sq_dev ----------

    #[test]
    fn sum_sq_dev_matches_scalar_for_all_tail_lengths() {
        for n in 0..=17 {
            let v = lcg_vec(n, 42 + n as u64, 10.0);
            let mu = 0.7;
            let expected: f64 = v.iter().map(|x| (x - mu) * (x - mu)).sum();
            assert_close(
                sum_sq_dev_simd(&v, mu),
                expected,
                1e-12,
                1e-12,
                &format!("n={n}"),
            );
        }
    }

    #[test]
    fn sum_sq_dev_zero_when_constant() {
        assert_eq!(sum_sq_dev_simd(&[3.0; 11], 3.0), 0.0);
    }

    #[test]
    fn sum_sq_dev_empty_is_zero() {
        assert_eq!(sum_sq_dev_simd(&[], 1.0), 0.0);
    }

    // ---------- accumulate_ln_p ----------

    fn ref_accumulate(acc: &mut [f64], vals: &[f64], mu: f64, coeff: f64) {
        for (a, &v) in acc.iter_mut().zip(vals) {
            *a += ref_ln_p((v - mu).abs() * coeff);
        }
    }

    #[test]
    fn accumulate_matches_reference_for_all_tail_lengths() {
        for n in 0..=17 {
            let vals = lcg_vec(n, 7 + n as u64, 5.0);
            let (mu, coeff) = (0.1, 0.8);
            let mut got = vec![0.0; n];
            let mut want = vec![0.0; n];
            accumulate_ln_p(&mut got, &vals, mu, coeff);
            ref_accumulate(&mut want, &vals, mu, coeff);
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_close(*g, *w, LN_TOL, 0.0, &format!("n={n} i={i}"));
            }
        }
    }

    #[test]
    fn accumulate_adds_to_existing_values() {
        let vals = lcg_vec(10, 3, 4.0);
        let mut got = vec![-5.0; 10];
        let mut want = vec![-5.0; 10];

        accumulate_ln_p(&mut got, &vals, 0.0, 1.0);
        ref_accumulate(&mut want, &vals, 0.0, 1.0);

        for (g, w) in got.iter().zip(&want) {
            assert_close(*g, *w, LN_TOL, 0.0, "pre-seeded acc");
        }
    }

    #[test]
    fn accumulate_sums_across_rows_and_stays_in_range() {
        let rows = 50;
        let n = 13;
        let mut got = vec![0.0; n];
        let mut want = vec![0.0; n];

        for r in 0..rows {
            let vals = lcg_vec(n, 100 + r as u64, 6.0);
            accumulate_ln_p(&mut got, &vals, 0.0, 0.9);
            ref_accumulate(&mut want, &vals, 0.0, 0.9);
        }

        for (g, w) in got.iter().zip(&want) {
            assert!(*g <= 1e-6, "sum of ln p must be <= 0, got {g}");
            assert!(*g >= LN_FLOOR.mul_add(f64::from(rows), -1e-6));
            // errors add across rows
            assert_close(*g, *w, LN_TOL * f64::from(rows), 0.0, "multi-row");
        }
    }

    #[test]
    fn accumulate_value_at_mu_contributes_zero() {
        let vals = [2.0; 8];
        let mut acc = vec![0.0; 8];
        accumulate_ln_p(&mut acc, &vals, 2.0, 1.0); // z = 0 -> p = 1 -> ln p = 0
        for a in acc {
            assert_close(a, 0.0, LN_TOL, 0.0, "z=0");
        }
    }

    #[test]
    fn accumulate_extreme_outlier_hits_floor() {
        let vals = [1.0e4, -1.0e4, 0.0, 0.0, 1.0e4];
        let mut acc = vec![0.0; 5];
        accumulate_ln_p(&mut acc, &vals, 0.0, 1.0);
        assert_eq!(acc[0], LN_FLOOR);
        assert_eq!(acc[1], LN_FLOOR);
        assert_eq!(acc[4], LN_FLOOR);
    }

    // ---------- gamma_lr_wh ----------

    const K: f64 = 1024.0;

    #[test]
    fn gamma_special_cases() {
        assert_eq!(gamma_lr_wh(K, 0.0), 0.0);
        assert_eq!(gamma_lr_wh(K, -3.0), 0.0);
        assert_eq!(gamma_lr_wh(K, f64::INFINITY), 1.0);
        assert_eq!(gamma_lr_wh(K, f64::NAN), 1.0);
    }

    #[test]
    fn gamma_always_in_unit_interval() {
        for &s in &[
            1e-300, 1e-3, 1.0, 100.0, 900.0, 1024.0, 1200.0, 5e3, 1e6, 1e300,
        ] {
            let g = gamma_lr_wh(K, s);
            assert!((0.0..=1.0).contains(&g), "s={s} -> {g}");
        }
    }

    #[test]
    fn gamma_is_monotonic_nondecreasing() {
        let mut prev = 0.0;
        for i in 0..400 {
            let s = f64::from(i).mul_add(1.5, 700.0); // spans well below to above k
            let g = gamma_lr_wh(K, s);
            assert!(g >= prev, "decreased at s={s}: {g} < {prev}");
            prev = g;
        }
    }

    #[test]
    fn gamma_near_half_at_mean() {
        // P(k, k) is just above 0.5 for large k
        assert_close(gamma_lr_wh(K, K), 0.5, 0.01, 0.0, "P(k,k)");
    }

    #[test]
    fn gamma_matches_exact_near_transition() {
        // within +/- 4 sd (sd of Gamma(k,1) is sqrt(k) = 32)
        for i in -40..=40 {
            let s = f64::from(i).mul_add(3.2, K);
            let exact = gamma::checked_gamma_lr(K, s).unwrap();
            assert_close(gamma_lr_wh(K, s), exact, 2.0e-3, 0.0, &format!("s={s}"));
        }
    }

    #[test]
    fn gamma_tails_agree_to_loose_tolerance() {
        for &s in &[800.0, 850.0, 1200.0, 1300.0] {
            let exact = gamma::checked_gamma_lr(K, s).unwrap();
            assert_close(
                gamma_lr_wh(K, s),
                exact,
                2.0e-3,
                0.0,
                &format!("tail s={s}"),
            );
        }
    }
}
