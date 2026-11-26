//! Performance benchmarks for stats operations.
#![allow(
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    reason = "allowed by benchmark construction"
)]

use criterion::{Criterion, Throughput, criterion_group};
use std::hint::black_box;

use rfi_broker::core::{simd_stats, stats};

use crate::common::generate_input_2d;

fn run_benchmarks(c: &mut Criterion) {
    let k = 1024.0_f64;

    // generated once; both groups reuse these arrays
    let inputs: Vec<_> = [(1024usize, 2048usize), (256, 512)]
        .iter()
        .map(|&(rows, cols)| (rows, cols, generate_input_2d(rows, cols, 0xDEAD_BEEF)))
        .collect();

    let mut fisher = c.benchmark_group("fisher");
    fisher.sample_size(20);

    for (rows, cols, input) in &inputs {
        let view = input.view();

        fisher.throughput(Throughput::Elements((rows * cols) as u64));
        fisher.bench_function(format!("{rows}x{cols}"), |b| {
            b.iter(|| black_box(stats::sk_fisher_chi2(black_box(&view), black_box(k))));
        });
    }
    fisher.finish();

    let mut singles = c.benchmark_group("operations");
    for (_, cols, input) in &inputs {
        // one row of the same array, as a contiguous slice
        let row = input.row(0);
        let row = row.as_slice().unwrap();

        singles.throughput(Throughput::Elements(*cols as u64));
        singles.bench_function(format!("median_{cols}"), |b| {
            b.iter(|| black_box(stats::median(black_box(row))));
        });

        let vals: Vec<f64> = row.to_vec();
        let mu = stats::median(&vals);
        let coeff = 1.0 / (2.0f64).sqrt() / 0.1; // arbitrary

        singles.bench_function(format!("sum_sq_dev_{cols}"), |b| {
            b.iter(|| black_box(simd_stats::sum_sq_dev_simd(black_box(&vals), black_box(mu))));
        });

        let mut acc = vec![0.0f64; *cols];
        singles.bench_function(format!("accumulate_ln_p_{cols}"), |b| {
            b.iter(|| {
                simd_stats::accumulate_ln_p(
                    black_box(&mut acc),
                    black_box(&vals),
                    black_box(mu),
                    black_box(coeff),
                );
            });
        });
    }
    singles.finish();
}

criterion_group!(benches, run_benchmarks);
