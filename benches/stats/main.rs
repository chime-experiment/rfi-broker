//! Performance benchmarks
mod common;
mod stats;

use criterion::criterion_main;

criterion_main!(stats::benches);
