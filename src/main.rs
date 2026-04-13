//! Entry point for the HTTP server binary.
//!
//! Parses CLI arguments, loads configuration file, builds a
//! multi-threaded tokio runtime, and runs the async server.
#![warn(clippy::pedantic)]

use std::net::SocketAddr;

use clap::Parser;
use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

mod datastate;
mod endpoints;
mod metrics;
mod packet;
mod ringbuffer;
mod server;
mod solar;

/// Command-line parser
#[derive(Parser)]
#[command(name = "RFI Receiver")]
#[command(version, about)] // Read from `Cargo.toml`
struct Cli {
    /// Address
    #[arg(short, long)]
    pub addr: SocketAddr,

    /// Address to listen for UDP packets
    #[arg(short, long)]
    pub udp_addr: SocketAddr,

    /// Number of worker threads
    #[arg(short, long, default_value_t = _default_nthreads())]
    pub threads: usize,
}

/// Returns the default number of work threads: the lesser of
/// the number of logical CPU cores and 4. Falls back to 1 if
/// the OS does not report available parallelism.
fn _default_nthreads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get().min(4))
}

/// Parses CLI, resolves config, and starts the server.
fn main() {
    // Set up logging
    tracing_subscriber::registry()
        // Default to `INFO` log level. Can be adjusted using RUST_LOG
        // environment variable
        .with(EnvFilter::from_default_env().add_directive(tracing::Level::INFO.into()))
        .with(tracing_journald::layer().ok()) // None is journald not available
        .with(tracing_subscriber::fmt::layer()) // Fallback to print to stdout
        .init();

    // Extract command-line options
    let cli = Cli::parse();
    tracing::debug!("Using {} worker threads", cli.threads);

    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(cli.threads)
        .enable_all()
        .build()
        .unwrap()
        .block_on(server::serve(cli.addr, cli.udp_addr));
}
