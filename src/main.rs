//! Entry point for the HTTP server binary.
//!
//! Parses CLI arguments, loads configuration file, builds a
//! multi-threaded tokio runtime, and runs the async server.

use std::net::SocketAddr;
use std::path::PathBuf;

use clap::Parser;

use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

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

    /// Config file
    #[arg(short, long)]
    pub config: PathBuf,
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
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .with(tracing_journald::layer().ok()) // None is journald not available
        .with(
            tracing_subscriber::fmt::layer()
                .with_thread_ids(true)
                .with_line_number(true),
        ) // Fallback to print to stdout with extra information
        .init();

    // Extract command-line options
    let cli = Cli::parse();

    // Load the config
    let Some(config_path) = cli.config.to_str() else {
        panic!("invalid config file path");
    };
    let config = rfi_receiver::config::load(config_path)
        .unwrap_or_else(|e| panic!("unable to load config file: {e}"));

    tracing::debug!("Using {} worker threads", cli.threads);

    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(cli.threads)
        .enable_all()
        .build()
        .unwrap()
        .block_on(rfi_receiver::server::serve(cli.addr, cli.udp_addr, config));
}
