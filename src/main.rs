//! Broker managing RFI-related tasks.
//!
//! Tasks include:
//! - Provides a per-feed likelihood that the feed is corrupted based on spectral kurtosis data
//! - Enables and disables RFI zeroing around solar noon
//! - Exports some Prometheus metrics

use std::net::SocketAddr;
use std::path::PathBuf;

use clap::Parser;
use eyre::{WrapErr, eyre};

mod app;
mod network;
mod state;
mod tasks;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

const DEFAULT_NUM_RECEIVERS: usize = 1;

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

    /// Config file
    #[arg(short, long)]
    pub config: Option<PathBuf>,

    /// Number of worker threads
    #[arg(short, long, default_value_t = default_nthreads())]
    pub threads: usize,

    /// Number of receiver threads
    #[arg(short, long, default_value_t = DEFAULT_NUM_RECEIVERS)]
    pub receivers: usize,
}

/// Returns the default number of work threads: the larger of
/// the number of logical CPU cores and 4. Falls back to 1 if
/// the OS does not report available parallelism.
fn default_nthreads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get().max(4))
}

/// Parses CLI, resolves config, and starts the server.
fn main() -> eyre::Result<()> {
    rustls_graviola::default_provider()
        .install_default()
        .map_err(|_| eyre!("failed to install default `rustls` crypto provider"))?;

    // Set up logging
    app::init_tracing();

    // Extract command-line options
    let cli = Cli::parse();

    // Load the config, accounting for the fact the both the argument and the
    // parsed result could be `None`
    let config = cli
        .config
        .as_ref()
        .and_then(|p| p.to_str())
        .map(crate::state::config::load)
        // transpose calls swap the order of Option and Result, with the end effect
        // of propagating errors occuring in `load` to the parent function
        .transpose()
        .wrap_err("failed to read config")?;

    tracing::info!("Using {} worker threads", cli.threads);

    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(cli.threads)
        .enable_all()
        .build()
        .wrap_err("failed to build async runtime")?
        .block_on(app::run(cli.addr, cli.udp_addr, config, cli.receivers))
        .wrap_err("server failed")?;

    Ok(())
}
