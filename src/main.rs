//! Entry point for the HTTP server binary.
//!
//! Parses CLI arguments, loads configuration file, builds a
//! multi-threaded tokio runtime, and runs the async server.

use std::net::SocketAddr;
use std::path::PathBuf;

use clap::Parser;
use eyre::WrapErr;

use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

mod config;
mod datastate;
mod endpoints;
mod events;
mod metrics;
mod packet;
mod ringbuffer;
mod server;

#[cfg(test)]
pub(crate) mod test_fixtures;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

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
fn main() -> eyre::Result<()> {
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

    // Load the config, accounting for the fact the both the argument and the
    // parsed result could be `None`
    let config = cli
        .config
        .as_ref()
        .and_then(|p| p.to_str())
        .map(config::load)
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
        .block_on(server::serve(cli.addr, cli.udp_addr, config))
        .wrap_err("server failed")?;

    Ok(())
}
