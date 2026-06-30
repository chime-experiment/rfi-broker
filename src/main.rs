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

use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

mod app;
mod buffer;
mod config;
mod endpoints;
mod metrics;
mod packet;
mod state;
mod stats;
mod tasks;

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
    #[arg(short, long, default_value_t = default_nthreads())]
    pub threads: usize,
}

/// Returns the default number of work threads: the larger of
/// the number of logical CPU cores and 4. Falls back to 1 if
/// the OS does not report available parallelism.
fn default_nthreads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get().max(4))
}

/// Check if process is controlled by systemd
fn is_controlled_by_systemd() -> bool {
    std::env::var("JOURNAL_STREAM").is_ok()
        || std::env::var("INVOCATION_ID").is_ok()
        || std::env::var("RFI_BROKER_JOURNALD_TRACING").is_ok_and(|v| v == "1")
}

/// Set up program logging.
///
/// Log formatting is adjusted depending on whether we are running
/// controlled by systemd or not.
fn init_tracing() {
    // Default to `INFO` log level. Can be adjusted using RUST_LOG environment variable
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let registry = tracing_subscriber::registry().with(env_filter);

    // set a human-readable format layer, which may or may not be used
    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_thread_ids(true)
        .with_line_number(true)
        .pretty();

    // Set the formatting layer depending on where we're writing logs
    if is_controlled_by_systemd() {
        match tracing_journald::layer() {
            Ok(layer) => {
                let layer = layer
                    .with_priority_mappings(tracing_journald::PriorityMappings {
                        // Map `INFO` -> Informational(6) instead of Notice(5) to prevent
                        // it from being rendered in bold font in journald
                        info: tracing_journald::Priority::Informational,
                        ..tracing_journald::PriorityMappings::new()
                    })
                    .with_field_prefix(None);
                registry.with(layer).init();
            }
            Err(e) => {
                // fall back to basic stdout formatting
                registry.with(fmt_layer).init();
                tracing::warn!(
                    error = ?e,
                    "journald logging was requested, but failed to set up layer:"
                );
            }
        }
    } else {
        // not system controlled - use stdout formatting
        registry.with(fmt_layer).init();
    }
}

/// Parses CLI, resolves config, and starts the server.
fn main() -> eyre::Result<()> {
    rustls_graviola::default_provider()
        .install_default()
        .map_err(|_| eyre!("failed to install default `rustls` crypto provider"))?;

    // Set up logging
    init_tracing();

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
        .block_on(app::run(cli.addr, cli.udp_addr, config))
        .wrap_err("server failed")?;

    Ok(())
}
