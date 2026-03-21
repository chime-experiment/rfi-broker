//! Entry point for the HTTP server binary.
//!
//! Parses CLI arguments, loads configuration file, builds a
//! multi-threaded tokio runtime, and runs the async server.

use clap::Parser;
use std::net::SocketAddr;
use std::path::PathBuf;

mod config;
mod datastate;
mod handlers;
mod header;
mod ringbuffer;
mod server;
mod udp;

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

    /// Path to a yaml config file
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
    // Extract command-line options
    let cli = Cli::parse();
    println!("Using {} worker threads", cli.threads);
    // Extract the config file
    let config = config::Config::from_file(&cli.config);
    // Create the data state
    let state = datastate::DataState::from_config_shared(&config);

    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(cli.threads)
        .enable_all()
        .build()
        .unwrap()
        .block_on(server::serve(cli.addr, cli.udp_addr, state));
}
