//! Entry point for the HTTP server binary.
//!
//! Parses CLI arguments, loads configuration file, builds a
//! multi-threaded tokio runtime, and runs the async server.
#![warn(clippy::pedantic)]

use std::net::SocketAddr;

use clap::Parser;

mod datastate;
mod endpoints;
mod frame;
mod metrics;
mod packet;
mod ringbuffer;
mod server;

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
    // Extract command-line options
    let cli = Cli::parse();
    println!("Using {} worker threads", cli.threads);

    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(cli.threads)
        .enable_all()
        .build()
        .unwrap()
        .block_on(server::serve(cli.addr, cli.udp_addr));
}
