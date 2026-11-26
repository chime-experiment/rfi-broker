//! Async application runner, intended to run in a tokio runtime.

use eyre::WrapErr;

use std::net::SocketAddr;
use std::sync::Arc;

use tokio::{net::TcpListener, task::JoinSet};

use crate::network::{
    router::make_router,
    udp::{NetworkActivity, construct_sock, packet_handler_task, packet_monitor_task},
};
use crate::state::AppConfig;
use crate::state::AppState;

/// Spawns all program tasks, including:
/// - rfi solar enable/disable task
/// - bad input metric computation task
/// - packet recv task
/// - http server
///
/// # Errors
/// - Errors if either the http or UDP recv address cannot be bound.
/// - Errors if any task fails
pub async fn run(
    http_addr: SocketAddr,
    udp_addr: SocketAddr,
    config: Option<AppConfig>,
    receiver_threads: usize,
) -> eyre::Result<()> {
    let state = AppState::default();
    let config: Option<Arc<AppConfig>> = config.map(Arc::new);

    // Start the solar event task
    let rfi_zeroing = tokio::spawn(crate::tasks::solar_event_task(
        Arc::clone(&state.metrics),
        config.map(|c| Arc::clone(&c)),
    ));

    // Start the bad input task
    let bad_inputs = tokio::spawn(crate::tasks::bad_input_task(
        Arc::clone(&state.buffers),
        Arc::clone(&state.computed),
    ));

    // Construct the socket and start the packet handling task. If this becomes
    // a bottleneck, it could be run in multiple threads
    let udp_sock = Arc::new(construct_sock(udp_addr).await?);
    // Construct an activity monitor for the network receive tasks
    let monitor = Arc::new(NetworkActivity::new());

    // Spawn multiple listeners
    let mut rx_tasks: JoinSet<(usize, Result<(), std::io::Error>)> = JoinSet::new();

    eyre::ensure!(
        receiver_threads > 0,
        "at least one packet receiver is required"
    );
    for i in 0..receiver_threads {
        let sock = Arc::clone(&udp_sock);
        let buffers = Arc::clone(&state.buffers);
        let metrics = Arc::clone(&state.metrics);
        let activity = Arc::clone(&monitor);
        rx_tasks.spawn(async move {
            (
                i,
                packet_handler_task(sock, activity, buffers, metrics).await,
            )
        });
    }

    // Also spawn the monitoring task
    let monitor_task = tokio::spawn(packet_monitor_task(
        Arc::clone(&state.buffers),
        Arc::clone(&monitor),
    ));

    let http_listener = TcpListener::bind(http_addr).await?;

    let http = tokio::spawn(axum::serve(http_listener, make_router(state)).into_future());
    tracing::info!(http_addr = ?http_addr, "started HTTP server: {http_addr}");

    tokio::select! {
        Some(result) = rx_tasks.join_next() => match result {
            Ok((i, Ok(()))) => eyre::bail!("packet handler {i} exited unexpectedly!"),
            Ok((i, Err(e))) => eyre::bail!("packet handler {i} failed: {e}"),
            Err(join_err) => eyre::bail!("packet handler panicked or was cancelled: {join_err}")
        },
        result = monitor_task => result?.wrap_err("network monitor failed"),
        result = http => result?.wrap_err("http server failed"),
        result = rfi_zeroing => result?.wrap_err("solar zeroing task failed"),
        result = bad_inputs => result?.wrap_err("bad input metric task failed"),
    }
}
