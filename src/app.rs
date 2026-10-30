//! Async application runner, intended to run in a tokio runtime.

use eyre::WrapErr;

use std::net::SocketAddr;
use std::sync::Arc;

use tokio::net::TcpListener;

use crate::network::{
    router::make_router,
    udp::{construct_sock, packet_handler_task},
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
    let udp_sock = construct_sock(udp_addr).await?;

    let packet_handler = tokio::spawn(packet_handler_task(
        udp_sock,
        Arc::clone(&state.buffers),
        Arc::clone(&state.metrics),
    ));

    let http_listener = TcpListener::bind(http_addr).await?;

    let http = tokio::spawn(axum::serve(http_listener, make_router(state)).into_future());
    tracing::info!(http_addr = ?http_addr, "started HTTP server");

    tokio::select! {
        result = packet_handler => result?.wrap_err("packet handler failed"),
        result = http => result?.wrap_err("http server failed"),
        result = rfi_zeroing => result?.wrap_err("solar zeroing task failed"),
        result = bad_inputs => result?.wrap_err("bad input metric task failed"),
    }
}
