//! Server, router, and UDP receiver.
//!
//! # Endpoints
//! - ``metadata``: Most recent packet header metadata
//! - ``data``: Dump all ringbuffers
//! - ``bad_input_likelihood``: per-input likelihood of a feed being corrupted

use std::future::IntoFuture;
use std::io::ErrorKind::WouldBlock;
use std::net::SocketAddr;
use std::sync::Arc;

use axum::{Router, routing::get};
use tokio::net::{TcpListener, UdpSocket};

use crate::datastate::{DataState, SharedDataState};
use crate::endpoints;
use crate::metrics::{Metrics, SharedMetrics, update_metrics};
use crate::packet::Packet;

/// Builds a [`Router`] containing all the endpoints we'd like to enable.
///
/// `store` is injected as Axum [`State`] so handlers can read the buffers.
fn router(state: SharedDataState, metrics: SharedMetrics) -> Router {
    let mut router = Router::new();
    router = router.route("/data", get(endpoints::data));
    router = router.route("/metadata", get(endpoints::metadata));
    router = router.route(
        "/bad_input_likelihood",
        get(endpoints::get_bad_input_likelihood),
    );
    router = router.route("/", get(endpoints::dump_bad_input_likelihood));

    let router = router.with_state(state);

    // Add metrics state
    let metrics_router = Router::new()
        .route("/metrics", get(endpoints::metrics))
        .with_state(metrics);

    router.merge(metrics_router)
}

/// Construct a UDP socket with a buffer large enough to handle
/// burst events.
async fn construct_sock(addr: SocketAddr) -> Result<UdpSocket, std::io::Error> {
    let socket = UdpSocket::bind(addr).await?;

    // Borrow the socket as a socket2 ref to increase buffer
    let sock_ref = socket2::SockRef::from(&socket);
    sock_ref.set_recv_buffer_size(8 * 1024 * 1024)?; // 8MB

    // log the actual recv buffer size
    let actual = sock_ref.recv_buffer_size()?;
    tracing::debug!("Request 8MB recv buffer, got {}MB", actual / 1024 / 1024);

    tracing::info!("UDP socket listening on {addr}");

    Ok(socket)
}

/// Drains a UDP socket buffer and pushes packets to the [`DataState`].
///
/// Runs indefinitely - intended to be run with [`tokio::spawn`].
/// Exits when all senders are dropped.
async fn packet_handler_task(
    sock: UdpSocket,
    metrics: SharedMetrics,
    state: SharedDataState,
) -> Result<(), std::io::Error> {
    // Allocate a buffer large enough for any valid UDP packet
    let mut buf = vec![0u8; u16::MAX as usize];

    let mut last_update_id: u64 = 0;

    loop {
        // Wait until socket is readable
        sock.readable().await?;
        // Pass through the entire os buffer
        loop {
            match sock.try_recv_from(&mut buf) {
                Ok((len, _)) => match Packet::parse(&buf[..len]) {
                    Ok(packet) => match state.push(packet) {
                        Ok(id) => {
                            if id != last_update_id {
                                last_update_id = id;
                                update_metrics(&metrics, &state);
                            }
                        }
                        Err(e) => {
                            metrics.packet_loss.inc_lost();
                            tracing::debug!("Error pushing packet to state: {e}");
                        }
                    },
                    Err(e) => {
                        metrics.packet_loss.inc_lost();
                        tracing::debug!("Error parsing packet: {e}");
                    }
                },
                Err(ref e) if e.kind() == WouldBlock => {
                    // No packet available so release the thread
                    break;
                }
                Err(e) => {
                    metrics.packet_loss.inc_lost();
                    tracing::debug!("UDP recv error: {e}");
                }
            }
            metrics.packet_loss.inc_total();
        }
    }
}

/// Spawns the UDP listener and HTTP server as ndependent tasks,
/// then waits for either to exit.
///
/// # Panics
/// Panics if either address cannot be bound.
pub async fn serve(http_addr: SocketAddr, udp_addr: SocketAddr) {
    let state: SharedDataState = Arc::new(DataState::default());
    let metrics: SharedMetrics = Arc::new(Metrics::new());

    // Construct the socket and start the packet handling task. If this becomes
    // a bottleneck, it could be run in multiple threads
    let udp_sock = construct_sock(udp_addr)
        .await
        .unwrap_or_else(|e| panic!("Failed to bind UDP listener on {udp_addr}: {e}"));

    let packet_handler = tokio::spawn(packet_handler_task(
        udp_sock,
        Arc::clone(&metrics),
        Arc::clone(&state),
    ));

    // Start the solar event task
    let solar = tokio::spawn(crate::solar::solar_event_task(Arc::clone(&metrics)));

    let listener = TcpListener::bind(http_addr)
        .await
        .unwrap_or_else(|e| panic!("Failed to bind HTTP listener on {http_addr}: {e}"));

    let http = tokio::spawn(axum::serve(listener, router(state, metrics)).into_future());
    tracing::info!("HTTP listening on {http_addr}");

    tokio::select! {
        _ = packet_handler => tracing::error!("Packet handler exited unexpectedly"),
        _ = http => tracing::error!("HTTP server exited unexpectedly"),
        _ = solar => tracing::error!("Solar event task exited unexpectedly"),
    }
}
