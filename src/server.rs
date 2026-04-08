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

use axum::{routing::get, Router};

use tokio::net::{TcpListener, UdpSocket};
use tokio::sync::mpsc;

use crate::datastate::{DataState, SharedDataState};
use crate::endpoints;
use crate::metrics::{update_metrics, Metrics, SharedMetrics};
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

/// Type for packet event message channel
type PacketEvent = Result<Vec<u8>, Box<dyn std::error::Error + Send>>;

/// Binds a UDP socket on `addr` and sends [`PacketEvent`] over `event_tx`.
///
/// Designed to handle short bursts of many packets. Individual packets are
/// pushed to the `event_tx` channel to be handled elsewhere.
///
/// Runs indefinitely; intended to be spawned with [`tokio::spawn`].
///
/// # Panics
/// Panics if the socket cannot be bound.
async fn packet_recv(addr: SocketAddr, event_tx: mpsc::Sender<PacketEvent>) -> std::io::Result<()> {
    let socket = UdpSocket::bind(addr).await?;

    println!("UDP listener bound to {addr}");

    // Allocate a buffer large enough for any valid UDP datagram.
    let mut buf = vec![0u8; u16::MAX as usize];
    // Need to set some metadata on first iteration, then check on
    // each subsequent iteration
    loop {
        // Wait until the socket is readable
        socket.readable().await?;
        // Pass through the entire OS buffer
        loop {
            match socket.try_recv_from(&mut buf) {
                Ok((len, _)) => {
                    let _ = event_tx.try_send(Ok(buf[..len].to_vec()));
                }
                Err(ref e) if e.kind() == WouldBlock => {
                    // No packet available, so assume that we've pulled
                    // everything from the OS buffer
                    break;
                }
                Err(e) => {
                    let _ = event_tx.try_send(Err(Box::new(e)));
                }
            }
        }
    }
}

/// Drains [`PacketEvent`]s from `rx` and updates metrics.
///
/// Runs indefinitely - intended to be run with [`tokio::spawn`].
/// Exits when all senders are dropped.
///
/// As-is, this assumes that metrics are very fast to compute, since
/// this triggers on every new packet. If we wanted to include more
/// complicated metrics, best approach is likely to switch to a fixed
/// cadence instead of packet event.
async fn packet_handler(
    mut rx: mpsc::Receiver<PacketEvent>,
    metrics: SharedMetrics,
    state: SharedDataState,
) {
    let mut last_update_id: u64 = 0;

    // NB: it's possible that this could become a bottleneck, in which
    // case we could make it multi-threaded
    while let Some(event) = rx.recv().await {
        // TODO: can we clean up this nested match?
        match event {
            Ok(bytes) => {
                let packet_id = match Packet::parse(&bytes) {
                    Ok(packet) => match state.push(&packet) {
                        Ok(id) => id,
                        Err(e) => {
                            eprintln!("Error pushing packet to state: {e}");
                            continue;
                        }
                    },
                    Err(e) => {
                        eprintln!("Error parsing packet: {e}");
                        continue;
                    }
                };
                // Only update metrics if the packet has a different
                // ID from the last one seen.
                if packet_id != last_update_id {
                    last_update_id = packet_id;
                    // Only update computationally cheap metrics
                    update_metrics(&metrics, &state);
                }
            }
            Err(err) => {
                eprintln!("{err}");
            } // TODO: increment a metric
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
    let metrics: SharedMetrics = Arc::new(Metrics::default());
    // Creates the mpsc channel used to send [`PacketEvent`]s from the UDP
    // listener to the metrics updater. Returns `(sender, receiver)`.
    // The channel is bounded to 2048 events; if the updater falls behind, senders
    // use [`try_send`](mpsc::Sender::try_send) and drop events rather than
    // blocking the UDP loop.
    let (packet_tx, packet_rx) = mpsc::channel::<PacketEvent>(2048);

    let packet_handler = tokio::spawn(packet_handler(
        packet_rx,
        Arc::clone(&metrics),
        Arc::clone(&state),
    ));
    let packet_recv = tokio::spawn(packet_recv(udp_addr, packet_tx));

    let listener = TcpListener::bind(http_addr)
        .await
        .unwrap_or_else(|e| panic!("Failed to bind HTTP listener on {http_addr}: {e}"));

    let http = tokio::spawn(axum::serve(listener, router(state, metrics)).into_future());
    println!("HTTP listening on {http_addr}");

    tokio::select! {
        _ = packet_handler => eprintln!("Packet handler exited unexpectedly"),
        _ = packet_recv => eprintln!("UDP receiver exited unexpectedly"),
        _ = http => eprintln!("HTTP server exited unexpectedly"),
    }
}
