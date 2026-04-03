//! Server, router, and UDP listener.
//!
//! # Endpoints
//! - ``metadata``: Most recent packet header metadata
//! - ``data``: Dump all ringbuffers
//! - ``bad_input_likelihood``: per-input likelihood of a feed being corrupted

use std::future::IntoFuture;
use std::net::SocketAddr;
use std::sync::Arc;

use axum::{routing::get, Router};

use tokio::net::{TcpListener, UdpSocket};
use tokio::sync::mpsc;

use crate::datastate::{DataState, SharedDataState};
use crate::endpoints;
use crate::metrics::{update_basic_metrics, Metrics, SharedMetrics};
use crate::packet::Packet;

/// Builds a [`Router`] containing all the endpoints we'd like to enable.
///
/// `store` is injected as Axum [`State`] so handlers can read the UDP buffers.
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

/// Signal sent whenever a packet is received.
enum PacketEvent {
    Received,
    Dropped,
}

/// Binds a UDP socket on `addr` and pushes decoded packets into `state`.
///
/// Also emits a [`PacketEvent`] whenever a new packet is received.
///
/// Runs indefinitely; intended to be spawned with [`tokio::spawn`].
/// Datagrams that fail to parse are silently discarded.
///
/// # Panics
/// Panics if the socket cannot be bound.
async fn udp_listener(
    addr: SocketAddr,
    state: SharedDataState,
    event_tx: mpsc::Sender<PacketEvent>,
) {
    let socket = UdpSocket::bind(addr)
        .await
        .unwrap_or_else(|e| panic!("Failed to bind UDP socket on {addr}: {e}"));

    println!("UDP listener bound to {addr}");

    // Allocate a buffer large enough for any valid UDP datagram.
    let mut buf = vec![0u8; u16::MAX as usize];
    // Need to set some metadata on first iteration, then check on
    // each subsequent iteration
    loop {
        let len = match socket.recv(&mut buf).await {
            Err(e) => {
                eprintln!("UDP recv error on {addr}: {e}");
                let _ = event_tx.try_send(PacketEvent::Dropped);
                continue;
            }
            Ok(len) => {
                let _ = event_tx.try_send(PacketEvent::Received);
                len
            }
        };

        // NB: this could be a bottleneck if packets are recieved faster than the
        // push can update
        match Packet::parse(&buf[..len]) {
            Ok(packet) => state
                .push(&packet)
                .unwrap_or_else(|e| eprintln!("Error pushing packet to app state: {e}")),
            Err(e) => eprintln!("Error parsing packet: {e}"),
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
async fn packet_event_handler(
    mut rx: mpsc::Receiver<PacketEvent>,
    metrics: SharedMetrics,
    state: SharedDataState,
) {
    let mut last_update_id: i64 = 0;
    while let Some(event) = rx.recv().await {
        match event {
            PacketEvent::Received => {
                // Only update metrics if the packet has a different
                // ID from the last one seen.
                let id = state.metadata.read().unwrap().id();
                if id != last_update_id {
                    last_update_id = id;
                    // Only update computationally cheap metrics
                    update_basic_metrics(&metrics, &state);
                }
            }
            PacketEvent::Dropped => {} // no-op
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
    // The channel is bounded to 256 events; if the updater falls behind, senders
    // use [`try_send`](mpsc::Sender::try_send) and drop events rather than
    // blocking the UDP loop.
    let (metrics_tx, metrics_rx) = mpsc::channel(256);

    let packet_event = tokio::spawn(packet_event_handler(
        metrics_rx,
        Arc::clone(&metrics),
        Arc::clone(&state),
    ));
    let udp = tokio::spawn(udp_listener(udp_addr, Arc::clone(&state), metrics_tx));

    let listener = TcpListener::bind(http_addr)
        .await
        .unwrap_or_else(|e| panic!("Failed to bind HTTP listener on {http_addr}: {e}"));

    let http = tokio::spawn(axum::serve(listener, router(state, metrics)).into_future());
    println!("HTTP listening on {http_addr}");

    tokio::select! {
        _ = packet_event => eprintln!("Packet received handler exited unexpectedly"),
        _ = udp  => eprintln!("UDP listener exited unexpectedly"),
        _ = http => eprintln!("HTTP server exited unexpectedly"),
    }
}
