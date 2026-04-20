//! Server, router, and UDP receiver.
//!
//! # Endpoints
//! - ``metadata``: Most recent packet header metadata
//! - ``metrics``: Application prometheus metrics
//! - ``data``: Dump all ringbuffers
//! - ``bad_input_likelihood``: per-input likelihood of a feed being corrupted

use std::io::ErrorKind::WouldBlock;
use std::net::SocketAddr;
use std::sync::Arc;

use eyre::WrapErr;

use axum::{Router, routing::get};
use tokio::net::{TcpListener, UdpSocket};

use crate::config::AppConfig;
use crate::datastate::{DataState, SharedDataState};
use crate::endpoints;
use crate::metrics::{Metrics, SharedMetrics, update_metrics};
use crate::packet::Packet;

/// Size in MB for the UDP socket buffer
const UDP_BUF_SIZE_MB: usize = 8;

/// Builds a [`Router`] containing all the endpoints we'd like to enable.
///
/// `state` and `metrics` are injected as Axum [`State`]s so handlers
/// can read them.
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

/// Construct a UDP socket with a buffer large enough to handle burst events.
async fn construct_sock(addr: SocketAddr) -> Result<UdpSocket, std::io::Error> {
    let socket = UdpSocket::bind(addr).await?;

    // Borrow the socket as a socket2 ref to increase buffer
    let sock_ref = socket2::SockRef::from(&socket);
    sock_ref.set_recv_buffer_size(UDP_BUF_SIZE_MB * 1024 * 1024)?;

    // log the actual recv buffer size
    #[allow(
        clippy::cast_precision_loss,
        reason = "buffer size should never be large enough to cause precision loss"
    )]
    let actual = sock_ref.recv_buffer_size()? as f64 / 1024_f64 / 1024_f64;
    tracing::debug!("Request {UDP_BUF_SIZE_MB}MB recv buffer, got {actual}MB");

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
                // Received a packet
                Ok((len, _)) => match Packet::parse(buf.get(..len).unwrap_or_default()) {
                    // Successfully parsed the packet
                    Ok(packet) => match state.push(packet) {
                        // data state push successfull
                        Ok(id) => {
                            metrics.packet_loss.inc_recv();
                            if id != last_update_id {
                                last_update_id = id;
                                update_metrics(&metrics, &state);
                            }
                        }
                        // failed to push to the data state
                        Err(e) => {
                            metrics.packet_loss.inc_lost();
                            tracing::warn!("Error pushing packet to state: {:#?}", e);
                        }
                    },
                    // failed to parse the packet
                    Err(e) => {
                        metrics.packet_loss.inc_lost();
                        tracing::warn!("Error parsing packet: {:#?}", e);
                    }
                },
                // no packet available - release the thread
                Err(ref e) if e.kind() == WouldBlock => {
                    break;
                }
                // error occured during UDP read
                Err(e) => {
                    metrics.packet_loss.inc_lost();
                    tracing::debug!("UDP recv error: {:#?}", e);
                }
            }
        }
    }
}

/// Spawns the UDP listener and HTTP server as ndependent tasks,
/// then waits for either to exit.
///
/// # Errors
/// Errors if either address cannot be bound.
pub async fn serve(
    http_addr: SocketAddr,
    udp_addr: SocketAddr,
    config: Option<AppConfig>,
) -> eyre::Result<()> {
    let state: SharedDataState = Arc::new(DataState::default());
    let metrics: SharedMetrics = Arc::new(Metrics::default());
    let config: Option<Arc<AppConfig>> = config.map(Arc::new);

    // Start the solar event task, if a config was provided
    let solar = config.map_or_else(
        || {
            tracing::debug!("Solar zeroing disabled - no config was provided.");
            tokio::spawn(std::future::pending()) // never resolves, effectively disabled
        },
        |cfg| {
            tokio::spawn(crate::events::solar_event_task(
                Arc::clone(&metrics),
                Arc::clone(&cfg),
            ))
        },
    );

    // Construct the socket and start the packet handling task. If this becomes
    // a bottleneck, it could be run in multiple threads
    let udp_sock = construct_sock(udp_addr).await?;

    let packet_handler = tokio::spawn(packet_handler_task(
        udp_sock,
        Arc::clone(&metrics),
        Arc::clone(&state),
    ));

    let http_listener = TcpListener::bind(http_addr).await?;

    let http = tokio::spawn(axum::serve(http_listener, router(state, metrics)).into_future());
    tracing::info!("HTTP listening on {http_addr}");

    // NB: each task will shut down right away if any task fails. This is mostly fine,
    // but it would result in an undesirable zeroing state
    // TODO: make the zeroing task more robust to failures
    tokio::select! {
        result = packet_handler => result?.wrap_err("packet handler failed"),
        result = http => result?.wrap_err("http server failed"),
        result = solar => result?.wrap_err("solar zeroing task failed"),
    }
}
