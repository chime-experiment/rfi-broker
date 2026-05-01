//! Server, router, and UDP receiver.
//!
//! # Endpoints
//! - ``metadata``: Most recent packet header metadata
//! - ``metrics``: Application prometheus metrics
//! - ``bad_input_likelihood``: per-input likelihood of a feed being corrupted
//!
//! # Debug endpoints
//! - ``data``: Pretty print of most recent buffer frame

use std::io::ErrorKind::WouldBlock;
use std::net::SocketAddr;
use std::sync::Arc;

use eyre::WrapErr;

#[cfg(debug_assertions)]
use axum::middleware::{self, Next};
use axum::{Router, routing::get};

use tokio::net::{TcpListener, UdpSocket};

use crate::config::AppConfig;
use crate::datastate::{DataState, SharedDataState};
use crate::endpoints;
use crate::metrics::{Metrics, SharedMetrics, update_metrics};
use crate::packet::Packet;

/// Size in MB for the UDP socket buffer
const UDP_BUF_SIZE_MB: usize = 8;

/// Middleware to emit a debug message every time an endpoint is triggered.
///
/// Only available in debug build.
#[cfg(debug_assertions)]
async fn debug_log_middleware(
    req: axum::http::Request<axum::body::Body>,
    next: Next,
) -> axum::response::Response {
    tracing::debug!(
        method = %req.method(),
        uri = %req.uri(),
        "-> request:"
    );

    let response = next.run(req).await;

    tracing::debug!(status = %response.status(), "<- response:");

    response
}

/// Builds a [`Router`] containing all the endpoints we'd like to enable.
///
/// `state` and `metrics` are injected as Axum [`State`]s so handlers
/// can read them.
fn make_router(state: SharedDataState, metrics: SharedMetrics) -> Router {
    // router using information from the data state
    let state_router = Router::new()
        .route("/metadata", get(endpoints::metadata))
        .route(
            "/bad_input_likelihood",
            get(endpoints::get_bad_input_likelihood),
        )
        .route("/", get(endpoints::dump_bad_input_likelihood));

    // debug-only endpoints
    #[cfg(debug_assertions)]
    let state_router = state_router.route("/data", get(endpoints::data));

    // Include the state
    let state_router = state_router.with_state(state);

    // router for metrics
    let metrics_router = Router::new()
        .route("/metrics", get(endpoints::metrics))
        .with_state(metrics);

    let router = Router::new().merge(state_router).merge(metrics_router);

    #[cfg(debug_assertions)]
    let router = router.layer(middleware::from_fn(debug_log_middleware));

    router
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
    tracing::info!(
        addr = ?addr,
        requested_MB = ?UDP_BUF_SIZE_MB,
        actual_MB = ?actual,
        "created UDP socket:",
    );

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

        if last_update_id == 0 {
            tracing::debug!("started receiving packets");
        }

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
                            tracing::warn!(error = ?e, "error pushing packet to shared state");
                        }
                    },
                    // failed to parse the packet
                    Err(e) => {
                        metrics.packet_loss.inc_lost();
                        tracing::warn!(error = ?e, "error parsing received packet");
                    }
                },
                // no packet available - release the thread
                Err(ref e) if e.kind() == WouldBlock => {
                    break;
                }
                // error occured during UDP read
                Err(e) => {
                    metrics.packet_loss.inc_lost();
                    tracing::debug!(error = ?e, "UDP recv error");
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
pub async fn run(
    http_addr: SocketAddr,
    udp_addr: SocketAddr,
    config: Option<AppConfig>,
) -> eyre::Result<()> {
    let state: SharedDataState = Arc::new(DataState::default());
    let metrics: SharedMetrics = Arc::new(Metrics::default());
    let config: Option<Arc<AppConfig>> = config.map(Arc::new);

    // Start the solar event task
    let solar = tokio::spawn(crate::tasks::solar_event_task(
        Arc::clone(&metrics),
        config.map(|c| Arc::clone(&c)),
    ));

    // Construct the socket and start the packet handling task. If this becomes
    // a bottleneck, it could be run in multiple threads
    let udp_sock = construct_sock(udp_addr).await?;

    let packet_handler = tokio::spawn(packet_handler_task(
        udp_sock,
        Arc::clone(&metrics),
        Arc::clone(&state),
    ));

    let http_listener = TcpListener::bind(http_addr).await?;

    let http = tokio::spawn(axum::serve(http_listener, make_router(state, metrics)).into_future());
    tracing::info!(addr = ?http_addr, "started HTTP server:");

    tokio::select! {
        result = packet_handler => result?.wrap_err("packet handler failed"),
        result = http => result?.wrap_err("http server failed"),
        result = solar => result?.wrap_err("solar zeroing task failed"),
    }
}
