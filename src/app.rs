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

use axum::{Router, routing::get};
#[cfg(debug_assertions)]
use {
    axum::middleware::{self, Next},
    axum::routing::post,
};

use tokio::net::{TcpListener, UdpSocket};
use tokio::time::{Duration, Instant, sleep_until};

use crate::config::AppConfig;
use crate::datastate::{DataState, SharedDataState};
use crate::endpoints;
use crate::metrics::{Metrics, SharedMetrics};
use crate::packet::Packet;

/// Size in MB for the UDP socket buffer
const UDP_BUF_SIZE_MB: usize = 8;
/// Time in seconds between packets before all pending packets are
/// flushed to the data state buffers
const DATA_STATE_FLUSH_TIMEOUT_SECONDS: u64 = 5;
/// Time in seconds before all data is cleared from the
/// data state buffers
const DATA_STATE_CLEAR_TIMEOUT_SECONDS: u64 = 60;

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
    let state_router = Router::new().route("/metadata", get(endpoints::metadata));

    // debug-only endpoints
    #[cfg(debug_assertions)]
    let state_router = state_router
        .route("/last-frame", get(endpoints::last_frame))
        .route("/write-data", post(endpoints::write_data));

    // Include the state
    let state_router = state_router.with_state(state);

    // router for metrics
    let metrics_router = Router::new()
        .route("/metrics", get(endpoints::metrics))
        .route(
            "/bad_input_likelihood",
            get(endpoints::get_bad_input_likelihood),
        )
        .route("/", get(endpoints::dump_bad_input_likelihood))
        .with_state(metrics);

    let router = Router::new().merge(state_router).merge(metrics_router);

    #[cfg(debug_assertions)]
    let router = router.layer(middleware::from_fn(debug_log_middleware));

    router
}

/// Construct a UDP socket with a buffer large enough to handle burst events.
async fn construct_sock(addr: SocketAddr) -> Result<UdpSocket, std::io::Error> {
    let socket = UdpSocket::bind(addr).await?;

    // Borrow the socket as a socket2 ref to increase buffer size
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
    // Record the first received packet
    let mut log_packet_recv: bool = true;
    // Allocate a buffer large enough for any valid UDP packet
    let mut buf = vec![0u8; u16::MAX as usize];

    // Record how long it's been since the last packet was received
    let flush_timeout = Duration::from_secs(DATA_STATE_FLUSH_TIMEOUT_SECONDS);
    let clear_timeout = Duration::from_secs(DATA_STATE_CLEAR_TIMEOUT_SECONDS);
    let flush_timer = sleep_until(Instant::now() + flush_timeout);
    let clear_timer = sleep_until(Instant::now() + clear_timeout);

    tokio::pin!(flush_timer);
    tokio::pin!(clear_timer);

    loop {
        tokio::select! {
            biased; // always poll the socket first
            // Wait until socket is readable. Making this asynchronous mean that this thread
            // will be released each time it drains the os buffer. This ends up being less
            // performant than having a permanent thread always listening, but the effect
            // is negligible for the amount of data that we're receiving. If we ever end up
            // wanting higher throughput, this should probably just become a fixed thread.
            result = sock.readable() => {
                result?;

                if log_packet_recv {
                    tracing::info!("started receiving packets");
                    log_packet_recv = false;
                }

                // drain the entire OS buffer
                loop {
                    // try to read a packet, breaking the inner loop if the
                    // os buffer is empty
                    let nbytes = match sock.try_recv_from(&mut buf) {
                        Ok((nbytes, _)) => Some(nbytes),
                        Err(ref e) if e.kind() == WouldBlock => break,
                        // something happened - log it and move on
                        Err(e) => {
                            metrics.packet_loss.inc_lost();
                            tracing::debug!(error = ?e, "UDP recv error:");
                            None
                        }
                    };

                    // Got a packet - reset the `flush` timer
                    flush_timer.as_mut().reset(Instant::now() + flush_timeout);
                    clear_timer.as_mut().reset(Instant::now() + clear_timeout);

                    // If there was an error in `try_recv_from`, go back and
                    // try to read another packet
                    let Some(nbytes) = nbytes else { continue };

                    Packet::parse(buf.get(..nbytes).unwrap_or_default())
                        // push to state if parse was successful
                        .and_then(|packet| state.push(packet))
                        .map_or_else(
                            // push failed - log it and move on
                            |e| {
                                metrics.packet_loss.inc_lost();
                                tracing::warn!(error = ?e, "error handling received packet:");
                            },
                            |_| metrics.packet_loss.inc_recv(),
                        );

                }
            }

            () = &mut flush_timer => {
                // Flush any pending data to the state buffer and reset the timer
                let nsamp = state.flush();

                if nsamp > 0 {
                    tracing::info!(
                        "No packets received for {DATA_STATE_FLUSH_TIMEOUT_SECONDS}s - \
                        flushed {nsamp} pending samples to state buffers"
                    );
                }
                // log the next time we start receiving packets
                log_packet_recv =  true;
            }

            () = &mut clear_timer => {
                // Remove all data from the state buffers
                let nsamp = state.clear();

                if nsamp > 0 {
                    tracing::info!("No packets received for {DATA_STATE_CLEAR_TIMEOUT_SECONDS}s - \
                        clearing {nsamp} samples from state buffers"
                    );
                }
                // log the next time we stat receiving packets
                log_packet_recv = true;
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
    let rfi_zeroing = tokio::spawn(crate::tasks::solar_event_task(
        Arc::clone(&metrics),
        config.map(|c| Arc::clone(&c)),
    ));

    // Start the bad input task
    let bad_inputs = tokio::spawn(crate::tasks::bad_input_task(
        Arc::clone(&state),
        Arc::clone(&metrics),
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
        result = rfi_zeroing => result?.wrap_err("solar zeroing task failed"),
        result = bad_inputs => result?.wrap_err("bad input metric task failed"),
    }
}
