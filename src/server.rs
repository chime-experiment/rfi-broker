//! HTTP server setup and server.
//!
//! [`router`] builds the Axum [`Router`] from a [`SharedDataState`], and [`serve`]
//! creates the shared [`RingBuffer`], spawns the UDP listener task, binds a
//! TCP listener, and runs the HTTP server to completion.

use std::future::IntoFuture;
use std::net::SocketAddr;
use std::sync::Arc;

use axum::{routing::get, Router};

use crate::datastate::SharedDataState;
use crate::handlers;

/// Builds a [`Router`] containing all the endpoints we'd like to enable.
///
/// `store` is injected as Axum [`State`] so handlers can read the UDP buffers.
pub fn router(state: SharedDataState) -> Router {
    let mut router = Router::new();
    router = router.route("/data", get(handlers::data));

    router.with_state(state)
}

/// Spawns the UDP listener and HTTP server as ndependent tasks,
/// then waits for either to exit.
///
/// # Panics
/// Panics if either address cannot be bound.
pub async fn serve(http_addr: SocketAddr, udp_addr: SocketAddr, state: SharedDataState) {
    let udp = tokio::spawn(crate::udp::run_listener(udp_addr, Arc::clone(&state)));

    let listener = tokio::net::TcpListener::bind(http_addr)
        .await
        .unwrap_or_else(|e| panic!("Failed to bind HTTP listener on {http_addr}: {e}"));

    let http = tokio::spawn(axum::serve(listener, router(state)).into_future());
    println!("HTTP listening on {http_addr}");

    tokio::select! {
        _ = udp  => eprintln!("UDP listener exited unexpectedly"),
        _ = http => eprintln!("HTTP server exited unexpectedly"),
    }
}
