//! Server, router, and UDP listener.

use std::future::IntoFuture;
use std::io::Cursor;
use std::net::SocketAddr;
use std::sync::Arc;

use axum::{routing::get, Router};
use binrw::BinRead;

use tokio::net::{TcpListener, UdpSocket};

use crate::datastate::SharedDataState;
use crate::handlers;
use crate::packet::Packet;

/// Builds a [`Router`] containing all the endpoints we'd like to enable.
///
/// `store` is injected as Axum [`State`] so handlers can read the UDP buffers.
pub fn router(state: SharedDataState) -> Router {
    let mut router = Router::new();
    router = router.route("/data", get(handlers::data));
    router = router.route("/inputs", get(handlers::get_bad_input_likelihood));
    router = router.route("/", get(handlers::dump_bad_input_likelihood));

    router.with_state(state)
}

/// Binds a UDP socket on `addr` and forwards decoded packets into `state`.
///
/// Runs indefinitely; intended to be spawned with [`tokio::spawn`].
/// Datagrams that fail to parse are silently discarded.
///
/// # Panics
/// Panics if the socket cannot be bound.
pub async fn udp_listener(addr: SocketAddr, state: SharedDataState) {
    let socket = UdpSocket::bind(addr)
        .await
        .unwrap_or_else(|e| panic!("Failed to bind UDP socket on {addr}: {e}"));

    println!("UDP listener bound to {addr}");

    // Allocate a buffer large enough for any valid UDP datagram.
    let mut buf = vec![0u8; u16::MAX as usize];
    // Need to set some metadata on first iteration, then check on
    // each subsequent iteration
    let mut first_packet = true;
    loop {
        let len = match socket.recv(&mut buf).await {
            Err(e) => {
                eprintln!("UDP recv error on {addr}: {e}");
                continue;
            }
            Ok(len) => len,
        };
        // Parse
        let mut cursor = Cursor::new(&buf[..len]);

        match Packet::read_le(&mut cursor) {
            Ok(packet) => state
                .push(&packet)
                .unwrap_or_else(|e| eprintln!("Error pushing packet to app state: {e}")),
            Err(e) => eprintln!("Error parsing packet: {e}"),
        }

        if first_packet {
            first_packet = false;
        }
    }
}

/// Spawns the UDP listener and HTTP server as ndependent tasks,
/// then waits for either to exit.
///
/// # Panics
/// Panics if either address cannot be bound.
pub async fn serve(http_addr: SocketAddr, udp_addr: SocketAddr, state: SharedDataState) {
    let udp = tokio::spawn(udp_listener(udp_addr, Arc::clone(&state)));

    let listener = TcpListener::bind(http_addr)
        .await
        .unwrap_or_else(|e| panic!("Failed to bind HTTP listener on {http_addr}: {e}"));

    let http = tokio::spawn(axum::serve(listener, router(state)).into_future());
    println!("HTTP listening on {http_addr}");

    tokio::select! {
        _ = udp  => eprintln!("UDP listener exited unexpectedly"),
        _ = http => eprintln!("HTTP server exited unexpectedly"),
    }
}
