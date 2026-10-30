//! UDP socket and packet parsing.

use std::sync::Arc;

use std::io::ErrorKind::WouldBlock;
use std::net::SocketAddr;

use tokio::net::UdpSocket;
use tokio::time::{Duration, Instant, sleep_until};

use super::packet::Packet;

/// Size in MB for the UDP socket buffer
const UDP_BUF_SIZE_MB: usize = 8;
/// Time in seconds between packets before all pending packets are
/// flushed to the data state buffers
const DATA_STATE_FLUSH_TIMEOUT_SECONDS: u64 = 5;

/// Construct a UDP socket with a buffer large enough to handle burst events.
pub async fn construct_sock(addr: SocketAddr) -> Result<UdpSocket, std::io::Error> {
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
        udp_addr = ?addr,
        requested_MB = ?UDP_BUF_SIZE_MB,
        actual_MB = ?actual,
        "created UDP socket",
    );

    Ok(socket)
}

/// Drains a UDP socket buffer and pushes packets to the [`Buffers`].
///
/// Runs indefinitely - intended to be run with [`tokio::spawn`].
pub async fn packet_handler_task(
    sock: UdpSocket,
    buffers: Arc<crate::state::Buffers>,
    metrics: Arc<crate::state::Metrics>,
) -> Result<(), std::io::Error> {
    // Record the first received packet
    let mut log_packet_recv: bool = true;
    // Allocate a buffer large enough for any valid UDP packet
    let mut buf = vec![0u8; u16::MAX as usize];

    // Record how long it's been since the last packet was received
    let flush_timeout = Duration::from_secs(DATA_STATE_FLUSH_TIMEOUT_SECONDS);
    let flush_timer = sleep_until(Instant::now() + flush_timeout);

    tokio::pin!(flush_timer);

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

                    // If there was an error in `try_recv_from`, go back and
                    // try to read another packet
                    let Some(nbytes) = nbytes else { continue };

                    Packet::parse(buf.get(..nbytes).unwrap_or_default())
                        // push to state if parse was successful
                        .and_then(|packet| buffers.push(packet))
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
                let nsamp = buffers.flush();
                // Reset the timer so we don't sit in this busy loop
                flush_timer.as_mut().reset(Instant::now() + flush_timeout);

                if nsamp > 0 {
                    tracing::info!(
                        "No packets received for {DATA_STATE_FLUSH_TIMEOUT_SECONDS}s - \
                        flushed {nsamp} pending samples to state buffers"
                    );
                }

                // log the next time we start receiving packets
                log_packet_recv =  true;
            }
        }
    }
}
