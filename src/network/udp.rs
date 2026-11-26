//! UDP socket and packet parsing.

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use std::io::ErrorKind::WouldBlock;
use std::net::SocketAddr;

use eyre::WrapErr;

use tokio::net::UdpSocket;
use tokio::time::{Duration, Instant, sleep};

use super::packet::Packet;
use rfi_broker::core::buffer::BufferError;

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
        "created UDP socket {addr}. Requested {UDP_BUF_SIZE_MB} MB, actual {actual} MB",
    );

    Ok(socket)
}

/// Tracks the duration since some sort of activity
pub struct NetworkActivity {
    start: Instant,
    last_ns: AtomicU64,
}

impl NetworkActivity {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
            last_ns: AtomicU64::new(0),
        }
    }
    pub fn touch(&self) -> eyre::Result<()> {
        let now = u64::try_from(self.start.elapsed().as_nanos())?;
        self.last_ns.fetch_max(now, Ordering::Relaxed); // never moves backwards
        Ok(())
    }
    pub fn idle_for(&self) -> eyre::Result<Duration> {
        let now = u64::try_from(self.start.elapsed().as_nanos())?;
        Ok(Duration::from_nanos(
            now.saturating_sub(self.last_ns.load(Ordering::Relaxed)),
        ))
    }
}

/// Tracks activity on consumer network receive tasks, and flushes state buffers
/// after a period of inactivity.
///
/// Runs indefinitely - intended to be run with [`tokio::spawn`].
pub async fn packet_monitor_task(
    buffers: Arc<crate::state::Buffers>,
    activity: Arc<NetworkActivity>,
) -> eyre::Result<()> {
    let timeout = Duration::from_secs(DATA_STATE_FLUSH_TIMEOUT_SECONDS);

    loop {
        let idle = activity.idle_for()?;
        let wait = if idle >= timeout {
            // no worker has seen a packet
            buffers.flush();
            timeout
        } else {
            timeout.checked_sub(idle).unwrap_or(Duration::from_secs(0))
        };

        sleep(wait).await;
    }
}

/// Drains a UDP socket buffer and pushes packets to the [`crate::state::Buffers`].
///
/// Runs indefinitely - intended to be run with [`tokio::spawn`].
pub async fn packet_handler_task(
    sock: Arc<UdpSocket>,
    activity: Arc<NetworkActivity>,
    buffers: Arc<crate::state::Buffers>,
    metrics: Arc<crate::state::Metrics>,
) -> eyre::Result<()> {
    // Record the first received packet
    let mut log_packet_recv: bool = true;
    // Allocate a buffer large enough for any valid UDP packet
    let mut buf = vec![0u8; u16::MAX as usize];

    loop {
        // Wait until socket is readable. Making this asynchronous mean that this thread
        // will be released each time it drains the os buffer. This ends up being less
        // performant than having a permanent thread always listening, but the effect
        // is negligible for the amount of data that we're receiving. If we ever end up
        // wanting higher throughput, this should probably just become a fixed thread.
        sock.readable().await.wrap_err("failed to read socket")?;

        if log_packet_recv {
            tracing::info!("started receiving packets");
            log_packet_recv = false;
        }

        // drain the entire OS buffer
        loop {
            // try to read a packet, breaking the inner loop if the
            // os buffer is empty
            let nbytes = match sock.try_recv_from(&mut buf) {
                Ok((nbytes, _)) => nbytes,
                Err(ref e) if e.kind() == WouldBlock => break,
                // something happened - log it and move on
                Err(e) => {
                    metrics.packet_loss.inc_lost();
                    tracing::warn!(error = ?e, "UDP recv error: {e}");
                    continue;
                }
            };

            // Got a packet - reset the activity timer
            activity.touch().map_err(std::io::Error::other)?;

            Packet::parse(buf.get(..nbytes).unwrap_or_default())
                // push to state if parse was successful
                .and_then(|packet| buffers.push(packet))
                .map_or_else(
                    // push failed - log it and move on
                    |e| {
                        if let Some(buf_err) = e.downcast_ref::<BufferError>() {
                            metrics.packet_loss.inc_skipped();
                            tracing::debug!("{:?}", buf_err.to_string());
                        } else {
                            metrics.packet_loss.inc_lost();
                            tracing::warn!(error = ?e, "error handling a packet: {e}");
                        }
                    },
                    |_| metrics.packet_loss.inc_recv(),
                );
        }
    }
}
