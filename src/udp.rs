//! UDP listener and packet parsing.
//!
//! # Wire format
//!
//! Every UDP datagram has a fixed-size header followed by a variable-length
//! payload of packed little-endian `f32` values. The Header format is defined
//! in [`Header`].

use std::net::SocketAddr;

use binrw::BinReaderExt;
use ndarray::{ArrayD, IxDyn};
use tokio::net::UdpSocket;

use crate::datastate::{SharedDataState, TypedArray, TypedBuffer};
use crate::header::Header;

/// Decoded datasets from a single UDP packet
pub struct ParsedPacket {
    pub header: Header,
    pub datasets: Vec<(String, TypedArray)>,
}

/// Parse bytes into a [`TypedArray`] based on the expected type
macro_rules! impl_typed_parse {
    ($( $variant:ident => $type:ty ), *) => {
        // Allow unused since this could in theory be called for something
        // that doesn't match a [`TypedBuffer`] variant
        #[allow(unused)]
        fn bytes_to_typed_array(
            bytes: &[u8],
            buf: &TypedBuffer
        ) -> Result<TypedArray, Box<dyn std::error::Error>> {
            match buf {
                $(
                    TypedBuffer::$variant(_) => {
                        let values: Vec<$type> = bytes
                            .chunks_exact(std::mem::size_of::<$type>())
                            .map(|b| {
                                <$type>::from_le_bytes(b.try_into().expect("Chunk shape is correct"))
                            })
                            .collect();
                        let arr = ArrayD::from_shape_vec(IxDyn(&buf.shape()), values)?;

                        Ok(TypedArray::$variant(arr))
                    }
                )*
                _ => Err("No matching array type".into()),
            }
        }
    }
}

impl_typed_parse! {
    U8 => u8,
    U16 => u16,
    U32 => u32,
    U64 => u64,
    F32 => f32,
    F64 => f64
}

/// Attempts to decode a datagram into a [`ParsedPacket`].
///
/// The header dims are cross-referenced against each ring buffer's declared
/// shape. The payload is sliced sequentially in config order, with each
/// dataset's byte footprint determined by its shape and element type.
///
/// `binrw` drives header deserialization via [`BinReaderExt::read_le`] on a
/// [`std::io::Cursor`], so no manual offset arithmetic is needed. The cursor
/// position after parsing is used as the payload offset, avoiding a separately
/// maintained `HEADER_SIZE` constant. Returns `None` if the header is invalid
/// (wrong magic, too short) or if the payload length does not exactly match
/// the dimensions declared in the header.
// TODO: preserve the expected state based on the first packet we get instead of
// fully re-checking every time
fn parse_packet(bytes: &[u8], state: &SharedDataState) -> Option<ParsedPacket> {
    let mut cursor = std::io::Cursor::new(bytes);
    // TODO: some additional checks on the header
    let header: Header = cursor.read_le().ok()?;
    let payload = &bytes[cursor.position() as usize..];

    if payload.len() as u32 != header.payload_length {
        eprintln!(
            "Mismatch between expected and received payload size. Expected {}, got {}",
            header.payload_length,
            payload.len(),
        );
        return None;
    }

    // Validate that expected buffers are consistant with the payload size
    let total_bytes: usize = state
        .buffers
        .values()
        .map(|buf| buf.shape().iter().product::<usize>() * buf.element_bytes())
        .sum();

    if payload.len() != total_bytes {
        eprintln!(
            "Mismatch between payload size {} and available buffer space {total_bytes}.",
            payload.len()
        );
        return None;
    }

    // Slice the payload and parse each dataset in order
    let mut offset: usize = 0;
    let mut datasets = Vec::with_capacity(state.buffers.len());

    for (name, buf) in &state.buffers {
        let n_elements: usize = buf.shape().iter().product();
        let n_bytes: usize = n_elements * buf.element_bytes();

        // Slice the payload for the expected size of the
        // next array, based on the order in `state`
        let chunk: &[u8] = payload.get(offset..offset + n_bytes)?;
        offset += n_bytes;

        if let Ok(array) = bytes_to_typed_array(chunk, buf) {
            datasets.push((name.clone(), array));
        }
    }

    Some(ParsedPacket {
        header: header,
        datasets: datasets,
    })
}

fn try_write_packet_to_state(
    bytes: &[u8],
    state: &SharedDataState,
    first_packet: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let parsed = parse_packet(bytes, state).ok_or("Failed to parse packet")?;

    for (name, array) in parsed.datasets {
        // Propagate if the dataset doesn't exist
        let tbuf = state
            .buffers
            .get(&name)
            .ok_or(format!("No matching dataset found with name `{name}`"))?;

        // Push to the array
        tbuf.push(array)?;
    }
    // Check the metadata
    if !first_packet {
        state
            .metadata
            .lock()
            .unwrap()
            .check_expected_equal(&parsed.header)?;
    }
    // Add the metadata since we got here successfully.
    // Need to momentarily get the guard
    state.metadata.lock().unwrap().update_from(&parsed.header)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Listener task
// ---------------------------------------------------------------------------

/// Binds a UDP socket on `addr` and forwards decoded packets into `state`.
///
/// Runs indefinitely; intended to be spawned with [`tokio::spawn`].
/// Datagrams that fail to parse are silently discarded.
///
/// # Panics
/// Panics if the socket cannot be bound.
pub async fn run_listener(addr: SocketAddr, state: SharedDataState) {
    let socket = UdpSocket::bind(addr)
        .await
        .unwrap_or_else(|e| panic!("Failed to bind UDP socket on {addr}: {e}"));

    println!("UDP listener bound to {addr}");

    // Allocate a buffer large enough for any valid UDP datagram.
    let mut buf = vec![0u8; u16::MAX as usize];
    // Need to set the header on first iteration, then check on
    // each subsequent iteration
    let mut first_packet = true;
    loop {
        match socket.recv(&mut buf).await {
            Ok(len) => {
                if let Err(e) = try_write_packet_to_state(&buf[..len], &state, first_packet) {
                    eprintln!("Error handling packet: {e}");
                }
            }
            Err(e) => eprintln!("UDP recv error: {e}"),
        }
        if first_packet {
            first_packet = false
        };
    }
}
