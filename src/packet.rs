//! Definition of the UDP packet.

// This header must match the one defined in `kotekan`:
// https://github.com/kotekan/kotekan/blob/chord/lib/utils/rfi_functions.h#L14
use std::io::Cursor;

use binrw::{BinRead, BinWrite};
use serde::Serialize;

/// The protocol version to accept. Packets with any other version number
/// are discarded.
const EXPECTED_VERSION: u16 = 2;

/// Packet-specific `stream_id` type
#[derive(BinRead, BinWrite, Debug, Default, Clone, Copy, PartialEq, Serialize)]
#[brw(little)]
#[allow(
    dead_code,
    non_camel_case_types,
    reason = "mirrors the type defined in kotekan"
)]
pub struct stream_t {
    id: u64,
}

/// Decoded header from a UDP datagram.
///
/// `#[derive(BinRead)]` with `#[br(little)]` instructs `binrw` to deserialize
/// each field in order from a little-endian byte stream, eliminating manual
/// offset arithmetic.
#[derive(BinRead, BinWrite, Debug, Default, PartialEq, Clone, Copy, Serialize)]
#[brw(little)]
pub struct Header {
    /// Protocol version number - must equal [`EXPECTED_VERSION`]
    #[br(assert(version == EXPECTED_VERSION, "unexpected version {}", version))]
    pub version: u16,
    /// Total payload length
    pub payload_length: u32,
    /// Time integration length of SK values.
    pub sk_step: u32,
    /// Number of elements/inputs
    pub num_elements: u32,
    /// Number of FPGA time samples in each frame
    pub samples_per_data_set: u32,
    /// Total number of system frequencies
    pub num_total_freq: u32,
    /// Number of local (per-packet) frequencies
    pub num_local_freq: u32,
    /// Number of frames integrated per-packet
    pub frames_per_packet: u32,
    /// FPGA sequence number of the first sample integrated into the packet
    pub seq_num: i64,
    /// Current `stream_id` value
    pub stream_id: stream_t,
}

impl Header {
    /// Get a numeric ID value for this packet
    #[must_use]
    pub const fn id(&self) -> i64 {
        self.seq_num
    }

    /// Check that values which *shouldn't* change are equal
    ///
    /// # Errors
    /// Errors if `self` and `other` are not equal for the
    /// expected fields
    pub fn check_expected_equal(&self, other: &Self) -> Result<(), String> {
        // Clone *other* and update the members that we expect
        // could have changed
        let mut other_c = *other; // Header is Copy
        other_c.seq_num = self.seq_num;
        other_c.stream_id.id = self.stream_id.id;

        if *self != other_c {
            return Err(format!(
                "Mismatched header values. Expected {self:?}, got {other:?}"
            ));
        }
        Ok(())
    }
}

/// Description of packet payload contents.
#[derive(BinRead, BinWrite, Debug, PartialEq)]
#[br(little, import { hdr: &Header })]
#[bw(little)]
pub struct Body {
    /// List of frequencies contained in this packet
    #[br(count = hdr.num_local_freq)]
    pub freq_ids: Vec<u32>,
    /// Fraction of flagged samples per frequency
    #[br(count = hdr.num_local_freq)]
    pub frac_flagged: Vec<f32>,
    /// Average SK per frequency
    #[br(count = hdr.num_local_freq)]
    pub sktilde_avg: Vec<f32>,
    /// Bad feed counter per frequency and element
    #[br(count = hdr.num_local_freq * hdr.num_elements)]
    pub bad_feed_counts: Vec<u8>,
}

/// Entire packet
#[derive(BinRead, BinWrite, Debug, PartialEq)]
#[brw(little)]
pub struct Packet {
    /// packet header
    pub header: Header,
    /// packet body
    #[br(args { hdr: &header })]
    pub body: Body,
}

impl Packet {
    /// Parse from bytes.
    ///
    /// # Errors
    /// Errors if the input buffer cannot be parsed.
    pub fn parse(buf: &[u8]) -> Result<Self, String> {
        let mut cursor = Cursor::new(&buf);

        Self::read_le(&mut cursor).map_err(|e| format!("Error parsing packet: {e}"))
    }

    /// Write to bytes
    ///
    /// # Errors
    /// Errors if writing fails
    pub fn to_vec(&self) -> Result<Vec<u8>, String> {
        let mut cursor = Cursor::new(Vec::new());

        self.write_le(&mut cursor)
            .map_err(|e| format!("failed to write to bytes: {e}"))?;

        Ok(cursor.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bin_roundtrip() {
        let header = Header {
            version: 2_u16,
            payload_length: 26_u32,
            sk_step: 8_u32,
            num_elements: 10_u32,
            samples_per_data_set: 32_u32,
            num_total_freq: 4_u32,
            num_local_freq: 2_u32,
            frames_per_packet: 2_u32,
            seq_num: 0_i64,
            stream_id: stream_t { id: 101_u64 },
        };

        let body = Body {
            freq_ids: vec![0, 1],
            frac_flagged: vec![0.2, 0.7],
            sktilde_avg: vec![1.3, 1.1],
            bad_feed_counts: vec![0u8; 20],
        };

        let packet = Packet { header, body };

        let bin = packet.to_vec().unwrap();
        let parsed = Packet::parse(&bin).unwrap();

        // NB: this shouldn't fail, but one possible reason would
        // be floating point error. Consider `approx` crate and
        // `assert_relative_eq!` if this is an issue.
        assert_eq!(packet, parsed);
    }
}
