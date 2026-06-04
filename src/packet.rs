//! Structure of incoming UDP packets, plus parsing.
//!
//! The packet must match the one defined in `kotekan`:
//! <https://github.com/kotekan/kotekan/blob/chord/lib/utils/rfi_functions.h#L14>

use std::io::Cursor;

use binrw::{BinRead, BinWrite};
use serde::Serialize;

use eyre::{WrapErr, bail};

/// The protocol version to accept. Packets with any other version number
/// are discarded.
const EXPECTED_VERSION: u16 = 2;

/// Packet body type specification.
///
/// We only handle a single packet type, so there's no need for
/// complicated trait implementations.
pub mod packet_types {
    pub type FreqIdType = u32;
    pub type FracFlaggedType = f32;
    pub type SkType = f32;
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
}

impl Header {
    /// Check that values which *shouldn't* change are equal.
    ///
    /// # Errors
    /// Errors if `self` and `other` are not equal for the
    /// expected fields.
    pub fn check_expected_equal(&self, other: &Self) -> eyre::Result<()> {
        // Clone *other* and update the members that we expect
        // could have changed
        let mut other_c = *other; // Header is Copy
        other_c.seq_num = self.seq_num;

        if *self != other_c {
            bail!("Mismatched header values. Expected {self:?}, got {other:?}");
        }
        Ok(())
    }
}

/// Description of packet payload contents.
#[derive(BinRead, BinWrite, Clone, Default, Debug, PartialEq)]
#[br(little, import { hdr: &Header })]
#[bw(little)]
pub struct Body {
    /// List of frequencies contained in this packet
    #[br(count = hdr.num_local_freq)]
    pub freq_ids: Vec<packet_types::FreqIdType>,
    /// Fraction of flagged samples per frequency
    #[br(count = hdr.num_local_freq)]
    pub frac_flagged: Vec<packet_types::FracFlaggedType>,
    /// Average SK per frequency
    #[br(count = hdr.num_local_freq)]
    pub sktilde_avg: Vec<packet_types::SkType>,
    /// Average SK per frequency and element
    #[br(count = hdr.num_local_freq * hdr.num_elements)]
    pub skbar_avg: Vec<packet_types::SkType>,
}

/// Entire packet
#[derive(BinRead, BinWrite, Clone, Debug, Default, PartialEq)]
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
    pub fn parse(buf: &[u8]) -> eyre::Result<Self> {
        let mut cursor = Cursor::new(&buf);

        Self::read_le(&mut cursor).wrap_err("failed to parse packet from bytes")
    }

    /// Write to bytes.
    ///
    /// # Errors
    /// Errors if writing fails.
    #[cfg(test)]
    pub fn to_vec(&self) -> eyre::Result<Vec<u8>> {
        let mut cursor = Cursor::new(Vec::new());

        self.write_le(&mut cursor)
            .wrap_err("failed to write packet to bytes")?;

        Ok(cursor.into_inner())
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    /// Generate `n` [`packet:Packet`]s with increasing frequency IDs.
    pub fn make_packets(nfreq: u32, npackets: u32) -> eyre::Result<Vec<Packet>> {
        if !nfreq.is_multiple_of(npackets) {
            bail!(
                "Number of packets: {npackets} does not evenly divide number of frequencies: {nfreq}"
            );
        }

        #[allow(
            clippy::integer_division,
            reason = "already checked that these are divisible"
        )]
        let nfreq_per_packet = nfreq / npackets;

        let mut packets = Vec::<Packet>::new();

        for i in 0..npackets {
            let header = Header {
                version: 2_u16,
                payload_length: 0_u32, // placeholder
                num_elements: 10_u32,
                samples_per_data_set: 32_u32,
                num_total_freq: nfreq,
                num_local_freq: nfreq_per_packet,
                frames_per_packet: 2_u32,
                seq_num: 0_i64,
            };

            let body = Body {
                freq_ids: (i * nfreq_per_packet..(i + 1) * nfreq_per_packet).collect(),
                frac_flagged: vec![0.2; nfreq_per_packet as usize],
                sktilde_avg: vec![1.3; nfreq_per_packet as usize],
                skbar_avg: vec![packet_types::SkType::default(); 10 * nfreq_per_packet as usize],
            };

            packets.push(Packet { header, body });
        }

        Ok(packets)
    }
    #[test]
    fn test_bin_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
        let packets = make_packets(3, 1)?;
        let packet = packets
            .first()
            .ok_or("packet was not constructed successfully")?;

        let bin = packet.to_vec().unwrap();
        let parsed = Packet::parse(&bin).unwrap();

        // NB: this shouldn't fail, but one possible reason would
        // be floating point error. Consider `approx` crate and
        // `assert_relative_eq!` if this is an issue.
        assert_eq!(packet, &parsed);

        Ok(())
    }
}
