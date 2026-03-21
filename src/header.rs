//! Definition of the UDP packet header.

// This header must match the one defined in `kotekan`:
// https://github.com/kotekan/kotekan/blob/chord/lib/utils/rfi_functions.h#L14

use binrw::BinRead;

/// The protocol version to accept. Packets with any other version number
/// are discarded.
const EXPECTED_VERSION: u16 = 2;

/// Packet-specific `stream_id` type
#[derive(BinRead, Debug, Default, Clone, PartialEq)]
#[allow(dead_code, non_camel_case_types)]
pub struct stream_t {
    id: u64,
}

/// Decoded header from a UDP datagram.
///
/// `#[derive(BinRead)]` with `#[br(little)]` instructs `binrw` to deserialize
/// each field in order from a little-endian byte stream, eliminating manual
/// offset arithmetic.
#[allow(dead_code)]
#[derive(BinRead, Debug, Default, PartialEq, Clone)]
#[br(little)]
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
    /// Current stream_id value
    pub stream_id: stream_t,
}

impl Header {
    /// Update values from another instance of [`Header`]
    pub fn update_from(&mut self, other: &Header) -> Result<(), Box<dyn std::error::Error>> {
        self.version = other.version;
        self.payload_length = other.payload_length;
        self.sk_step = other.sk_step;
        self.num_elements = other.num_elements;
        self.samples_per_data_set = other.samples_per_data_set;
        self.num_total_freq = other.num_total_freq;
        self.num_local_freq = other.num_local_freq;
        self.frames_per_packet = other.frames_per_packet;
        self.seq_num = other.seq_num;
        self.stream_id.id = other.stream_id.id;

        Ok(())
    }

    /// Check that values which *shouldn't* change wont change
    pub fn check_expected_equal(&self, other: &Header) -> Result<(), String> {
        // Clone *other* and update the members that we expect
        // could have changed
        let mut other_c = other.clone();
        other_c.seq_num = self.seq_num;
        other_c.stream_id.id = self.stream_id.id;

        if self != &other_c {
            return Err(format!(
                "Mismatched header values. Expected {self:?}, got {other:?}"
            ));
        }
        Ok(())
    }
}
