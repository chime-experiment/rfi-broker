//! Fixtures for unit tests

use crate::packet;

/// Generate a [`packet:Packet`] with specific frequency IDs.
pub fn packet(freq_ids: Vec<u32>) -> packet::Packet {
    let nsamp = freq_ids.len();

    let header = packet::Header {
        version: 2_u16,
        payload_length: 26_u32,
        sk_step: 8_u32,
        num_elements: 10_u32,
        samples_per_data_set: 32_u32,
        num_total_freq: 4_u32,
        num_local_freq: 2_u32,
        frames_per_packet: 2_u32,
        seq_num: 0_i64,
    };

    let body = packet::Body {
        freq_ids,
        frac_flagged: vec![0.2; nsamp],
        sktilde_avg: vec![1.3; nsamp],
        bad_feed_counts: vec![0u8; 10 * nsamp],
    };

    packet::Packet { header, body }
}
