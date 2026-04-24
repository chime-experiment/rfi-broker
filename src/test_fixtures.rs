//! Fixtures for unit tests

use eyre::bail;

use crate::packet;

/// Generate `n` [`packet:Packet`]s with increasing frequency IDs.
pub fn make_packets(nfreq: u32, npackets: u32) -> eyre::Result<Vec<packet::Packet>> {
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

    let mut packets = Vec::<packet::Packet>::new();

    for i in 0..npackets {
        let header = packet::Header {
            version: 2_u16,
            payload_length: 0_u32, // placeholder
            sk_step: 8_u32,
            num_elements: 10_u32,
            samples_per_data_set: 32_u32,
            num_total_freq: nfreq,
            num_local_freq: nfreq_per_packet,
            frames_per_packet: 2_u32,
            seq_num: 0_i64,
        };

        let body = packet::Body {
            freq_ids: (i * nfreq_per_packet..(i + 1) * nfreq_per_packet).collect(),
            frac_flagged: vec![0.2; nfreq_per_packet as usize],
            sktilde_avg: vec![1.3; nfreq_per_packet as usize],
            bad_feed_counts: vec![0u8; 10 * nfreq_per_packet as usize],
        };

        packets.push(packet::Packet { header, body });
    }

    Ok(packets)
}
