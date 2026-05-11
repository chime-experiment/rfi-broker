//! [`DataState`] implementation holding a ringbuffer for each dataset.
//!
//! Each buffer is independently typed and locked, so reads on one dataset
//! never block reads or writes on another.
//!
//! This is an application-specific state, and not meant to be used as part
//! of a library. It interfaces directly with [`crate::packet::Packet`].
//!
//! Designed to be wrapped in a [`std::sync::Arc`] for easy use with
//! `axum` and `tokio`.

use std::sync::{Arc, OnceLock};

use eyre::OptionExt;
use parking_lot::Mutex;

use crate::packet::{Body, Header, Packet, packet_types};
use crate::ringbuffer::RingBuffer;

/// Shared state for application data.
///
/// Datasets are application specific, and matches those expected
/// in [`crate::packet::Packet`].
#[derive(Default, Debug)]
pub struct DataState {
    /// Fixed instance of the packet header, whose values should
    /// be set by the first valid packet
    pub metadata: OnceLock<Mutex<Header>>,
    /// Ringbuffers holding associated datasets from the
    /// packet body. Implements `Default`.
    pub frac_flagged: OnceLock<RingBuffer<packet_types::FracFlaggedType>>,
    pub sktilde_avg: OnceLock<RingBuffer<packet_types::SkTildeType>>,
    pub bad_feed_counts: OnceLock<RingBuffer<packet_types::BadFeedType>>,
}

pub type SharedDataState = Arc<DataState>;

impl DataState {
    /// Push a packet to the state, initializing on first push.
    pub fn push(&self, packet: Packet) -> eyre::Result<u64> {
        let body: Body = packet.body;
        let header: Header = packet.header;

        // Check that the metadata is as-expected and initialize otherwise
        self.metadata
            .get_or_init(|| Mutex::new(header))
            .lock()
            .check_expected_equal(&packet.header)?;

        // Convert the frequency indices into the expected type
        let indices: Vec<usize> = body.freq_ids.iter().map(|&x| x as usize).collect();
        let id = header.seq_num.cast_unsigned();
        let axis: usize = 0;

        // Push to each ringbuffer, initializing if this is the first push
        self.frac_flagged
            .get_or_init(|| {
                RingBuffer::<packet_types::FracFlaggedType>::new(vec![
                    header.num_total_freq as usize,
                ])
            })
            .push_vec(body.frac_flagged, id, &indices, axis)?;

        self.sktilde_avg
            .get_or_init(|| {
                RingBuffer::<packet_types::SkTildeType>::new(vec![header.num_total_freq as usize])
            })
            .push_vec(body.sktilde_avg, id, &indices, axis)?;

        self.bad_feed_counts
            .get_or_init(|| {
                RingBuffer::<packet_types::BadFeedType>::new(vec![
                    header.num_total_freq as usize,
                    header.num_elements as usize,
                ])
            })
            .push_vec(body.bad_feed_counts, id, &indices, axis)?;

        // Update the metadata since we got here
        let meta = self
            .metadata
            .get()
            .ok_or_eyre("unexpected failure accessing existing metadata")?;
        *meta.lock() = header;

        Ok(id)
    }

    /// Flush all buffers
    pub fn flush(&self) -> usize {
        let mut nflushed = 0;

        if let Some(buf) = self.frac_flagged.get() {
            nflushed += buf.flush();
        }
        if let Some(buf) = self.sktilde_avg.get() {
            nflushed += buf.flush();
        }
        if let Some(buf) = self.bad_feed_counts.get() {
            nflushed += buf.flush();
        }

        nflushed
    }

    /// Clear all buffers
    pub fn clear(&self) -> usize {
        let mut ncleared = 0;

        if let Some(buf) = self.frac_flagged.get() {
            ncleared += buf.clear();
        }
        if let Some(buf) = self.sktilde_avg.get() {
            ncleared += buf.clear();
        }
        if let Some(buf) = self.bad_feed_counts.get() {
            ncleared += buf.clear();
        }

        ncleared
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packet::tests::make_packets;

    /// Test that packets are successfully parsed and pushed into
    /// the corresponding [`RingBuffer`]s.
    #[test]
    fn test_push_packets() -> Result<(), Box<dyn std::error::Error>> {
        let state = DataState::default();

        // Produce and push a couple of packets
        let packets = make_packets(4, 2)?;
        assert_eq!(packets.len(), 2);

        for packet in &packets {
            state.push(packet.clone())?;
        }

        // There should now be some data in the various buffers
        assert!(state.metadata.get().is_some());
        assert!(state.frac_flagged.get().is_some());
        assert!(state.sktilde_avg.get().is_some());
        assert!(state.bad_feed_counts.get().is_some());

        // Check that the metadata is what we expect
        let meta = state.metadata.get().ok_or("error getting metadata")?.lock();
        let header = packets
            .get(1)
            .ok_or("packets not successfully constructed")?
            .header;
        meta.check_expected_equal(&header)?;

        // Check that specific metadata values are correct
        assert_eq!(meta.num_total_freq, 4);
        assert_eq!(meta.num_local_freq, 2);
        assert_eq!(meta.num_elements, 10);

        // Check that each buffer has been initialized
        let frac_flagged = state
            .frac_flagged
            .get()
            .ok_or("error getting `frac_flagged`")?;

        let sktilde_avg = state
            .sktilde_avg
            .get()
            .ok_or("error getting `sktilde_avg`")?;

        let bad_feed_counts = state
            .bad_feed_counts
            .get()
            .ok_or("error getting `bad_feed_counts`")?;

        // Check that each buffer has the correct shape
        assert_eq!(*frac_flagged.shape(), [4]);
        assert_eq!(*sktilde_avg.shape(), [4]);
        assert_eq!(*bad_feed_counts.shape(), [4, 10]);

        // Check that a complete frame has been pushed to each buffer
        assert_eq!(frac_flagged.len(), 1);
        assert_eq!(frac_flagged.queue_len(), 0);

        assert_eq!(sktilde_avg.len(), 1);
        assert_eq!(sktilde_avg.queue_len(), 0);

        assert_eq!(bad_feed_counts.len(), 1);
        assert_eq!(bad_feed_counts.queue_len(), 0);

        Ok(())
    }
}
