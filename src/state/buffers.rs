//! Implements global state objects.

use std::sync::OnceLock;

use eyre::OptionExt;
use parking_lot::Mutex;

use crate::core::buffer::Buffer;
use crate::network::packet::{Body, Header, Packet, packet_types};

/// Broadcast channel depth for frame events
const TX_BUFFER_CAPACITY: usize = 32;

/// Store for application data buffers.
///
/// Datasets are application specific, and matches those expected
/// in [`crate::packet::Packet`].
///
/// Buffers are created lazily - only instantiated when a packet is
/// received and parsed. This is implemented via a ``OnceLock``.
///
/// Each buffer is independently typed and locked, so reads/writes on one
/// dataset never blocks another.
#[derive(Default, Debug)]
pub struct Buffers {
    /// Fixed instance of the packet header, whose values should
    /// be set by the first valid packet
    pub metadata: OnceLock<Mutex<Header>>,
    /// Ringbuffers holding associated datasets from the
    /// packet body. Implements `Default`.
    pub frac_flagged: OnceLock<Buffer<packet_types::FracFlaggedType, TX_BUFFER_CAPACITY>>,
    pub sktilde_avg: OnceLock<Buffer<packet_types::SkType, TX_BUFFER_CAPACITY>>,
    pub skbar_avg: OnceLock<Buffer<packet_types::SkType, TX_BUFFER_CAPACITY>>,
}

impl Buffers {
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
                Buffer::<packet_types::FracFlaggedType, TX_BUFFER_CAPACITY>::new(vec![
                    header.num_total_freq as usize,
                ])
            })
            .push_vec(body.frac_flagged, id, &indices, axis)?;

        self.sktilde_avg
            .get_or_init(|| {
                Buffer::<packet_types::SkType, TX_BUFFER_CAPACITY>::new(vec![
                    header.num_total_freq as usize,
                ])
            })
            .push_vec(body.sktilde_avg, id, &indices, axis)?;

        self.skbar_avg
            .get_or_init(|| {
                Buffer::<packet_types::SkType, TX_BUFFER_CAPACITY>::new(vec![
                    header.num_total_freq as usize,
                    header.num_elements as usize,
                ])
            })
            .push_vec(body.skbar_avg, id, &indices, axis)?;

        // Update the metadata since we got here
        let meta = self
            .metadata
            .get()
            .ok_or_eyre("unexpected failure accessing existing metadata")?;
        *meta.lock() = header;

        Ok(id)
    }

    /// Flush all buffers - that is, push all partial frames.
    pub fn flush(&self) -> usize {
        let mut nflushed = 0;

        if let Some(buf) = self.frac_flagged.get() {
            nflushed += buf.flush();
        }
        if let Some(buf) = self.sktilde_avg.get() {
            nflushed += buf.flush();
        }
        if let Some(buf) = self.skbar_avg.get() {
            nflushed += buf.flush();
        }

        nflushed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::packet::tests::make_packets;

    /// Test that are successfully initialized when the first packet
    /// is pushed.
    #[test]
    fn test_init_buffers() -> Result<(), Box<dyn std::error::Error>> {
        let state = Buffers::default();

        let packets = make_packets(4, 2)?;
        assert_eq!(packets.len(), 2);

        for packet in &packets {
            state.push(packet.clone())?;
        }
        // There should now be some data in the various buffers
        assert!(state.metadata.get().is_some());
        assert!(state.frac_flagged.get().is_some());
        assert!(state.sktilde_avg.get().is_some());
        assert!(state.skbar_avg.get().is_some());

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

        let skbar_avg = state.skbar_avg.get().ok_or("error getting `skbar_avg`")?;

        // Check that each buffer has the correct shape
        assert_eq!(*frac_flagged.shape(), [4]);
        assert_eq!(*sktilde_avg.shape(), [4]);
        assert_eq!(*skbar_avg.shape(), [4, 10]);

        Ok(())
    }

    /// Test that packets are successfully parsed and pushed into
    /// the corresponding [`Buffer`]s.
    #[test]
    fn test_push_packets() -> Result<(), Box<dyn std::error::Error>> {
        let state = std::sync::Arc::new(Buffers::default());

        // Push some packets to the buffer to initialize
        let packets = make_packets(4, 2)?;
        for packet in &packets {
            state.push(packet.clone())?;
        }
        state.flush();

        // Check that each buffer has been initialized
        let frac_flagged = state
            .frac_flagged
            .get()
            .ok_or("error getting `frac_flagged`")?;
        let sktilde_avg = state
            .sktilde_avg
            .get()
            .ok_or("error getting `sktilde_avg`")?;
        let skbar_avg = state.skbar_avg.get().ok_or("error getting `skbar_avg`")?;

        // Check that a complete frame has been pushed to each buffer
        assert!(
            frac_flagged.last_frame().is_some(),
            "`frac_flagged` frame was not pushed"
        );
        assert!(
            sktilde_avg.last_frame().is_some(),
            "`sktilde_avg` frame was not pushed"
        );
        assert!(
            skbar_avg.last_frame().is_some(),
            "`skbar_avg` frame was not pushed"
        );

        Ok(())
    }
}
