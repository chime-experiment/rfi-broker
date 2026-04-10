//! [`DataState`] implementation holding a ringbuffer for each dataset.
//!
//! Each buffer is independently typed and locked, so reads on one dataset
//! never block reads or writes on another.

use std::sync::{Arc, OnceLock};

use parking_lot::Mutex;

use crate::packet::{Body, Header, Packet};
use crate::ringbuffer::RingBuffer;

/// Hold an arbitrary number of `[TypedBuffer]`s.
#[derive(Default, Debug)]
pub struct DataState {
    /// Fixed instance of the packet header, whose values should
    /// be set by the first valid packet
    pub metadata: OnceLock<Mutex<Header>>,
    /// Ringbuffers holding associated datasets from the
    /// packet body. Implements `Default`.
    pub frac_flagged: OnceLock<RingBuffer<f32>>,
    pub sktilde_avg: OnceLock<RingBuffer<f32>>,
    pub bad_feed_counts: OnceLock<RingBuffer<u8>>,
}

pub type SharedDataState = Arc<DataState>;

impl DataState {
    /// Push a packet to the state, initializing on first push.
    pub fn push(&self, packet: Packet) -> Result<u64, String> {
        // Push to each ringbuffer
        let body: Body = packet.body;
        let header: Header = packet.header;

        // Check that the metadata is as-expected
        self.metadata
            .get_or_init(|| Mutex::new(header))
            .lock()
            .check_expected_equal(&packet.header)?;

        // Convert the frequency indices into the expected type
        let indices: Vec<usize> = body.freq_ids.iter().map(|&x| x as usize).collect();
        let id = header.id().cast_unsigned();
        let axis: usize = 0;

        self.frac_flagged
            .get_or_init(|| RingBuffer::<f32>::new(vec![header.num_local_freq as usize]))
            .push_vec(body.frac_flagged, id, &indices, axis)?;

        self.sktilde_avg
            .get_or_init(|| RingBuffer::<f32>::new(vec![header.num_local_freq as usize]))
            .push_vec(body.sktilde_avg, id, &indices, axis)?;

        self.bad_feed_counts
            .get_or_init(|| {
                RingBuffer::<u8>::new(vec![
                    header.num_local_freq as usize,
                    header.num_elements as usize,
                ])
            })
            .push_vec(body.bad_feed_counts, id, &indices, axis)?;

        // Update the metadata since we got here
        *self.metadata.get().unwrap().lock() = header;

        Ok(id)
    }
}
