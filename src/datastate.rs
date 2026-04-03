//! [`DataState`] implementation holding a ringbuffer for each dataset.
//!
//! Each buffer is independently typed and locked, so reads on one dataset
//! never block reads or writes on another.

use std::sync::{Arc, OnceLock, RwLock};

use crate::packet::{Body, Header, Packet};
use crate::ringbuffer::RingBuffer;

/// Hold an arbitrary number of `[TypedBuffer]`s.
#[derive(Default, Debug)]
pub struct DataState {
    /// Private flag used to determine if this state has
    /// been initialized. Defaults to `false` from #[derive(Default)]
    initialized: OnceLock<bool>,
    /// Fixed instance of the packet header, whose values should
    /// be set by the first valid packet
    pub metadata: RwLock<Header>,
    /// Ringbuffers holding associated datasets from the
    /// packet body
    pub frac_flagged: RingBuffer<f32>,
    pub sktilde_avg: RingBuffer<f32>,
    pub bad_feed_counts: RingBuffer<u8>,
}

pub type SharedDataState = Arc<DataState>;

impl DataState {
    fn is_initialized(&self) -> bool {
        self.initialized.get().is_some()
    }

    /// Initialize a default state from a packet
    fn init(&self, packet: &Packet) -> Result<&Self, String> {
        if self.is_initialized() {
            return Err("State has already been initialized!".into());
        }
        let header: &Header = &packet.header;
        // Set the metadata, which is behind a mutex
        *self.metadata.write().unwrap() = packet.header;
        // Reset the buffers, which handle their own internal mutex's
        self.frac_flagged
            .init(vec![header.num_local_freq as usize])?;
        self.sktilde_avg
            .init(vec![header.num_local_freq as usize])?;
        self.bad_feed_counts.init(vec![
            header.num_local_freq as usize,
            header.num_elements as usize,
        ])?;

        self.initialized.set(true).ok();

        Ok(self)
    }

    /// Push a packet to an existing state
    pub fn push(&self, packet: &Packet) -> Result<(), String> {
        // Don't actually care about the result of `init` here
        let _ = self.init(packet);
        // Check that the metadata is as-expected
        self.metadata
            .read()
            .unwrap()
            .check_expected_equal(&packet.header)?;

        // Push to each ringbuffer
        let body: &Body = &packet.body;
        let header: &Header = &packet.header;

        // Convert the frequency indices into the expected type
        let indices: Vec<usize> = body.freq_ids.iter().map(|&x| x as usize).collect();
        let id = header.id().cast_unsigned();
        let axis: usize = 0;

        self.frac_flagged.push_vec(
            &body.frac_flagged,
            self.frac_flagged.frame_shape().unwrap(),
            id,
            &indices,
            axis,
        )?;

        self.sktilde_avg.push_vec(
            &body.sktilde_avg,
            self.sktilde_avg.frame_shape().unwrap(),
            id,
            &indices,
            axis,
        )?;

        self.bad_feed_counts.push_vec(
            &body.bad_feed_counts,
            self.bad_feed_counts.frame_shape().unwrap(),
            id,
            &indices,
            axis,
        )?;

        // Update the metadata since we got here
        let mut guard = self.metadata.write().unwrap();
        *guard = *header;

        Ok(())
    }
}
