//! [`DataState`] implementation holding a ringbuffer for each dataset.
//!
//! Each buffer is independently typed and locked, so reads on one dataset
//! never block reads or writes on another.

use std::sync::{Arc, Mutex, OnceLock};

use crate::packet::{Body, Header, Packet};
use crate::ringbuffer::SharedRingBuffer;

/// Hold an arbitrary number of `[TypedBuffer]`s.
#[derive(Default, Debug)]
pub struct DataState {
    /// Private flag used to determine if this state has
    /// been initialized. Defaults to `false` from #[derive(Default)]
    initialized: OnceLock<bool>,
    /// Fixed instance of the packet header, whose values should
    /// be set by the first valid packet
    pub metadata: Mutex<Header>,
    /// Ringbuffers holding associated datasets from the
    /// packet body
    pub frac_flagged: SharedRingBuffer<f32>,
    pub sktilde_avg: SharedRingBuffer<f32>,
    pub bad_feed_counts: SharedRingBuffer<u8>,
}

pub type SharedDataState = Arc<DataState>;

impl DataState {
    /// Initialize a default state from a packet
    fn init(&self, packet: &Packet) {
        if self.initialized.get().is_some() {
            return;
        }
        let header: &Header = &packet.header;
        // Set the metadata, which is behind a mutex
        *self.metadata.lock().unwrap() = packet.header;
        // Reset the buffers, which handle their own internal mutex's
        self.frac_flagged
            .reset(vec![header.num_local_freq as usize]);
        self.sktilde_avg.reset(vec![header.num_local_freq as usize]);
        self.bad_feed_counts.reset(vec![
            header.num_local_freq as usize,
            header.num_elements as usize,
        ]);

        self.initialized.set(true).ok();
    }

    /// Create a new datastate from a parsed packet.
    #[allow(dead_code)]
    fn new(packet: &Packet) -> Self {
        let state = Self::default();
        state.init(packet);

        state
    }

    /// New shared datastate
    #[allow(dead_code)]
    fn new_shared(packet: &Packet) -> SharedDataState {
        Arc::new(Self::new(packet))
    }

    /// Default shared
    pub fn default_shared() -> SharedDataState {
        Arc::new(Self::default())
    }

    /// Push a packet to an existing state
    pub fn push(&self, packet: &Packet) -> Result<(), Box<dyn std::error::Error>> {
        self.init(packet); // No behaviour if state is initialized
                           // Check that the metadata is as-expected
        self.metadata
            .lock()
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
        let mut guard = self.metadata.lock().unwrap();
        *guard = *header;

        Ok(())
    }
}
