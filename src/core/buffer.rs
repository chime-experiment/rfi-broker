//! Generic shared buffer of array frames.
//!
//! The shape and dimensions are fixed at construction time; frames with
//! a mismatched shape are dropped on push.

use std::cmp::Ordering;

use parking_lot::{Mutex, RwLock};
use std::sync::{Arc, OnceLock};

use tokio::sync::broadcast;
#[cfg(any(debug_assertions, test))]
use {
    ndarray::Array2,
    tokio::time::{Duration, sleep},
};

use eyre::{OptionExt, WrapErr, bail, eyre};

use ndarray::{ArrayD, ArrayViewD, Axis, IxDyn};
use num_traits::Num;

/// Constants managing how/when a partial frame should stop accumulating
/// samples and get moved into the buffer
const PARTIAL_FRAME_BUFFER_CAPACITY: usize = 64;
const MIN_FRAME_SAMPLE_COUNT: u64 = 1;

/// Single [`Buffer`] frame.
///
/// Contains an array, mask, and ID.
#[derive(Clone, Debug, PartialEq)]
pub struct Frame<T> {
    /// Numeric identifier
    pub sequence_id: u64,
    /// Arbitrary-sized array
    pub array: ArrayD<T>,
    /// Track how many slices of the array exist
    pub mask: Vec<bool>,
    /// Track how many samples have already been written
    /// relative to the max number
    sample_count: u64,
    max_sample_count: u64,
    /// Track the axis over which this frame can be split
    axis: usize,
}

impl<T> Frame<T>
where
    T: Num + Clone,
{
    /// Create a new frame from an id, shape, and split axis.
    ///
    /// Frame array and mask are fully initialized as zeros/false values.
    fn new(sequence_id: impl Into<u64>, shape: &[usize], axis: usize) -> eyre::Result<Self> {
        // Validate the incoming axis
        let axlen = *shape
            .to_vec()
            .get(axis)
            .ok_or_else(|| eyre!("axis {axis} is invalid for expected frame shape"))?;

        Ok(Self {
            sequence_id: sequence_id.into(),
            array: ArrayD::<T>::zeros(IxDyn(shape)),
            mask: vec![false; axlen],
            sample_count: 0,
            max_sample_count: axlen as u64,
            axis,
        })
    }

    /// Insert a chunk of data into the frame.
    ///
    /// The chunk shape must match the frame shape along all axes other
    /// than the split. `indices` references the indices along the split
    /// axis where `chunk` should be written to. `indices` are not required
    /// to be contiguous.
    fn insert_chunk(&mut self, indices: &[usize], chunk: &ArrayViewD<T>) -> eyre::Result<bool> {
        if chunk
            .shape()
            .get(self.axis)
            .is_none_or(|x| *x != indices.len())
        {
            bail!(
                "number of indices does not match chunk shape on axis {}: {} != {:?}",
                self.axis,
                indices.len(),
                chunk.shape(),
            );
        }
        // Don't assume that indices are contiguous
        let axis = Axis(self.axis);

        for (ii, idx) in indices.iter().enumerate() {
            // Check that the indices make sense, and that this sample
            // hasn't already been received
            let Some(m_sl) = self.mask.get_mut(*idx) else {
                bail!(
                    "invalid index {idx} on axis {} for frame shape {:?}",
                    self.axis,
                    self.array.shape()
                );
            };

            if *m_sl {
                bail!("tried to insert an index which has already been written: {idx}");
            }
            // Insert to the array and update the mask
            self.array
                .index_axis_mut(axis, *idx)
                .assign(&chunk.index_axis(axis, ii));
            // Also record that these indices have been written
            *m_sl = true;
            self.sample_count += 1;
        }

        Ok(self.sample_count == self.max_sample_count)
    }
}

pub type SharedFrame<T> = Arc<Frame<T>>;

/// Buffer slot containing a partial [`Frame`]
#[derive(Debug)]
struct Slot<T> {
    /// Current sequency ID occupying this slot
    id: Option<u64>,
    /// Current [`Frame`] in this slow
    frame: Option<Frame<T>>, //
}

impl<T> Slot<T> {
    const fn empty() -> Self {
        Self {
            id: None,
            frame: None,
        }
    }
}

/// Buffer of decoded partial frames, shared across tasks.
///
/// Once a [`Frame`] is complete, it gets pushed to all subscribers,
/// which are registered by calling `.subscribe()`.
///
/// The broadcast channel length is set at compile time by the
/// `N` parameter.
///
/// All frames are required to have the same shape, which is fixed at
/// construction. Pushes that violate this are dropped.
#[derive(Debug)]
pub struct Buffer<T, const N: usize> {
    /// Expected shape of each frame
    frame_shape: Vec<usize>,
    /// Store partial frames
    slots: Box<[Mutex<Slot<T>>]>,
    /// The number of frame `id`s which this slot can hold.
    /// Relevant when the difference between `id`s > 1
    slot_width: usize,
    first_id: OnceLock<usize>,
    /// Holds the most recent frame for quick access
    last_frame: OnceLock<RwLock<SharedFrame<T>>>,
    /// broadcast sender for new frame events
    tx: broadcast::Sender<SharedFrame<T>>,
    /// Hold a lock while sending and updating the
    /// last sent frame
    send_lock: Mutex<()>,
}

impl<T, const N: usize> Buffer<T, N>
where
    T: Num + Clone,
{
    /// Create a new buffer with a fixed shape.
    pub fn new(frame_shape: Vec<usize>, slot_width: usize) -> Self {
        let (tx, _) = broadcast::channel(N);

        let slots: Box<[Mutex<Slot<T>>]> = (0..PARTIAL_FRAME_BUFFER_CAPACITY)
            .map(|_| Mutex::new(Slot::empty()))
            .collect();

        Self {
            frame_shape,
            slots,
            slot_width,
            first_id: OnceLock::<usize>::new(),
            last_frame: OnceLock::<RwLock<SharedFrame<T>>>::default(),
            tx,
            send_lock: Mutex::new(()),
        }
    }

    /// Get the most recently pushed frame.
    pub fn last_frame(&self) -> Option<SharedFrame<T>> {
        self.last_frame.get().map(|lock| lock.read().clone())
    }

    /// Subscribe to a new frame event broadcast.
    ///
    /// The subscriber received a new ``Arc<Frame>`` each time the
    /// new frame is created and pushed to the buffer.
    ///
    /// Because this sends an `Arc`, the underlying data array is not
    /// cloned.
    pub fn subscribe(&self) -> broadcast::Receiver<SharedFrame<T>> {
        self.tx.subscribe()
    }

    /// Sends a frame to all subscribers and update the
    /// internal `last_frame`.
    fn send_frame(&self, frame: Frame<T>) {
        // log the case where the frame isn't full
        if frame.sample_count != frame.max_sample_count {
            tracing::debug!(
                "pushed incomplete frame; sample count={:?}/{:?}",
                frame.sample_count,
                frame.max_sample_count
            );
        }

        let shared_frame = Arc::new(frame);

        // hold the send lock while updating and sending
        let _guard = self.send_lock.lock();

        // NB: there's a double write for the first frame received
        *self
            .last_frame
            .get_or_init(|| RwLock::new(Arc::clone(&shared_frame)))
            .write() = Arc::clone(&shared_frame);

        // send to all subscribers, if they exist
        if self.tx.receiver_count() > 0 {
            let _ = self.tx.send(shared_frame).inspect_err(|_| {
                tracing::warn!(
                    "broadcast send failed; receivers may have disconnected (receiver_count={:?})",
                    self.tx.receiver_count(),
                );
            });
        }
    }

    /// Push all partial frames to subscribers.
    pub fn flush(&self) -> usize {
        let mut num_sent: usize = 0;
        let mut last_id: u64 = 0;
        let mut start_id: usize = 0;

        // find the earliest sequence id
        for (i, slot) in self.slots.iter().enumerate() {
            if let Some(id) = slot.lock().id {
                if id < last_id {
                    start_id = i;
                } else {
                    last_id = id;
                }
            }
        }

        // run through all of the slots, starting at `start_id`
        for slot in self
            .slots
            .iter()
            .cycle()
            .skip(start_id)
            .take(self.slots.len())
        {
            if let Some(f) = slot.lock().frame.take() {
                self.send_frame(f);
                num_sent += 1;
            }
        }

        num_sent
    }

    /// Add an array to a frame and push the frame if it is full.
    ///
    /// If the frame is full, push to subscribers. Otherwise, store
    /// in a map under the assumption that the rest of the frame will be
    /// received.
    ///
    /// Frames which are never filled will eventually get pushed to the frame if
    /// a sufficient number of samples have been received, or if `flush()`
    /// is called.
    ///
    /// Assumes that frame `id`s are monotonically increasing.
    fn push_array(
        &self,
        array: &ArrayD<T>,
        sequence_id: impl Into<u64>,
        indices: &[usize],
        axis: usize,
    ) -> eyre::Result<u64> {
        // NB: there's some complicated slot indexing here, because we need
        // to assume that `sequence_id` can have a step greater than one. So,
        // we need to map the `sequence_id` into a bin corresponding to a
        // a specific buffer slot
        let key: u64 = sequence_id.into();
        #[allow(
            clippy::integer_division,
            clippy::cast_possible_truncation,
            reason = "integer values too small for truncation"
        )]
        let (incremental_index, first_key) = {
            let first_key = self.first_id.get_or_init(|| {
                tracing::info!(
                    first_sequence_id = ?key,
                    "received first sample with sequence ID {key}"
                );
                key as usize
            });
            (((key as usize - first_key) / self.slot_width), first_key)
        };
        // wrap based on the length of the buffer
        let slot_index = incremental_index % PARTIAL_FRAME_BUFFER_CAPACITY;

        let mut evicted: Option<Frame<T>> = None;

        let result: eyre::Result<Option<Frame<T>>> = (|| {
            let mut slot = self
                .slots
                .get(slot_index)
                .ok_or_eyre("unexpected failure accessing slow {slot_index}")?
                .lock();

            #[allow(
                clippy::integer_division,
                clippy::cast_possible_truncation,
                reason = "desired behaviour"
            )]
            // get the incremental slot index of the current slot `id`
            let slot_id_compare = slot
                .id
                .map(|id| incremental_index.cmp(&((id as usize - first_key) / self.slot_width)));

            match slot_id_compare {
                #[allow(
                    clippy::unwrap_used,
                    reason = "unwrap guaranteed to succeed in this arm"
                )]
                // `id` should be monotonically increasing, so drop the sample
                // if it's too old
                Some(Ordering::Less) => bail!(
                    "tried to push data with id {key} older than the slot's current id \
                    group {} (first_id={first_key}) (index={slot_index})",
                    slot.id.unwrap()
                ),
                Some(Ordering::Equal) => {
                    if slot.frame.is_none() {
                        bail!("frame with id group {key} was already completed!");
                    }
                }
                _ => {
                    // make a new frame and insert it
                    let new_frame = Frame::new(key, &self.frame_shape, axis)?;
                    evicted = slot.frame.replace(new_frame);
                    slot.id = Some(key);
                }
            }

            let frame = slot
                .frame
                .as_mut()
                .ok_or_eyre("unexpected failure extracting partial frame")?;

            let ready = frame.insert_chunk(indices, &array.view())?;

            Ok(if ready { slot.frame.take() } else { None })
        })();

        if let Some(old) = evicted {
            if old.sample_count >= MIN_FRAME_SAMPLE_COUNT {
                self.send_frame(old);
            } else {
                tracing::debug!(
                    "Dropped frame with sequence number {} because sample count {} is below \
                    threshold {MIN_FRAME_SAMPLE_COUNT}",
                    old.sequence_id,
                    old.sample_count,
                );
            }
        }

        if let Some(full) = result? {
            self.send_frame(full);
        }

        Ok(key)
    }

    /// Add a ``Vec`` to the buffer, converting it into [`ArrayD`].
    ///
    /// The ``Vec`` is consumed to avoid a copy.
    pub fn push_vec(
        &self,
        vec: Vec<T>,
        id: impl Into<u64>,
        indices: &[usize],
        axis: usize,
    ) -> eyre::Result<u64> {
        // Sort out the shape of this chunk
        let mut shape = self.frame_shape.clone();
        let ax_shape = shape.get_mut(axis).ok_or_else(|| {
            eyre!(
                "Axis `{axis}` is out of bounds for shape {:?}",
                self.frame_shape
            )
        })?;
        *ax_shape = indices.len();

        let arr =
            ArrayD::from_shape_vec(shape, vec).wrap_err("failed to construct array from vec")?;

        self.push_array(&arr, id, indices, axis)
    }
}

#[cfg(any(debug_assertions, test))]
impl<T, const N: usize> Buffer<T, N>
where
    T: Num + Clone,
{
    /// Return the shape of each frame.
    pub const fn shape(&self) -> &Vec<usize> {
        &self.frame_shape
    }

    /// Accumulate `n` frames and return as a [`Vec`].
    pub async fn accumulate(
        &self,
        n: usize,
        item_timeout: impl Into<Option<Duration>>,
    ) -> eyre::Result<Vec<SharedFrame<T>>> {
        // subscribe to the internal sender
        let mut rx = self.tx.subscribe();
        let mut buf = Vec::with_capacity(n);

        let item_timeout = item_timeout.into().unwrap_or(Duration::from_millis(50));

        while buf.len() < n {
            tokio::select! {
                biased; // always poll the buffer first
                result = rx.recv() => {
                    match result {
                        Ok(frame) => buf.push(frame),
                        Err(broadcast::error::RecvError::Lagged(skipped)) => {
                            bail!("receive buffer overflowed - {skipped} frames were missed");
                        }
                        Err(broadcast::error::RecvError::Closed) => {
                            bail!("sender was dropped before accumulation was complete");
                        }
                    }
                }
                () = sleep(item_timeout) => {
                    bail!("timed out while waiting for frames");
                }
            }
        }

        Ok(buf)
    }
}

#[cfg(any(debug_assertions, test))]
/// Return an `N+1` dimensional [`ArrayD`] stacked over an axis, or `None`
/// if no frames available.
///
/// Returns `None` if any errors occur while stacking.
pub fn stack_buffer_array<T>(
    buffer: &[SharedFrame<T>],
    axis: impl Into<Option<usize>>,
) -> Option<ArrayD<T>>
where
    T: Num + Clone,
{
    // `stack` requires views
    let views: Vec<ArrayViewD<T>> = buffer.iter().map(|f| f.array.view()).collect();

    let ax = axis.into().map_or(Axis(0), Axis);

    ndarray::stack(ax, &views).ok()
}

#[cfg(any(debug_assertions, test))]
/// Stack the frame masks, creating a new outermost axis.
pub fn stack_buffer_mask<T>(buffer: &[SharedFrame<T>]) -> Option<Array2<u8>>
where
    T: Num + Clone,
{
    // Sort out the shape
    let ncols = buffer.first()?.mask.len();
    let nrows = buffer.len();
    // Masks are 1-dimensional, so concatenate the first axis. This means
    // that the sample axis is the slowest varying, so have to transpose
    // if this isn't the desired layout
    let flat_vec: Vec<u8> = buffer
        .iter()
        .flat_map(|f| f.mask.iter().map(|&x| u8::from(x))) // return u8 instead of bool
        .collect();

    ndarray::Array2::<u8>::from_shape_vec((nrows, ncols), flat_vec).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::s;

    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "casts on small positive integers"
    )]
    #[tokio::test]
    /// Test that subscribers received a frame as expected, and that both the
    /// pushed and sent frames match
    async fn test_subscribe() -> Result<(), Box<dyn std::error::Error>> {
        let frame_shape = vec![3, 12];
        // Create a new empty buffer
        let buf = Buffer::<f32, 4>::new(frame_shape, 1);
        // Subscribe to the buffer for new frame events
        let mut rx1 = buf.subscribe();
        let mut rx2 = buf.subscribe();

        // Expected array which will be pushed to the buffer in chunks
        let expected_arr = ArrayD::<f32>::from_shape_fn(IxDyn(&[3, 12]), |idx| idx[0] as f32);

        // Push the array to the buffer in chunks
        for (i, row_view) in expected_arr.axis_iter(Axis(0)).enumerate() {
            let chunk: Vec<f32> = row_view.iter().copied().collect();
            buf.push_vec(chunk, 0_u64, &[i], 0)?;
        }

        // Confirm that the subscriber has received the new frame
        let frame1: SharedFrame<f32> = rx1.recv().await?;
        let frame2: SharedFrame<f32> = rx2.recv().await?;

        assert_abs_diff_eq!(frame1.array, expected_arr);
        assert_abs_diff_eq!(frame2.array, expected_arr);
        assert_eq!(*frame1, *frame2);

        Ok(())
    }

    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "casts on small positive integers"
    )]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    /// Test that the mask and arrays are stacked properly
    async fn test_accumulate() -> Result<(), Box<dyn std::error::Error>> {
        let frame_shape = vec![3, 12];
        //i Create a new buffer. Wrap in an Arc so we can pass to the receive task
        let buf = Arc::new(Buffer::<f32, 4>::new(frame_shape, 1));
        // create an array to push
        let arr = ArrayD::<f32>::from_shape_fn(IxDyn(&[2, 12]), |idx| idx[1] as f32);
        // create an array to compare with, since there's an extra row
        let mut arr_compare = ArrayD::<f32>::zeros(IxDyn(&[3, 12]));
        arr_compare.slice_mut(s![..2, ..]).assign(&arr);

        // spawn the `accumulate` call, and wait until the task has actually spawned
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let buf_clone = Arc::clone(&buf);
        let handle = tokio::spawn(async move {
            tx.send(()).unwrap();
            buf_clone.accumulate(2, Duration::from_secs(1)).await
        });

        // sleep for a moment to ensure that the accumulate call is listening
        sleep(Duration::from_millis(500)).await;
        // wait for spawned task
        rx.await?;

        // Push the partial arrays to the buffer
        for i in 0..2 {
            buf.push_array(&arr.clone(), i as u64, &[0, 1], 0)?;
        }
        // flush frames to the buffer to ensure that everything has been sent
        buf.flush();

        // grab the received vec
        let received = handle.await??;

        // assert that the last frame was flushed
        assert!(!received.is_empty(), "frame was never received");
        assert_eq!(received.last().unwrap(), &buf.last_frame().unwrap());

        // Stack both buffers over the 0th axis
        let arr_stack = stack_buffer_array(&received, 0).unwrap();
        let mask_stack = stack_buffer_mask(&received).unwrap();

        assert_eq!(arr_stack.shape(), &[2, 3, 12]);
        assert_eq!(mask_stack.shape(), &[2, 3]);

        // Check that each row of the stacked arrays are as expected
        for i in 0..received.len() {
            assert_eq!(arr_stack.index_axis(Axis(0), i), arr_compare);
            assert_eq!(mask_stack.row(i).to_vec(), vec![1u8, 1u8, 0u8]);
        }

        // Finally, confirm that stacking over different axes also works as expected
        assert_eq!(
            stack_buffer_array(&received, 1).unwrap().shape(),
            &[3, 2, 12]
        );
        assert_eq!(
            stack_buffer_array(&received, 2).unwrap().shape(),
            &[3, 12, 2]
        );

        Ok(())
    }
}
