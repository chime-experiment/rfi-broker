//! Generic [`DataState`] holding multiple typed [`SharedRingbuffer`] instances.
//!
//! [`TypedBuffer`] and [`TypedArray`] erase the element type behind an enum so
//! that buffers of different types can coexist in the same ordered map. The map
//! preserves dataset insertion order.
//!
//! Each buffer is independently typed and locked, so reads on one dataset
//! never block reads or writes on another.

use std::sync::{Arc, Mutex};

use serde_json::{Error, Value};

use indexmap::IndexMap;
use ndarray::ArrayD;

use crate::config::{Config, DType};
use crate::header::Header;
use crate::ringbuffer::{RingBuffer, SharedRingBuffer};

/// Define and implement [`TypedBuffer`] and [`TypedArray`] for various
/// numeric types. Mostly just exposes the underlying [`RingBuffer`] methods.
macro_rules! define_typed_variants {
    ($( $variant:ident => $type:ty ),*) => {
        /// A ringbuffer whose element type is determined at runtime.
        pub enum TypedBuffer { $( $variant(SharedRingBuffer<$type>), )* }
        /// An array whose element type is determined at runtime.
        pub enum TypedArray { $( $variant(ArrayD<$type>), )* }


        #[allow(dead_code)]
        impl TypedBuffer {
            /// Buffer name
            pub fn name(&self) -> &String {
                match self { $( TypedBuffer::$variant(b) => &b.name, )* }
            }

            /// Dimension names
            pub fn dims(&self) -> &Vec<String> {
                match self { $( TypedBuffer::$variant(b) => &b.dims, )* }
            }

            /// The expected shape of each buffer
            pub fn shape(&self) -> &Vec<usize> {
                match self { $( TypedBuffer::$variant(b) => &b.shape, )* }
            }

            /// The byte size of a single element for this buffer's type
            pub fn element_bytes(&self) -> usize {
                match self { $( TypedBuffer::$variant(_) => std::mem::size_of::<$type>(), )* }
            }

            /// Serialize the underlying ringbuffer
            pub fn serialize(&self) -> Result<Vec<Value>, Error> {
                match self { $( TypedBuffer::$variant(b) => b.serialize(), )* }
            }

            /// Push to the underlying ringbuffer
            pub fn push(&self, arr: TypedArray) -> Result<(), String> {
                match (self, arr) {
                    $(
                        (TypedBuffer::$variant(b), TypedArray::$variant(arr)) => b.push(arr),
                    )*
                    // Need to return as a `String` to satisfy the type returned
                    // by [`RingBuffer::push`]
                    _ => Err(String::from("Array has no matching type.")),
                }

            }
        }
    }
}

define_typed_variants! {
    F32 => f32,
    F64 => f64,
    U8 => u8,
    U16 => u16,
    U32 => u32,
    U64 => u64
}

/// Hold an arbitrary number of `[TypedBuffer]`s.
pub struct DataState {
    /// Fixed instance of the packet header, whose values should
    /// be set by the first valid packet
    pub metadata: Mutex<Header>,
    /// Ordered map from dataset name to its typed ringbuffer
    pub buffers: IndexMap<String, TypedBuffer>,
}

/// Convenience alias for the reference-counted [`DataState`].
pub type SharedDataState = Arc<DataState>;

macro_rules! impl_data_state {
    ($( $variant:ident => $type:ty ), *) => {
        impl DataState {
            /// Constructs a [`DataState`] from the datasets declared in `config`,
            /// building a typed ringbuffer for each.
            ///
            /// # Panics
            /// Panics if any dataset references an unknown dimension name.
            pub fn from_config(config: &Config) -> Self {
                let mut buffers = IndexMap::new();
                for ds in &config.datasets {
                    // Make sure that
                    let shape = config.resolve_dataset_shape(&ds).unwrap_or_else(|| {
                        panic!("Invalid dimension config for dataset {}", ds.name)
                    });

                    let buffer = match ds.dtype {
                        $(
                            DType::$variant => TypedBuffer::$variant(RingBuffer::new_shared(
                                ds.name.clone(),
                                ds.dims.clone(),
                                shape,
                            )),
                        )*
                    };

                    buffers.insert(ds.name.clone(), buffer);
                }

                // Header information only gets filled when a
                // packet is received.
                Self {
                    metadata: Mutex::new(Header::default()),
                    buffers: buffers
                }
            }

            /// Constructs a new [`SharedDataState`]
            pub fn from_config_shared(config: &Config) -> SharedDataState {
                Arc::new(Self::from_config(&config))
            }
        }
    };
}

impl_data_state! {
    F32 => f32,
    F64 => f64,
    U8 => u8,
    U16 => u16,
    U32 => u32,
    U64 => u64
}
