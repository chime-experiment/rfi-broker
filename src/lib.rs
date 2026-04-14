//! Main application interface and module definitions.
//!
//! Required for better interfacing with tests
pub(crate) mod datastate;
pub(crate) mod endpoints;
pub(crate) mod metrics;
pub(crate) mod ringbuffer;
pub(crate) mod solar;

pub mod config;
pub mod packet;
pub mod server;
