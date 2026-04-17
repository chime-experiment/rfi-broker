//! Main application interface and module definitions.
//!
//! Required for better interfacing with tests

pub(crate) mod datastate;
pub(crate) mod endpoints;
pub(crate) mod events;
pub(crate) mod metrics;
pub(crate) mod ringbuffer;

pub mod config;
pub mod packet;
pub mod server;

#[cfg(test)]
pub(crate) mod test_fixtures;
