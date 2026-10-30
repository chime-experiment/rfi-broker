//! Server, router, and UDP receiver.
//!
//! # Endpoints
//! - ``metadata``: Most recent packet header metadata
//! - ``metrics``: Application prometheus metrics
//! - ``human-metrics``: Some human-readable metrics
//! - ``bad_input_likelihood``: per-input likelihood of a feed being corrupted
//!
//! ## Debug-only
//! - ``last-frame``: Pretty print of most recent buffer frame
//! - ``write-buffers``: POST with `path` + `nsamples` to accumulate and write SK buffers

pub mod endpoints;
pub mod packet;
pub mod router;
pub mod udp;
