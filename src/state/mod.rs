//! Global app state, metrics, and configuration.
//!
//! Implements a global [`AppState`] and member states, including
//! - [`Computed`] - computed/derived quantities
//! - [`Metrics`] - application Prometheus metrics
//! - [`Buffers`] - ringbuffers for each incoming dataset

pub mod config;
mod metric_types;
#[allow(clippy::module_inception, reason = "Not a public module")]
mod state;

pub use config::AppConfig;
pub use state::{AppState, Buffers, Computed, Metrics};
