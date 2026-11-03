//! Global app state, metrics, and configuration.
//!
//! Implements a global [`AppState`] and member states, including
//! - [`Computed`] - computed/derived quantities
//! - [`Metrics`] - application Prometheus metrics
//! - [`Buffers`] - ringbuffers for each incoming dataset

mod buffers;
pub mod config;
mod metrics;

pub use buffers::Buffers;
pub use config::AppConfig;
pub use metrics::{Computed, Metrics};

/// Shared application state.
///
/// Intended to be wrapped with [`std::sync::Arc`] for use with
/// [`tokio`] and [`axum`].
#[derive(Default, Clone)]
pub struct AppState {
    /// Application metrics
    pub metrics: std::sync::Arc<Metrics>,
    /// Computed quantities,
    pub computed: std::sync::Arc<Computed>,
    /// Application buffers
    pub buffers: std::sync::Arc<Buffers>,
}
