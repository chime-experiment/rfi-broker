//! Program tracing settings.

use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

/// Check if process is controlled by systemd
fn is_controlled_by_systemd() -> bool {
    std::env::var("JOURNAL_STREAM").is_ok()
        || std::env::var("INVOCATION_ID").is_ok()
        || std::env::var("RFI_BROKER_JOURNALD_TRACING").is_ok_and(|v| v == "1")
}

/// Set up program logging.
///
/// Log formatting is adjusted depending on whether we are running
/// controlled by systemd or not.
pub fn init_tracing() {
    // Default to `INFO` log level. Can be adjusted using RUST_LOG environment variable
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let registry = tracing_subscriber::registry().with(env_filter);

    // set a human-readable format layer, which may or may not be used
    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_thread_ids(true)
        .with_line_number(true)
        .pretty();

    // Set the formatting layer depending on where we're writing logs
    if is_controlled_by_systemd() {
        match tracing_journald::layer() {
            Ok(layer) => {
                let layer = layer
                    .with_priority_mappings(tracing_journald::PriorityMappings {
                        // Map `INFO` -> Informational(6) instead of Notice(5) to prevent
                        // it from being rendered in bold font in journald
                        info: tracing_journald::Priority::Informational,
                        ..tracing_journald::PriorityMappings::new()
                    })
                    .with_field_prefix(None);
                registry.with(layer).init();
            }
            Err(e) => {
                // fall back to basic stdout formatting
                registry.with(fmt_layer).init();
                tracing::warn!(
                    error = ?e,
                    "journald logging was requested, but failed to set up layer: {e}"
                );
            }
        }
    } else {
        // not system controlled - use stdout formatting
        registry.with(fmt_layer).init();
    }
}
