//! Application configuration.
//!
//! Parses a `.toml` config file into individual config structs.
//! The complete config file is represented as [`AppConfig`], where
//! each member is a section. Each section is optional, and gets parsed
//! into the corresponding struct.

/// Telescope coordinate configuration
#[derive(Debug, serde::Deserialize)]
pub struct TelescopeCoordinates {
    pub latitude: f64,
    pub longitude: f64,
    pub altitude: f64,
}

/// RFI zeroing event configuration
#[derive(Debug, serde::Deserialize)]
pub struct RFIZeroingConfig {
    pub downtime: u64,
    pub hostname: String,
    pub target: String,
    pub first_stage: String,
    pub second_stage: String,
}

/// Appplication config.
///
/// All sections here should be optional.
#[derive(Debug, serde::Deserialize)]
pub struct AppConfig {
    pub telescope: Option<TelescopeCoordinates>,
    pub zeroing: Option<RFIZeroingConfig>,
}

/// Shared [`Config`] type.
pub type SharedAppConfig = std::sync::Arc<AppConfig>;

/// Load a [`Config`] from a toml file.
///
/// # Errors
/// Returns [`config:;ConfigError`] if the file can't be read.
pub fn load(config_path: &str) -> Result<AppConfig, config::ConfigError> {
    // Read and resolve the config file
    config::Config::builder()
        .add_source(config::File::with_name(config_path))
        .build()?
        .try_deserialize::<AppConfig>()
}
