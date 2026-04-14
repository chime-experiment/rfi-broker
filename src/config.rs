//! Application configuration

/// Telescope config
#[derive(Debug, serde::Deserialize)]
pub struct TelescopeCoordinates {
    pub latitude: f64,
    pub longitude: f64,
    pub altitude: f64,
}

/// Endpoint config
#[derive(Debug, serde::Deserialize)]
pub struct RFIZeroingConfig {
    pub downtime: u64,
    pub hostname: String,
    pub target: String,
    pub first_stage: String,
    pub second_stage: String,
}

/// Overall config
#[derive(Debug, serde::Deserialize)]
pub struct AppConfig {
    pub telescope: TelescopeCoordinates,
    pub zeroing: RFIZeroingConfig,
}

pub type SharedAppConfig = std::sync::Arc<AppConfig>;

/// Load a [`Config`] from a toml file.
///
/// # Errors
/// Returns [`config:;ConfigError`] if the file can't be read.
pub fn load(config_path: &str) -> Result<AppConfig, config::ConfigError> {
    // Read and resolve the config file
    let config: config::Config = config::Config::builder()
        .add_source(config::File::with_name(config_path))
        .build()?;

    config.try_deserialize::<AppConfig>()
}
