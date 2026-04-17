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
    pub telescope: Option<TelescopeCoordinates>,
    pub zeroing: Option<RFIZeroingConfig>,
}

pub type SharedAppConfig = std::sync::Arc<AppConfig>;

/// Load a [`Config`] from a toml file.
///
/// # Errors
/// Returns [`config:;ConfigError`] if the file can't be read.
pub fn load(config_path: &str) -> Result<Option<AppConfig>, config::ConfigError> {
    // Read and resolve the config file
    let conf = config::Config::builder()
        .add_source(config::File::with_name(config_path))
        .build()?
        .try_deserialize::<AppConfig>()?;

    // Both config sections must be provided together, but it's ok
    // for neither to be provided
    match (&conf.telescope, &conf.zeroing) {
        (Some(_), Some(_)) => Ok(Some(conf)),
        (None, None) => Ok(None),
        _ => Err(config::ConfigError::NotFound(
            "`telescope` and `zeroing config sections must either \
            be provided or excluded together.`"
                .to_string(),
        )),
    }
}
