//! [`DataState`] configuration, loaded from a YAML file.
//!
//! Extracts information about the expected datasets in a UDP packet
//! from a YAML config file.

use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

/// Supported dataset element types. Must be a subset of datatypes supported
/// by [`datastore::TypedBuffer`]
// NB: We end up repeating this list of supported types in a few
// places. Maybe there's a way to improve?
#[derive(Deserialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum DType {
    F32,
    F64,
    U8,
    U16,
    U32,
    U64,
}

/// One array dimension: a name and its size
#[derive(Deserialize, Clone, Debug)]
pub struct DimensionConfig {
    /// Label
    pub name: String,
    /// Number of elements
    pub size: usize,
}

/// One dataset within a packet: its name, the ordered list of dimension names
/// it uses (referencing the shared `dimensions` pool), and its element type.
#[derive(Deserialize, Clone, Debug)]
pub struct DatasetConfig {
    /// Identifier for this dataset
    pub name: String,
    /// Ordered list of dimension names
    pub dims: Vec<String>,
    /// Element type
    pub dtype: DType,
}

/// Top-level server configuration.
#[derive(Deserialize, Debug)]
pub struct Config {
    /// Declared shape of incomiung UDP array frame
    pub dimensions: Vec<DimensionConfig>,
    /// Ordered list of datasets expected in each UDP packet
    pub datasets: Vec<DatasetConfig>,
}

impl Config {
    /// Reads and deserializes a [`Config`] from a YAML file at `path`.
    ///
    /// # Panics
    /// Panics if the file cannot be read or if its contents are not valid YAML.
    pub fn from_file(path: &Path) -> Self {
        let text = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("Failed to read config file {}: {e}", path.display()));
        serde_yml::from_str(&text)
            .unwrap_or_else(|e| panic!("Failed to parse config file {}: {e}", path.display()))
    }

    /// Resolves the shape for a given [`DatasetConfig`].
    pub fn resolve_dataset_shape(&self, dataset: &DatasetConfig) -> Option<Vec<usize>> {
        // Map between dimension name and size
        let dimmap: HashMap<String, usize> = self
            .dimensions
            .iter()
            .map(|dim| (dim.name.clone(), dim.size))
            .collect();

        dataset
            .dims
            .iter()
            .map(|name| dimmap.get(name).cloned())
            .collect()
    }
}
