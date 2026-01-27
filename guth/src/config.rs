use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlowConfig {
    pub dim: i64,
    pub depth: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlowLmTransformerConfig {
    pub hidden_scale: i64,
    pub max_period: i64,
    pub d_model: i64,
    pub num_heads: i64,
    pub num_layers: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LookupTableConfig {
    pub dim: i64,
    pub n_bins: i64,
    pub tokenizer: String,
    pub tokenizer_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlowLmConfig {
    pub dtype: String,
    pub flow: FlowConfig,
    pub transformer: FlowLmTransformerConfig,
    pub lookup_table: LookupTableConfig,
    pub weights_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeanetConfig {
    pub dimension: i64,
    pub channels: i64,
    pub n_filters: i64,
    pub n_residual_layers: i64,
    pub ratios: Vec<i64>,
    pub kernel_size: i64,
    pub residual_kernel_size: i64,
    pub last_kernel_size: i64,
    pub dilation_base: i64,
    pub pad_mode: String,
    pub compress: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MimiTransformerConfig {
    pub d_model: i64,
    pub input_dimension: i64,
    pub output_dimensions: Vec<i64>,
    pub num_heads: i64,
    pub num_layers: i64,
    pub layer_scale: f32,
    pub context: i64,
    #[serde(default = "default_mimi_max_period")]
    pub max_period: f32,
    pub dim_feedforward: i64,
}

fn default_mimi_max_period() -> f32 {
    10000.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuantizerConfig {
    pub dimension: i64,
    pub output_dimension: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MimiConfig {
    pub dtype: String,
    pub sample_rate: i64,
    pub channels: i64,
    pub frame_rate: f32,
    pub seanet: SeanetConfig,
    pub transformer: MimiTransformerConfig,
    pub quantizer: QuantizerConfig,
    pub weights_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub flow_lm: FlowLmConfig,
    pub mimi: MimiConfig,
    pub weights_path: Option<String>,
    pub weights_path_without_voice_cloning: Option<String>,
}

pub fn load_config(path: impl AsRef<Path>) -> anyhow::Result<Config> {
    let path = path.as_ref();
    if !path.exists() {
        anyhow::bail!("Config file not found: {}", path.display());
    }

    let data = fs::read_to_string(path)?;
    let config: Config = serde_yaml::from_str(&data)?;
    Ok(config)
}

pub fn resolve_relative_path(config_path: &Path, maybe_relative: &str) -> PathBuf {
    let candidate = Path::new(maybe_relative);
    if candidate.is_absolute() {
        return candidate.to_path_buf();
    }
    config_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(candidate)
}
