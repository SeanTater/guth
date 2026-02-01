//! Configuration types for TTS model architecture and weights.
//!
//! Configurations are typically loaded from YAML files using [`load_config`].

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// Flow matching configuration (dimensionality and depth).
pub struct FlowConfig {
    /// Hidden size inside the flow network.
    pub dim: i64,
    /// Number of residual blocks.
    pub depth: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// Transformer hyperparameters for the FlowLM backbone.
pub struct FlowLmTransformerConfig {
    /// Feed-forward expansion factor.
    pub hidden_scale: i64,
    /// RoPE base period.
    pub max_period: i64,
    /// Model width.
    pub d_model: i64,
    /// Number of attention heads.
    pub num_heads: i64,
    /// Number of transformer layers.
    pub num_layers: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// Lookup-table tokenizer and embedding settings.
pub struct LookupTableConfig {
    /// Embedding dimension for token IDs.
    pub dim: i64,
    /// Vocabulary size (number of bins).
    pub n_bins: i64,
    /// Tokenizer name (informational).
    pub tokenizer: String,
    /// Path or `hf://` URL to SentencePiece model.
    pub tokenizer_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// Full FlowLM configuration and optional weights.
pub struct FlowLmConfig {
    /// Tensor dtype string (e.g., "f32", "bf16").
    pub dtype: String,
    /// Flow network configuration.
    pub flow: FlowConfig,
    /// Transformer configuration.
    pub transformer: FlowLmTransformerConfig,
    /// Text lookup-table configuration.
    pub lookup_table: LookupTableConfig,
    /// Optional weights path for FlowLM-only checkpoints.
    pub weights_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// SEANet encoder/decoder configuration for the Mimi codec.
pub struct SeanetConfig {
    /// Latent dimension produced by the encoder.
    pub dimension: i64,
    /// Number of audio channels.
    pub channels: i64,
    /// Base number of convolutional filters.
    pub n_filters: i64,
    /// Residual layers per down/up block.
    pub n_residual_layers: i64,
    /// Stride ratios for downsampling/upsampling.
    pub ratios: Vec<i64>,
    /// First/last convolution kernel sizes.
    pub kernel_size: i64,
    /// Residual block kernel size.
    pub residual_kernel_size: i64,
    /// Final convolution kernel size.
    pub last_kernel_size: i64,
    /// Dilation base for residual blocks.
    pub dilation_base: i64,
    /// Padding mode string ("constant" or "replicate").
    pub pad_mode: String,
    /// Compression factor inside residual blocks.
    pub compress: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// Transformer settings used inside Mimi for temporal modeling.
pub struct MimiTransformerConfig {
    /// Model width.
    pub d_model: i64,
    /// Input projection dimension.
    pub input_dimension: i64,
    /// Output projection dimensions (one per head).
    pub output_dimensions: Vec<i64>,
    /// Number of attention heads.
    pub num_heads: i64,
    /// Number of transformer layers.
    pub num_layers: i64,
    /// Optional LayerScale initialization value.
    pub layer_scale: f32,
    /// Attention context length (0 for full).
    pub context: i64,
    /// RoPE base period.
    #[serde(default = "default_mimi_max_period")]
    pub max_period: f32,
    /// Feed-forward hidden size.
    pub dim_feedforward: i64,
}

/// Default RoPE max period for Mimi when not specified in config.
fn default_mimi_max_period() -> f32 {
    10000.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// Quantizer configuration for Mimi (used in weight shapes).
pub struct QuantizerConfig {
    /// Latent dimension.
    pub dimension: i64,
    /// Output dimension used by dummy quantizer.
    pub output_dimension: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// Mimi codec configuration and optional weights.
pub struct MimiConfig {
    /// Tensor dtype string (e.g., "f32", "bf16").
    pub dtype: String,
    /// Audio sample rate in Hz.
    pub sample_rate: i64,
    /// Number of audio channels.
    pub channels: i64,
    /// Frame rate in Hz (latents per second).
    pub frame_rate: f32,
    /// SEANet encoder/decoder configuration.
    pub seanet: SeanetConfig,
    /// Transformer configuration.
    pub transformer: MimiTransformerConfig,
    /// Quantizer configuration.
    pub quantizer: QuantizerConfig,
    /// Optional weights path for Mimi-only checkpoints.
    pub weights_path: Option<String>,
}

/// Top-level TTS model configuration.
///
/// Load from YAML using [`load_config`]. Weight paths can be local files or HuggingFace
/// URLs using the `hf://` scheme (e.g., `hf://kyutai/hibiki-v0.2/model.safetensors`).
///
/// # Example YAML
///
/// ```yaml
/// weights_path: "hf://kyutai/hibiki-v0.2/model.safetensors"
/// flow_lm:
///   dtype: bf16
///   flow: { dim: 1024, depth: 6 }
///   transformer: { d_model: 1024, num_heads: 16, num_layers: 24, ... }
///   lookup_table: { dim: 1024, n_bins: 8192, tokenizer: sentencepiece, ... }
/// mimi:
///   dtype: bf16
///   sample_rate: 24000
///   channels: 1
///   frame_rate: 12.5
///   # ... seanet, transformer, quantizer configs
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// Top-level configuration for the full TTS stack.
pub struct Config {
    /// FlowLM transformer configuration
    pub flow_lm: FlowLmConfig,
    /// Mimi neural codec configuration
    pub mimi: MimiConfig,
    /// Path to combined model weights (supports `hf://` URLs)
    pub weights_path: Option<String>,
    /// Fallback weights without voice cloning support
    pub weights_path_without_voice_cloning: Option<String>,
}

/// Load a model configuration from a YAML file.
///
/// # Errors
///
/// Returns an error if the file doesn't exist or contains invalid YAML.
pub fn load_config(path: impl AsRef<Path>) -> anyhow::Result<Config> {
    let path = path.as_ref();
    if !path.exists() {
        anyhow::bail!("Config file not found: {}", path.display());
    }

    let data = fs::read_to_string(path)?;
    let config: Config = serde_yaml::from_str(&data)?;
    Ok(config)
}

/// Resolve a possibly relative path against a config file location.
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
