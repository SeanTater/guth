//! SafeTensors weight loading and name mapping utilities.
//!
//! These helpers translate between Python checkpoint naming conventions and the
//! Rust module layout used by this crate.

use anyhow::Result;
use safetensors::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Raw tensor payload extracted from a SafeTensors file.
#[derive(Debug, Clone)]
pub struct TensorData {
    /// Scalar dtype in the file.
    pub dtype: Dtype,
    /// Shape as a list of dimensions.
    pub shape: Vec<usize>,
    /// Raw byte buffer in row-major order.
    pub data: Vec<u8>,
}

impl TensorData {
    /// Create TensorData from a safetensors TensorView.
    pub fn from_safetensor(tensor: safetensors::tensor::TensorView<'_>) -> Self {
        Self {
            dtype: tensor.dtype(),
            shape: tensor.shape().to_vec(),
            data: tensor.data().to_vec(),
        }
    }
}

/// Load a FlowLM checkpoint and map names into Rust module paths.
pub fn load_flow_lm_state_dict(path: impl AsRef<Path>) -> Result<HashMap<String, TensorData>> {
    let path = path.as_ref();
    let bytes = fs::read(path)?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let mut state = HashMap::new();

    for name in tensors.names() {
        let rest = name.strip_prefix("flow_lm.").unwrap_or(name);
        if let Some(mapped) = map_flow_lm_name(rest) {
            let tensor = tensors.tensor(name)?;
            state.insert(mapped, TensorData::from_safetensor(tensor));
        }
    }

    Ok(state)
}

/// Load a Mimi checkpoint and map names into Rust module paths.
pub fn load_mimi_state_dict(path: impl AsRef<Path>) -> Result<HashMap<String, TensorData>> {
    let path = path.as_ref();
    let bytes = fs::read(path)?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let mut state = HashMap::new();

    for name in tensors.names() {
        if name.starts_with("model.quantizer.vq.") || name == "model.quantizer.logvar_proj.weight" {
            continue;
        }

        if let Some(mapped) = map_mimi_name(name) {
            let tensor = tensors.tensor(name)?;
            state.insert(mapped, TensorData::from_safetensor(tensor));
        }
    }

    Ok(state)
}

/// Combined state dictionary for the full TTS model.
#[derive(Debug)]
pub struct TtsStateDict {
    /// FlowLM tensors keyed by Rust module path.
    pub flow_lm: HashMap<String, TensorData>,
    /// Mimi tensors keyed by Rust module path.
    pub mimi: HashMap<String, TensorData>,
    /// Optional speaker projection weights (voice conditioning).
    pub speaker_proj_weight: Option<TensorData>,
}

/// Load a combined TTS checkpoint with FlowLM + Mimi weights.
pub fn load_tts_state_dict(path: impl AsRef<Path>) -> Result<TtsStateDict> {
    let path = path.as_ref();
    let bytes = fs::read(path)?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let mut flow_lm = HashMap::new();
    let mut mimi = HashMap::new();
    let mut speaker_proj_weight = None;

    for name in tensors.names() {
        if name == "speaker_proj_weight" {
            let tensor = tensors.tensor(name)?;
            speaker_proj_weight = Some(TensorData::from_safetensor(tensor));
            continue;
        }

        if let Some(rest) = name.strip_prefix("flow_lm.") {
            if let Some(mapped) = map_flow_lm_name(rest) {
                let tensor = tensors.tensor(name)?;
                let tensor_data = TensorData::from_safetensor(tensor);
                if mapped == "speaker_proj_weight" {
                    speaker_proj_weight = Some(tensor_data.clone());
                }
                flow_lm.insert(mapped, tensor_data);
            }
            continue;
        }

        if let Some(rest) = name.strip_prefix("mimi.") {
            if let Some(mapped) = map_mimi_name(rest) {
                let tensor = tensors.tensor(name)?;
                mimi.insert(mapped, TensorData::from_safetensor(tensor));
            }
        }
    }

    Ok(TtsStateDict {
        flow_lm,
        mimi,
        speaker_proj_weight,
    })
}

/// Map FlowLM checkpoint tensor names into Rust module paths.
fn map_flow_lm_name(name: &str) -> Option<String> {
    // Mapping rules are intentionally explicit to make debugging weight loading easy.
    const FLOW_SKIP_EXACT: &[&str] = &[
        "condition_provider.conditioners.transcript_in_segment.learnt_padding",
        "condition_provider.conditioners.speaker_wavs.learnt_padding",
    ];
    const FLOW_SKIP_PREFIXES: &[&str] = &["flow.w_s_t."];
    const FLOW_RENAME_EXACT: &[(&str, &str)] = &[
        (
            "condition_provider.conditioners.transcript_in_segment.embed.weight",
            "conditioner.embed.weight",
        ),
        (
            "condition_provider.conditioners.speaker_wavs.output_proj.weight",
            "speaker_proj_weight",
        ),
    ];

    if FLOW_SKIP_EXACT.iter().any(|skip| *skip == name)
        || FLOW_SKIP_PREFIXES.iter().any(|prefix| name.starts_with(prefix))
    {
        return None;
    }

    if let Some(mapped) = map_exact(name, FLOW_RENAME_EXACT) {
        return Some(mapped);
    }

    Some(name.to_string())
}

/// Map Mimi checkpoint tensor names into Rust module paths.
fn map_mimi_name(name: &str) -> Option<String> {
    // Mapping rules are intentionally explicit to make debugging weight loading easy.
    const MIMI_SKIP_EXACT: &[&str] = &["quantizer.logvar_proj.weight"];
    const MIMI_SKIP_PREFIXES: &[&str] = &["quantizer.vq."];
    const MIMI_RENAME_EXACT: &[(&str, &str)] =
        &[("quantizer.output_proj.weight", "quantizer.weight")];
    const MIMI_PREFIX_MAP: &[(&str, &str)] = &[
        ("encoder_transformer.transformer.", "encoder_transformer."),
        ("decoder_transformer.transformer.", "decoder_transformer."),
        // Original: downsample.conv.conv.weight -> Target: downsample.conv.weight
        ("downsample.conv.conv.", "downsample.conv."),
        // Original: upsample.convtr.convtr.weight -> Target: upsample.conv.weight
        ("upsample.convtr.convtr.", "upsample.conv."),
        // Fallback: downsample.conv.weight -> downsample.conv.weight (if no double prefix)
        ("downsample.conv.", "downsample.conv."),
        // Fallback: upsample.convtr.weight -> upsample.conv.weight (if no double prefix)
        ("upsample.convtr.", "upsample.conv."),
    ];

    let name = name.strip_prefix("model.").unwrap_or(name);

    if MIMI_SKIP_EXACT.iter().any(|skip| *skip == name)
        || MIMI_SKIP_PREFIXES.iter().any(|prefix| name.starts_with(prefix))
    {
        return None;
    }

    if let Some(mapped) = map_exact(name, MIMI_RENAME_EXACT) {
        return Some(mapped);
    }

    if let Some(rest) = name.strip_prefix("encoder.model.") {
        return map_seanet_layer("encoder", rest);
    }
    if let Some(rest) = name.strip_prefix("decoder.model.") {
        return map_seanet_layer("decoder", rest);
    }

    if let Some(mapped) = map_prefix(name, MIMI_PREFIX_MAP) {
        return Some(mapped);
    }

    Some(name.to_string())
}

/// Apply exact-match rename rules.
fn map_exact(name: &str, rules: &[(&str, &str)]) -> Option<String> {
    for (from, to) in rules {
        if name == *from {
            return Some((*to).to_string());
        }
    }
    None
}

/// Apply prefix-based rename rules.
fn map_prefix(name: &str, rules: &[(&str, &str)]) -> Option<String> {
    for (prefix, target) in rules {
        if let Some(rest) = name.strip_prefix(prefix) {
            return Some(format!("{target}{rest}"));
        }
    }
    None
}

/// Map a SEANet layer name into the Rust `SeanetLayer` index format.
fn map_seanet_layer(prefix: &str, rest: &str) -> Option<String> {
    let mut parts = rest.split('.');
    let layer_idx = parts.next()?;
    let tail: Vec<&str> = parts.collect();
    match tail.as_slice() {
        ["conv", "weight"] => Some(format!("{prefix}.layers.{layer_idx}.conv.weight")),
        ["conv", "bias"] => Some(format!("{prefix}.layers.{layer_idx}.conv.bias")),
        ["convtr", "weight"] => Some(format!("{prefix}.layers.{layer_idx}.conv_transpose.weight")),
        ["convtr", "bias"] => Some(format!("{prefix}.layers.{layer_idx}.conv_transpose.bias")),
        ["block", block_idx, "conv", "weight"] => {
            let idx: usize = block_idx.parse().ok()?;
            let conv_idx = idx / 2;
            Some(format!(
                "{prefix}.layers.{layer_idx}.resblock.{conv_idx}.weight"
            ))
        }
        ["block", block_idx, "conv", "bias"] => {
            let idx: usize = block_idx.parse().ok()?;
            let conv_idx = idx / 2;
            Some(format!(
                "{prefix}.layers.{layer_idx}.resblock.{conv_idx}.bias"
            ))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::serialize;
    use safetensors::tensor::TensorView;
    use std::collections::HashMap;
    use std::fs;

    /// Write a temporary SafeTensors file from named views.
    fn write_safetensors(path: &Path, tensors: HashMap<String, TensorView<'_>>) {
        let bytes = serialize(&tensors, &None).expect("serialize safetensors");
        fs::write(path, bytes).expect("write safetensors");
    }

    #[test]
    fn flow_lm_filters_and_renames() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("flow.safetensors");

        let data: Vec<u8> = vec![0.0f32, 1.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let tensor = TensorView::new(Dtype::F32, vec![2], &data).expect("tensor view");

        let tensors: HashMap<String, TensorView<'_>> = vec![
            (
                "condition_provider.conditioners.transcript_in_segment.embed.weight".to_string(),
                tensor.clone(),
            ),
            (
                "condition_provider.conditioners.speaker_wavs.output_proj.weight".to_string(),
                tensor.clone(),
            ),
            ("flow.w_s_t.skip".to_string(), tensor),
        ]
        .into_iter()
        .collect();

        write_safetensors(&path, tensors);

        let state = load_flow_lm_state_dict(&path).expect("load flow lm state");
        assert!(state.contains_key("conditioner.embed.weight"));
        assert!(state.contains_key("speaker_proj_weight"));
        assert!(!state.contains_key("flow.w_s_t.skip"));
    }

    #[test]
    fn mimi_filters_and_strips_prefix() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("mimi.safetensors");

        let data: Vec<u8> = vec![0.0f32]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let tensor = TensorView::new(Dtype::F32, vec![1], &data).expect("tensor view");

        let tensors: HashMap<String, TensorView<'_>> = vec![
            ("model.encoder.weight".to_string(), tensor.clone()),
            ("model.quantizer.vq.skip".to_string(), tensor.clone()),
            ("model.quantizer.logvar_proj.weight".to_string(), tensor),
        ]
        .into_iter()
        .collect();

        write_safetensors(&path, tensors);

        let state = load_mimi_state_dict(&path).expect("load mimi state");
        assert!(state.contains_key("encoder.weight"));
        assert!(!state.contains_key("model.encoder.weight"));
        assert!(!state.contains_key("quantizer.vq.skip"));
        assert!(!state.contains_key("quantizer.logvar_proj.weight"));
    }
}
