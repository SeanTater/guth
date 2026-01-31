use anyhow::Result;
use safetensors::{SafeTensors, Dtype};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct TensorData {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

pub fn load_flow_lm_state_dict(path: impl AsRef<Path>) -> Result<HashMap<String, TensorData>> {
    let path = path.as_ref();
    let bytes = fs::read(path)?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let mut state = HashMap::new();

    for name in tensors.names() {
        let name = name.strip_prefix("flow_lm.").unwrap_or(name);
        if name.starts_with("flow.w_s_t.")
            || name
                == "condition_provider.conditioners.transcript_in_segment.learnt_padding"
            || name == "condition_provider.conditioners.speaker_wavs.learnt_padding"
        {
            continue;
        }

        let mut new_name = name.to_string();
        if name == "condition_provider.conditioners.transcript_in_segment.embed.weight" {
            new_name = "conditioner.embed.weight".to_string();
        }
        if name == "condition_provider.conditioners.speaker_wavs.output_proj.weight" {
            new_name = "speaker_proj_weight".to_string();
        }

        let tensor = tensors.tensor(name)?;
        state.insert(
            new_name,
            TensorData {
                dtype: tensor.dtype(),
                shape: tensor.shape().iter().map(|v| *v as usize).collect(),
                data: tensor.data().to_vec(),
            },
        );
    }

    Ok(state)
}

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
            state.insert(
                mapped,
                TensorData {
                    dtype: tensor.dtype(),
                    shape: tensor.shape().iter().map(|v| *v as usize).collect(),
                    data: tensor.data().to_vec(),
                },
            );
        }
    }

    Ok(state)
}

#[derive(Debug)]
pub struct TtsStateDict {
    pub flow_lm: HashMap<String, TensorData>,
    pub mimi: HashMap<String, TensorData>,
    pub speaker_proj_weight: Option<TensorData>,
}

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
            speaker_proj_weight = Some(TensorData {
                dtype: tensor.dtype(),
                shape: tensor.shape().iter().map(|v| *v as usize).collect(),
                data: tensor.data().to_vec(),
            });
            continue;
        }

        if let Some(rest) = name.strip_prefix("flow_lm.") {
            if rest.starts_with("flow.w_s_t.")
                || rest
                    == "condition_provider.conditioners.transcript_in_segment.learnt_padding"
                || rest == "condition_provider.conditioners.speaker_wavs.learnt_padding"
            {
                continue;
            }

            let mut new_name = rest.to_string();
            if rest == "condition_provider.conditioners.transcript_in_segment.embed.weight" {
                new_name = "conditioner.embed.weight".to_string();
            }
            if rest == "condition_provider.conditioners.speaker_wavs.output_proj.weight" {
                new_name = "speaker_proj_weight".to_string();
            }
            let tensor = tensors.tensor(name)?;
            if rest == "speaker_proj_weight" || new_name == "speaker_proj_weight" {
                speaker_proj_weight = Some(TensorData {
                    dtype: tensor.dtype(),
                    shape: tensor.shape().iter().map(|v| *v as usize).collect(),
                    data: tensor.data().to_vec(),
                });
            }
            flow_lm.insert(
                new_name,
                TensorData {
                    dtype: tensor.dtype(),
                    shape: tensor.shape().iter().map(|v| *v as usize).collect(),
                    data: tensor.data().to_vec(),
                },
            );
            continue;
        }

        if let Some(rest) = name.strip_prefix("mimi.") {
            if let Some(mapped) = map_mimi_name(rest) {
                let tensor = tensors.tensor(name)?;
                mimi.insert(
                    mapped,
                    TensorData {
                        dtype: tensor.dtype(),
                        shape: tensor.shape().iter().map(|v| *v as usize).collect(),
                        data: tensor.data().to_vec(),
                    },
                );
            }
        }
    }

    Ok(TtsStateDict {
        flow_lm,
        mimi,
        speaker_proj_weight,
    })
}

fn map_mimi_name(name: &str) -> Option<String> {
    let name = name.strip_prefix("model.").unwrap_or(name);
    if name.starts_with("quantizer.vq.") || name == "quantizer.logvar_proj.weight" {
        return None;
    }
    if name == "quantizer.output_proj.weight" {
        return Some("quantizer.weight".to_string());
    }
    if let Some(rest) = name.strip_prefix("encoder.model.") {
        return map_seanet_layer("encoder", rest);
    }
    if let Some(rest) = name.strip_prefix("decoder.model.") {
        return map_seanet_layer("decoder", rest);
    }
    if let Some(rest) = name.strip_prefix("encoder_transformer.transformer.") {
        return Some(format!("encoder_transformer.{rest}"));
    }
    if let Some(rest) = name.strip_prefix("decoder_transformer.transformer.") {
        return Some(format!("decoder_transformer.{rest}"));
    }
    if let Some(rest) = name.strip_prefix("encoder_transformer.") {
        return Some(format!("encoder_transformer.{rest}"));
    }
    if let Some(rest) = name.strip_prefix("decoder_transformer.") {
        return Some(format!("decoder_transformer.{rest}"));
    }
    // Original: downsample.conv.conv.weight -> Target: downsample.conv.weight
    if let Some(rest) = name.strip_prefix("downsample.conv.conv.") {
        return Some(format!("downsample.conv.{rest}"));
    }
    // Original: upsample.convtr.convtr.weight -> Target: upsample.conv.weight
    if let Some(rest) = name.strip_prefix("upsample.convtr.convtr.") {
        return Some(format!("upsample.conv.{rest}"));
    }
    // Fallback: downsample.conv.weight -> downsample.conv.weight (if no double prefix)
    if let Some(rest) = name.strip_prefix("downsample.conv.") {
        return Some(format!("downsample.conv.{rest}"));
    }
    // Fallback: upsample.convtr.weight -> upsample.conv.weight (if no double prefix)
    if let Some(rest) = name.strip_prefix("upsample.convtr.") {
        return Some(format!("upsample.conv.{rest}"));
    }
    Some(name.to_string())
}

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
    use safetensors::tensor::TensorView;
    use safetensors::serialize;
    use std::collections::HashMap;
    use std::fs;

    fn write_safetensors(path: &Path, tensors: HashMap<String, TensorView<'_>>) {
        let bytes = serialize(&tensors, &None).expect("serialize safetensors");
        fs::write(path, bytes).expect("write safetensors");
    }

    #[test]
    fn flow_lm_filters_and_renames() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("flow.safetensors");

        let data: Vec<u8> = vec![0.0f32, 1.0].into_iter().flat_map(f32::to_le_bytes).collect();
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

        let data: Vec<u8> = vec![0.0f32].into_iter().flat_map(f32::to_le_bytes).collect();
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
