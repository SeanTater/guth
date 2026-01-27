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

        let stripped = name.strip_prefix("model.").unwrap_or(name);
        let tensor = tensors.tensor(name)?;
        state.insert(
            stripped.to_string(),
            TensorData {
                dtype: tensor.dtype(),
                shape: tensor.shape().iter().map(|v| *v as usize).collect(),
                data: tensor.data().to_vec(),
            },
        );
    }

    Ok(state)
}
