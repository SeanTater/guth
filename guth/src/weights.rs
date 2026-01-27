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
