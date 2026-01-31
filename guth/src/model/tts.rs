use crate::model::flow_lm::{FlowLmModel, FlowLmState};
use crate::model::mimi::{MimiModel, MimiState};
use crate::config::Config;
use crate::download::download_if_necessary;
use crate::weights::{load_flow_lm_state_dict, load_mimi_state_dict, load_tts_state_dict};
use burn::tensor::backend::Backend;
use burn::tensor::module::linear;
use burn::tensor::{Bool, ElementConversion, Int, Tensor};
use std::sync::mpsc::{self, Receiver};

#[derive(Debug)]
pub struct TtsState<B: Backend> {
    pub flow_lm: FlowLmState<B>,
    pub mimi: MimiState<B>,
}

#[derive(Debug)]
pub struct TtsModel<B: Backend> {
    pub flow_lm: FlowLmModel<B>,
    pub mimi: MimiModel<B>,
    pub speaker_proj_weight: Tensor<B, 2>,
    pub temp: f32,
    pub lsd_decode_steps: usize,
    pub noise_clamp: Option<f32>,
    pub eos_threshold: f32,
}

impl<B: Backend + 'static> TtsModel<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        flow_lm: FlowLmModel<B>,
        mimi: MimiModel<B>,
        speaker_proj_weight: Tensor<B, 2>,
        temp: f32,
        lsd_decode_steps: usize,
        noise_clamp: Option<f32>,
        eos_threshold: f32,
    ) -> Self {
        Self {
            flow_lm,
            mimi,
            speaker_proj_weight,
            temp,
            lsd_decode_steps,
            noise_clamp,
            eos_threshold,
        }
    }

    pub fn from_config(
        config: &Config,
        temp: f32,
        lsd_decode_steps: usize,
        noise_clamp: Option<f32>,
        eos_threshold: f32,
        device: &B::Device,
    ) -> anyhow::Result<Self> {
        let latent_dim = config.mimi.quantizer.dimension as usize;
        let mut flow_lm = FlowLmModel::from_config(&config.flow_lm, latent_dim, device)?;
        let mut mimi = MimiModel::from_config(&config.mimi, device);

        let mut speaker_proj_weight =
            Tensor::<B, 2>::zeros([flow_lm.dim, flow_lm.ldim], device);

        if let Some(weights_path) = config.weights_path.as_ref() {
            let weights_path = match download_if_necessary(weights_path) {
                Ok(path) => path,
                Err(err) => {
                    if let Some(fallback) = config.weights_path_without_voice_cloning.as_ref() {
                        download_if_necessary(fallback)?
                    } else {
                        return Err(err);
                    }
                }
            };
            let tts_state = load_tts_state_dict(&weights_path)?;
            if !tts_state.flow_lm.is_empty() {
                flow_lm.load_state_dict(&tts_state.flow_lm, device)?;
            }
            if !tts_state.mimi.is_empty() {
                mimi.load_state_dict(&tts_state.mimi, device)?;
            }
            if let Some(weight) = tts_state.speaker_proj_weight {
                speaker_proj_weight = tensor2_from_data(&weight, device)?;
            } else if let Some(weight) = tts_state.flow_lm.get("speaker_proj_weight") {
                speaker_proj_weight = tensor2_from_data(weight, device)?;
            }
            let speaker_proj_weight = speaker_proj_weight.transpose();
            return Ok(Self::new(
                flow_lm,
                mimi,
                speaker_proj_weight,
                temp,
                lsd_decode_steps,
                noise_clamp,
                eos_threshold,
            ));
        }

        if let Some(flow_path) = config.flow_lm.weights_path.as_ref() {
            if config.mimi.weights_path.is_none() {
                anyhow::bail!("mimi.weights_path must be set when flow_lm.weights_path is provided");
            }
            let flow_path = download_if_necessary(flow_path)?;
            let flow_state = load_flow_lm_state_dict(flow_path)?;
            flow_lm.load_state_dict(&flow_state, device)?;
            if let Some(weight) = flow_state.get("speaker_proj_weight") {
                speaker_proj_weight = tensor2_from_data(weight, device)?;
            }
        }

        if let Some(mimi_path) = config.mimi.weights_path.as_ref() {
            if config.flow_lm.weights_path.is_none() {
                anyhow::bail!("flow_lm.weights_path must be set when mimi.weights_path is provided");
            }
            let mimi_path = download_if_necessary(mimi_path)?;
            let mimi_state = load_mimi_state_dict(mimi_path)?;
            mimi.load_state_dict(&mimi_state, device)?;
        }

        let speaker_proj_weight = speaker_proj_weight.transpose();
        Ok(Self::new(
            flow_lm,
            mimi,
            speaker_proj_weight,
            temp,
            lsd_decode_steps,
            noise_clamp,
            eos_threshold,
        ))
    }

    pub fn init_state(
        &self,
        batch_size: usize,
        flow_sequence_length: usize,
        mimi_sequence_length: usize,
        device: &B::Device,
    ) -> TtsState<B> {
        let flow_lm = self.flow_lm.init_state(batch_size, flow_sequence_length);
        let mimi = self.mimi.init_state(batch_size, mimi_sequence_length, device);
        TtsState { flow_lm, mimi }
    }

    pub fn generate_latents_from_tokens(
        &self,
        tokens: Tensor<B, 2, Int>,
        state: &mut TtsState<B>,
        max_gen_len: usize,
        frames_after_eos: usize,
    ) -> anyhow::Result<(Tensor<B, 3>, Tensor<B, 2, Bool>)> {
        self.run_flow_lm_and_increment(
            state,
            Some(tokens.clone()),
            None,
            None,
        )?;

        let [batch, _] = tokens.dims();
        let device = tokens.device();
        let mut backbone_input = Tensor::<B, 3>::full(
            [batch, 1, self.flow_lm.ldim],
            f32::NAN,
            &device,
        );

        let mut latents = Vec::with_capacity(max_gen_len);
        let mut eos_flags = Vec::with_capacity(max_gen_len);
        let mut eos_step: Option<usize> = None;

        for step in 0..max_gen_len {
            let (latent, is_eos) =
                self.run_flow_lm_and_increment(state, None, Some(backbone_input.clone()), None)?;
            if eos_step.is_none() {
                let eos_any = is_eos.clone().any().into_scalar();
                if eos_any.elem() {
                    eos_step = Some(step);
                }
            }

            backbone_input = latent.clone();
            latents.push(latent);
            eos_flags.push(is_eos);

            if let Some(eos_step) = eos_step {
                if step >= eos_step + frames_after_eos {
                    break;
                }
            }
        }

        let latents = Tensor::cat(latents, 1);
        let eos_flags = Tensor::cat(eos_flags, 1);
        Ok((latents, eos_flags))
    }

    pub fn decode_latents(&self, latents: Tensor<B, 3>, state: &mut TtsState<B>) -> Tensor<B, 3> {
        let [_batch, steps, _] = latents.dims();
        let mut audio_chunks = Vec::with_capacity(steps);
        for idx in 0..steps {
            let latent = latents.clone().narrow(1, idx, 1);
            let audio = self.decode_latent_step(latent, &mut state.mimi);
            audio_chunks.push(audio);
        }
        Tensor::cat(audio_chunks, 2)
    }

    pub fn generate_audio_from_tokens(
        &self,
        tokens: Tensor<B, 2, Int>,
        state: &mut TtsState<B>,
        max_gen_len: usize,
        frames_after_eos: usize,
    ) -> anyhow::Result<(Tensor<B, 3>, Tensor<B, 2, Bool>, Tensor<B, 3>)> {
        let (latents, eos) =
            self.generate_latents_from_tokens(tokens, state, max_gen_len, frames_after_eos)?;
        let audio = self.decode_latents(latents.clone(), state);
        Ok((latents, eos, audio))
    }

    /// Spawn a background worker to generate audio chunks and stream them over a channel.
    ///
    /// Returns a receiver that yields audio chunks. If an internal error occurs during
    /// generation, the channel will close early. Errors are logged to stderr.
    pub fn generate_audio_stream(
        self,
        tokens: Tensor<B, 2, Int>,
        max_gen_len: usize,
        frames_after_eos: usize,
    ) -> Receiver<Tensor<B, 3>>
    where
        B: Backend + Send + Sync + 'static,
    {
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let device = tokens.device();
            let flow_len = tokens.dims()[1] + max_gen_len + 1;
            let mut state = self.init_state(1, flow_len, max_gen_len, &device);

            // Initial text conditioning - no audio conditioning, so dimension mismatch cannot occur.
            if let Err(e) = self.run_flow_lm_and_increment(&mut state, Some(tokens), None, None) {
                eprintln!("Error in generate_audio_stream: {e}");
                return;
            }

            let mut backbone_input = Tensor::<B, 3>::full(
                [1, 1, self.flow_lm.ldim],
                f32::NAN,
                &device,
            );
            let mut eos_step: Option<usize> = None;
            for step in 0..max_gen_len {
                // Generation loop - no audio conditioning, so dimension mismatch cannot occur.
                let (latent, is_eos) = match self.run_flow_lm_and_increment(
                    &mut state,
                    None,
                    Some(backbone_input.clone()),
                    None,
                ) {
                    Ok(result) => result,
                    Err(e) => {
                        eprintln!("Error in generate_audio_stream: {e}");
                        break;
                    }
                };
                if eos_step.is_none() {
                    let eos_any = is_eos.clone().any().into_scalar();
                    if eos_any.elem() {
                        eos_step = Some(step);
                    }
                }
                if let Some(eos_step) = eos_step {
                    if step >= eos_step + frames_after_eos {
                        break;
                    }
                }

                let latent_step = latent.clone().reshape([1, 1, self.flow_lm.ldim]);
                let audio = self.decode_latent_step(latent_step, &mut state.mimi);
                if tx.send(audio).is_err() {
                    break;
                }
                backbone_input = latent.reshape([1, 1, self.flow_lm.ldim]);
            }
        });
        rx
    }

    pub fn condition_on_audio(
        &self,
        audio_prompt: Tensor<B, 3>,
        state: &mut TtsState<B>,
    ) -> anyhow::Result<()> {
        let latents = self.mimi.encode_to_latent(audio_prompt);
        let latents = latents.swap_dims(1, 2);
        let conditioning = linear(latents, self.speaker_proj_weight.clone(), None);
        self.run_flow_lm_and_increment(state, None, None, Some(conditioning))?;
        Ok(())
    }

    /// Apply pre-computed voice conditioning to the model state.
    ///
    /// The conditioning tensor should be a 3D tensor of shape `[batch, seq, dim]`
    /// where `dim` matches the transformer's d_model. This tensor is typically
    /// created by `voice encode` command which encodes audio through mimi and
    /// projects it with the speaker projection weight.
    pub fn condition_on_precomputed(
        &self,
        conditioning: Tensor<B, 3>,
        state: &mut TtsState<B>,
    ) -> anyhow::Result<()> {
        self.run_flow_lm_and_increment(state, None, None, Some(conditioning))?;
        Ok(())
    }

    fn run_flow_lm_and_increment(
        &self,
        state: &mut TtsState<B>,
        text_tokens: Option<Tensor<B, 2, Int>>,
        backbone_input_latents: Option<Tensor<B, 3>>,
        audio_conditioning: Option<Tensor<B, 3>>,
    ) -> anyhow::Result<(Tensor<B, 3>, Tensor<B, 2, Bool>)> {
        // FlowLM manages its own KV cache append; no explicit increment needed here.
        let device = self.flow_lm.bos_emb.device();

        let text_tokens = text_tokens.unwrap_or_else(|| {
            empty_tensor2_int::<B>(1, 0, &device)
        });
        let backbone_input_latents = backbone_input_latents.unwrap_or_else(|| {
            empty_tensor3::<B>(1, 0, self.flow_lm.ldim, &device)
        });
        let text_len = text_tokens.dims()[1];
        let backbone_len = backbone_input_latents.dims()[1];

        let text_embeddings = if text_len == 0 {
            empty_tensor3::<B>(1, 0, self.flow_lm.dim, &device)
        } else {
            self.flow_lm.conditioner.forward_tokens(text_tokens.clone())
        };
        let mut audio_conditioning = audio_conditioning.unwrap_or_else(|| {
            let dim = if text_len == 0 {
                self.flow_lm.dim
            } else {
                text_embeddings.dims()[2]
            };
            empty_tensor3::<B>(1, 0, dim, &device)
        });
        let audio_len = audio_conditioning.dims()[1];
        if text_embeddings.dims()[2] != audio_conditioning.dims()[2] {
            if audio_len == 0 {
                audio_conditioning = empty_tensor3::<B>(1, 0, text_embeddings.dims()[2], &device);
            } else {
                anyhow::bail!(
                    "Audio conditioning dim {} does not match text embedding dim {}",
                    audio_conditioning.dims()[2],
                    text_embeddings.dims()[2]
                );
            }
        }
        let text_embeddings = Tensor::cat(vec![text_embeddings, audio_conditioning.clone()], 1);

        let (latent, is_eos) = self.flow_lm.sample_next_latent(
            backbone_input_latents,
            text_embeddings,
            &mut state.flow_lm,
            self.lsd_decode_steps,
            self.temp,
            self.noise_clamp,
            self.eos_threshold,
        );

        let _ = (text_len, backbone_len, audio_len);

        let latent_dims = latent.dims();
        let latent = latent.reshape([latent_dims[0], 1, latent_dims[1]]);
        Ok((latent, is_eos))
    }

    fn decode_latent_step(
        &self,
        latent: Tensor<B, 3>,
        state: &mut MimiState<B>,
    ) -> Tensor<B, 3> {
        let emb_mean = self
            .flow_lm
            .emb_mean
            .clone()
            .reshape([1, 1, self.flow_lm.ldim]);
        let emb_std = self
            .flow_lm
            .emb_std
            .clone()
            .reshape([1, 1, self.flow_lm.ldim]);
        let decoded = latent.mul(emb_std).add(emb_mean);
        let transposed = decoded.swap_dims(1, 2);
        let quantized = self.mimi.quantizer.forward(transposed);
        let audio = self.mimi.decode_from_latent(quantized, state);
        // Increment by upsample stride, not 1, to match Python's behavior
        // The decoder transformer processes `upsample_stride` upsampled time steps per input frame
        self.mimi.increment_step(state, self.mimi.upsample_stride());
        audio
    }
}

fn tensor2_from_data<B: Backend>(
    tensor: &crate::weights::TensorData,
    device: &B::Device,
) -> anyhow::Result<Tensor<B, 2>> {
    let shape: [usize; 2] = tensor.shape.clone().try_into().map_err(|_| {
        anyhow::anyhow!("Expected 2D tensor, got shape {:?}", tensor.shape)
    })?;
    let mut values = Vec::new();
    match tensor.dtype {
        safetensors::Dtype::F32 => {
            values.reserve(tensor.data.len() / 4);
            for chunk in tensor.data.chunks_exact(4) {
                values.push(f32::from_le_bytes(chunk.try_into().unwrap()));
            }
        }
        safetensors::Dtype::BF16 => {
            values.reserve(tensor.data.len() / 2);
            for chunk in tensor.data.chunks_exact(2) {
                let bits = u16::from_le_bytes(chunk.try_into().unwrap()) as u32;
                values.push(f32::from_bits(bits << 16));
            }
        }
        _ => anyhow::bail!("Unsupported dtype {:?}", tensor.dtype),
    }
    Ok(Tensor::from_data(burn::tensor::TensorData::new(values, shape), device))
}

fn empty_tensor2_int<B: Backend>(batch: usize, len: usize, device: &B::Device) -> Tensor<B, 2, Int> {
    if len == 0 {
        let data: Vec<i64> = Vec::new();
        let td = burn::tensor::TensorData::new(data, [batch, len]);
        Tensor::from_data(td, device)
    } else {
        Tensor::<B, 2, Int>::zeros([batch, len], device)
    }
}

fn empty_tensor3<B: Backend>(
    batch: usize,
    seq: usize,
    dim: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    if seq == 0 {
        let data: Vec<f32> = Vec::new();
        let td = burn::tensor::TensorData::new(data, [batch, seq, dim]);
        Tensor::from_data(td, device)
    } else {
        Tensor::<B, 3>::zeros([batch, seq, dim], device)
    }
}

#[cfg(test)]
mod tests {
    use super::tensor2_from_data;
    use burn::tensor::Tensor;
    use burn_ndarray::{NdArray, NdArrayDevice};
    use safetensors::Dtype;

    #[test]
    fn tensor2_from_data_decodes_bf16() {
        let values = [1.0_f32, -0.5_f32, 2.25_f32];
        let mut data = Vec::new();
        let mut expected = Vec::new();
        for value in values {
            let bits = value.to_bits();
            let bf16 = (bits >> 16) as u16;
            data.extend_from_slice(&bf16.to_le_bytes());
            expected.push(f32::from_bits((bf16 as u32) << 16));
        }
        let tensor = crate::weights::TensorData {
            dtype: Dtype::BF16,
            shape: vec![1, expected.len()],
            data,
        };
        let device = NdArrayDevice::default();
        let decoded: Tensor<NdArray<f32>, 2> =
            tensor2_from_data(&tensor, &device).expect("decode bf16");
        let decoded_data = decoded.to_data();
        let decoded_values = decoded_data.as_slice::<f32>().expect("slice");
        assert_eq!(decoded_values, expected.as_slice());
    }

    #[test]
    fn empty_tensor_helpers_return_zero_length() {
        let device = NdArrayDevice::default();
        let empty2 = super::empty_tensor2_int::<NdArray<f32>>(1, 0, &device);
        assert_eq!(empty2.dims(), [1, 0]);
        let empty3 = super::empty_tensor3::<NdArray<f32>>(1, 0, 4, &device);
        assert_eq!(empty3.dims(), [1, 0, 4]);
    }
}
