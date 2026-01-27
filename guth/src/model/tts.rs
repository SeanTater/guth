use crate::model::flow_lm::{FlowLmModel, FlowLmState};
use crate::model::mimi::{MimiModel, MimiState};
use burn::tensor::backend::Backend;
use burn::tensor::module::linear;
use burn::tensor::{Bool, ElementConversion, Int, Tensor};

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

impl<B: Backend> TtsModel<B> {
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
    ) -> (Tensor<B, 3>, Tensor<B, 2, Bool>) {
        self.run_flow_lm_and_increment(
            state,
            Some(tokens.clone()),
            None,
            None,
        );

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
                self.run_flow_lm_and_increment(state, None, Some(backbone_input.clone()), None);
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

            backbone_input = latent.clone();
            latents.push(latent);
            eos_flags.push(is_eos);
        }

        let latents = Tensor::cat(latents, 1);
        let eos_flags = Tensor::cat(eos_flags, 1);
        (latents, eos_flags)
    }

    pub fn decode_latents(&self, latents: Tensor<B, 3>, state: &mut TtsState<B>) -> Tensor<B, 3> {
        let [_batch, steps, _] = latents.dims();
        let mut audio_chunks = Vec::with_capacity(steps);
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

        for idx in 0..steps {
            let latent = latents.clone().narrow(1, idx, 1);
            let decoded = latent.mul(emb_std.clone()).add(emb_mean.clone());
            let transposed = decoded.swap_dims(1, 2);
            let quantized = self.mimi.quantizer.forward(transposed);
            let audio = self.mimi.decode_from_latent(quantized, &mut state.mimi);
            self.mimi.increment_step(&mut state.mimi, 1);
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
    ) -> (Tensor<B, 3>, Tensor<B, 2, Bool>, Tensor<B, 3>) {
        let (latents, eos) =
            self.generate_latents_from_tokens(tokens, state, max_gen_len, frames_after_eos);
        let audio = self.decode_latents(latents.clone(), state);
        (latents, eos, audio)
    }

    pub fn condition_on_audio(&self, audio_prompt: Tensor<B, 3>, state: &mut TtsState<B>) {
        let latents = self.mimi.encode_to_latent(audio_prompt);
        let latents = latents.swap_dims(1, 2);
        let conditioning = linear(latents, self.speaker_proj_weight.clone(), None);
        self.run_flow_lm_and_increment(state, None, None, Some(conditioning));
    }

    fn run_flow_lm_and_increment(
        &self,
        state: &mut TtsState<B>,
        text_tokens: Option<Tensor<B, 2, Int>>,
        backbone_input_latents: Option<Tensor<B, 3>>,
        audio_conditioning: Option<Tensor<B, 3>>,
    ) -> (Tensor<B, 3>, Tensor<B, 2, Bool>) {
        let device = self.flow_lm.bos_emb.device();

        let text_tokens = text_tokens.unwrap_or_else(|| {
            Tensor::<B, 2, Int>::zeros([1, 0], &device)
        });
        let backbone_input_latents = backbone_input_latents.unwrap_or_else(|| {
            Tensor::<B, 3>::zeros([1, 0, self.flow_lm.ldim], &device)
        });
        let audio_conditioning = audio_conditioning.unwrap_or_else(|| {
            Tensor::<B, 3>::zeros([1, 0, self.flow_lm.dim], &device)
        });

        let text_len = text_tokens.dims()[1];
        let backbone_len = backbone_input_latents.dims()[1];
        let audio_len = audio_conditioning.dims()[1];

        let text_embeddings = if text_len == 0 {
            Tensor::<B, 3>::zeros([1, 0, self.flow_lm.dim], &device)
        } else {
            self.flow_lm.conditioner.forward_tokens(text_tokens.clone())
        };
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
        (latent, is_eos)
    }
}
