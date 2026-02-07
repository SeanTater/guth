//! High-level runtime wrapper for TTS model bootstrapping.
//!
//! This module provides a stable, convenience-focused API for constructing a
//! model, preparing text tokens, and applying voice conditioning. It is intended
//! to reduce boilerplate in CLIs and language bindings.

use crate::audio::io::WavIo;
use crate::audio::resample::AudioResampler;
use crate::conditioner::text::TextTokenizer;
use crate::config::{load_config, Config};
use crate::model::tts::{TtsModel, TtsState};
use crate::perf::{self, Metric};
use anyhow::Result;
use burn::tensor::backend::Backend;
use burn::tensor::module::linear;
use burn::tensor::{Bool, Int, Tensor, TensorData};
use std::path::Path;
use std::sync::mpsc::Receiver;

/// Generation parameters used when building a runtime.
#[derive(Debug, Clone, Copy)]
pub struct RuntimeParams {
    /// Sampling temperature (0.0 = deterministic).
    pub temp: f32,
    /// Number of LSD decode steps.
    pub lsd_decode_steps: usize,
    /// Optional noise clamp applied during sampling.
    pub noise_clamp: Option<f32>,
    /// End-of-sequence threshold.
    pub eos_threshold: f32,
}

/// Standard return type for batch audio generation.
pub type AudioGenerationResult<B> = (Tensor<B, 3>, Tensor<B, 2, Bool>, Tensor<B, 3>);

impl RuntimeParams {
    /// Build a parameter bundle for runtime creation.
    pub fn new(
        temp: f32,
        lsd_decode_steps: usize,
        noise_clamp: Option<f32>,
        eos_threshold: f32,
    ) -> Self {
        Self {
            temp,
            lsd_decode_steps,
            noise_clamp,
            eos_threshold,
        }
    }
}

impl Default for RuntimeParams {
    /// Defaults tuned for typical inference usage.
    fn default() -> Self {
        Self {
            temp: 0.7,
            lsd_decode_steps: 2,
            noise_clamp: None,
            eos_threshold: 0.0,
        }
    }
}

/// High-level TTS runtime that owns the model and configuration.
#[derive(Debug)]
pub struct TtsRuntime<B: Backend> {
    config: Config,
    model: TtsModel<B>,
}

impl<B: Backend + 'static> TtsRuntime<B> {
    /// Create a runtime from a config path.
    pub fn from_config_path(
        path: impl AsRef<Path>,
        params: RuntimeParams,
        device: &B::Device,
    ) -> Result<Self> {
        let _span = perf::span(Metric::RuntimeFromConfigPath);
        let config = load_config(path)?;
        Self::from_config(&config, params, device)
    }

    /// Create a runtime from an already-loaded config.
    pub fn from_config(config: &Config, params: RuntimeParams, device: &B::Device) -> Result<Self> {
        let _span = perf::span(Metric::RuntimeFromConfig);
        let model = TtsModel::from_config(
            config,
            params.temp,
            params.lsd_decode_steps,
            params.noise_clamp,
            params.eos_threshold,
            device,
        )?;
        Ok(Self {
            config: config.clone(),
            model,
        })
    }

    /// Access the loaded configuration.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Access the underlying model (advanced usage).
    pub fn model(&self) -> &TtsModel<B> {
        &self.model
    }

    /// Returns true when speaker projection weights are available.
    pub fn voice_cloning_supported(&self) -> bool {
        self.model.voice_cloning_supported
    }

    /// Prepare text into tokens and a suggested `frames_after_eos`.
    pub fn prepare_tokens(&self, text: &str) -> Result<(Tensor<B, 2, Int>, usize)> {
        let _span = perf::span(Metric::RuntimePrepareTokens);
        let tokenizer = self
            .model
            .flow_lm
            .conditioner
            .tokenizer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Tokenizer not loaded"))?;
        let (prepared, frames_after_eos) = TextTokenizer::prepare_text_prompt(text)?;
        let tokens = tokenizer.encode(&prepared)?;
        let tokens = tokens.into_iter().map(|v| v as i64).collect::<Vec<_>>();
        let device = self.model.flow_lm.bos_emb.device();
        let tokens_len = tokens.len();
        let tokens =
            Tensor::<B, 2, Int>::from_data(TensorData::new(tokens, [1, tokens_len]), &device);
        perf::add_count(Metric::TtsTokens, tokens_len as u64);
        Ok((tokens, frames_after_eos))
    }

    /// Initialize generation state sized for the provided tokens and max length.
    pub fn init_state_for_tokens(
        &self,
        tokens: &Tensor<B, 2, Int>,
        max_gen_len: usize,
    ) -> TtsState<B> {
        let device = tokens.device();
        let flow_len = tokens.dims()[1] + max_gen_len + 1;
        self.model
            .init_state(tokens.dims()[0], flow_len, max_gen_len, &device)
    }

    /// Generate audio from tokens in a single batch.
    pub fn generate_audio_from_tokens(
        &self,
        tokens: Tensor<B, 2, Int>,
        state: &mut TtsState<B>,
        max_gen_len: usize,
        frames_after_eos: usize,
    ) -> Result<AudioGenerationResult<B>> {
        self.model
            .generate_audio_from_tokens(tokens, state, max_gen_len, frames_after_eos)
    }

    /// Stream audio chunks from tokens using a pre-initialized state.
    ///
    /// Consumes the runtime because the model is moved to a background thread.
    pub fn generate_audio_stream_with_state(
        self,
        tokens: Tensor<B, 2, Int>,
        state: TtsState<B>,
        max_gen_len: usize,
        frames_after_eos: usize,
    ) -> Receiver<Tensor<B, 3>>
    where
        B: Backend + Send + Sync + 'static,
    {
        self.model
            .generate_audio_stream_with_state(tokens, state, max_gen_len, frames_after_eos)
    }

    /// Stream audio chunks from tokens with a fresh state.
    ///
    /// Consumes the runtime because the model is moved to a background thread.
    pub fn generate_audio_stream(
        self,
        tokens: Tensor<B, 2, Int>,
        max_gen_len: usize,
        frames_after_eos: usize,
    ) -> Receiver<Tensor<B, 3>>
    where
        B: Backend + Send + Sync + 'static,
    {
        self.model
            .generate_audio_stream(tokens, max_gen_len, frames_after_eos)
    }

    /// Condition the state on a precomputed conditioning tensor.
    pub fn condition_on_precomputed(
        &self,
        conditioning: Tensor<B, 3>,
        state: &mut TtsState<B>,
    ) -> Result<()> {
        self.model.condition_on_precomputed(conditioning, state)
    }

    /// Condition the model state on an audio file path.
    pub fn condition_on_audio_path(
        &self,
        path: impl AsRef<Path>,
        state: &mut TtsState<B>,
    ) -> Result<()> {
        let _span = perf::span(Metric::RuntimeConditionOnAudioPath);
        let (samples, sample_rate) = WavIo::read_audio(path)?;
        self.condition_on_audio_samples(samples, sample_rate, state)
    }

    /// Condition the model state on raw audio samples.
    pub fn condition_on_audio_samples(
        &self,
        samples: Vec<Vec<f32>>,
        sample_rate: u32,
        state: &mut TtsState<B>,
    ) -> Result<()> {
        let _span = perf::span(Metric::RuntimeConditionOnAudioSamples);
        let prompt = AudioResampler::convert_audio(
            samples,
            sample_rate,
            self.config.mimi.sample_rate as u32,
            self.config.mimi.channels as usize,
        )?;
        let device = self.model.flow_lm.bos_emb.device();
        let prompt_tensor = tensor_from_audio::<B>(prompt, &device);
        self.model.condition_on_audio(prompt_tensor, state)
    }

    /// Compute a conditioning tensor from an audio file path.
    pub fn conditioning_from_audio_path(&self, path: impl AsRef<Path>) -> Result<Tensor<B, 3>> {
        let _span = perf::span(Metric::RuntimeConditioningFromAudioPath);
        let (samples, sample_rate) = WavIo::read_audio(path)?;
        self.conditioning_from_audio_samples(samples, sample_rate)
    }

    /// Compute a conditioning tensor from an audio file path, truncating if needed.
    pub fn conditioning_from_audio_path_with_truncate(
        &self,
        path: impl AsRef<Path>,
        max_seconds: f32,
    ) -> Result<(Tensor<B, 3>, bool)> {
        let (samples, sample_rate) = WavIo::read_audio(path)?;
        self.conditioning_from_audio_samples_with_truncate(samples, sample_rate, max_seconds)
    }

    /// Compute a conditioning tensor from raw audio samples.
    pub fn conditioning_from_audio_samples(
        &self,
        samples: Vec<Vec<f32>>,
        sample_rate: u32,
    ) -> Result<Tensor<B, 3>> {
        let _span = perf::span(Metric::RuntimeConditioningFromAudioSamples);
        let prompt = AudioResampler::convert_audio(
            samples,
            sample_rate,
            self.config.mimi.sample_rate as u32,
            self.config.mimi.channels as usize,
        )?;
        let device = self.model.flow_lm.bos_emb.device();
        let prompt_tensor = tensor_from_audio::<B>(prompt, &device);
        Ok(compute_conditioning(&self.model, prompt_tensor))
    }

    /// Compute a conditioning tensor from raw audio samples, truncating if needed.
    pub fn conditioning_from_audio_samples_with_truncate(
        &self,
        samples: Vec<Vec<f32>>,
        sample_rate: u32,
        max_seconds: f32,
    ) -> Result<(Tensor<B, 3>, bool)> {
        let mut prompt = AudioResampler::convert_audio(
            samples,
            sample_rate,
            self.config.mimi.sample_rate as u32,
            self.config.mimi.channels as usize,
        )?;
        // Truncate long prompts to keep conditioning stable and aligned with the Python CLI.
        let max_samples = (max_seconds * self.config.mimi.sample_rate as f32).round() as usize;
        let truncated = truncate_audio_samples(&mut prompt, max_samples);
        let device = self.model.flow_lm.bos_emb.device();
        let prompt_tensor = tensor_from_audio::<B>(prompt, &device);
        Ok((compute_conditioning(&self.model, prompt_tensor), truncated))
    }
}

/// Convert per-channel samples into a `[1, channels, samples]` tensor.
fn tensor_from_audio<B: Backend>(samples: Vec<Vec<f32>>, device: &B::Device) -> Tensor<B, 3> {
    let channels = samples.len();
    let len = samples[0].len();
    let mut flat = Vec::with_capacity(channels * len);
    for channel in samples {
        flat.extend(channel);
    }
    Tensor::from_data(TensorData::new(flat, [1, channels, len]), device)
}

fn truncate_audio_samples(samples: &mut [Vec<f32>], max_samples: usize) -> bool {
    if samples.is_empty() {
        return false;
    }
    let current_len = samples[0].len();
    if current_len <= max_samples || max_samples == 0 {
        return false;
    }
    for channel in samples {
        channel.truncate(max_samples);
    }
    true
}

/// Project Mimi latents into FlowLM conditioning space.
fn compute_conditioning<B: Backend>(tts: &TtsModel<B>, audio_prompt: Tensor<B, 3>) -> Tensor<B, 3> {
    let latents = tts.mimi.encode_to_latent(audio_prompt);
    let latents = latents.swap_dims(1, 2);
    linear(latents, tts.speaker_proj_weight.clone(), None)
}
