//! # guth - Fast CPU-Optimized Text-to-Speech
//!
//! A Rust implementation of flow-matching text-to-speech, optimized for CPU inference.
//! This library provides the core TTS pipeline used by the `pocket-tts` project.
//!
//! ## Architecture Overview
//!
//! The TTS pipeline consists of three main components:
//!
//! 1. **Text Conditioning** ([`TextTokenizer`]): Converts text to token sequences and embeds
//!    them into the model's latent space (similar to PyTorch's `nn.Embedding`).
//!
//! 2. **FlowLM**: A flow-matching transformer that autoregressively generates latent audio
//!    frames from text embeddings. Uses classifier-free guidance and can optionally condition
//!    on speaker audio for voice cloning.
//!
//! 3. **Mimi Codec**: A neural audio codec that decodes the latent frames into PCM waveforms.
//!    Based on the SEANet architecture with streaming support.
//!
//! ## Quick Start
//!
//! ```no_run
//! use burn_ndarray::{NdArray, NdArrayDevice};
//! use burn::tensor::{Int, Tensor, TensorData};
//! use guth::{load_config, TtsModel, TextTokenizer};
//!
//! // Load configuration and model
//! let config = load_config("config.yaml").unwrap();
//! let device = NdArrayDevice::default();
//! let tts = TtsModel::<NdArray<f32>>::from_config(
//!     &config,
//!     0.7,  // temperature
//!     2,    // LSD decode steps
//!     None, // noise clamp
//!     0.0,  // EOS threshold
//!     &device,
//! ).unwrap();
//!
//! // Prepare text and tokenize
//! let tokenizer = tts.flow_lm.conditioner.tokenizer.as_ref().unwrap();
//! let (prepared, frames_after_eos) = TextTokenizer::prepare_text_prompt("Hello, world!").unwrap();
//! let tokens = tokenizer.encode(&prepared).unwrap();
//! let tokens: Vec<i64> = tokens.into_iter().map(|t| t as i64).collect();
//! let tokens = Tensor::<NdArray<f32>, 2, Int>::from_data(
//!     TensorData::new(tokens.clone(), [1, tokens.len()]),
//!     &device,
//! );
//!
//! // Generate audio (batch mode)
//! let max_gen_len = 256;
//! let mut state = tts.init_state(1, tokens.dims()[1] + max_gen_len + 1, max_gen_len, &device);
//! let (_latents, _eos, audio) = tts.generate_audio_from_tokens(
//!     tokens, &mut state, max_gen_len, frames_after_eos,
//! ).unwrap();
//! ```
//!
//! ## Streaming Generation
//!
//! For real-time applications, use streaming generation which yields audio chunks as they
//! are produced:
//!
//! ```no_run
//! # use burn_ndarray::{NdArray, NdArrayDevice};
//! # use burn::tensor::{Int, Tensor, TensorData};
//! # use guth::{load_config, TtsModel};
//! # let config = load_config("config.yaml").unwrap();
//! # let device = NdArrayDevice::default();
//! # let tts = TtsModel::<NdArray<f32>>::from_config(&config, 0.7, 2, None, 0.0, &device).unwrap();
//! # let tokens = Tensor::<NdArray<f32>, 2, Int>::zeros([1, 10], &device);
//! // Streaming returns a channel receiver
//! let receiver = tts.generate_audio_stream(tokens, 256, 8);
//!
//! for audio_chunk in receiver {
//!     // Process each chunk as it arrives
//!     // audio_chunk shape: [batch, channels, samples]
//!     println!("Got chunk with {} samples", audio_chunk.dims()[2]);
//! }
//! ```
//!
//! ## Voice Cloning
//!
//! The model supports voice cloning by conditioning on a reference audio sample:
//!
//! ```no_run
//! # use burn_ndarray::{NdArray, NdArrayDevice};
//! # use burn::tensor::{Int, Tensor, TensorData};
//! # use guth::{load_config, TtsModel};
//! # let config = load_config("config.yaml").unwrap();
//! # let device = NdArrayDevice::default();
//! # let tts = TtsModel::<NdArray<f32>>::from_config(&config, 0.7, 2, None, 0.0, &device).unwrap();
//! # let tokens = Tensor::<NdArray<f32>, 2, Int>::zeros([1, 10], &device);
//! // Load reference audio (must be at model's sample rate)
//! let reference_audio: Tensor<NdArray<f32>, 3> = // ... load audio
//! # Tensor::zeros([1, 1, 24000], &device);
//!
//! // Initialize state and condition on reference
//! let mut state = tts.init_state(1, 512, 256, &device);
//! tts.condition_on_audio(reference_audio, &mut state).unwrap();
//!
//! // Generate with the conditioned state
//! let receiver = tts.generate_audio_stream_with_state(tokens, state, 256, 8);
//! ```
//!
//! ## Configuration
//!
//! Models are configured via YAML files that specify architecture parameters and weight paths.
//! Weights can be loaded from local files or automatically downloaded from HuggingFace Hub
//! using the `hf://` URL scheme.
//!
//! See [`Config`] for the full configuration structure.

// Public modules - these are part of the stable API
pub mod audio;
pub mod config;
pub mod download;

// Internal modules - exposed for integration tests but not part of stable API.
// These may change without notice between versions.
#[doc(hidden)]
pub mod conditioner;
#[doc(hidden)]
pub mod model;
#[doc(hidden)]
pub mod modules;
#[doc(hidden)]
pub mod state;
#[doc(hidden)]
pub mod weights;

// Re-exports forming the public API
pub use conditioner::text::TextTokenizer;
pub use config::{load_config, Config};
pub use download::download_if_necessary;
pub use model::tts::{TtsModel, TtsState};
