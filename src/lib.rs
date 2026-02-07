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
//! For most applications, [`TtsRuntime`] provides the most convenient and stable API.
//!
//! ```no_run
//! use burn_ndarray::{NdArray, NdArrayDevice};
//! use guth::{load_config, RuntimeParams, TtsRuntime};
//!
//! // Load configuration and model
//! let config = load_config("config.yaml").unwrap();
//! let device = NdArrayDevice::default();
//! let params = RuntimeParams::new(0.7, 2, None, 0.0);
//! let runtime = TtsRuntime::<NdArray<f32>>::from_config(&config, params, &device).unwrap();
//!
//! // Prepare text and tokenize
//! let (tokens, frames_after_eos) = runtime.prepare_tokens("Hello, world!").unwrap();
//!
//! // Generate audio (batch mode)
//! let max_gen_len = 256;
//! let mut state = runtime.init_state_for_tokens(&tokens, max_gen_len);
//! let (_latents, _eos, audio) = runtime
//!     .generate_audio_from_tokens(tokens, &mut state, max_gen_len, frames_after_eos)
//!     .unwrap();
//! ```
//!
//! ## Streaming Generation
//!
//! For real-time applications, use streaming generation which yields audio chunks as they
//! are produced:
//!
//! ```no_run
//! # use burn_ndarray::{NdArray, NdArrayDevice};
//! # use guth::{load_config, RuntimeParams, TtsRuntime};
//! # let config = load_config("config.yaml").unwrap();
//! # let device = NdArrayDevice::default();
//! # let runtime = TtsRuntime::<NdArray<f32>>::from_config(&config, RuntimeParams::default(), &device).unwrap();
//! // Streaming returns a channel receiver
//! let (tokens, frames_after_eos) = runtime.prepare_tokens("Hello from streaming!").unwrap();
//! let receiver = runtime.generate_audio_stream(tokens, 256, frames_after_eos);
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
//! # use guth::{load_config, RuntimeParams, TtsRuntime};
//! # let config = load_config("config.yaml").unwrap();
//! # let device = NdArrayDevice::default();
//! # let runtime = TtsRuntime::<NdArray<f32>>::from_config(&config, RuntimeParams::default(), &device).unwrap();
//! let (tokens, frames_after_eos) = runtime.prepare_tokens("Voice cloning example").unwrap();
//! let max_gen_len = 256;
//! let mut state = runtime.init_state_for_tokens(&tokens, max_gen_len);
//!
//! // Condition on a reference audio file
//! runtime.condition_on_audio_path("voice.wav", &mut state).unwrap();
//!
//! // Generate with the conditioned state
//! let receiver = runtime.generate_audio_stream_with_state(tokens, state, max_gen_len, frames_after_eos);
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
#[doc(hidden)]
pub mod perf;
pub mod runtime;

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
pub use runtime::{AudioGenerationResult, RuntimeParams, TtsRuntime};
