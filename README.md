# guth

Fast CPU-optimized text-to-speech library using flow matching.

## Overview

`guth` is a Rust implementation of a flow-matching TTS pipeline, designed for efficient CPU inference. It powers the `pocket-tts` project.

### Architecture

The TTS pipeline consists of three components:

1. **Text Conditioning**: Tokenizes text using SentencePiece and embeds it into the model's latent space
2. **FlowLM**: Flow-matching transformer that autoregressively generates latent audio frames
3. **Mimi Codec**: Neural audio codec (SEANet-based) that decodes latents to PCM waveforms

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
guth = "0.1"
burn-ndarray = "0.20"  # or another Burn backend
```

## Quick Start

Use `TtsRuntime` for a convenience wrapper around config + model loading:

```rust
use burn_ndarray::{NdArray, NdArrayDevice};
use guth::{load_config, RuntimeParams, TtsRuntime};

// Load model from config
let config = load_config("config.yaml")?;
let device = NdArrayDevice::default();
let params = RuntimeParams::new(0.7, 2, None, 0.0);
let runtime = TtsRuntime::<NdArray<f32>>::from_config(&config, params, &device)?;

// Prepare and tokenize text
let (tokens, frames_after_eos) = runtime.prepare_tokens("Hello, world!")?;

// Generate audio
let max_gen_len = 256;
let mut state = runtime.init_state_for_tokens(&tokens, max_gen_len);
let (_latents, _eos, audio) =
    runtime.generate_audio_from_tokens(tokens, &mut state, max_gen_len, frames_after_eos)?;
// audio shape: [batch, channels, samples]
```

## Streaming Generation

For real-time applications:

```rust
// Assume `runtime` is created as in Quick Start
let (tokens, frames_after_eos) = runtime.prepare_tokens("Hello from streaming!")?;
let receiver = runtime.generate_audio_stream(tokens, 256, frames_after_eos);
for audio_chunk in receiver {
    // Process each chunk as it arrives
}
```

## Voice Cloning

Condition on reference audio for voice cloning:

```rust
// Assume `runtime` is created as in Quick Start
let (tokens, frames_after_eos) = runtime.prepare_tokens("Voice cloning example")?;
let mut state = runtime.init_state_for_tokens(&tokens, 256);
runtime.condition_on_audio_path("voice.wav", &mut state)?;
let receiver = runtime.generate_audio_stream_with_state(tokens, state, 256, frames_after_eos);
```

## Configuration

Models are configured via YAML files. Weight paths support:

- Local files: `/path/to/model.safetensors`
- HuggingFace Hub: `hf://owner/repo/model.safetensors`
- HuggingFace with revision: `hf://owner/repo/model.safetensors@v1.0`

## License

See the repository root for license information.
