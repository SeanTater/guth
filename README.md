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

```rust
use burn_ndarray::{NdArray, NdArrayDevice};
use burn::tensor::{Int, Tensor, TensorData};
use guth::{load_config, TtsModel, TextTokenizer};

// Load model from config
let config = load_config("config.yaml")?;
let device = NdArrayDevice::default();
let tts = TtsModel::<NdArray<f32>>::from_config(
    &config,
    0.7,   // temperature
    2,     // LSD decode steps
    None,  // noise clamp
    0.0,   // EOS threshold
    &device,
)?;

// Prepare and tokenize text
let tokenizer = tts.flow_lm.conditioner.tokenizer.as_ref().unwrap();
let (prepared, frames_after_eos) = TextTokenizer::prepare_text_prompt("Hello, world!")?;
let tokens = tokenizer.encode(&prepared)?;
let tokens: Vec<i64> = tokens.into_iter().map(|t| t as i64).collect();
let tokens = Tensor::<NdArray<f32>, 2, Int>::from_data(
    TensorData::new(tokens.clone(), [1, tokens.len()]),
    &device,
);

// Generate audio
let max_gen_len = 256;
let mut state = tts.init_state(1, tokens.dims()[1] + max_gen_len + 1, max_gen_len, &device);
let (_latents, _eos, audio) = tts.generate_audio_from_tokens(
    tokens, &mut state, max_gen_len, frames_after_eos,
)?;
// audio shape: [batch, channels, samples]
```

## Streaming Generation

For real-time applications:

```rust
let receiver = tts.generate_audio_stream(tokens, 256, 8);
for audio_chunk in receiver {
    // Process each chunk as it arrives
}
```

## Voice Cloning

Condition on reference audio for voice cloning:

```rust
let mut state = tts.init_state(1, 512, 256, &device);
tts.condition_on_audio(reference_audio, &mut state)?;
let receiver = tts.generate_audio_stream_with_state(tokens, state, 256, 8);
```

## Configuration

Models are configured via YAML files. Weight paths support:

- Local files: `/path/to/model.safetensors`
- HuggingFace Hub: `hf://owner/repo/model.safetensors`
- HuggingFace with revision: `hf://owner/repo/model.safetensors@v1.0`

## License

See the repository root for license information.
