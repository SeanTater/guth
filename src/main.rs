//! Command-line interface for the Rust TTS implementation.
//!
//! The CLI wraps the core model to provide speech synthesis, voice encoding,
//! model downloads, and basic audio conversion utilities.

#![recursion_limit = "256"]

use anyhow::Result;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use burn_ndarray::{NdArray, NdArrayDevice};
use clap::{Parser, Subcommand};
use clap::ValueEnum;
use guth::audio::io::StreamingWavWriter;
use guth::audio::io::WavIo;
use guth::audio::resample::AudioResampler;
use guth::config::load_config;
use guth::perf;
use guth::runtime::{RuntimeParams, TtsRuntime};
use safetensors::Dtype;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[cfg(feature = "backend-wgpu")]
use burn_wgpu::graphics::AutoGraphicsApi;
#[cfg(feature = "backend-wgpu")]
use burn_wgpu::{init_setup, Wgpu, WgpuDevice};

const VOICE_CLONING_UNSUPPORTED: &str = "Voice cloning weights are unavailable. \
Use --voice with a built-in voice, or download the full weights for voice cloning.";

/// Supported compute backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "kebab-case")]
enum BackendChoice {
    /// Use the WGPU backend (GPU acceleration when available).
    Wgpu,
    /// Use the ndarray backend (CPU).
    Ndarray,
}

#[cfg(feature = "backend-wgpu")]
const DEFAULT_BACKEND: BackendChoice = BackendChoice::Wgpu;
#[cfg(not(feature = "backend-wgpu"))]
const DEFAULT_BACKEND: BackendChoice = BackendChoice::Ndarray;

/// Top-level CLI options.
#[derive(Parser)]
#[command(name = "guth")]
#[command(about = "Rust rewrite of pocket-tts", long_about = None)]
struct Cli {
    /// Print performance summary at the end of the run.
    #[arg(long, short, global = true)]
    verbose: bool,
    /// Compute backend to use.
    #[arg(long, value_enum, default_value_t = DEFAULT_BACKEND, global = true)]
    backend: BackendChoice,
    /// Subcommand to execute.
    #[command(subcommand)]
    command: Commands,
}

/// CLI subcommands.
#[derive(Subcommand)]
enum Commands {
    /// Generate speech from text.
    Say {
        /// Text prompt to synthesize.
        text: String,
        /// Name of a precomputed voice embedding.
        #[arg(long)]
        voice: Option<String>,
        /// Path to a custom voice audio file or voice prompt safetensors.
        #[arg(long)]
        voice_file: Option<PathBuf>,
        /// Output WAV file path.
        #[arg(long)]
        output: Option<PathBuf>,
        /// Model configuration YAML.
        #[arg(long, default_value = "python/pocket_tts/config/b6369a24.yaml")]
        config: PathBuf,
        /// Sampling temperature.
        #[arg(long)]
        temp: Option<f32>,
        /// Langevin decode steps.
        #[arg(long)]
        lsd_decode_steps: Option<usize>,
        /// Optional noise clamp.
        #[arg(long)]
        noise_clamp: Option<f32>,
        /// End-of-sequence threshold.
        #[arg(long)]
        eos_threshold: Option<f32>,
        /// Maximum latent frames to generate.
        #[arg(long)]
        max_gen_len: Option<usize>,
        /// Frames to continue after EOS detection.
        #[arg(long)]
        frames_after_eos: Option<usize>,
        /// Stream audio to disk as it is generated.
        #[arg(long)]
        stream: bool,
        /// Print per-chunk progress for streaming.
        #[arg(long)]
        progress: bool,
    },
    /// List available voices.
    Voices,
    /// Voice-related subcommands.
    Voice {
        /// Voice subcommand to execute.
        #[command(subcommand)]
        command: VoiceCommands,
    },
    /// List available models.
    Models,
    /// Download model artifacts.
    Download {
        /// Model name to download.
        model: String,
    },
    /// Audio utility subcommands.
    Audio {
        /// Audio subcommand to execute.
        #[command(subcommand)]
        command: AudioCommands,
    },
}

/// Audio utility commands.
#[derive(Subcommand)]
enum AudioCommands {
    /// Convert sample rate and channel count.
    Convert {
        /// Input audio file path.
        #[arg(long)]
        input: PathBuf,
        /// Output WAV path.
        #[arg(long)]
        output: PathBuf,
        /// Target sample rate in Hz.
        #[arg(long)]
        to_rate: u32,
        /// Target channel count.
        #[arg(long)]
        to_channels: usize,
    },
}

/// Voice-related commands.
#[derive(Subcommand)]
enum VoiceCommands {
    /// Encode a voice prompt to a conditioning tensor.
    Encode {
        /// Input audio file.
        #[arg(long)]
        input: PathBuf,
        /// Output safetensors path.
        #[arg(long)]
        output: PathBuf,
        /// Model configuration YAML.
        #[arg(long, default_value = "python/pocket_tts/config/b6369a24.yaml")]
        config: PathBuf,
        /// Truncate long audio prompts before encoding.
        #[arg(long)]
        truncate: bool,
        /// Maximum duration in seconds when truncating.
        #[arg(long, default_value_t = 30.0)]
        truncate_seconds: f32,
    },
}

#[derive(Debug, Clone)]
struct SayArgs {
    text: String,
    voice: Option<String>,
    voice_file: Option<PathBuf>,
    output: Option<PathBuf>,
    config: PathBuf,
    temp: Option<f32>,
    lsd_decode_steps: Option<usize>,
    noise_clamp: Option<f32>,
    eos_threshold: Option<f32>,
    max_gen_len: Option<usize>,
    frames_after_eos: Option<usize>,
    stream: bool,
    progress: bool,
}

#[derive(Debug, Clone)]
struct VoiceEncodeArgs {
    input: PathBuf,
    output: PathBuf,
    config: PathBuf,
    truncate: bool,
    truncate_seconds: f32,
}

/// Entry point for the CLI.
fn main() -> Result<()> {
    let cli = Cli::parse();
    let verbose = cli.verbose;
    let backend = cli.backend;

    match cli.command {
        Commands::Say {
            text,
            voice,
            voice_file,
            output,
            config,
            temp,
            lsd_decode_steps,
            noise_clamp,
            eos_threshold,
            max_gen_len,
            frames_after_eos,
            stream,
            progress,
        } => {
            let args = SayArgs {
                text,
                voice,
                voice_file,
                output,
                config,
                temp,
                lsd_decode_steps,
                noise_clamp,
                eos_threshold,
                max_gen_len,
                frames_after_eos,
                stream,
                progress,
            };
            let interrupted = Arc::new(AtomicBool::new(false));
            let interrupt_flag = Arc::clone(&interrupted);
            ctrlc::set_handler(move || {
                interrupt_flag.store(true, Ordering::SeqCst);
            })?;
            match backend {
                BackendChoice::Wgpu => {
                    #[cfg(feature = "backend-wgpu")]
                    {
                        let device = WgpuDevice::default();
                        init_setup::<AutoGraphicsApi>(&device, Default::default());
                        run_say::<Wgpu>(args, &device, interrupted)?;
                    }
                    #[cfg(not(feature = "backend-wgpu"))]
                    {
                        let _ = args;
                        anyhow::bail!("WGPU backend not enabled; build with --features backend-wgpu");
                    }
                }
                BackendChoice::Ndarray => {
                    let device = NdArrayDevice::default();
                    run_say::<NdArray<f32>>(args, &device, interrupted)?;
                }
            }
        }
        Commands::Voices => {
            let all_voices = available_voices();
            for voice in &all_voices {
                let available = resolve_voice_path(voice).is_some();
                if available {
                    println!("{voice}");
                } else {
                    println!("{voice} (not installed)");
                }
            }
        }
        Commands::Voice { command } => match command {
            VoiceCommands::Encode {
                input,
                output,
                config,
                truncate,
                truncate_seconds,
            } => {
                let args = VoiceEncodeArgs {
                    input,
                    output,
                    config,
                    truncate,
                    truncate_seconds,
                };
                match backend {
                    BackendChoice::Wgpu => {
                        #[cfg(feature = "backend-wgpu")]
                        {
                            let device = WgpuDevice::default();
                            init_setup::<AutoGraphicsApi>(&device, Default::default());
                            run_voice_encode::<Wgpu>(args, &device)?;
                        }
                        #[cfg(not(feature = "backend-wgpu"))]
                        {
                            let _ = args;
                            anyhow::bail!(
                                "WGPU backend not enabled; build with --features backend-wgpu"
                            );
                        }
                    }
                    BackendChoice::Ndarray => {
                        let device = NdArrayDevice::default();
                        run_voice_encode::<NdArray<f32>>(args, &device)?;
                    }
                }
            }
        },
        Commands::Models => {
            for model in available_models() {
                println!("{model}");
            }
        }
        Commands::Download { model } => {
            download_model(&model)?;
        }
        Commands::Audio { command } => match command {
            AudioCommands::Convert {
                input,
                output,
                to_rate,
                to_channels,
            } => {
                let (samples, sample_rate) = WavIo::read_audio(input)?;
                let converted =
                    AudioResampler::convert_audio(samples, sample_rate, to_rate, to_channels)?;
                WavIo::write_wav(output, &converted, to_rate)?;
            }
        },
    }

    if verbose {
        eprintln!("{}", perf::report());
    }

    Ok(())
}

fn run_say<B: Backend + Send + Sync + 'static>(
    args: SayArgs,
    device: &B::Device,
    interrupted: Arc<AtomicBool>,
) -> Result<()> {
    let config_path = resolve_config_path(args.config);
    let params = RuntimeParams::new(
        args.temp.unwrap_or(0.0),
        args.lsd_decode_steps.unwrap_or(2),
        args.noise_clamp,
        args.eos_threshold.unwrap_or(-4.0),
    );
    let runtime = TtsRuntime::<B>::from_config_path(&config_path, params, device)?;
    let flow_dim = runtime.model().flow_lm.dim;
    if interrupted.load(Ordering::SeqCst) {
        anyhow::bail!("Interrupted");
    }
    let (tokens, frames_after_eos_guess) = runtime.prepare_tokens(&args.text)?;

    let max_gen_len = args.max_gen_len.unwrap_or(256);
    let frames_after_eos = args.frames_after_eos.unwrap_or(frames_after_eos_guess);
    let mut state = runtime.init_state_for_tokens(&tokens, max_gen_len);

    if let Some(voice_name) = args.voice {
        let voice_path = resolve_voice_path(&voice_name).ok_or_else(|| {
            anyhow::anyhow!(
                "Voice '{voice_name}' not found. Run `guth voices` to see available voices."
            )
        })?;
        let conditioning = load_conditioning_tensor::<B>(&voice_path, device, flow_dim)?;
        runtime.condition_on_precomputed(conditioning, &mut state)?;
    } else if let Some(voice_path) = args.voice_file {
        if !runtime.voice_cloning_supported() {
            anyhow::bail!(VOICE_CLONING_UNSUPPORTED);
        }
        if is_safetensors_path(&voice_path) {
            let conditioning = load_conditioning_tensor::<B>(&voice_path, device, flow_dim)?;
            runtime.condition_on_precomputed(conditioning, &mut state)?;
        } else {
            runtime.condition_on_audio_path(&voice_path, &mut state)?;
        }
    }

    let output_path = args.output.ok_or_else(|| anyhow::anyhow!("--output is required"))?;
    let sample_rate = runtime.config().mimi.sample_rate as u32;
    let channels = runtime.config().mimi.channels as usize;
    if args.stream {
        let receiver = runtime.generate_audio_stream_with_state(
            tokens,
            state,
            max_gen_len,
            frames_after_eos,
        );
        let mut writer = StreamingWavWriter::create(output_path, sample_rate, channels)?;
        for (chunk_idx, chunk) in receiver.into_iter().enumerate() {
            if interrupted.load(Ordering::SeqCst) {
                writer.finalize()?;
                anyhow::bail!("Interrupted");
            }
            let audio_vec = audio_to_vec(chunk);
            writer.write_chunk(&audio_vec)?;
            if args.progress {
                eprintln!("wrote chunk {chunk_idx}");
            }
        }
        writer.finalize()?;
    } else {
        let (_latents, _eos, audio) =
            runtime.generate_audio_from_tokens(tokens, &mut state, max_gen_len, frames_after_eos)?;
        if interrupted.load(Ordering::SeqCst) {
            anyhow::bail!("Interrupted");
        }
        let audio_vec = audio_to_vec(audio);
        WavIo::write_wav(output_path, &audio_vec, sample_rate)?;
    }

    Ok(())
}

fn run_voice_encode<B: Backend>(args: VoiceEncodeArgs, device: &B::Device) -> Result<()> {
    let params = RuntimeParams::new(0.0, 2, None, 0.0);
    let runtime = TtsRuntime::<B>::from_config_path(
        resolve_config_path(args.config),
        params,
        device,
    )?;
    if !runtime.voice_cloning_supported() {
        anyhow::bail!(VOICE_CLONING_UNSUPPORTED);
    }
    if args.truncate && args.truncate_seconds <= 0.0 {
        anyhow::bail!("--truncate-seconds must be > 0");
    }
    let (conditioning, truncated) = if args.truncate {
        runtime.conditioning_from_audio_path_with_truncate(args.input, args.truncate_seconds)?
    } else {
        (runtime.conditioning_from_audio_path(args.input)?, false)
    };
    if truncated {
        eprintln!("Truncated voice prompt to {:.1}s", args.truncate_seconds);
    }
    save_tensor(&args.output, "audio_prompt", conditioning)
}

/// Convert a `[1, channels, samples]` tensor into per-channel vectors.
fn audio_to_vec<B: Backend>(audio: Tensor<B, 3>) -> Vec<Vec<f32>> {
    let data = audio.to_data();
    let values = data.as_slice::<f32>().expect("audio data");
    let shape = data.shape.clone();
    let batch = shape[0];
    let channels = shape[1];
    let len = shape[2];
    assert_eq!(batch, 1);
    let mut output = vec![vec![0.0_f32; len]; channels];
    for c in 0..channels {
        for t in 0..len {
            output[c][t] = values[c * len + t];
        }
    }
    output
}

/// Save a 3D tensor to a SafeTensors file.
fn save_tensor<B: Backend>(path: &Path, name: &str, tensor: Tensor<B, 3>) -> Result<()> {
    let data = tensor.to_data();
    let values = data.as_slice::<f32>().expect("tensor data");
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    let view = safetensors::tensor::TensorView::new(Dtype::F32, data.shape.clone(), &bytes)?;
    let mut tensors = HashMap::new();
    tensors.insert(name.to_string(), view);
    let serialized = safetensors::serialize(&tensors, &None)?;
    std::fs::write(path, serialized)?;
    Ok(())
}

fn is_safetensors_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("safetensors"))
}

/// Load a conditioning tensor from a SafeTensors file.
fn load_conditioning_tensor<B: Backend>(
    path: &Path,
    device: &B::Device,
    expected_dim: usize,
) -> Result<Tensor<B, 3>> {
    let data = std::fs::read(path)?;
    let safetensors = safetensors::SafeTensors::deserialize(&data)?;
    let tensor_view = match safetensors.tensor("audio_prompt") {
        Ok(tensor) => tensor,
        Err(_) => safetensors
            .tensor("conditioning")
            .map_err(|e| anyhow::anyhow!("Failed to load conditioning tensor: {e}"))?,
    };
    let shape = tensor_view.shape();
    if shape.len() != 3 {
        anyhow::bail!("Expected 3D tensor, got {}D", shape.len());
    }
    let mut values = Vec::new();
    match tensor_view.dtype() {
        safetensors::Dtype::F32 => {
            values.reserve(tensor_view.data().len() / 4);
            for chunk in tensor_view.data().chunks_exact(4) {
                values.push(f32::from_le_bytes(chunk.try_into().unwrap()));
            }
        }
        safetensors::Dtype::BF16 => {
            values.reserve(tensor_view.data().len() / 2);
            for chunk in tensor_view.data().chunks_exact(2) {
                let bits = u16::from_le_bytes(chunk.try_into().unwrap()) as u32;
                values.push(f32::from_bits(bits << 16));
            }
        }
        other => anyhow::bail!("Unsupported conditioning dtype {:?}", other),
    }

    // Conditioning tensors are expected to be [batch, seq, dim]. Some legacy
    // exports used [batch, dim, seq], so detect and swap to avoid silent misuse.
    let batch = shape[0];
    let dim0 = shape[1];
    let dim1 = shape[2];
    let conditioning = Tensor::from_data(TensorData::new(values, [batch, dim0, dim1]), device);
    if dim1 == expected_dim {
        Ok(conditioning)
    } else if dim0 == expected_dim {
        eprintln!(
            "Warning: conditioning tensor is [batch, dim, seq]; swapping to [batch, seq, dim]."
        );
        Ok(conditioning.swap_dims(1, 2))
    } else {
        anyhow::bail!(
            "Conditioning dim mismatch: expected last dim {expected_dim}, got shape {shape:?}"
        );
    }
}

/// Resolve a built-in voice name to an on-disk file path.
fn resolve_voice_path(voice_name: &str) -> Option<PathBuf> {
    // Resolve precomputed voices from the repository voices directory.
    let candidate = format!("voices/{voice_name}.safetensors");
    let path = PathBuf::from(&candidate);
    if path.exists() {
        return Some(path);
    }
    None
}

/// Return the list of known model names.
fn available_models() -> Vec<&'static str> {
    vec!["b6369a24"]
}

/// Return the list of built-in voice names.
fn available_voices() -> Vec<&'static str> {
    vec![
        "alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma",
    ]
}

/// Download model artifacts referenced by the config.
fn download_model(model: &str) -> Result<()> {
    let config_path = match model {
        "b6369a24" => PathBuf::from("python/pocket_tts/config/b6369a24.yaml"),
        _ => anyhow::bail!("Unknown model {model}"),
    };
    let cfg = load_config(resolve_config_path(config_path))?;

    if let Some(path) = cfg.weights_path.as_ref() {
        let _ = guth::download::download_if_necessary(path)?;
    }
    if let Some(path) = cfg.weights_path_without_voice_cloning.as_ref() {
        let _ = guth::download::download_if_necessary(path)?;
    }
    if let Some(path) = cfg.flow_lm.weights_path.as_ref() {
        let _ = guth::download::download_if_necessary(path)?;
    }
    if let Some(path) = cfg.mimi.weights_path.as_ref() {
        let _ = guth::download::download_if_necessary(path)?;
    }
    let _ = guth::download::download_if_necessary(&cfg.flow_lm.lookup_table.tokenizer_path)?;
    println!("Downloaded model artifacts for {model}");
    Ok(())
}

/// Resolve a config path, falling back to parent directory if needed.
fn resolve_config_path(path: PathBuf) -> PathBuf {
    if path.exists() {
        return path;
    }
    let candidate = PathBuf::from("..").join(&path);
    if candidate.exists() {
        return candidate;
    }
    path
}
