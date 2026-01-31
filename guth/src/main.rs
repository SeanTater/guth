use anyhow::Result;
use clap::{Parser, Subcommand};
use guth::audio::io::WavIo;
use guth::audio::io::StreamingWavWriter;
use guth::audio::resample::AudioResampler;
use guth::config::load_config;
use guth::model::tts::TtsModel;
use guth::conditioner::text::TextTokenizer;
use burn::tensor::{Int, Tensor, TensorData};
use burn_ndarray::{NdArray, NdArrayDevice};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::collections::HashMap;
use safetensors::Dtype;

#[derive(Parser)]
#[command(name = "guth")]
#[command(about = "Rust rewrite of pocket-tts", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Say {
        text: String,
        #[arg(long)]
        voice: Option<String>,
        #[arg(long)]
        voice_file: Option<PathBuf>,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long, default_value = "pocket_tts/config/b6369a24.yaml")]
        config: PathBuf,
        #[arg(long)]
        temp: Option<f32>,
        #[arg(long)]
        lsd_decode_steps: Option<usize>,
        #[arg(long)]
        noise_clamp: Option<f32>,
        #[arg(long)]
        eos_threshold: Option<f32>,
        #[arg(long)]
        max_gen_len: Option<usize>,
        #[arg(long)]
        frames_after_eos: Option<usize>,
        #[arg(long)]
        stream: bool,
        #[arg(long)]
        progress: bool,
    },
    Voices,
    Voice {
        #[command(subcommand)]
        command: VoiceCommands,
    },
    Models,
    Download {
        model: String,
    },
    Audio {
        #[command(subcommand)]
        command: AudioCommands,
    },
}

#[derive(Subcommand)]
enum AudioCommands {
    Convert {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        to_rate: u32,
        #[arg(long)]
        to_channels: usize,
    },
}

#[derive(Subcommand)]
enum VoiceCommands {
    Encode {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value = "pocket_tts/config/b6369a24.yaml")]
        config: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

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
            let interrupted = Arc::new(AtomicBool::new(false));
            let interrupt_flag = Arc::clone(&interrupted);
            ctrlc::set_handler(move || {
                interrupt_flag.store(true, Ordering::SeqCst);
            })?;
            let config_path = resolve_config_path(config);
            let cfg = load_config(&config_path)?;
            let device = NdArrayDevice::default();
            let tts = TtsModel::<NdArray<f32>>::from_config(
                &cfg,
                temp.unwrap_or(0.0),
                lsd_decode_steps.unwrap_or(2),
                noise_clamp,
                eos_threshold.unwrap_or(-4.0),
                &device,
            )?;

            let tokenizer = tts
                .flow_lm
                .conditioner
                .tokenizer
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Tokenizer not loaded"))?;
            let (prepared, frames_after_eos_guess) = TextTokenizer::prepare_text_prompt(&text)?;
            let tokens = tokenizer.encode(&prepared)?;
            let tokens = tokens.into_iter().map(|v| v as i64).collect::<Vec<_>>();
            let tokens_len = tokens.len();
            let tokens = Tensor::<NdArray<f32>, 2, Int>::from_data(
                TensorData::new(tokens, [1, tokens_len]),
                &device,
            );

            let max_gen_len = max_gen_len.unwrap_or(256);
            let frames_after_eos = frames_after_eos.unwrap_or(frames_after_eos_guess);
            let mut state = tts.init_state(1, tokens.dims()[1] + max_gen_len + 1, max_gen_len, &device);

            // Apply voice conditioning if specified
            if let Some(voice_name) = voice {
                // Predefined voice - load pre-computed conditioning tensor
                let voice_path = resolve_voice_path(&voice_name).ok_or_else(|| {
                    anyhow::anyhow!(
                        "Voice '{voice_name}' not found. Run `guth voices` to see available voices."
                    )
                })?;
                let conditioning = load_conditioning_tensor(&voice_path, &device)?;
                tts.condition_on_precomputed(conditioning, &mut state)?;
            } else if let Some(voice_path) = voice_file {
                // Custom voice file - compute conditioning from audio
                let (samples, sample_rate) = WavIo::read_audio(&voice_path)?;
                let prompt = AudioResampler::convert_audio(
                    samples,
                    sample_rate,
                    cfg.mimi.sample_rate as u32,
                    cfg.mimi.channels as usize,
                )?;
                let prompt_tensor = tensor_from_audio(prompt, &device);
                tts.condition_on_audio(prompt_tensor, &mut state)?;
            }

            let output_path = output.ok_or_else(|| anyhow::anyhow!("--output is required"))?;
            if stream {
                let receiver = tts.generate_audio_stream(tokens, max_gen_len, frames_after_eos);
                let mut writer = StreamingWavWriter::create(
                    output_path,
                    cfg.mimi.sample_rate as u32,
                    cfg.mimi.channels as usize,
                )?;
                let mut chunk_idx = 0usize;
                for chunk in receiver {
                    if interrupted.load(Ordering::SeqCst) {
                        writer.finalize()?;
                        anyhow::bail!("Interrupted");
                    }
                    let audio_vec = audio_to_vec(chunk);
                    writer.write_chunk(&audio_vec)?;
                    if progress {
                        eprintln!("wrote chunk {chunk_idx}");
                    }
                    chunk_idx += 1;
                }
                writer.finalize()?;
            } else {
                let (_latents, _eos, audio) = tts.generate_audio_from_tokens(
                    tokens,
                    &mut state,
                    max_gen_len,
                    frames_after_eos,
                )?;
                if interrupted.load(Ordering::SeqCst) {
                    anyhow::bail!("Interrupted");
                }
                let audio_vec = audio_to_vec(audio);
                WavIo::write_wav(output_path, &audio_vec, cfg.mimi.sample_rate as u32)?;
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
            VoiceCommands::Encode { input, output, config } => {
                let cfg = load_config(resolve_config_path(config))?;
                let device = NdArrayDevice::default();
                let tts = TtsModel::<NdArray<f32>>::from_config(
                    &cfg,
                    0.0,
                    2,
                    None,
                    0.0,
                    &device,
                )?;
                let (samples, sample_rate) = WavIo::read_audio(input)?;
                let prompt = AudioResampler::convert_audio(
                    samples,
                    sample_rate,
                    cfg.mimi.sample_rate as u32,
                    cfg.mimi.channels as usize,
                )?;
                let prompt_tensor = tensor_from_audio(prompt, &device);
                let conditioning = compute_conditioning(&tts, prompt_tensor);
                save_tensor(&output, "conditioning", conditioning)?;
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
                let converted = AudioResampler::convert_audio(
                    samples,
                    sample_rate,
                    to_rate,
                    to_channels,
                )?;
                WavIo::write_wav(output, &converted, to_rate)?;
            }
        },
    }

    Ok(())
}

fn tensor_from_audio(
    samples: Vec<Vec<f32>>,
    device: &NdArrayDevice,
) -> Tensor<NdArray<f32>, 3> {
    let channels = samples.len();
    let len = samples[0].len();
    let mut flat = Vec::with_capacity(channels * len);
    for channel in samples {
        flat.extend(channel);
    }
    Tensor::from_data(TensorData::new(flat, [1, channels, len]), device)
}

fn audio_to_vec(audio: Tensor<NdArray<f32>, 3>) -> Vec<Vec<f32>> {
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

fn compute_conditioning(
    tts: &TtsModel<NdArray<f32>>,
    audio_prompt: Tensor<NdArray<f32>, 3>,
) -> Tensor<NdArray<f32>, 3> {
    let latents = tts.mimi.encode_to_latent(audio_prompt);
    let latents = latents.swap_dims(1, 2);
    burn::tensor::module::linear(latents, tts.speaker_proj_weight.clone(), None)
}

fn save_tensor(path: &PathBuf, name: &str, tensor: Tensor<NdArray<f32>, 3>) -> Result<()> {
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

fn load_conditioning_tensor(
    path: &PathBuf,
    device: &NdArrayDevice,
) -> Result<Tensor<NdArray<f32>, 3>> {
    let data = std::fs::read(path)?;
    let safetensors = safetensors::SafeTensors::deserialize(&data)?;
    let tensor_view = safetensors
        .tensor("audio_prompt")
        .map_err(|e| anyhow::anyhow!("Failed to load conditioning tensor: {e}"))?;
    let shape = tensor_view.shape();
    if shape.len() != 3 {
        anyhow::bail!("Expected 3D tensor, got {}D", shape.len());
    }
    let mut values = Vec::with_capacity(tensor_view.data().len() / 4);
    for chunk in tensor_view.data().chunks_exact(4) {
        values.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(Tensor::from_data(
        TensorData::new(values, [shape[0], shape[1], shape[2]]),
        device,
    ))
}

fn resolve_voice_path(voice_name: &str) -> Option<PathBuf> {
    // Try multiple locations for voice files
    let candidates = [
        format!("pocket_tts/voices/{voice_name}.safetensors"),
        format!("voices/{voice_name}.safetensors"),
    ];
    for candidate in candidates {
        let path = PathBuf::from(&candidate);
        if path.exists() {
            return Some(path);
        }
    }
    None
}

fn available_models() -> Vec<&'static str> {
    vec!["b6369a24"]
}

fn available_voices() -> Vec<&'static str> {
    vec![
        "alba",
        "marius",
        "javert",
        "jean",
        "fantine",
        "cosette",
        "eponine",
        "azelma",
    ]
}

fn download_model(model: &str) -> Result<()> {
    let config_path = match model {
        "b6369a24" => PathBuf::from("pocket_tts/config/b6369a24.yaml"),
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
