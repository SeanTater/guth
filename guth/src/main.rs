use anyhow::Result;
use clap::{Parser, Subcommand};
use guth::audio::io::WavIo;
use guth::audio::resample::AudioResampler;
use guth::config::load_config;
use guth::model::tts::TtsModel;
use guth::conditioner::text::TextTokenizer;
use burn::tensor::{Int, Tensor, TensorData};
use burn_ndarray::{NdArray, NdArrayDevice};
use std::path::PathBuf;

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
    },
    Voices,
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
        } => {
            if voice.is_some() {
                anyhow::bail!("Predefined voices are not wired yet; use --voice-file for now.");
            }
            let config_path = config;
            let cfg = load_config(&config_path)?;
            let device = NdArrayDevice::default();
            let tts = TtsModel::<NdArray<f32>>::from_config(
                &cfg,
                temp.unwrap_or(0.0),
                lsd_decode_steps.unwrap_or(2),
                noise_clamp,
                eos_threshold.unwrap_or(0.0),
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

            let mut state = tts.init_state(
                1,
                tokens.dims()[1] + 256,
                256,
                &device,
            );

            if let Some(voice_path) = voice_file {
                let (samples, sample_rate) = WavIo::read_wav(&voice_path)?;
                let prompt = AudioResampler::convert_audio(
                    samples,
                    sample_rate,
                    cfg.mimi.sample_rate as u32,
                    cfg.mimi.channels as usize,
                )?;
                let prompt_tensor = tensor_from_audio(prompt, &device);
                tts.condition_on_audio(prompt_tensor, &mut state);
            }

            let (_latents, _eos, audio) = tts.generate_audio_from_tokens(
                tokens,
                &mut state,
                256,
                frames_after_eos_guess,
            );

            let output_path = output.ok_or_else(|| anyhow::anyhow!("--output is required"))?;
            let audio_vec = audio_to_vec(audio);
            WavIo::write_wav(output_path, &audio_vec, cfg.mimi.sample_rate as u32)?;
        }
        Commands::Voices => {
            println!("List voices (todo)");
        }
        Commands::Models => {
            println!("List models (todo)");
        }
        Commands::Download { model } => {
            println!("Download model: {model}");
        }
        Commands::Audio { command } => match command {
            AudioCommands::Convert {
                input,
                output,
                to_rate,
                to_channels,
            } => {
                let (samples, sample_rate) = WavIo::read_wav(input)?;
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
