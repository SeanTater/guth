//! Export intermediate encoding stages for parity testing with Python.
//!
//! This binary processes an audio file through the Mimi encoder and prints
//! statistics at each stage, matching the Python export_encoding_stages.py script.
//!
//! Usage: cargo run --release --bin export_encoding_stages -- [audio_path]

use anyhow::Result;
use burn::tensor::{Tensor, TensorData};
use burn_ndarray::{NdArray, NdArrayDevice};
use guth::audio::io::WavIo;
use guth::audio::resample::AudioResampler;
use guth::config::load_config;
use guth::model::tts::TtsModel;
use serde::Serialize;

type B = NdArray<f32>;

#[derive(Serialize)]
struct TensorStats {
    shape: Vec<usize>,
    rms: f32,
    mean: f32,
    std: f32,
    min: f32,
    max: f32,
}

impl TensorStats {
    fn from_tensor<const D: usize>(tensor: &Tensor<B, D>) -> Self {
        let data = tensor.to_data();
        let values = data.as_slice::<f32>().expect("f32 slice");
        let n = values.len() as f32;

        let sum: f32 = values.iter().sum();
        let mean = sum / n;

        let sum_sq: f32 = values.iter().map(|x| x * x).sum();
        let rms = (sum_sq / n).sqrt();

        let variance: f32 = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
        let std = variance.sqrt();

        let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        Self {
            shape: tensor.dims().to_vec(),
            rms,
            mean,
            std,
            min,
            max,
        }
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let audio_path = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("tests/fixtures/voices/sean.ogg");

    let config_path = args
        .get(2)
        .map(|s| s.as_str())
        .unwrap_or("python/pocket_tts/config/b6369a24.yaml");

    println!("Loading config from: {config_path}");
    let config = load_config(config_path)?;

    let device = NdArrayDevice::default();
    println!("Building model...");

    // Build TTS model to get Mimi with weights loaded
    let tts = TtsModel::<B>::from_config(&config, 0.0, 2, None, 0.0, &device)?;
    let mimi = &tts.mimi;

    println!("Reading audio from: {audio_path}");
    let (samples, sample_rate) = WavIo::read_audio(audio_path)?;
    println!(
        "  Original: {} channels, {} samples, {} Hz",
        samples.len(),
        samples[0].len(),
        sample_rate
    );

    // Resample to target sample rate
    let target_sr = config.mimi.sample_rate as u32;
    let samples = AudioResampler::convert_audio(samples, sample_rate, target_sr, 1)?;
    println!(
        "  After resample: {} channels, {} samples, {} Hz",
        samples.len(),
        samples[0].len(),
        target_sr
    );

    // Truncate to 5 seconds
    let max_samples = (5.0 * target_sr as f64) as usize;
    let samples: Vec<Vec<f32>> = samples
        .into_iter()
        .map(|mut ch| {
            ch.truncate(max_samples);
            ch
        })
        .collect();
    println!(
        "  After truncate: {} samples ({:.2}s)",
        samples[0].len(),
        samples[0].len() as f64 / target_sr as f64
    );

    // Convert to tensor [batch=1, channels=1, length]
    let audio_data: Vec<f32> = samples.into_iter().flatten().collect();
    let length = audio_data.len();
    let audio_tensor: Tensor<B, 3> =
        Tensor::from_data(TensorData::new(audio_data, [1, 1, length]), &device);

    println!("\n=== Input Audio ===");
    let input_stats = TensorStats::from_tensor(&audio_tensor);
    println!(
        "  Shape: {:?}, RMS: {:.6}",
        input_stats.shape, input_stats.rms
    );

    // Step 1: Compute padding
    let frame_size = mimi.frame_size();
    let extra_padding = get_extra_padding_for_conv1d(length, frame_size, frame_size, 0);
    println!("\n=== Padding ===");
    println!("  Frame size: {frame_size}");
    println!("  Extra padding: {extra_padding}");

    let padded = if extra_padding == 0 {
        audio_tensor.clone()
    } else {
        let padding: Tensor<B, 3> = Tensor::zeros([1, 1, extra_padding], &device);
        Tensor::cat(vec![audio_tensor.clone(), padding], 2)
    };
    let padded_stats = TensorStats::from_tensor(&padded);
    println!(
        "  Padded shape: {:?}, RMS: {:.6}",
        padded_stats.shape, padded_stats.rms
    );

    // Step 2: SEANet encoder
    println!("\n=== SEANet Encoder ===");
    let mut encoder_state = mimi.encoder.init_state(1);
    let seanet_out = mimi
        .encoder
        .forward(padded.clone(), &mut encoder_state)
        .expect("seanet encoder");
    let seanet_stats = TensorStats::from_tensor(&seanet_out);
    println!(
        "  Shape: {:?}, RMS: {:.6}, mean: {:.6}, std: {:.6}",
        seanet_stats.shape, seanet_stats.rms, seanet_stats.mean, seanet_stats.std
    );
    println!(
        "  Range: [{:.6}, {:.6}]",
        seanet_stats.min, seanet_stats.max
    );

    // Step 3: Encoder transformer
    println!("\n=== Encoder Transformer ===");
    let mut transformer_state = mimi.encoder_transformer.init_state(1, &device);
    let mut outputs = mimi
        .encoder_transformer
        .forward(seanet_out.clone(), &mut transformer_state);
    let transformer_out = outputs.remove(0);
    let transformer_stats = TensorStats::from_tensor(&transformer_out);
    println!(
        "  Shape: {:?}, RMS: {:.6}, mean: {:.6}, std: {:.6}",
        transformer_stats.shape,
        transformer_stats.rms,
        transformer_stats.mean,
        transformer_stats.std
    );
    println!(
        "  Range: [{:.6}, {:.6}]",
        transformer_stats.min, transformer_stats.max
    );

    // Step 4: Downsample to frame rate
    println!("\n=== Downsampling ===");
    let final_latent = if let Some(downsample) = &mimi.downsample {
        let mut downsample_state = downsample.init_state(1, &device);
        downsample.forward(transformer_out.clone(), &mut downsample_state)
    } else {
        transformer_out.clone()
    };
    let final_stats = TensorStats::from_tensor(&final_latent);
    println!(
        "  Shape: {:?}, RMS: {:.6}, mean: {:.6}, std: {:.6}",
        final_stats.shape, final_stats.rms, final_stats.mean, final_stats.std
    );

    // Step 5: Project to conditioning
    println!("\n=== Conditioning Projection ===");
    let latents = final_latent.swap_dims(1, 2); // [B, T, C]
    let conditioning = burn::tensor::module::linear(latents, tts.speaker_proj_weight.clone(), None);
    let cond_stats = TensorStats::from_tensor(&conditioning);
    println!(
        "  Shape: {:?}, RMS: {:.6}, mean: {:.6}, std: {:.6}",
        cond_stats.shape, cond_stats.rms, cond_stats.mean, cond_stats.std
    );
    println!("  Range: [{:.6}, {:.6}]", cond_stats.min, cond_stats.max);

    // Print comparison with Python reference
    println!("\n=== Comparison with Python Reference ===");
    println!("Stage               | Python RMS  | Rust RMS    | Diff");
    println!("--------------------|-------------|-------------|--------");
    println!(
        "Input audio         | {:.6}    | {:.6}    | {:.6}",
        0.044243,
        input_stats.rms,
        (0.044243 - input_stats.rms).abs()
    );
    println!(
        "Padded audio        | {:.6}    | {:.6}    | {:.6}",
        0.044067,
        padded_stats.rms,
        (0.044067 - padded_stats.rms).abs()
    );
    println!(
        "SEANet output       | {:.6}    | {:.6}    | {:.6}",
        0.012102,
        seanet_stats.rms,
        (0.012102 - seanet_stats.rms).abs()
    );
    println!(
        "Transformer output  | {:.6}    | {:.6}    | {:.6}",
        0.020091,
        transformer_stats.rms,
        (0.020091 - transformer_stats.rms).abs()
    );
    println!(
        "Final latent        | {:.6}    | {:.6}    | {:.6}",
        0.575767,
        final_stats.rms,
        (0.575767 - final_stats.rms).abs()
    );
    println!(
        "Conditioning        | {:.6}    | {:.6}    | {:.6}",
        0.093773,
        cond_stats.rms,
        (0.093773 - cond_stats.rms).abs()
    );

    Ok(())
}

/// Compute extra padding for conv1d to ensure the last window is complete.
fn get_extra_padding_for_conv1d(
    length: usize,
    kernel_size: usize,
    stride: usize,
    padding_total: usize,
) -> usize {
    let n_frames =
        (length as f64 - kernel_size as f64 + padding_total as f64) / stride as f64 + 1.0;
    let ceil_frames = n_frames.ceil() as usize;
    let ideal_length = if ceil_frames == 0 {
        0
    } else {
        (ceil_frames - 1) * stride + kernel_size - padding_total
    };
    ideal_length.saturating_sub(length)
}
