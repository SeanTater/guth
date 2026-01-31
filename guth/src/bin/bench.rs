use anyhow::Result;
use burn::tensor::{Int, Tensor, TensorData};
use burn::tensor::backend::Backend;
use guth::conditioner::text::TextTokenizer;
use guth::config::load_config;
use guth::model::tts::TtsModel;
use std::time::Instant;

#[derive(Debug, Clone, Copy)]
struct BenchArgs {
    iters: usize,
    max_gen_len: usize,
    stream: bool,
}

fn parse_args() -> (String, BenchArgs) {
    let mut config_path = "tests/fixtures/tts_integration_config.yaml".to_string();
    let mut iters = 3usize;
    let mut max_gen_len = 32usize;
    let mut stream = false;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--config" => {
                if let Some(value) = args.next() {
                    config_path = value;
                }
            }
            "--iters" => {
                if let Some(value) = args.next() {
                    iters = value.parse().unwrap_or(iters);
                }
            }
            "--max-gen-len" => {
                if let Some(value) = args.next() {
                    max_gen_len = value.parse().unwrap_or(max_gen_len);
                }
            }
            "--stream" => {
                stream = true;
            }
            _ => {}
        }
    }

    (config_path, BenchArgs { iters, max_gen_len, stream })
}

fn run_bench<B: Backend>(
    name: &str,
    device: &B::Device,
    config_path: &str,
    args: BenchArgs,
) -> Result<()> {
    let config = load_config(config_path)?;
    let tts = TtsModel::<B>::from_config(&config, 0.0, 2, None, 0.0, device)?;

    let tokenizer = tts
        .flow_lm
        .conditioner
        .tokenizer
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Tokenizer not loaded"))?;
    let (prepared, frames_after_eos_guess) =
        TextTokenizer::prepare_text_prompt("Benchmarking the Rust TTS backend.")?;
    let tokens = tokenizer.encode(&prepared)?;
    let tokens_len = tokens.len();
    let tokens = tokens.into_iter().map(|v| v as i64).collect::<Vec<_>>();
    let tokens = Tensor::<B, 2, Int>::from_data(
        TensorData::new(tokens, [1, tokens_len]),
        device,
    );

    let frames_after_eos = frames_after_eos_guess;
    let mut durations = Vec::with_capacity(args.iters);

    if args.stream {
        let start = Instant::now();
        let receiver = tts.generate_audio_stream(
            tokens.clone(),
            args.max_gen_len,
            frames_after_eos,
        );
        for _chunk in receiver {}
        durations.push(start.elapsed());
    } else {
        for _ in 0..args.iters {
            let mut state = tts.init_state(
                1,
                tokens.dims()[1] + args.max_gen_len + 1,
                args.max_gen_len,
                device,
            );
            let start = Instant::now();
            let _ = tts.generate_audio_from_tokens(
                tokens.clone(),
                &mut state,
                args.max_gen_len,
                frames_after_eos,
            )?;
            durations.push(start.elapsed());
        }
    }

    let total_ms: f64 = durations.iter().map(|d| d.as_secs_f64() * 1000.0).sum();
    let avg_ms = total_ms / durations.len() as f64;
    println!(
        "{name}: avg {:.2} ms over {} iters (max_gen_len={}, stream={})",
        avg_ms,
        args.iters,
        args.max_gen_len,
        args.stream
    );
    Ok(())
}

fn main() -> Result<()> {
    let (config_path, args) = parse_args();

    #[cfg(feature = "backend-ndarray")]
    {
        use burn_ndarray::{NdArray, NdArrayDevice};
        let device = NdArrayDevice::default();
        run_bench::<NdArray<f32>>("ndarray", &device, &config_path, args)?;
    }

    #[cfg(feature = "backend-cpu")]
    {
        use burn_cpu::{Cpu, CpuDevice};
        let device = CpuDevice::default();
        run_bench::<Cpu>("cpu", &device, &config_path, args)?;
    }

    #[cfg(feature = "backend-wgpu")]
    {
        use burn_wgpu::graphics::AutoGraphicsApi;
        use burn_wgpu::{init_setup, Wgpu, WgpuDevice};
        let device = WgpuDevice::default();
        init_setup::<AutoGraphicsApi>(&device, Default::default());
        run_bench::<Wgpu>("wgpu", &device, &config_path, args)?;
    }
    Ok(())
}
