use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use guth::conditioner::text::TextTokenizer;
use guth::config::load_config;
use guth::model::tts::TtsModel;

const BACKEND_CONFIG: &str = "tests/fixtures/backend_bench_config.yaml";

fn run_single_step<B: Backend>(device: &B::Device) {
    let config = load_config(BACKEND_CONFIG).expect("load config");
    let tts = TtsModel::<B>::from_config(&config, 0.0, 2, None, 0.0, device).expect("build tts");

    let tokenizer = tts
        .flow_lm
        .conditioner
        .tokenizer
        .as_ref()
        .expect("tokenizer");
    let (prepared, _frames_after_eos_guess) =
        TextTokenizer::prepare_text_prompt("hi").expect("prepare");
    let tokens = tokenizer.encode(&prepared).expect("encode");
    let tokens_len = tokens.len();
    let tokens = tokens.into_iter().map(|v| v as i64).collect::<Vec<_>>();
    let tokens = Tensor::<B, 2, Int>::from_data(TensorData::new(tokens, [1, tokens_len]), device);

    let max_gen_len = 1;
    let frames_after_eos = 0;
    let mut state = tts.init_state(1, tokens.dims()[1] + max_gen_len + 1, max_gen_len, device);
    let _ = tts
        .generate_latents_from_tokens(tokens, &mut state, max_gen_len, frames_after_eos)
        .expect("generate latents");
}

#[cfg(feature = "backend-cpu")]
#[test]
fn backend_cpu_generate_does_not_panic() {
    use burn_cpu::{Cpu, CpuDevice};
    let device = CpuDevice::default();
    run_single_step::<Cpu>(&device);
}

#[cfg(feature = "backend-wgpu")]
#[test]
fn backend_wgpu_generate_does_not_panic() {
    use burn_wgpu::graphics::AutoGraphicsApi;
    use burn_wgpu::{init_setup, Wgpu, WgpuDevice};
    let device = WgpuDevice::default();
    init_setup::<AutoGraphicsApi>(&device, Default::default());
    run_single_step::<Wgpu>(&device);
}
