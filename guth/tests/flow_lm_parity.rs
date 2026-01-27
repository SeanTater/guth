use burn::tensor::{Bool, Int, Tensor, TensorData, Tolerance};
use burn_ndarray::{NdArray, NdArrayDevice};
use serde::Deserialize;

use guth::conditioner::text::LutConditioner;
use guth::model::flow_lm::{FlowLmModel, FlowLmState};
use guth::modules::flow_net::{SimpleMlpAdaLn, SimpleMlpAdaLnConfig};
use guth::modules::transformer::{
    StreamingTransformer, StreamingTransformerConfig, StreamingTransformerLayerConfig,
};
use guth::state::StreamingModule;
use guth::weights::load_flow_lm_state_dict;

const FIXTURE_DIR: &str = "tests/fixtures";
type TestBackend = NdArray<f32>;

#[derive(Debug, Deserialize)]
struct FlowLmFixture {
    config: FlowLmConfig,
    conditioner: ConditionerFixture,
    sequence: Vec<Vec<Vec<f32>>>,
    sequence_nan_mask: Vec<Vec<Vec<bool>>>,
    input_linear: Vec<Vec<Vec<f32>>>,
    transformer_input: Vec<Vec<Vec<f32>>>,
    transformer_raw: Vec<Vec<Vec<f32>>>,
    transformer_normed: Vec<Vec<Vec<f32>>>,
    transformer_trimmed: Vec<Vec<Vec<f32>>>,
    transformer_last: Vec<Vec<f32>>,
    latent: Vec<Vec<f32>>,
    eos: Vec<Vec<bool>>,
}

#[derive(Debug, Deserialize)]
struct FlowLmConfig {
    ldim: usize,
    dim: usize,
    num_heads: usize,
    num_layers: usize,
    ffn_dim: usize,
    max_period: f32,
}

#[derive(Debug, Deserialize)]
struct ConditionerFixture {
    n_bins: usize,
    tokens: Vec<Vec<i64>>,
    text_embeddings: Vec<Vec<Vec<f32>>>,
}

#[derive(Debug, Deserialize)]
struct FlowNetConfigFixture {
    config: FlowNetConfig,
}

#[derive(Debug, Deserialize)]
struct FlowNetConfig {
    in_channels: usize,
    model_channels: usize,
    out_channels: usize,
    cond_channels: usize,
    num_res_blocks: usize,
    num_time_conds: usize,
    frequency_embedding_size: usize,
    max_period: f32,
}

fn read_fixture<T: for<'de> Deserialize<'de>>(name: &str) -> T {
    let path = format!("{FIXTURE_DIR}/{name}");
    let data = std::fs::read_to_string(path).expect("fixture read");
    serde_json::from_str(&data).expect("fixture parse")
}

fn tensor2(device: &NdArrayDevice, data: Vec<Vec<f32>>) -> Tensor<TestBackend, 2> {
    let rows = data.len();
    let cols = data.first().map(|row| row.len()).unwrap_or(0);
    let flat: Vec<f32> = data.into_iter().flatten().collect();
    let td = TensorData::new(flat, [rows, cols]);
    Tensor::from_data(td, device)
}

fn tensor2_int(device: &NdArrayDevice, data: Vec<Vec<i64>>) -> Tensor<TestBackend, 2, Int> {
    let rows = data.len();
    let cols = data.first().map(|row| row.len()).unwrap_or(0);
    let flat: Vec<i64> = data.into_iter().flatten().collect();
    let td = TensorData::new(flat, [rows, cols]);
    Tensor::from_data(td, device)
}

fn tensor3(device: &NdArrayDevice, data: Vec<Vec<Vec<f32>>>) -> Tensor<TestBackend, 3> {
    let b = data.len();
    let c = data[0].len();
    let t = data[0][0].len();
    let mut flat = Vec::with_capacity(b * c * t);
    for bb in data {
        for cc in bb {
            for val in cc {
                flat.push(val);
            }
        }
    }
    let td = TensorData::new(flat, [b, c, t]);
    Tensor::from_data(td, device)
}

fn tensor3_bool(device: &NdArrayDevice, data: Vec<Vec<Vec<bool>>>) -> Tensor<TestBackend, 3, Bool> {
    let b = data.len();
    let c = data[0].len();
    let t = data[0][0].len();
    let mut flat = Vec::with_capacity(b * c * t);
    for bb in data {
        for cc in bb {
            for val in cc {
                flat.push(val);
            }
        }
    }
    let td = TensorData::new(flat, [b, c, t]);
    Tensor::from_data(td, device)
}

fn linear_3d(
    linear: &burn_nn::Linear<TestBackend>,
    input: Tensor<TestBackend, 3>,
) -> Tensor<TestBackend, 3> {
    let [batch, seq, dim] = input.dims();
    let flat = input.reshape([batch * seq, dim]);
    let output = linear.forward(flat);
    let out_dim = output.dims()[1];
    output.reshape([batch, seq, out_dim])
}

fn norm_3d(
    norm: &burn_nn::LayerNorm<TestBackend>,
    input: Tensor<TestBackend, 3>,
) -> Tensor<TestBackend, 3> {
    let [batch, seq, dim] = input.dims();
    let flat = input.reshape([batch * seq, dim]);
    let output = norm.forward(flat);
    output.reshape([batch, seq, dim])
}

fn build_flow_net(device: &NdArrayDevice) -> SimpleMlpAdaLn<TestBackend> {
    let fixture: FlowNetConfigFixture = read_fixture("flow_net.json");

    let mut config = SimpleMlpAdaLnConfig::new(
        fixture.config.in_channels,
        fixture.config.model_channels,
        fixture.config.out_channels,
        fixture.config.cond_channels,
    );
    config.num_res_blocks = fixture.config.num_res_blocks;
    config.num_time_conds = fixture.config.num_time_conds;
    config.time_embed.frequency_embedding_size = fixture.config.frequency_embedding_size;
    config.time_embed.max_period = fixture.config.max_period;

    SimpleMlpAdaLn::<TestBackend>::new(config, device)
}

#[test]
fn flow_lm_matches_fixture() {
    let device = NdArrayDevice::default();
    let fixture: FlowLmFixture = read_fixture("flow_lm_model.json");

    let embed =
        burn_nn::EmbeddingConfig::new(fixture.conditioner.n_bins, fixture.config.dim)
            .init::<TestBackend>(&device);
    let conditioner = LutConditioner {
        tokenizer: None,
        embed,
        dim: fixture.config.dim,
        output_dim: fixture.config.dim,
    };

    let transformer_config = StreamingTransformerConfig {
        num_layers: fixture.config.num_layers,
        layer: StreamingTransformerLayerConfig {
            d_model: fixture.config.dim,
            num_heads: fixture.config.num_heads,
            ffn_dim: fixture.config.ffn_dim,
            causal: true,
            context: None,
            rope_max_seq: Some(64),
            rope_theta: fixture.config.max_period,
            layer_scale: None,
        },
    };
    let transformer = StreamingTransformer::<TestBackend>::new(transformer_config, &device);
    let flow_net = build_flow_net(&device);
    let input_linear = burn_nn::LinearConfig::new(fixture.config.ldim, fixture.config.dim)
        .with_bias(false)
        .init::<TestBackend>(&device);
    let out_norm = burn_nn::LayerNormConfig::new(fixture.config.dim)
        .init::<TestBackend>(&device);
    let out_eos = burn_nn::LinearConfig::new(fixture.config.dim, 1)
        .init::<TestBackend>(&device);
    let emb_mean = Tensor::<TestBackend, 1>::zeros([fixture.config.ldim], &device);
    let emb_std = Tensor::<TestBackend, 1>::zeros([fixture.config.ldim], &device);
    let bos_emb = Tensor::<TestBackend, 1>::zeros([fixture.config.ldim], &device);

    let mut model = FlowLmModel::new(
        conditioner,
        flow_net,
        transformer,
        input_linear,
        out_norm,
        out_eos,
        emb_std,
        emb_mean,
        bos_emb,
        fixture.config.dim,
        fixture.config.ldim,
    );

    let state = load_flow_lm_state_dict("tests/fixtures/flow_lm_state.safetensors")
        .expect("load flow lm state dict");
    model
        .load_state_dict(&state, &device)
        .expect("apply flow lm state dict");

    let tokens = tensor2_int(&device, fixture.conditioner.tokens);
    let text_embeddings = model.conditioner.forward_tokens(tokens);
    text_embeddings.to_data().assert_approx_eq(
        &tensor3(&device, fixture.conditioner.text_embeddings).to_data(),
        Tolerance::<f32>::absolute(1e-4).set_relative(1e-4),
    );

    let sequence_clean = tensor3(&device, fixture.sequence);
    let nan_mask = tensor3_bool(&device, fixture.sequence_nan_mask);
    let nan_fill = Tensor::<TestBackend, 3>::full(sequence_clean.dims(), f32::NAN, &device);
    let sequence = sequence_clean.mask_where(nan_mask.clone(), nan_fill);
    let mut state = FlowLmState {
        transformer: model
            .transformer
            .init_state(1, text_embeddings.dims()[1] + sequence.dims()[1]),
    };

    let bos = model
        .bos_emb
        .clone()
        .reshape([1, 1, fixture.config.ldim])
        .repeat_dim(0, 1)
        .repeat_dim(1, sequence.dims()[1]);
    let sequence_with_bos = sequence.clone().mask_where(nan_mask, bos);
    let input_linear_out = linear_3d(&model.input_linear, sequence_with_bos);
    input_linear_out.to_data().assert_approx_eq(
        &tensor3(&device, fixture.input_linear).to_data(),
        Tolerance::<f32>::absolute(1e-4).set_relative(1e-4),
    );

    let transformer_input = Tensor::cat(vec![text_embeddings.clone(), input_linear_out], 1);
    transformer_input.to_data().assert_approx_eq(
        &tensor3(&device, fixture.transformer_input).to_data(),
        Tolerance::<f32>::absolute(1e-4).set_relative(1e-4),
    );

    let mut transformer_state = model
        .transformer
        .init_state(1, transformer_input.dims()[1]);
    let transformer_raw = model.transformer.forward(transformer_input, &mut transformer_state);
    transformer_raw.to_data().assert_approx_eq(
        &tensor3(&device, fixture.transformer_raw).to_data(),
        Tolerance::<f32>::absolute(1e-4).set_relative(1e-4),
    );

    let transformer_normed = norm_3d(&model.out_norm, transformer_raw);
    transformer_normed.to_data().assert_approx_eq(
        &tensor3(&device, fixture.transformer_normed).to_data(),
        Tolerance::<f32>::absolute(1e-4).set_relative(1e-4),
    );

    let total_len = transformer_normed.dims()[1];
    let transformer_trimmed =
        transformer_normed.narrow(1, total_len - sequence.dims()[1], sequence.dims()[1]);
    transformer_trimmed.to_data().assert_approx_eq(
        &tensor3(&device, fixture.transformer_trimmed).to_data(),
        Tolerance::<f32>::absolute(1e-4).set_relative(1e-4),
    );

    let transformer_last = transformer_trimmed
        .narrow(1, sequence.dims()[1] - 1, 1)
        .reshape([1, fixture.config.dim]);
    transformer_last.to_data().assert_approx_eq(
        &tensor2(&device, fixture.transformer_last).to_data(),
        Tolerance::<f32>::absolute(1e-4).set_relative(1e-4),
    );

    let (latent, eos) = model.forward(sequence, text_embeddings, &mut state, 2, 0.0, None, 0.0);

    latent.to_data().assert_approx_eq(
        &tensor2(&device, fixture.latent).to_data(),
        Tolerance::<f32>::absolute(1e-4).set_relative(1e-4),
    );

    let eos_data = eos.to_data();
    let eos_slice = eos_data.as_slice::<bool>().expect("eos slice");
    let expected: Vec<bool> = fixture.eos.into_iter().flatten().collect();
    assert_eq!(eos_slice, expected.as_slice());
}
