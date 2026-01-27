use burn::module::Param;
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

const FIXTURE_DIR: &str = "tests/fixtures";
type TestBackend = NdArray<f32>;

#[derive(Debug, Deserialize)]
struct FlowLmFixture {
    config: FlowLmConfig,
    conditioner: ConditionerFixture,
    input_linear_weight: Vec<Vec<f32>>,
    bos_emb: Vec<f32>,
    emb_mean: Vec<f32>,
    emb_std: Vec<f32>,
    transformer: TransformerFixture,
    out_norm: NormFixture,
    out_eos: LinearFixture,
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
    embed_weight: Vec<Vec<f32>>,
    tokens: Vec<Vec<i64>>,
    text_embeddings: Vec<Vec<Vec<f32>>>,
}

#[derive(Debug, Deserialize)]
struct TransformerFixture {
    layers: Vec<TransformerLayerFixture>,
}

#[derive(Debug, Deserialize)]
struct TransformerLayerFixture {
    self_attn: SelfAttnFixture,
    norm1: NormFixture,
    norm2: NormFixture,
    linear1: LinearFixture,
    linear2: LinearFixture,
}

#[derive(Debug, Deserialize)]
struct SelfAttnFixture {
    in_proj: LinearFixture,
    out_proj: LinearFixture,
}

#[derive(Debug, Deserialize)]
struct LinearFixture {
    weight: Vec<Vec<f32>>,
    bias: Option<Vec<f32>>,
}

#[derive(Debug, Deserialize)]
struct NormFixture {
    gamma: Vec<f32>,
    beta: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct FlowNetFixture {
    config: FlowNetConfig,
    weights: FlowNetWeights,
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

#[derive(Debug, Deserialize)]
struct FlowNetWeights {
    input_proj: LinearWeights,
    cond_embed: LinearWeights,
    time_embed: Vec<TimeEmbedWeights>,
    res_blocks: Vec<ResBlockWeights>,
    final_layer: FinalLayerWeights,
}

#[derive(Debug, Deserialize)]
struct LinearWeights {
    weight: Vec<Vec<f32>>,
    bias: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct NormWeights {
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
}

#[derive(Debug, Deserialize)]
struct TimeEmbedWeights {
    proj_in: LinearWeights,
    proj_out: LinearWeights,
    rms_weight: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct ResBlockWeights {
    norm: NormWeights,
    mlp_in: LinearWeights,
    mlp_out: LinearWeights,
    modulation: LinearWeights,
}

#[derive(Debug, Deserialize)]
struct FinalLayerWeights {
    norm: NormWeights,
    linear: LinearWeights,
    modulation: LinearWeights,
}

fn read_fixture<T: for<'de> Deserialize<'de>>(name: &str) -> T {
    let path = format!("{FIXTURE_DIR}/{name}");
    let data = std::fs::read_to_string(path).expect("fixture read");
    serde_json::from_str(&data).expect("fixture parse")
}

fn tensor1(device: &NdArrayDevice, data: Vec<f32>) -> Tensor<TestBackend, 1> {
    let len = data.len();
    let td = TensorData::new(data, [len]);
    Tensor::from_data(td, device)
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

fn apply_linear(linear: &mut burn_nn::Linear<TestBackend>, weights: &LinearFixture, device: &NdArrayDevice) {
    let out_dim = weights.weight.len();
    let in_dim = weights.weight[0].len();
    let mut flat = Vec::with_capacity(in_dim * out_dim);
    for c in 0..in_dim {
        for r in 0..out_dim {
            flat.push(weights.weight[r][c]);
        }
    }
    let weight = Tensor::from_data(TensorData::new(flat, [in_dim, out_dim]), device);
    linear.weight = Param::from_tensor(weight);
    if let Some(bias) = &weights.bias {
        let bias = Tensor::from_data(TensorData::new(bias.clone(), [bias.len()]), device);
        linear.bias = Some(Param::from_tensor(bias));
    }
}

fn apply_norm(norm: &mut burn_nn::LayerNorm<TestBackend>, weights: &NormFixture, device: &NdArrayDevice) {
    let gamma = tensor1(device, weights.gamma.clone());
    let beta = tensor1(device, weights.beta.clone());
    norm.gamma = Param::from_tensor(gamma);
    norm.beta = Some(Param::from_tensor(beta));
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

fn apply_flow_linear(
    linear: &mut burn_nn::Linear<TestBackend>,
    weights: &LinearWeights,
    device: &NdArrayDevice,
) {
    let rows = weights.weight.len();
    let cols = weights.weight[0].len();
    let flat: Vec<f32> = weights.weight.clone().into_iter().flatten().collect();
    let weight = Tensor::from_data(TensorData::new(flat, [rows, cols]), device);
    linear.weight = Param::from_tensor(weight);
    let bias = Tensor::from_data(TensorData::new(weights.bias.clone(), [weights.bias.len()]), device);
    linear.bias = Some(Param::from_tensor(bias));
}

fn apply_flow_norm(
    norm: &mut burn_nn::LayerNorm<TestBackend>,
    weights: &NormWeights,
    device: &NdArrayDevice,
) {
    let gamma = tensor1(device, weights.weight.clone());
    norm.gamma = Param::from_tensor(gamma);
    norm.beta = weights.bias.as_ref().map(|bias| {
        let beta = tensor1(device, bias.clone());
        Param::from_tensor(beta)
    });
}

fn apply_rms(norm: &mut guth::modules::flow_net::RmsNorm<TestBackend>, weights: &[f32], device: &NdArrayDevice) {
    let gamma = tensor1(device, weights.to_vec());
    norm.gamma = Param::from_tensor(gamma);
}

fn build_flow_net(device: &NdArrayDevice) -> SimpleMlpAdaLn<TestBackend> {
    let fixture: FlowNetFixture = read_fixture("flow_net.json");

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

    let mut model = SimpleMlpAdaLn::<TestBackend>::new(config, device);
    apply_flow_linear(&mut model.input_proj, &fixture.weights.input_proj, device);
    apply_flow_linear(&mut model.cond_embed, &fixture.weights.cond_embed, device);

    for (embedder, weights) in model.time_embed.iter_mut().zip(fixture.weights.time_embed.iter()) {
        apply_flow_linear(&mut embedder.proj_in, &weights.proj_in, device);
        apply_flow_linear(&mut embedder.proj_out, &weights.proj_out, device);
        apply_rms(&mut embedder.norm, &weights.rms_weight, device);
    }

    for (block, weights) in model.res_blocks.iter_mut().zip(fixture.weights.res_blocks.iter()) {
        apply_flow_norm(&mut block.norm, &weights.norm, device);
        apply_flow_linear(&mut block.mlp_in, &weights.mlp_in, device);
        apply_flow_linear(&mut block.mlp_out, &weights.mlp_out, device);
        apply_flow_linear(&mut block.mod_linear, &weights.modulation, device);
    }

    apply_flow_norm(&mut model.final_layer.norm, &fixture.weights.final_layer.norm, device);
    apply_flow_linear(&mut model.final_layer.linear, &fixture.weights.final_layer.linear, device);
    apply_flow_linear(
        &mut model.final_layer.mod_linear,
        &fixture.weights.final_layer.modulation,
        device,
    );

    model
}

#[test]
fn flow_lm_matches_fixture() {
    let device = NdArrayDevice::default();
    let fixture: FlowLmFixture = read_fixture("flow_lm_model.json");

    let embed_weight = tensor2(&device, fixture.conditioner.embed_weight);
    let embed_rows = embed_weight.dims()[0];
    let embed_cols = embed_weight.dims()[1];
    let embed = burn_nn::EmbeddingConfig::new(embed_rows, embed_cols).init::<TestBackend>(&device);
    let mut conditioner = LutConditioner {
        tokenizer: None,
        embed,
        dim: fixture.config.dim,
        output_dim: fixture.config.dim,
    };
    conditioner.embed.weight = Param::from_tensor(embed_weight);

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
    let mut transformer = StreamingTransformer::<TestBackend>::new(transformer_config, &device);

    for (layer, weights) in transformer.layers.iter_mut().zip(fixture.transformer.layers.iter()) {
        apply_linear(&mut layer.qkv, &weights.self_attn.in_proj, &device);
        apply_linear(&mut layer.out_proj, &weights.self_attn.out_proj, &device);
        apply_norm(&mut layer.norm1, &weights.norm1, &device);
        apply_norm(&mut layer.norm2, &weights.norm2, &device);
        apply_linear(&mut layer.ffn_in, &weights.linear1, &device);
        apply_linear(&mut layer.ffn_out, &weights.linear2, &device);
    }

    let flow_net = build_flow_net(&device);

    let mut input_linear = burn_nn::LinearConfig::new(fixture.config.ldim, fixture.config.dim)
        .with_bias(false)
        .init::<TestBackend>(&device);
    apply_linear(
        &mut input_linear,
        &LinearFixture {
            weight: fixture.input_linear_weight,
            bias: None,
        },
        &device,
    );

    let mut out_norm = burn_nn::LayerNormConfig::new(fixture.config.dim)
        .init::<TestBackend>(&device);
    apply_norm(&mut out_norm, &fixture.out_norm, &device);

    let mut out_eos = burn_nn::LinearConfig::new(fixture.config.dim, 1)
        .init::<TestBackend>(&device);
    apply_linear(&mut out_eos, &fixture.out_eos, &device);

    let emb_mean = tensor1(&device, fixture.emb_mean);
    let emb_std = tensor1(&device, fixture.emb_std);
    let bos_emb = tensor1(&device, fixture.bos_emb);

    let tokens = tensor2_int(&device, fixture.conditioner.tokens);
    let text_embeddings = conditioner.forward_tokens(tokens);
    text_embeddings.to_data().assert_approx_eq(
        &tensor3(&device, fixture.conditioner.text_embeddings).to_data(),
        Tolerance::<f32>::absolute(1e-4).set_relative(1e-4),
    );

    let model = FlowLmModel::new(
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
