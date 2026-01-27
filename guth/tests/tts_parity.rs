use burn::module::Param;
use burn::tensor::{Int, Tensor, TensorData, Tolerance};
use burn_ndarray::{NdArray, NdArrayDevice};
use serde::Deserialize;

use guth::conditioner::text::LutConditioner;
use guth::model::flow_lm::FlowLmModel;
use guth::model::mimi::MimiModel;
use guth::model::tts::TtsModel;
use guth::modules::dummy_quantizer::DummyQuantizer;
use guth::modules::flow_net::{SimpleMlpAdaLn, SimpleMlpAdaLnConfig};
use guth::modules::mimi_transformer::{
    MimiProjectedTransformer, MimiProjectedTransformerConfig, MimiTransformerConfig, ProjectedOutput,
};
use guth::modules::seanet::{SeanetDecoder, SeanetEncoder, SeanetLayer, SeanetResnetBlock};
use guth::modules::streaming_conv::{PaddingMode, StreamingConv1dOp, StreamingConvConfig, StreamingConvTranspose1dOp};
use guth::modules::transformer::{
    StreamingTransformer, StreamingTransformerConfig, StreamingTransformerLayerConfig,
};

const FIXTURE_DIR: &str = "tests/fixtures";
type TestBackend = NdArray<f32>;

#[derive(Debug, Deserialize)]
struct TtsFixture {
    config: FlowLmConfig,
    flow_net: FlowNetFixture,
    conditioner: ConditionerFixture,
    input_linear_weight: Vec<Vec<f32>>,
    bos_emb: Vec<f32>,
    emb_mean: Vec<f32>,
    emb_std: Vec<f32>,
    speaker_proj_weight: Vec<Vec<f32>>,
    transformer: TransformerFixture,
    out_norm: NormFixture,
    out_eos: LinearFixture,
    generation: GenerationFixture,
    latents: Vec<Vec<Vec<f32>>>,
    eos: Vec<Vec<bool>>,
    audio_full: Vec<Vec<Vec<f32>>>,
    quantizer_weight: Vec<Vec<Vec<f32>>>,
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
    text: String,
    tokens: Vec<Vec<i64>>,
}

#[derive(Debug, Deserialize)]
struct GenerationFixture {
    max_gen_len: usize,
    frames_after_eos: usize,
    temp: f32,
    lsd_decode_steps: usize,
    noise_clamp: Option<f32>,
    eos_threshold: f32,
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

#[derive(Deserialize)]
struct MimiFixture {
    config: MimiConfigFixture,
    encoder_transformer: ProjectedTransformerFixture,
    decoder_transformer: ProjectedTransformerFixture,
}

#[derive(Deserialize)]
struct MimiConfigFixture {
    sample_rate: usize,
    frame_rate: f32,
    encoder_frame_rate: f32,
    channels: usize,
    dimension: usize,
    transformer: MimiTransformerFixtureConfig,
}

#[derive(Deserialize)]
struct MimiTransformerFixtureConfig {
    input_dimension: usize,
    output_dimensions: Vec<usize>,
    d_model: usize,
    num_heads: usize,
    num_layers: usize,
    layer_scale: f32,
    context: usize,
    max_period: f32,
    dim_feedforward: usize,
}

#[derive(Deserialize)]
struct ProjectedTransformerFixture {
    input_proj: ProjectedInputFixture,
    output_projs: Vec<ProjectedOutputFixture>,
    layers: Vec<MimiTransformerLayerFixture>,
}

#[derive(Deserialize)]
struct ProjectedInputFixture {
    kind: String,
    weight: Option<Vec<Vec<f32>>>,
}

#[derive(Deserialize)]
struct ProjectedOutputFixture {
    kind: String,
    weight: Option<Vec<Vec<f32>>>,
}

#[derive(Deserialize)]
struct MimiTransformerLayerFixture {
    self_attn: SelfAttnFixture,
    norm1: NormFixture,
    norm2: NormFixture,
    linear1: LinearFixture,
    linear2: LinearFixture,
    layer_scale_1: LayerScaleFixture,
    layer_scale_2: LayerScaleFixture,
}

#[derive(Deserialize)]
struct LayerScaleFixture {
    scale: Vec<f32>,
}

#[derive(Deserialize)]
struct SeanetFixture {
    layers: Vec<LayerFixture>,
}

#[derive(Deserialize)]
#[serde(tag = "kind")]
enum LayerFixture {
    #[serde(rename = "conv1d")]
    Conv1d {
        config: ConvConfigFixture,
        weight: Vec<Vec<Vec<f32>>>,
        bias: Vec<f32>,
    },
    #[serde(rename = "conv_transpose")]
    ConvTranspose {
        config: ConvConfigFixture,
        weight: Vec<Vec<Vec<f32>>>,
        bias: Vec<f32>,
    },
    #[serde(rename = "elu")]
    Elu,
    #[serde(rename = "resblock")]
    ResBlock { convs: Vec<ConvLayerFixture> },
}

#[derive(Deserialize)]
struct ConvLayerFixture {
    config: ConvConfigFixture,
    weight: Vec<Vec<Vec<f32>>>,
    bias: Vec<f32>,
}

#[derive(Deserialize)]
struct ConvConfigFixture {
    kernel_size: usize,
    stride: usize,
    dilation: usize,
    padding: usize,
    pad_mode: Option<String>,
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

fn apply_linear_raw(linear: &mut burn_nn::Linear<TestBackend>, weight: Vec<Vec<f32>>, device: &NdArrayDevice) {
    let out_dim = weight.len();
    let in_dim = weight[0].len();
    let mut flat = Vec::with_capacity(in_dim * out_dim);
    for c in 0..in_dim {
        for r in 0..out_dim {
            flat.push(weight[r][c]);
        }
    }
    let weight = Tensor::from_data(TensorData::new(flat, [in_dim, out_dim]), device);
    linear.weight = Param::from_tensor(weight);
}

fn apply_norm(norm: &mut burn_nn::LayerNorm<TestBackend>, weights: &NormFixture, device: &NdArrayDevice) {
    let gamma = tensor1(device, weights.gamma.clone());
    let beta = tensor1(device, weights.beta.clone());
    norm.gamma = Param::from_tensor(gamma);
    norm.beta = Some(Param::from_tensor(beta));
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

fn build_flow_net(device: &NdArrayDevice, fixture: FlowNetFixture) -> SimpleMlpAdaLn<TestBackend> {
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

fn build_conv(
    config: &ConvConfigFixture,
    weight: Vec<Vec<Vec<f32>>>,
    bias: Vec<f32>,
    device: &NdArrayDevice,
) -> StreamingConv1dOp<TestBackend> {
    let pad_mode = match config.pad_mode.as_deref() {
        Some("replicate") => PaddingMode::Replicate,
        _ => PaddingMode::Constant,
    };
    let conv_config = StreamingConvConfig {
        kernel_size: config.kernel_size,
        stride: config.stride,
        dilation: config.dilation,
        padding: config.padding,
        pad_mode,
        groups: 1,
    };
    let weight = tensor3(device, weight);
    let bias = tensor1(device, bias);
    StreamingConv1dOp::new(conv_config, weight, Some(bias))
}

fn build_convtr(
    config: &ConvConfigFixture,
    weight: Vec<Vec<Vec<f32>>>,
    bias: Vec<f32>,
    device: &NdArrayDevice,
) -> StreamingConvTranspose1dOp<TestBackend> {
    let conv_config = StreamingConvConfig {
        kernel_size: config.kernel_size,
        stride: config.stride,
        dilation: config.dilation,
        padding: config.padding,
        pad_mode: PaddingMode::Constant,
        groups: 1,
    };
    let weight = tensor3(device, weight);
    let bias = tensor1(device, bias);
    StreamingConvTranspose1dOp::new(conv_config, weight, Some(bias))
}

fn build_seanet_encoder(device: &NdArrayDevice) -> SeanetEncoder<TestBackend> {
    let fixture: SeanetFixture = read_fixture("seanet_encoder.json");
    let mut layers = Vec::new();
    for layer in fixture.layers {
        match layer {
            LayerFixture::Conv1d { config, weight, bias } => {
                layers.push(SeanetLayer::Conv1d(build_conv(&config, weight, bias, device)));
            }
            LayerFixture::Elu => layers.push(SeanetLayer::Elu),
            LayerFixture::ResBlock { convs } => {
                let mut block_layers = Vec::new();
                for conv in convs {
                    block_layers.push(build_conv(&conv.config, conv.weight, conv.bias, device));
                }
                layers.push(SeanetLayer::ResBlock(SeanetResnetBlock::new(block_layers)));
            }
            _ => unreachable!("unexpected layer in encoder"),
        }
    }
    SeanetEncoder::new(layers)
}

fn build_seanet_decoder(device: &NdArrayDevice) -> SeanetDecoder<TestBackend> {
    let fixture: SeanetFixture = read_fixture("seanet_decoder.json");
    let mut layers = Vec::new();
    for layer in fixture.layers {
        match layer {
            LayerFixture::ConvTranspose { config, weight, bias } => layers.push(SeanetLayer::ConvTranspose1d(
                build_convtr(&config, weight, bias, device),
            )),
            LayerFixture::Conv1d { config, weight, bias } => {
                layers.push(SeanetLayer::Conv1d(build_conv(&config, weight, bias, device)));
            }
            LayerFixture::Elu => layers.push(SeanetLayer::Elu),
            LayerFixture::ResBlock { convs } => {
                let mut block_layers = Vec::new();
                for conv in convs {
                    block_layers.push(build_conv(&conv.config, conv.weight, conv.bias, device));
                }
                layers.push(SeanetLayer::ResBlock(SeanetResnetBlock::new(block_layers)));
            }
        }
    }
    SeanetDecoder::new(layers)
}

fn build_projected_transformer(
    fixture: &ProjectedTransformerFixture,
    config: &MimiTransformerFixtureConfig,
    device: &NdArrayDevice,
) -> MimiProjectedTransformer<TestBackend> {
    let transformer_config = MimiTransformerConfig {
        d_model: config.d_model,
        num_heads: config.num_heads,
        num_layers: config.num_layers,
        layer_scale: config.layer_scale,
        context: config.context,
        max_period: config.max_period,
        dim_feedforward: config.dim_feedforward,
    };

    let projected_config = MimiProjectedTransformerConfig {
        input_dim: config.input_dimension,
        output_dims: config.output_dimensions.clone(),
        transformer: transformer_config,
    };
    let mut projected = MimiProjectedTransformer::<TestBackend>::new(projected_config, device);

    if fixture.input_proj.kind == "linear" {
        let weight = fixture
            .input_proj
            .weight
            .clone()
            .expect("input proj weight");
        let input_proj = projected.input_proj.as_mut().expect("input proj");
        apply_linear_raw(input_proj, weight, device);
    }

    for (index, output) in fixture.output_projs.iter().enumerate() {
        match (&output.kind[..], projected.output_projs.get_mut(index)) {
            ("identity", Some(ProjectedOutput::Identity)) => {}
            ("linear", Some(ProjectedOutput::Linear(linear))) => {
                let weight = output.weight.clone().expect("output proj weight");
                apply_linear_raw(linear, weight, device);
            }
            _ => panic!("unexpected output proj"),
        }
    }

    for (layer, fixture_layer) in projected
        .transformer
        .layers
        .iter_mut()
        .zip(fixture.layers.iter())
    {
        apply_linear_raw(&mut layer.self_attn.in_proj, fixture_layer.self_attn.in_proj.weight.clone(), device);
        apply_linear_raw(&mut layer.self_attn.out_proj, fixture_layer.self_attn.out_proj.weight.clone(), device);
        apply_norm(&mut layer.norm1, &fixture_layer.norm1, device);
        apply_norm(&mut layer.norm2, &fixture_layer.norm2, device);
        apply_linear_raw(&mut layer.linear1, fixture_layer.linear1.weight.clone(), device);
        apply_linear_raw(&mut layer.linear2, fixture_layer.linear2.weight.clone(), device);
        if let Some(scale) = layer.layer_scale_1.as_mut() {
            let scale_values = tensor1(device, fixture_layer.layer_scale_1.scale.clone());
            scale.scale = scale_values;
        }
        if let Some(scale) = layer.layer_scale_2.as_mut() {
            let scale_values = tensor1(device, fixture_layer.layer_scale_2.scale.clone());
            scale.scale = scale_values;
        }
    }

    projected
}

#[test]
fn tts_model_matches_fixture() {
    let device = NdArrayDevice::default();
    let fixture: TtsFixture = read_fixture("tts_model.json");
    let mimi_fixture: MimiFixture = read_fixture("mimi_model.json");

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

    let flow_net = build_flow_net(&device, fixture.flow_net);

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

    let flow_lm = FlowLmModel::new(
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

    let encoder = build_seanet_encoder(&device);
    let decoder = build_seanet_decoder(&device);
    let encoder_transformer =
        build_projected_transformer(&mimi_fixture.encoder_transformer, &mimi_fixture.config.transformer, &device);
    let decoder_transformer =
        build_projected_transformer(&mimi_fixture.decoder_transformer, &mimi_fixture.config.transformer, &device);
    let quantizer_weight = tensor3(&device, fixture.quantizer_weight);
    let quantizer = DummyQuantizer::new(quantizer_weight);

    let mimi = MimiModel::new(
        encoder,
        decoder,
        encoder_transformer,
        decoder_transformer,
        quantizer,
        mimi_fixture.config.frame_rate,
        mimi_fixture.config.encoder_frame_rate,
        mimi_fixture.config.sample_rate,
        mimi_fixture.config.channels,
        mimi_fixture.config.dimension,
        None,
        None,
    );

    let speaker_proj_weight = tensor2(&device, fixture.speaker_proj_weight);
    let tts = TtsModel::new(
        flow_lm,
        mimi,
        speaker_proj_weight,
        fixture.generation.temp,
        fixture.generation.lsd_decode_steps,
        fixture.generation.noise_clamp,
        fixture.generation.eos_threshold,
    );

    let tokens = tensor2_int(&device, fixture.conditioner.tokens);
    let flow_len = tokens.dims()[1] + fixture.generation.max_gen_len + 1;
    let mut state = tts.init_state(1, flow_len, fixture.generation.max_gen_len, &device);

    let (latents, eos, audio) = tts.generate_audio_from_tokens(
        tokens,
        &mut state,
        fixture.generation.max_gen_len,
        fixture.generation.frames_after_eos,
    );

    latents.to_data().assert_approx_eq(
        &tensor3(&device, fixture.latents).to_data(),
        Tolerance::<f32>::absolute(1e-4).set_relative(1e-4),
    );
    let eos_data = eos.to_data();
    let eos_slice = eos_data.as_slice::<bool>().expect("eos slice");
    let expected: Vec<bool> = fixture.eos.into_iter().flatten().collect();
    assert_eq!(eos_slice, expected.as_slice());

    audio.to_data().assert_approx_eq(
        &tensor3(&device, fixture.audio_full).to_data(),
        Tolerance::<f32>::absolute(1e-4).set_relative(1e-4),
    );
}
