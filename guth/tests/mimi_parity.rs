use burn::module::Param;
use burn::tensor::{Tensor, TensorData};
use burn_ndarray::{NdArray, NdArrayDevice};
use serde::Deserialize;

use guth::model::mimi::MimiModel;
use guth::modules::dummy_quantizer::DummyQuantizer;
use guth::modules::mimi_transformer::{
    MimiProjectedTransformer, MimiProjectedTransformerConfig, MimiTransformerConfig, ProjectedOutput,
};
use guth::modules::seanet::{SeanetDecoder, SeanetEncoder, SeanetLayer};
use guth::modules::streaming_conv::{PaddingMode, StreamingConv1dOp, StreamingConvConfig, StreamingConvTranspose1dOp};

const FIXTURE_DIR: &str = "tests/fixtures";
type TestBackend = NdArray<f32>;

#[derive(Deserialize)]
struct DummyQuantizerFixture {
    dimension: usize,
    output_dimension: usize,
    weight: Vec<Vec<Vec<f32>>>,
    input: Vec<Vec<Vec<f32>>>,
    output: Vec<Vec<Vec<f32>>>,
}

#[derive(Deserialize)]
struct MimiFixture {
    config: MimiConfigFixture,
    encoder_transformer: ProjectedTransformerFixture,
    decoder_transformer: ProjectedTransformerFixture,
    audio_input: Vec<Vec<Vec<f32>>>,
    latent_output: Vec<Vec<Vec<f32>>>,
    latent_input: Vec<Vec<Vec<f32>>>,
    audio_output: Vec<Vec<Vec<f32>>>,
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
    layers: Vec<TransformerLayerFixture>,
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
struct TransformerLayerFixture {
    self_attn: SelfAttnFixture,
    norm1: NormFixture,
    norm2: NormFixture,
    linear1: LinearFixture,
    linear2: LinearFixture,
    layer_scale_1: LayerScaleFixture,
    layer_scale_2: LayerScaleFixture,
}

#[derive(Deserialize)]
struct SelfAttnFixture {
    in_proj: LinearFixture,
    out_proj: LinearFixture,
}

#[derive(Deserialize)]
struct LinearFixture {
    weight: Vec<Vec<f32>>,
}

#[derive(Deserialize)]
struct NormFixture {
    gamma: Vec<f32>,
    beta: Vec<f32>,
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

fn tensor1_from_vec(data: Vec<f32>, device: &NdArrayDevice) -> Tensor<TestBackend, 1> {
    let len = data.len();
    let td = TensorData::new(data, vec![len]);
    Tensor::from_data(td, device)
}

fn tensor2_from_nested(data: Vec<Vec<f32>>, device: &NdArrayDevice) -> Tensor<TestBackend, 2> {
    let rows = data.len();
    let cols = data[0].len();
    let flat: Vec<f32> = data.into_iter().flatten().collect();
    let td = TensorData::new(flat, vec![rows, cols]);
    Tensor::from_data(td, device)
}

fn tensor3_from_nested(data: Vec<Vec<Vec<f32>>>, device: &NdArrayDevice) -> Tensor<TestBackend, 3> {
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
    let td = TensorData::new(flat, vec![b, c, t]);
    Tensor::from_data(td, device)
}

fn assert_close(a: &TensorData, b: &TensorData, tol: f32) {
    let a_slice = a.as_slice::<f32>().expect("a slice");
    let b_slice = b.as_slice::<f32>().expect("b slice");
    assert_eq!(a_slice.len(), b_slice.len());
    for (idx, (x, y)) in a_slice.iter().zip(b_slice.iter()).enumerate() {
        if (x - y).abs() > tol {
            panic!("mismatch at {idx}: {x} vs {y}");
        }
    }
}

fn build_conv(config: &ConvConfigFixture, weight: Vec<Vec<Vec<f32>>>, bias: Vec<f32>, device: &NdArrayDevice) -> StreamingConv1dOp<TestBackend> {
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
    let weight = tensor3_from_nested(weight, device);
    let bias = tensor1_from_vec(bias, device);
    StreamingConv1dOp::new(conv_config, weight, Some(bias))
}

fn build_conv_transpose(
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
    let weight = tensor3_from_nested(weight, device);
    let bias = tensor1_from_vec(bias, device);
    StreamingConvTranspose1dOp::new(conv_config, weight, Some(bias))
}

fn build_seanet_encoder(device: &NdArrayDevice) -> SeanetEncoder<TestBackend> {
    let fixture: SeanetFixture = read_fixture("seanet_encoder.json");
    let mut layers = Vec::new();
    for layer in fixture.layers {
        match layer {
            LayerFixture::Conv1d { config, weight, bias } => {
                let conv = build_conv(&config, weight, bias, device);
                layers.push(SeanetLayer::Conv1d(conv));
            }
            LayerFixture::Elu => layers.push(SeanetLayer::Elu),
            LayerFixture::ResBlock { convs } => {
                let mut res_layers = Vec::new();
                for conv in convs {
                    let res = build_conv(&conv.config, conv.weight, conv.bias, device);
                    res_layers.push(res);
                }
                layers.push(SeanetLayer::ResBlock(res_layers));
            }
            _ => {}
        }
    }
    SeanetEncoder::new(layers)
}

fn build_seanet_decoder(device: &NdArrayDevice) -> SeanetDecoder<TestBackend> {
    let fixture: SeanetFixture = read_fixture("seanet_decoder.json");
    let mut layers = Vec::new();
    for layer in fixture.layers {
        match layer {
            LayerFixture::ConvTranspose { config, weight, bias } => {
                let conv = build_conv_transpose(&config, weight, bias, device);
                layers.push(SeanetLayer::ConvTranspose1d(conv));
            }
            LayerFixture::Conv1d { config, weight, bias } => {
                let conv = build_conv(&config, weight, bias, device);
                layers.push(SeanetLayer::Conv1d(conv));
            }
            LayerFixture::Elu => layers.push(SeanetLayer::Elu),
            LayerFixture::ResBlock { convs } => {
                let mut res_layers = Vec::new();
                for conv in convs {
                    let res = build_conv(&conv.config, conv.weight, conv.bias, device);
                    res_layers.push(res);
                }
                layers.push(SeanetLayer::ResBlock(res_layers));
            }
        }
    }
    SeanetDecoder::new(layers)
}

fn apply_linear(linear: &mut burn_nn::Linear<TestBackend>, weight: Vec<Vec<f32>>, device: &NdArrayDevice) {
    let rows = weight.len();
    let cols = weight[0].len();
    let flat: Vec<f32> = weight.into_iter().flatten().collect();
    let weight = Tensor::from_data(TensorData::new(flat, [rows, cols]), device);
    linear.weight = Param::from_tensor(weight);
}

fn apply_layer_norm(norm: &mut burn_nn::LayerNorm<TestBackend>, weights: &NormFixture, device: &NdArrayDevice) {
    let gamma = tensor1_from_vec(weights.gamma.clone(), device);
    let beta = tensor1_from_vec(weights.beta.clone(), device);
    norm.gamma = Param::from_tensor(gamma);
    norm.beta = Some(Param::from_tensor(beta));
}

fn apply_layer_scale(scale: &mut guth::modules::layer_scale::LayerScale<TestBackend>, weights: &LayerScaleFixture, device: &NdArrayDevice) {
    let weight = tensor1_from_vec(weights.scale.clone(), device);
    scale.scale = Param::from_tensor(weight);
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
    let mut projected = MimiProjectedTransformer::new(projected_config, device);

    if fixture.input_proj.kind == "linear" {
        let weight = fixture
            .input_proj
            .weight
            .clone()
            .expect("input proj weight");
        let input_proj = projected.input_proj.as_mut().expect("input proj");
        apply_linear(input_proj, weight, device);
    }

    for (index, output) in fixture.output_projs.iter().enumerate() {
        match (&output.kind[..], projected.output_projs.get_mut(index)) {
            ("identity", Some(ProjectedOutput::Identity)) => {}
            ("linear", Some(ProjectedOutput::Linear(linear))) => {
                let weight = output.weight.clone().expect("output proj weight");
                apply_linear(linear, weight, device);
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
        apply_linear(&mut layer.self_attn.in_proj, fixture_layer.self_attn.in_proj.weight.clone(), device);
        apply_linear(&mut layer.self_attn.out_proj, fixture_layer.self_attn.out_proj.weight.clone(), device);
        apply_layer_norm(&mut layer.norm1, &fixture_layer.norm1, device);
        apply_layer_norm(&mut layer.norm2, &fixture_layer.norm2, device);
        apply_linear(&mut layer.linear1, fixture_layer.linear1.weight.clone(), device);
        apply_linear(&mut layer.linear2, fixture_layer.linear2.weight.clone(), device);
        if let Some(scale) = layer.layer_scale_1.as_mut() {
            apply_layer_scale(scale, &fixture_layer.layer_scale_1, device);
        }
        if let Some(scale) = layer.layer_scale_2.as_mut() {
            apply_layer_scale(scale, &fixture_layer.layer_scale_2, device);
        }
    }

    projected
}

#[test]
fn dummy_quantizer_matches_fixture() {
    let device = NdArrayDevice::default();
    let fixture: DummyQuantizerFixture = read_fixture("dummy_quantizer.json");
    assert_eq!(fixture.weight.len(), fixture.output_dimension);

    let weight = tensor3_from_nested(fixture.weight, &device);
    let quantizer = DummyQuantizer::new(weight);
    let input = tensor3_from_nested(fixture.input, &device);
    let output = quantizer.forward(input).to_data();
    let expected = tensor3_from_nested(fixture.output, &device).to_data();
    assert_close(&output, &expected, 1e-4);
}

#[test]
fn mimi_encode_decode_matches_fixture() {
    let device = NdArrayDevice::default();
    let fixture: MimiFixture = read_fixture("mimi_model.json");

    let encoder = build_seanet_encoder(&device);
    let decoder = build_seanet_decoder(&device);
    let encoder_transformer =
        build_projected_transformer(&fixture.encoder_transformer, &fixture.config.transformer, &device);
    let decoder_transformer =
        build_projected_transformer(&fixture.decoder_transformer, &fixture.config.transformer, &device);

    let quantizer_weight = Tensor::<TestBackend, 3>::zeros([1, 1, 1], &device);
    let quantizer = DummyQuantizer::new(quantizer_weight);

    let mimi = MimiModel::new(
        encoder,
        decoder,
        encoder_transformer,
        decoder_transformer,
        quantizer,
        fixture.config.frame_rate,
        fixture.config.encoder_frame_rate,
        fixture.config.sample_rate,
        fixture.config.channels,
        fixture.config.dimension,
    );

    let audio_input = tensor3_from_nested(fixture.audio_input, &device);
    let latent_output = mimi.encode_to_latent(audio_input).to_data();
    let expected_latent = tensor3_from_nested(fixture.latent_output, &device).to_data();
    assert_close(&latent_output, &expected_latent, 1e-4);

    let latent_input = tensor3_from_nested(fixture.latent_input, &device);
    let mut state = mimi.init_state(1, latent_input.dims()[2]);
    let audio_output = mimi.decode_from_latent(latent_input, &mut state).to_data();
    let expected_audio = tensor3_from_nested(fixture.audio_output, &device).to_data();
    assert_close(&audio_output, &expected_audio, 1e-4);
}

#[test]
fn mimi_streaming_decode_matches_batch() {
    let device = NdArrayDevice::default();
    let fixture: MimiFixture = read_fixture("mimi_model.json");

    let encoder = build_seanet_encoder(&device);
    let decoder = build_seanet_decoder(&device);
    let encoder_transformer =
        build_projected_transformer(&fixture.encoder_transformer, &fixture.config.transformer, &device);
    let decoder_transformer =
        build_projected_transformer(&fixture.decoder_transformer, &fixture.config.transformer, &device);
    let quantizer_weight = Tensor::<TestBackend, 3>::zeros([1, 1, 1], &device);
    let quantizer = DummyQuantizer::new(quantizer_weight);

    let mimi = MimiModel::new(
        encoder,
        decoder,
        encoder_transformer,
        decoder_transformer,
        quantizer,
        fixture.config.frame_rate,
        fixture.config.encoder_frame_rate,
        fixture.config.sample_rate,
        fixture.config.channels,
        fixture.config.dimension,
    );

    let latent_input = tensor3_from_nested(fixture.latent_input, &device);
    let mut batch_state = mimi.init_state(1, latent_input.dims()[2]);
    let batch_audio = mimi.decode_from_latent(latent_input.clone(), &mut batch_state);

    let mut stream_state = mimi.init_state(1, latent_input.dims()[2]);
    let chunk1 = latent_input.clone().narrow(2, 0, 1);
    let chunk2 = latent_input.clone().narrow(2, 1, latent_input.dims()[2] - 1);
    let out1 = mimi.decode_from_latent(chunk1, &mut stream_state);
    mimi.increment_step(&mut stream_state, 1);
    let out2 = mimi.decode_from_latent(chunk2, &mut stream_state);
    let stream_audio = Tensor::cat(vec![out1, out2], 2);

    let diff = batch_audio.sub(stream_audio).abs().to_data();
    let diff_values = diff.as_slice::<f32>().expect("diff slice");
    let max_diff = diff_values
        .iter()
        .copied()
        .fold(0.0_f32, |acc, val| acc.max(val));
    assert!(max_diff < 1e-4, "max diff {max_diff}");
}
