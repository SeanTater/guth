use burn::tensor::{Tensor, TensorData};
use burn_ndarray::NdArrayDevice;
use serde::Deserialize;

use crate::common::{assert_close, read_fixture, tensor3, TestBackend};
use guth::model::mimi::MimiModel;
use guth::modules::dummy_quantizer::DummyQuantizer;
use guth::modules::mimi_transformer::{
    MimiProjectedTransformer, MimiProjectedTransformerConfig, MimiTransformerConfig,
};
use guth::modules::seanet::{SeanetDecoder, SeanetEncoder, SeanetLayer, SeanetResnetBlock};
use guth::modules::streaming_conv::{
    PaddingMode, StreamingConv1dOp, StreamingConvConfig, StreamingConvTranspose1dOp,
};
use guth::weights::load_mimi_state_dict;

#[derive(Deserialize)]
struct DummyQuantizerFixture {
    input: Vec<Vec<Vec<f32>>>,
    output: Vec<Vec<Vec<f32>>>,
}

#[derive(Deserialize)]
struct MimiFixture {
    config: MimiConfigFixture,
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
    let out_channels = weight.len();
    let in_channels = weight[0].len();
    let kernel = weight[0][0].len();
    let weight = Tensor::<TestBackend, 3>::zeros([out_channels, in_channels, kernel], device);
    let bias = Tensor::<TestBackend, 1>::zeros([bias.len()], device);
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
    let out_channels = weight.len();
    let in_channels = weight[0].len();
    let kernel = weight[0][0].len();
    let weight = Tensor::<TestBackend, 3>::zeros([out_channels, in_channels, kernel], device);
    let bias = Tensor::<TestBackend, 1>::zeros([bias.len()], device);
    StreamingConvTranspose1dOp::new(conv_config, weight, Some(bias))
}

fn build_seanet_encoder(device: &NdArrayDevice) -> SeanetEncoder<TestBackend> {
    let fixture: SeanetFixture = read_fixture("seanet_encoder.json");
    let mut layers = Vec::new();
    for layer in fixture.layers {
        match layer {
            LayerFixture::Conv1d {
                config,
                weight,
                bias,
            } => {
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
                layers.push(SeanetLayer::ResBlock(SeanetResnetBlock::new(res_layers)));
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
            LayerFixture::ConvTranspose {
                config,
                weight,
                bias,
            } => {
                let conv = build_conv_transpose(&config, weight, bias, device);
                layers.push(SeanetLayer::ConvTranspose1d(conv));
            }
            LayerFixture::Conv1d {
                config,
                weight,
                bias,
            } => {
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
                layers.push(SeanetLayer::ResBlock(SeanetResnetBlock::new(res_layers)));
            }
        }
    }
    SeanetDecoder::new(layers)
}

fn build_projected_transformer(
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
    MimiProjectedTransformer::new(projected_config, device)
}

fn build_mimi(fixture: &MimiFixture, device: &NdArrayDevice) -> MimiModel<TestBackend> {
    let encoder = build_seanet_encoder(device);
    let decoder = build_seanet_decoder(device);
    let encoder_transformer = build_projected_transformer(&fixture.config.transformer, device);
    let decoder_transformer = build_projected_transformer(&fixture.config.transformer, device);
    let quantizer_weight = Tensor::<TestBackend, 3>::zeros([1, 1, 1], device);
    let quantizer = DummyQuantizer::new(quantizer_weight);

    let mut mimi = MimiModel::new(
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
        None,
        None,
    );

    let state = load_mimi_state_dict("tests/fixtures/mimi_state.safetensors")
        .expect("load mimi state dict");
    mimi.load_state_dict(&state, device)
        .expect("apply mimi state dict");
    mimi
}

fn tensor3_from_state_data(
    data: &guth::weights::TensorData,
    device: &NdArrayDevice,
) -> Tensor<TestBackend, 3> {
    let shape = [data.shape[0], data.shape[1], data.shape[2]];
    let mut values = Vec::with_capacity(data.data.len() / 4);
    for chunk in data.data.chunks_exact(4) {
        values.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    Tensor::from_data(TensorData::new(values, shape), device)
}

#[test]
fn dummy_quantizer_matches_fixture() {
    let device = NdArrayDevice::default();
    let fixture: DummyQuantizerFixture = read_fixture("dummy_quantizer.json");

    let state = load_mimi_state_dict("tests/fixtures/mimi_state.safetensors")
        .expect("load mimi state dict");
    let weight_data = state.get("quantizer.weight").expect("quantizer weight");
    let weight = tensor3_from_state_data(weight_data, &device);
    let quantizer = DummyQuantizer::new(weight);
    let input = tensor3(fixture.input, &device);
    let output = quantizer.forward(input).to_data();
    let expected = tensor3(fixture.output, &device).to_data();
    assert_close(&output, &expected, 1e-4);
}

#[test]
fn mimi_encode_decode_matches_fixture() {
    let device = NdArrayDevice::default();
    let fixture: MimiFixture = read_fixture("mimi_model.json");

    let mimi = build_mimi(&fixture, &device);

    let audio_input = tensor3(fixture.audio_input, &device);
    let latent_output = mimi.encode_to_latent(audio_input).to_data();
    let expected_latent = tensor3(fixture.latent_output, &device).to_data();
    assert_close(&latent_output, &expected_latent, 1e-4);

    let latent_input = tensor3(fixture.latent_input, &device);
    let mut state = mimi.init_state(1, latent_input.dims()[2], &device);
    let audio_output = mimi.decode_from_latent(latent_input, &mut state).to_data();
    let expected_audio = tensor3(fixture.audio_output, &device).to_data();
    assert_close(&audio_output, &expected_audio, 1e-4);
}

#[test]
fn mimi_streaming_decode_matches_batch() {
    let device = NdArrayDevice::default();
    let fixture: MimiFixture = read_fixture("mimi_model.json");

    let mimi = build_mimi(&fixture, &device);

    let latent_input = tensor3(fixture.latent_input, &device);
    let mut batch_state = mimi.init_state(1, latent_input.dims()[2], &device);
    let batch_audio = mimi.decode_from_latent(latent_input.clone(), &mut batch_state);

    let mut stream_state = mimi.init_state(1, latent_input.dims()[2], &device);
    let chunk1 = latent_input.clone().narrow(2, 0, 2);
    let chunk2 = latent_input
        .clone()
        .narrow(2, 2, latent_input.dims()[2] - 2);
    let out1 = mimi.decode_from_latent(chunk1, &mut stream_state);
    mimi.increment_step(&mut stream_state, 2);
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

#[test]
fn mimi_streaming_decode_unaligned_chunks_matches_batch() {
    let device = NdArrayDevice::default();
    let fixture: MimiFixture = read_fixture("mimi_model.json");

    let mimi = build_mimi(&fixture, &device);

    let latent_input = tensor3(fixture.latent_input, &device);
    let mut batch_state = mimi.init_state(1, latent_input.dims()[2], &device);
    let batch_audio = mimi.decode_from_latent(latent_input.clone(), &mut batch_state);

    let mut stream_state = mimi.init_state(1, latent_input.dims()[2], &device);
    let first_len = 1;
    let chunk1 = latent_input.clone().narrow(2, 0, first_len);
    let chunk2 = latent_input
        .clone()
        .narrow(2, first_len, latent_input.dims()[2] - first_len);
    let out1 = mimi.decode_from_latent(chunk1, &mut stream_state);
    mimi.increment_step(&mut stream_state, first_len);
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

#[test]
fn mimi_transformer_streaming_matches_batch() {
    let device = NdArrayDevice::default();
    let fixture: MimiFixture = read_fixture("mimi_model.json");

    let mimi = build_mimi(&fixture, &device);
    let transformer = mimi.decoder_transformer;
    let input = tensor3(fixture.latent_input, &device);

    let mut batch_state = transformer.init_state(1, &device);
    let batch_output = transformer
        .forward(input.clone(), &mut batch_state)
        .remove(0);

    let mut stream_state = transformer.init_state(1, &device);
    let chunk1 = input.clone().narrow(2, 0, 2);
    let chunk2 = input.clone().narrow(2, 2, input.dims()[2] - 2);
    let out1 = transformer.forward(chunk1, &mut stream_state).remove(0);
    transformer.increment_step(&mut stream_state, 2);
    let out2 = transformer.forward(chunk2, &mut stream_state).remove(0);
    let stream_output = Tensor::cat(vec![out1, out2], 2);

    let diff = batch_output.sub(stream_output).abs().to_data();
    let diff_values = diff.as_slice::<f32>().expect("diff slice");
    let max_diff = diff_values
        .iter()
        .copied()
        .fold(0.0_f32, |acc, val| acc.max(val));
    assert!(max_diff < 1e-4, "max diff {max_diff}");
}
