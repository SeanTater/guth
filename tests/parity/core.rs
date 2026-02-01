use burn::tensor::Tensor;
use burn_ndarray::NdArrayDevice;
use serde::Deserialize;

use crate::common::{assert_close, read_fixture, tensor1, tensor3, tensor4, TestBackend};
use guth::modules::streaming_conv::{
    PaddingMode, StreamingConv1dOp, StreamingConvConfig, StreamingConvState,
    StreamingConvTranspose1dOp,
};
use guth::modules::streaming_mha::{StreamingMha, StreamingMhaConfig, StreamingMhaOp};
use guth::state::StreamingModule;

#[derive(Deserialize)]
struct Conv1dFixture {
    config: Conv1dConfigFixture,
    weight: Vec<Vec<Vec<f32>>>,
    bias: Vec<f32>,
    chunks: Vec<Vec<Vec<Vec<f32>>>>,
    outputs: Vec<Vec<Vec<Vec<f32>>>>,
}

#[derive(Deserialize)]
struct Conv1dConfigFixture {
    kernel_size: usize,
    stride: usize,
    dilation: usize,
    padding: usize,
    pad_mode: String,
}

#[derive(Deserialize)]
struct ConvTransposeFixture {
    config: ConvTransposeConfigFixture,
    weight: Vec<Vec<Vec<f32>>>,
    bias: Vec<f32>,
    chunk: Vec<Vec<Vec<f32>>>,
    emit: Vec<Vec<Vec<f32>>>,
    flush: Vec<Vec<Vec<f32>>>,
}

#[derive(Deserialize)]
struct ConvTransposeConfigFixture {
    kernel_size: usize,
    stride: usize,
    dilation: usize,
    padding: usize,
}

#[derive(Deserialize)]
struct AttentionFixture {
    config: AttentionConfigFixture,
    keys: Vec<Vec<Vec<Vec<f32>>>>,
    values: Vec<Vec<Vec<Vec<f32>>>>,
    queries: Vec<Vec<Vec<Vec<f32>>>>,
    output: Vec<Vec<Vec<Vec<f32>>>>,
}

#[derive(Deserialize)]
struct AttentionConfigFixture {
    num_heads: usize,
    head_dim: usize,
    causal: bool,
}

fn init_conv_state(
    config: &StreamingConvConfig,
    weight: &Tensor<TestBackend, 3>,
    device: &NdArrayDevice,
) -> StreamingConvState<TestBackend> {
    let effective_kernel = config.dilation * (config.kernel_size.saturating_sub(1)) + 1;
    let history_len = effective_kernel.saturating_sub(config.stride);
    let mut state = StreamingConvState::default();
    if history_len == 0 {
        return state;
    }
    let in_channels = weight.dims()[1];
    let history = Tensor::zeros([1, in_channels, history_len], device);
    state.history = Some(history);
    state
}

fn init_conv_transpose_state(
    config: &StreamingConvConfig,
    weight: &Tensor<TestBackend, 3>,
    device: &NdArrayDevice,
) -> StreamingConvState<TestBackend> {
    let history_len = config.kernel_size.saturating_sub(config.stride);
    let mut state = StreamingConvState::default();
    if history_len == 0 {
        return state;
    }
    let out_channels = weight.dims()[1];
    let history = Tensor::zeros([1, out_channels, history_len], device);
    state.history = Some(history);
    state
}

#[test]
fn parity_conv1d_streaming() {
    let device = NdArrayDevice::default();
    let fixture: Conv1dFixture = read_fixture("conv1d.json");
    let pad_mode = match fixture.config.pad_mode.as_str() {
        "Replicate" => PaddingMode::Replicate,
        _ => PaddingMode::Constant,
    };
    let config = StreamingConvConfig {
        kernel_size: fixture.config.kernel_size,
        stride: fixture.config.stride,
        dilation: fixture.config.dilation,
        padding: fixture.config.padding,
        pad_mode,
        groups: 1,
    };
    let weight = tensor3(fixture.weight, &device);
    let bias = tensor1(fixture.bias, &device);
    let op = StreamingConv1dOp::new(config, weight, Some(bias));
    let mut state = init_conv_state(&op.config, &op.weight, &device);

    for (chunk, expected) in fixture.chunks.into_iter().zip(fixture.outputs.into_iter()) {
        let input = tensor3(chunk, &device);
        let output = op.forward(&mut state, input).to_data();
        let expected = tensor3(expected, &device).to_data();
        assert_close(&output, &expected, 1e-4);
    }
}

#[test]
fn parity_conv_transpose_streaming() {
    let device = NdArrayDevice::default();
    let fixture: ConvTransposeFixture = read_fixture("conv_transpose.json");
    let config = StreamingConvConfig {
        kernel_size: fixture.config.kernel_size,
        stride: fixture.config.stride,
        dilation: fixture.config.dilation,
        padding: fixture.config.padding,
        pad_mode: PaddingMode::Constant,
        groups: 1,
    };
    let weight = tensor3(fixture.weight, &device);
    let bias = tensor1(fixture.bias, &device);
    let op = StreamingConvTranspose1dOp::new(config, weight, Some(bias));
    let mut state = init_conv_transpose_state(&op.config, &op.weight, &device);

    let input = tensor3(fixture.chunk, &device);
    let output = op.forward(&mut state, input).unwrap().to_data();
    let expected = tensor3(fixture.emit, &device).to_data();
    assert_close(&output, &expected, 1e-4);

    let flush = op
        .forward(
            &mut state,
            Tensor::<TestBackend, 3>::zeros([1, 1, 0], &device),
        )
        .unwrap()
        .to_data();
    let expected_flush = tensor3(fixture.flush, &device).to_data();
    assert_close(&flush, &expected_flush, 1e-4);
}

#[test]
fn parity_attention() {
    let device = NdArrayDevice::default();
    for fixture_name in [
        "attention.json",
        "attention_causal.json",
        "attention_multihead.json",
    ] {
        let fixture: AttentionFixture = read_fixture(fixture_name);
        let config = StreamingMhaConfig {
            num_heads: fixture.config.num_heads,
            head_dim: fixture.config.head_dim,
            causal: fixture.config.causal,
            ..Default::default()
        };
        let op = StreamingMhaOp::<TestBackend>::new(config, &device);
        let mut state = StreamingMha::default().init_state(1, 0);

        let keys = tensor4(fixture.keys, &device);
        let values = tensor4(fixture.values, &device);
        op.append_kv(&mut state, keys, values);

        let queries = tensor4(fixture.queries, &device);
        let output = op.attention(&state, queries).to_data();
        let expected = tensor4(fixture.output, &device).to_data();
        assert_close(&output, &expected, 1e-4);
    }
}
