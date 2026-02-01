use crate::modules::streaming_conv::{
    StreamingConv1dOp, StreamingConvState, StreamingConvTranspose1dOp,
};
use crate::state::StreamStep;
use anyhow::Result;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

fn elu<B: Backend>(input: Tensor<B, 3>, alpha: f32) -> Tensor<B, 3> {
    let mask = input.clone().greater_equal_elem(0.0);
    let neg = input.clone().exp().sub_scalar(1.0).mul_scalar(alpha);
    neg.mask_where(mask, input)
}

#[derive(Debug, Clone)]
pub struct SeanetResnetBlock<B: Backend> {
    pub(crate) convs: Vec<StreamingConv1dOp<B>>,
}

#[derive(Debug, Clone)]
pub struct SeanetResnetBlockState<B: Backend> {
    convs: Vec<StreamingConvState<B>>,
}

impl<B: Backend> SeanetResnetBlock<B> {
    pub fn new(convs: Vec<StreamingConv1dOp<B>>) -> Self {
        Self { convs }
    }

    pub fn init_state(&self, batch_size: usize) -> SeanetResnetBlockState<B> {
        SeanetResnetBlockState {
            convs: self
                .convs
                .iter()
                .map(|conv| init_conv_state(conv, batch_size))
                .collect(),
        }
    }

    pub fn forward(
        &self,
        input: Tensor<B, 3>,
        state: &mut SeanetResnetBlockState<B>,
    ) -> Tensor<B, 3> {
        let mut value = input.clone();
        for (conv, conv_state) in self.convs.iter().zip(state.convs.iter_mut()) {
            value = elu(value, 1.0);
            value = conv.forward(conv_state, value);
        }
        input + value
    }
}

#[derive(Debug, Clone)]
pub enum SeanetLayer<B: Backend> {
    Elu,
    Conv1d(StreamingConv1dOp<B>),
    ConvTranspose1d(StreamingConvTranspose1dOp<B>),
    ResBlock(SeanetResnetBlock<B>),
}

#[derive(Debug, Clone)]
pub enum SeanetLayerState<B: Backend> {
    None,
    Conv1d(StreamingConvState<B>),
    ConvTranspose1d(StreamingConvState<B>),
    ResBlock(SeanetResnetBlockState<B>),
}

#[derive(Debug, Clone)]
pub struct SeanetState<B: Backend> {
    layers: Vec<SeanetLayerState<B>>,
}

impl<B: Backend> SeanetState<B> {
    fn new(layers: Vec<SeanetLayerState<B>>) -> Self {
        Self { layers }
    }
}

#[derive(Debug, Clone)]
pub struct SeanetEncoder<B: Backend> {
    pub(crate) layers: Vec<SeanetLayer<B>>,
}

#[derive(Debug, Clone)]
pub struct SeanetDecoder<B: Backend> {
    pub(crate) layers: Vec<SeanetLayer<B>>,
}

impl<B: Backend> SeanetEncoder<B> {
    pub fn new(layers: Vec<SeanetLayer<B>>) -> Self {
        Self { layers }
    }

    pub fn init_state(&self, batch_size: usize) -> SeanetState<B> {
        SeanetState::new(init_layer_states(&self.layers, batch_size))
    }

    pub fn forward(
        &self,
        mut input: Tensor<B, 3>,
        state: &mut SeanetState<B>,
    ) -> Result<Tensor<B, 3>> {
        for (layer, layer_state) in self.layers.iter().zip(state.layers.iter_mut()) {
            input = apply_layer(layer, layer_state, input)?;
        }
        Ok(input)
    }
}

impl<B: Backend> SeanetDecoder<B> {
    pub fn new(layers: Vec<SeanetLayer<B>>) -> Self {
        Self { layers }
    }

    pub fn init_state(&self, batch_size: usize) -> SeanetState<B> {
        SeanetState::new(init_layer_states(&self.layers, batch_size))
    }

    pub fn forward(
        &self,
        mut input: Tensor<B, 3>,
        state: &mut SeanetState<B>,
    ) -> Result<Tensor<B, 3>> {
        for (layer, layer_state) in self.layers.iter().zip(state.layers.iter_mut()) {
            input = apply_layer(layer, layer_state, input)?;
        }
        Ok(input)
    }
}

fn init_layer_states<B: Backend>(
    layers: &[SeanetLayer<B>],
    batch_size: usize,
) -> Vec<SeanetLayerState<B>> {
    layers
        .iter()
        .map(|layer| match layer {
            SeanetLayer::Elu => SeanetLayerState::None,
            SeanetLayer::Conv1d(conv) => {
                SeanetLayerState::Conv1d(init_conv_state(conv, batch_size))
            }
            SeanetLayer::ConvTranspose1d(conv) => {
                SeanetLayerState::ConvTranspose1d(init_conv_transpose_state(conv, batch_size))
            }
            SeanetLayer::ResBlock(block) => {
                SeanetLayerState::ResBlock(block.init_state(batch_size))
            }
        })
        .collect()
}

fn init_conv_state<B: Backend>(
    conv: &StreamingConv1dOp<B>,
    batch_size: usize,
) -> StreamingConvState<B> {
    let effective_kernel = conv.config.dilation * (conv.config.kernel_size.saturating_sub(1)) + 1;
    let history_len = effective_kernel.saturating_sub(conv.config.stride);
    let mut state = StreamingConvState::default();
    if history_len == 0 {
        return state;
    }
    let in_channels = conv.weight.dims()[1];
    let device = conv.weight.device();
    let history = Tensor::zeros([batch_size, in_channels, history_len], &device);
    state.history = Some(history);
    state.step = StreamStep::new();
    state
}

fn init_conv_transpose_state<B: Backend>(
    conv: &StreamingConvTranspose1dOp<B>,
    batch_size: usize,
) -> StreamingConvState<B> {
    let history_len = conv.config.kernel_size.saturating_sub(conv.config.stride);
    let mut state = StreamingConvState::default();
    if history_len == 0 {
        return state;
    }
    let out_channels = conv.weight.dims()[1];
    let device = conv.weight.device();
    let history = Tensor::zeros([batch_size, out_channels, history_len], &device);
    state.history = Some(history);
    state.step = StreamStep::new();
    state
}

fn apply_layer<B: Backend>(
    layer: &SeanetLayer<B>,
    state: &mut SeanetLayerState<B>,
    input: Tensor<B, 3>,
) -> Result<Tensor<B, 3>> {
    match (layer, state) {
        (SeanetLayer::Elu, _) => Ok(elu(input, 1.0)),
        (SeanetLayer::Conv1d(conv), SeanetLayerState::Conv1d(state)) => {
            Ok(conv.forward(state, input))
        }
        (SeanetLayer::ConvTranspose1d(conv), SeanetLayerState::ConvTranspose1d(state)) => {
            conv.forward(state, input)
        }
        (SeanetLayer::ResBlock(block), SeanetLayerState::ResBlock(state)) => {
            Ok(block.forward(input, state))
        }
        _ => Err(anyhow::anyhow!("Seanet layer/state mismatch")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modules::streaming_conv::{PaddingMode, StreamingConvConfig};
    use burn::tensor::{TensorData, Tolerance};
    use burn_ndarray::{NdArray, NdArrayDevice};
    use serde::Deserialize;
    use std::fs;
    use std::path::PathBuf;

    type TestBackend = NdArray<f32>;

    #[allow(dead_code)]
    #[derive(Debug, Deserialize)]
    struct SeanetFixture {
        config: SeanetConfigFixture,
        input: Vec<Vec<Vec<f32>>>,
        layers: Vec<LayerFixture>,
        output: Vec<Vec<Vec<f32>>>,
    }

    #[allow(dead_code)]
    #[derive(Debug, Deserialize)]
    struct SeanetConfigFixture {
        channels: usize,
        dimension: usize,
        n_filters: usize,
        n_residual_layers: usize,
        ratios: Vec<usize>,
        kernel_size: usize,
        last_kernel_size: usize,
        residual_kernel_size: usize,
        dilation_base: usize,
        pad_mode: String,
        compress: usize,
    }

    #[derive(Debug, Deserialize)]
    #[serde(tag = "kind")]
    enum LayerFixture {
        #[serde(rename = "elu")]
        Elu,
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
        #[serde(rename = "resblock")]
        ResBlock { convs: Vec<ConvLayerFixture> },
    }

    #[derive(Debug, Deserialize, Clone)]
    struct ConvLayerFixture {
        config: ConvConfigFixture,
        weight: Vec<Vec<Vec<f32>>>,
        bias: Vec<f32>,
    }

    #[derive(Debug, Deserialize, Clone)]
    struct ConvConfigFixture {
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        padding: usize,
        #[serde(default)]
        pad_mode: String,
    }

    fn fixture_path(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join(name)
    }

    fn load_fixture(name: &str) -> SeanetFixture {
        let data = fs::read_to_string(fixture_path(name)).expect("fixture read");
        serde_json::from_str(&data).expect("fixture parse")
    }

    fn tensor3(device: &NdArrayDevice, data: Vec<Vec<Vec<f32>>>) -> Tensor<TestBackend, 3> {
        let b = data.len();
        let c = data[0].len();
        let t = data[0][0].len();
        let flat: Vec<f32> = data.into_iter().flatten().flatten().collect();
        let td = TensorData::new(flat, [b, c, t]);
        Tensor::from_data(td, device)
    }

    fn tensor1(device: &NdArrayDevice, data: Vec<f32>) -> Tensor<TestBackend, 1> {
        let len = data.len();
        Tensor::from_data(TensorData::new(data, [len]), device)
    }

    fn pad_mode(name: &str) -> PaddingMode {
        match name {
            "replicate" | "Replicate" => PaddingMode::Replicate,
            _ => PaddingMode::Constant,
        }
    }

    fn build_conv(
        fixture: &ConvLayerFixture,
        device: &NdArrayDevice,
    ) -> StreamingConv1dOp<TestBackend> {
        let config = StreamingConvConfig {
            kernel_size: fixture.config.kernel_size,
            stride: fixture.config.stride,
            dilation: fixture.config.dilation,
            padding: fixture.config.padding,
            pad_mode: pad_mode(&fixture.config.pad_mode),
            groups: 1,
        };
        let weight = tensor3(device, fixture.weight.clone());
        let bias = if fixture.bias.is_empty() {
            None
        } else {
            Some(tensor1(device, fixture.bias.clone()))
        };
        StreamingConv1dOp::new(config, weight, bias)
    }

    fn build_conv_transpose(
        fixture: &ConvConfigFixture,
        weight: Vec<Vec<Vec<f32>>>,
        bias: Vec<f32>,
        device: &NdArrayDevice,
    ) -> StreamingConvTranspose1dOp<TestBackend> {
        let config = StreamingConvConfig {
            kernel_size: fixture.kernel_size,
            stride: fixture.stride,
            dilation: fixture.dilation,
            padding: fixture.padding,
            pad_mode: PaddingMode::Constant,
            groups: 1,
        };
        let weight = tensor3(device, weight);
        let bias = if bias.is_empty() {
            None
        } else {
            Some(tensor1(device, bias))
        };
        StreamingConvTranspose1dOp::new(config, weight, bias)
    }

    fn build_layers(
        fixture: &SeanetFixture,
        device: &NdArrayDevice,
    ) -> Vec<SeanetLayer<TestBackend>> {
        fixture
            .layers
            .iter()
            .map(|layer| match layer {
                LayerFixture::Elu => SeanetLayer::Elu,
                LayerFixture::Conv1d {
                    config,
                    weight,
                    bias,
                } => {
                    let conv_fixture = ConvLayerFixture {
                        config: config.clone(),
                        weight: weight.clone(),
                        bias: bias.clone(),
                    };
                    SeanetLayer::Conv1d(build_conv(&conv_fixture, device))
                }
                LayerFixture::ConvTranspose {
                    config,
                    weight,
                    bias,
                } => SeanetLayer::ConvTranspose1d(build_conv_transpose(
                    config,
                    weight.clone(),
                    bias.clone(),
                    device,
                )),
                LayerFixture::ResBlock { convs } => {
                    let convs = convs.iter().map(|conv| build_conv(conv, device)).collect();
                    SeanetLayer::ResBlock(SeanetResnetBlock::new(convs))
                }
            })
            .collect()
    }

    #[test]
    fn seanet_encoder_matches_fixture() {
        let fixture = load_fixture("seanet_encoder.json");
        let device = NdArrayDevice::default();
        let layers = build_layers(&fixture, &device);
        let encoder = SeanetEncoder::new(layers);
        let mut state = encoder.init_state(1);
        let input = tensor3(&device, fixture.input);
        let output = encoder.forward(input, &mut state).expect("forward");
        let expected = tensor3(&device, fixture.output).to_data();
        output
            .to_data()
            .assert_approx_eq(&expected, Tolerance::<f32>::absolute(1e-4));
    }

    #[test]
    fn seanet_decoder_matches_fixture() {
        let fixture = load_fixture("seanet_decoder.json");
        let device = NdArrayDevice::default();
        let layers = build_layers(&fixture, &device);
        let decoder = SeanetDecoder::new(layers);
        let mut state = decoder.init_state(1);
        let input = tensor3(&device, fixture.input);
        let output = decoder.forward(input, &mut state).expect("forward");
        let expected = tensor3(&device, fixture.output).to_data();
        output
            .to_data()
            .assert_approx_eq(&expected, Tolerance::<f32>::absolute(1e-4));
    }
}
