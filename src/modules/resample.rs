use crate::modules::streaming_conv::{
    PaddingMode, StreamingConv1dOp, StreamingConvConfig, StreamingConvState,
    StreamingConvTranspose1dOp,
};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct ConvDownsample1d<B: Backend> {
    pub conv: StreamingConv1dOp<B>,
}

#[derive(Debug, Clone)]
pub struct ConvDownsample1dState<B: Backend> {
    pub conv: StreamingConvState<B>,
}

#[derive(Debug, Clone)]
pub struct ConvTrUpsample1d<B: Backend> {
    pub conv: StreamingConvTranspose1dOp<B>,
}

#[derive(Debug, Clone)]
pub struct ConvTrUpsample1dState<B: Backend> {
    pub conv: StreamingConvState<B>,
}

fn init_conv_state<B: Backend>(
    config: &StreamingConvConfig,
    batch_size: usize,
    in_channels: usize,
    device: &B::Device,
) -> StreamingConvState<B> {
    let effective_kernel = config.dilation * (config.kernel_size.saturating_sub(1)) + 1;
    let history_len = effective_kernel.saturating_sub(config.stride);
    let mut state = StreamingConvState::default();
    if history_len == 0 {
        return state;
    }
    state.history = Some(Tensor::zeros(
        [batch_size, in_channels, history_len],
        device,
    ));
    state
}

fn init_conv_transpose_state<B: Backend>(
    config: &StreamingConvConfig,
    batch_size: usize,
    out_channels: usize,
    device: &B::Device,
) -> StreamingConvState<B> {
    let history_len = config.kernel_size.saturating_sub(config.stride);
    let mut state = StreamingConvState::default();
    if history_len == 0 {
        return state;
    }
    state.history = Some(Tensor::zeros(
        [batch_size, out_channels, history_len],
        device,
    ));
    state
}

impl<B: Backend> ConvDownsample1d<B> {
    pub fn new(stride: usize, weight: Tensor<B, 3>) -> Self {
        let config = StreamingConvConfig {
            kernel_size: 2 * stride,
            stride,
            dilation: 1,
            padding: 0,
            pad_mode: PaddingMode::Replicate,
            groups: 1,
        };
        Self {
            conv: StreamingConv1dOp::new(config, weight, None),
        }
    }

    pub fn init_state(&self, batch_size: usize, device: &B::Device) -> ConvDownsample1dState<B> {
        let in_channels = self.conv.weight.dims()[1];
        let state = init_conv_state(&self.conv.config, batch_size, in_channels, device);
        ConvDownsample1dState { conv: state }
    }

    pub fn forward(
        &self,
        input: Tensor<B, 3>,
        state: &mut ConvDownsample1dState<B>,
    ) -> Tensor<B, 3> {
        self.conv.forward(&mut state.conv, input)
    }
}

impl<B: Backend> ConvTrUpsample1d<B> {
    pub fn new(stride: usize, weight: Tensor<B, 3>, groups: usize) -> Self {
        let config = StreamingConvConfig {
            kernel_size: 2 * stride,
            stride,
            dilation: 1,
            padding: 0,
            pad_mode: PaddingMode::Constant,
            groups,
        };
        Self {
            conv: StreamingConvTranspose1dOp::new(config, weight, None),
        }
    }

    pub fn init_state(&self, batch_size: usize, device: &B::Device) -> ConvTrUpsample1dState<B> {
        let out_channels = self.conv.weight.dims()[1] * self.conv.config.groups;
        let state = init_conv_transpose_state(&self.conv.config, batch_size, out_channels, device);
        ConvTrUpsample1dState { conv: state }
    }

    pub fn forward(
        &self,
        input: Tensor<B, 3>,
        state: &mut ConvTrUpsample1dState<B>,
    ) -> anyhow::Result<Tensor<B, 3>> {
        self.conv.forward(&mut state.conv, input)
    }
}
