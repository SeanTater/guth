//! Streaming-friendly 1D convolution primitives.
//!
//! These operators keep a small history buffer so they can process audio in
//! chunks while producing the same result as full batch convolution.

use crate::state::StreamingModule;
use anyhow::Result;
use burn::tensor::{
    backend::Backend,
    module::{conv1d, conv_transpose1d},
    ops::{ConvOptions, ConvTransposeOptions, PadMode},
    s, Tensor,
};

/// Marker type for streaming 1D convolution.
#[derive(Debug, Default)]
pub struct StreamingConv1d;

/// Marker type for streaming 1D transposed convolution.
#[derive(Debug, Default)]
pub struct StreamingConvTranspose1d;

/// Streaming convolution state (history buffer).
#[derive(Debug, Clone)]
pub struct StreamingConvState<B: Backend> {
    /// Cached input/output history for overlap.
    pub history: Option<Tensor<B, 3>>,
    /// Track first chunk for replicate padding parity with the Python implementation.
    pub is_first: bool,
}

impl<B: Backend> Default for StreamingConvState<B> {
    /// Create an empty streaming state with no history.
    fn default() -> Self {
        Self {
            history: None,
            is_first: true,
        }
    }
}

/// Padding modes supported by streaming convs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaddingMode {
    /// Pad with zeros.
    Constant,
    /// Replicate edge values.
    Replicate,
}

/// Configuration for streaming convolution ops.
#[derive(Debug, Clone)]
pub struct StreamingConvConfig {
    /// Kernel size in samples.
    pub kernel_size: usize,
    /// Stride in samples.
    pub stride: usize,
    /// Dilation factor.
    pub dilation: usize,
    /// Explicit padding on the left.
    pub padding: usize,
    /// Padding mode.
    pub pad_mode: PaddingMode,
    /// Number of groups.
    pub groups: usize,
}

impl Default for StreamingConvConfig {
    /// Default to a 1x1 convolution with no padding.
    fn default() -> Self {
        Self {
            kernel_size: 1,
            stride: 1,
            dilation: 1,
            padding: 0,
            pad_mode: PaddingMode::Constant,
            groups: 1,
        }
    }
}

/// 1D convolution op with streaming support.
#[derive(Debug, Clone)]
pub struct StreamingConv1dOp<B: Backend> {
    /// Convolution config.
    pub config: StreamingConvConfig,
    /// Weight tensor `[out, in, kernel]`.
    pub weight: Tensor<B, 3>,
    /// Optional bias `[out]`.
    pub bias: Option<Tensor<B, 1>>,
}

impl<B: Backend> StreamingConv1dOp<B> {
    /// Create a new streaming conv op.
    pub fn new(
        config: StreamingConvConfig,
        weight: Tensor<B, 3>,
        bias: Option<Tensor<B, 1>>,
    ) -> Self {
        Self {
            config,
            weight,
            bias,
        }
    }

    /// Apply convolution to a streaming chunk, updating history.
    pub fn forward(&self, state: &mut StreamingConvState<B>, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let input_len = input.dims()[2];
        if input_len == 0 {
            return input;
        }

        let history_len = state.history.as_ref().map(|h| h.dims()[2]).unwrap_or(0);
        if self.config.pad_mode == PaddingMode::Replicate && state.is_first && history_len > 0 {
            // Python StreamingConv1d fills the initial history by repeating the first sample
            // when pad_mode is replicate; do the same to keep Mimi encode parity.
            let first = input.clone().narrow(2, 0, 1).repeat_dim(2, history_len);
            if let Some(history) = state.history.as_mut() {
                *history = first;
            } else {
                state.history = Some(first);
            }
        }
        let mut extended = if let Some(history) = state.history.take() {
            Tensor::cat(vec![history, input], 2)
        } else {
            input
        };

        if self.config.padding > 0 {
            let pad_mode = match self.config.pad_mode {
                PaddingMode::Constant => PadMode::Constant(0.0),
                PaddingMode::Replicate => PadMode::Edge,
            };
            extended = extended.pad((self.config.padding, 0, 0, 0), pad_mode);
        }

        let output = conv1d(
            extended.clone(),
            self.weight.clone(),
            self.bias.clone(),
            ConvOptions::new(
                [self.config.stride],
                [0],
                [self.config.dilation],
                self.config.groups,
            ),
        );

        let out_hist = conv_output_len(
            history_len + self.config.padding,
            self.config.kernel_size,
            self.config.stride,
            self.config.dilation,
            0,
        );
        let out_total = output.dims()[2];
        let out_new_len = out_total.saturating_sub(out_hist);
        let output = if out_new_len == 0 {
            let device = output.device();
            Tensor::zeros([output.dims()[0], output.dims()[1], 0], &device)
        } else {
            output.narrow(2, out_hist, out_new_len)
        };

        let effective_kernel =
            self.config.dilation * (self.config.kernel_size.saturating_sub(1)) + 1;
        let history_keep = effective_kernel.saturating_sub(self.config.stride);
        if history_keep > 0 {
            let extended_len = extended.dims()[2];
            let start = extended_len.saturating_sub(history_keep);
            state.history = Some(extended.narrow(2, start, history_keep));
        } else {
            state.history = None;
        }
        state.is_first = false;

        output
    }
}

/// 1D transposed convolution op with streaming support.
#[derive(Debug, Clone)]
pub struct StreamingConvTranspose1dOp<B: Backend> {
    /// Convolution config.
    pub config: StreamingConvConfig,
    /// Weight tensor `[in, out, kernel]`.
    pub weight: Tensor<B, 3>,
    /// Optional bias `[out]`.
    pub bias: Option<Tensor<B, 1>>,
}

impl<B: Backend> StreamingConvTranspose1dOp<B> {
    /// Create a new streaming transposed conv op.
    pub fn new(
        config: StreamingConvConfig,
        weight: Tensor<B, 3>,
        bias: Option<Tensor<B, 1>>,
    ) -> Self {
        Self {
            config,
            weight,
            bias,
        }
    }

    /// Apply transposed convolution to a streaming chunk.
    pub fn forward(
        &self,
        state: &mut StreamingConvState<B>,
        input: Tensor<B, 3>,
    ) -> Result<Tensor<B, 3>> {
        let input_len = input.dims()[2];
        if input_len == 0 {
            if let Some(history) = state.history.take() {
                return Ok(history);
            }
            return Ok(input);
        }

        let output = conv_transpose1d(
            input,
            self.weight.clone(),
            self.bias.clone(),
            ConvTransposeOptions::new(
                [self.config.stride],
                [self.config.padding],
                [0],
                [self.config.dilation],
                self.config.groups,
            ),
        );

        let output = if let Some(history) = state.history.take() {
            let hist_len = history.dims()[2];
            if hist_len > 0 {
                let prefix = output.clone().narrow(2, 0, hist_len);
                let merged = prefix + history;
                output.slice_assign(s![.., .., 0..hist_len], merged)
            } else {
                output
            }
        } else {
            output
        };

        let tail_len = self.config.kernel_size.saturating_sub(self.config.stride);
        let total_len = output.dims()[2];
        if tail_len == 0 {
            return Ok(output);
        }
        if total_len <= tail_len {
            let dims = output.dims();
            let device = output.device();
            state.history = Some(output);
            return Ok(Tensor::zeros([dims[0], dims[1], 0], &device));
        }
        let emit_len = total_len - tail_len;
        let emit = output.clone().narrow(2, 0, emit_len);
        let tail = output.narrow(2, emit_len, tail_len);
        let tail = if let Some(bias) = &self.bias {
            let out_channels = bias.dims()[0];
            tail.sub(bias.clone().reshape([1, out_channels, 1]))
        } else {
            tail
        };
        state.history = Some(tail);

        Ok(emit)
    }
}

impl<B: Backend> StreamingModule<B> for StreamingConv1d {
    type State = StreamingConvState<B>;

    /// Initialize an empty streaming state.
    fn init_state(&self, _batch_size: usize, _sequence_length: usize) -> Self::State {
        StreamingConvState::default()
    }

    /// Increment step counter for this state.
    fn increment_step(&self, state: &mut Self::State, increment: usize) {
        let _ = (state, increment);
    }
}

impl<B: Backend> StreamingModule<B> for StreamingConvTranspose1d {
    type State = StreamingConvState<B>;

    /// Initialize an empty streaming state.
    fn init_state(&self, _batch_size: usize, _sequence_length: usize) -> Self::State {
        StreamingConvState::default()
    }

    /// Increment step counter for this state.
    fn increment_step(&self, state: &mut Self::State, increment: usize) {
        let _ = (state, increment);
    }
}

/// Compute the output length of a 1D convolution.
fn conv_output_len(
    input_len: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
    padding: usize,
) -> usize {
    if input_len == 0 {
        return 0;
    }
    let kernel_extent = dilation * (kernel.saturating_sub(1)) + 1;
    let padded = input_len + 2 * padding;
    if padded < kernel_extent {
        return 0;
    }
    (padded - kernel_extent) / stride + 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::{NdArray, NdArrayDevice};

    type TestBackend = NdArray<f32>;

    #[test]
    fn streaming_conv_forward_accumulates_history() {
        let device = NdArrayDevice::default();
        let weight = Tensor::<TestBackend, 3>::from_floats([[[0.0, 1.0]]], &device);
        let bias = Some(Tensor::<TestBackend, 1>::from_floats([0.0], &device));
        let config = StreamingConvConfig {
            kernel_size: 2,
            stride: 1,
            dilation: 1,
            padding: 1,
            pad_mode: PaddingMode::Constant,
            groups: 1,
        };
        let op = StreamingConv1dOp::new(config, weight, bias);
        let mut state = StreamingConv1d.init_state(1, 0);

        let input = Tensor::<TestBackend, 3>::from_floats([[[1.0, 2.0, 3.0]]], &device);
        let first = op.forward(&mut state, input).to_data();
        assert_eq!(first.as_slice::<f32>().unwrap(), &[1.0, 2.0, 3.0]);

        let input = Tensor::<TestBackend, 3>::from_floats([[[4.0]]], &device);
        let second = op.forward(&mut state, input).to_data();
        assert_eq!(second.as_slice::<f32>().unwrap(), &[4.0]);
    }

    #[test]
    fn streaming_conv_replicate_padding() {
        let device = NdArrayDevice::default();
        let weight = Tensor::<TestBackend, 3>::from_floats([[[1.0, 1.0]]], &device);
        let bias = Some(Tensor::<TestBackend, 1>::from_floats([0.0], &device));
        let config = StreamingConvConfig {
            kernel_size: 2,
            stride: 1,
            dilation: 1,
            padding: 1,
            pad_mode: PaddingMode::Replicate,
            groups: 1,
        };
        let op = StreamingConv1dOp::new(config, weight, bias);
        let mut state = StreamingConv1d.init_state(1, 0);

        let input = Tensor::<TestBackend, 3>::from_floats([[[2.0, 3.0]]], &device);
        let output = op.forward(&mut state, input).to_data();
        assert_eq!(output.as_slice::<f32>().unwrap(), &[4.0, 5.0]);
    }

    #[test]
    fn streaming_conv_transpose_stride_padding() {
        let device = NdArrayDevice::default();
        let weight = Tensor::<TestBackend, 3>::from_floats([[[1.0, 1.0]]], &device);
        let bias = Some(Tensor::<TestBackend, 1>::from_floats([0.0], &device));
        let config = StreamingConvConfig {
            kernel_size: 2,
            stride: 1,
            dilation: 1,
            padding: 0,
            pad_mode: PaddingMode::Constant,
            groups: 1,
        };
        let op = StreamingConvTranspose1dOp::new(config, weight, bias);
        let mut state = StreamingConv1d.init_state(1, 0);

        let input = Tensor::<TestBackend, 3>::from_floats([[[1.0, 1.0]]], &device);
        let output = op.forward(&mut state, input).unwrap().to_data();
        assert_eq!(output.as_slice::<f32>().unwrap(), &[1.0, 2.0]);

        let flush = op
            .forward(
                &mut state,
                Tensor::<TestBackend, 3>::zeros([1, 1, 0], &device),
            )
            .unwrap()
            .to_data();
        assert_eq!(flush.as_slice::<f32>().unwrap(), &[1.0]);
    }
}
