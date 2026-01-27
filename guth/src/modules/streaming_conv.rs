use crate::state::{StreamStep, StreamingModule};
use anyhow::Result;
use burn::tensor::backend::Backend;
use burn::tensor::module::{conv1d, conv_transpose1d};
use burn::tensor::ops::{ConvOptions, ConvTransposeOptions, PadMode};
use burn::tensor::{s, Tensor};

#[derive(Debug, Default)]
pub struct StreamingConv1d;

#[derive(Debug, Default)]
pub struct StreamingConvTranspose1d;

#[derive(Debug, Clone)]
pub struct StreamingConvState<B: Backend> {
    pub step: StreamStep,
    pub history: Option<Tensor<B, 3>>,
}

impl<B: Backend> Default for StreamingConvState<B> {
    fn default() -> Self {
        Self {
            step: StreamStep::new(),
            history: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaddingMode {
    Constant,
    Replicate,
}

#[derive(Debug, Clone)]
pub struct StreamingConvConfig {
    pub kernel_size: usize,
    pub stride: usize,
    pub dilation: usize,
    pub padding: usize,
    pub pad_mode: PaddingMode,
    pub groups: usize,
}

impl Default for StreamingConvConfig {
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

#[derive(Debug, Clone)]
pub struct StreamingConv1dOp<B: Backend> {
    pub config: StreamingConvConfig,
    pub weight: Tensor<B, 3>,
    pub bias: Option<Tensor<B, 1>>,
}

impl<B: Backend> StreamingConv1dOp<B> {
    pub fn new(config: StreamingConvConfig, weight: Tensor<B, 3>, bias: Option<Tensor<B, 1>>) -> Self {
        Self {
            config,
            weight,
            bias,
        }
    }

    pub fn forward(&self, state: &mut StreamingConvState<B>, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let input_len = input.dims()[2];
        if input_len == 0 {
            return input;
        }

        let history_len = state.history.as_ref().map(|h| h.dims()[2]).unwrap_or(0);
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

        let effective_kernel = self.config.dilation * (self.config.kernel_size.saturating_sub(1)) + 1;
        let history_keep = effective_kernel.saturating_sub(self.config.stride);
        if history_keep > 0 {
            let extended_len = extended.dims()[2];
            let start = extended_len.saturating_sub(history_keep);
            state.history = Some(extended.narrow(2, start, history_keep));
        } else {
            state.history = None;
        }

        output
    }
}

#[derive(Debug, Clone)]
pub struct StreamingConvTranspose1dOp<B: Backend> {
    pub config: StreamingConvConfig,
    pub weight: Tensor<B, 3>,
    pub bias: Option<Tensor<B, 1>>,
}

impl<B: Backend> StreamingConvTranspose1dOp<B> {
    pub fn new(config: StreamingConvConfig, weight: Tensor<B, 3>, bias: Option<Tensor<B, 1>>) -> Self {
        Self {
            config,
            weight,
            bias,
        }
    }

    pub fn forward(&self, state: &mut StreamingConvState<B>, input: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
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
        state.history = Some(tail);

        Ok(emit)
    }
}

impl<B: Backend> StreamingModule<B> for StreamingConv1d {
    type State = StreamingConvState<B>;

    fn init_state(&self, _batch_size: usize, _sequence_length: usize) -> Self::State {
        StreamingConvState::default()
    }

    fn increment_step(&self, state: &mut Self::State, increment: usize) {
        state.step.increment(increment);
    }
}

impl<B: Backend> StreamingModule<B> for StreamingConvTranspose1d {
    type State = StreamingConvState<B>;

    fn init_state(&self, _batch_size: usize, _sequence_length: usize) -> Self::State {
        StreamingConvState::default()
    }

    fn increment_step(&self, state: &mut Self::State, increment: usize) {
        state.step.increment(increment);
    }
}

fn conv_output_len(input_len: usize, kernel: usize, stride: usize, dilation: usize, padding: usize) -> usize {
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
        let mut state = StreamingConv1d::default().init_state(1, 0);

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
        let mut state = StreamingConv1d::default().init_state(1, 0);

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
        let mut state = StreamingConv1d::default().init_state(1, 0);

        let input = Tensor::<TestBackend, 3>::from_floats([[[1.0, 1.0]]], &device);
        let output = op.forward(&mut state, input).unwrap().to_data();
        assert_eq!(output.as_slice::<f32>().unwrap(), &[1.0, 2.0]);

        let flush = op
            .forward(&mut state, Tensor::<TestBackend, 3>::zeros([1, 1, 0], &device))
            .unwrap()
            .to_data();
        assert_eq!(flush.as_slice::<f32>().unwrap(), &[1.0]);
    }
}
