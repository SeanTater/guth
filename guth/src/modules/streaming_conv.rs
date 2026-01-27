use crate::state::{StreamStep, StreamingModule};
use anyhow::Result;

#[derive(Debug, Default)]
pub struct StreamingConv1d;

#[derive(Debug, Default)]
pub struct StreamingConvTranspose1d;

#[derive(Debug, Clone)]
pub struct StreamingConvState {
    pub step: StreamStep,
    pub history: Vec<Vec<f32>>,
}

impl Default for StreamingConvState {
    fn default() -> Self {
        Self {
            step: StreamStep::new(),
            history: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StreamingConvConfig {
    pub kernel_size: usize,
    pub stride: usize,
    pub dilation: usize,
    pub padding: usize,
    pub pad_mode: PaddingMode,
}

impl Default for StreamingConvConfig {
    fn default() -> Self {
        Self {
            kernel_size: 1,
            stride: 1,
            dilation: 1,
            padding: 0,
            pad_mode: PaddingMode::Constant,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaddingMode {
    Constant,
    Replicate,
}

#[derive(Debug, Clone)]
pub struct StreamingConvKernel {
    pub weights: Vec<Vec<f32>>,
    pub bias: Option<Vec<f32>>,
}

impl StreamingConvKernel {
    pub fn new(weights: Vec<Vec<f32>>, bias: Option<Vec<f32>>) -> Result<Self> {
        if weights.is_empty() || weights[0].is_empty() {
            anyhow::bail!("kernel weights cannot be empty");
        }
        Ok(Self { weights, bias })
    }
}

#[derive(Debug, Clone)]
pub struct StreamingConv1dOp {
    pub config: StreamingConvConfig,
    pub kernel: StreamingConvKernel,
}

impl StreamingConv1dOp {
    pub fn new(config: StreamingConvConfig, kernel: StreamingConvKernel) -> Self {
        Self { config, kernel }
    }

    pub fn forward(&self, state: &mut StreamingConvState, input: &[Vec<f32>]) -> Vec<Vec<f32>> {
        if input.is_empty() {
            return Vec::new();
        }

        let k = self.config.kernel_size;
        let dilation = self.config.dilation.max(1);
        let stride = self.config.stride.max(1);
        let padding = self.config.padding;
        let channels = input[0].len();

        let history_len = (k - 1) * dilation;
        let mut extended = Vec::with_capacity(state.history.len() + input.len());
        extended.extend_from_slice(&state.history);
        extended.extend_from_slice(input);

        let mut outputs = Vec::new();
        let total_len = extended.len();
        for idx in 0..input.len() {
            let position = state.history.len() + idx;
            if position % stride != 0 {
                continue;
            }

            let mut acc = vec![0.0f32; channels];
            for tap in 0..k {
                let offset = tap * dilation;
                let target_idx = position as isize - offset as isize;
                let target_idx = target_idx - padding as isize;
                let sample = if target_idx < 0 {
                    match self.config.pad_mode {
                        PaddingMode::Constant => vec![0.0f32; channels],
                        PaddingMode::Replicate => extended.first().cloned().unwrap_or(vec![0.0; channels]),
                    }
                } else if target_idx as usize >= total_len {
                    match self.config.pad_mode {
                        PaddingMode::Constant => vec![0.0f32; channels],
                        PaddingMode::Replicate => extended.last().cloned().unwrap_or(vec![0.0; channels]),
                    }
                } else {
                    extended[target_idx as usize].clone()
                };
                for ch in 0..channels {
                    let weight = self.kernel.weights[tap][ch];
                    acc[ch] += weight * sample[ch];
                }
            }
            if let Some(bias) = &self.kernel.bias {
                for ch in 0..channels {
                    acc[ch] += bias[ch];
                }
            }
            outputs.push(acc);
        }

        if history_len > 0 {
            let start = total_len.saturating_sub(history_len);
            state.history.clear();
            state.history.extend_from_slice(&extended[start..]);
        }

        outputs
    }
}

#[derive(Debug, Clone)]
pub struct StreamingConvTranspose1dOp {
    pub config: StreamingConvConfig,
    pub kernel: StreamingConvKernel,
}

impl StreamingConvTranspose1dOp {
    pub fn new(config: StreamingConvConfig, kernel: StreamingConvKernel) -> Self {
        Self { config, kernel }
    }

    pub fn forward(&self, state: &mut StreamingConvState, input: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        if input.is_empty() {
            if state.history.is_empty() {
                return Ok(Vec::new());
            }
            let tail = std::mem::take(&mut state.history);
            return Ok(tail);
        }

        let k = self.config.kernel_size;
        let channels = input[0].len();
        let output_len = (input.len().saturating_sub(1)) * self.config.stride + k;
        let output_len = output_len.saturating_sub(2 * self.config.padding);
        let mut output = vec![vec![0.0f32; channels]; output_len.max(1)];

        for (i, sample) in input.iter().enumerate() {
            let base = i * self.config.stride;
            for tap in 0..k {
                let out_idx = base + tap;
                if out_idx >= output_len {
                    continue;
                }
                for ch in 0..channels {
                    let weight = self.kernel.weights[tap][ch];
                    output[out_idx][ch] += sample[ch] * weight;
                }
            }
        }

        if !state.history.is_empty() {
            for (idx, value) in state.history.iter().enumerate() {
                if idx < output.len() {
                    for ch in 0..channels {
                        output[idx][ch] += value[ch];
                    }
                }
            }
        }

        let tail_len = k - 1;
        let emit_len = output.len().saturating_sub(tail_len);
        let mut emit = output[..emit_len].to_vec();
        let tail = output[emit_len..].to_vec();
        state.history = tail;

        if let Some(bias) = &self.kernel.bias {
            for frame in &mut emit {
                for ch in 0..channels {
                    frame[ch] += bias[ch];
                }
            }
        }

        Ok(emit)
    }
}

impl StreamingModule for StreamingConv1d {
    type State = StreamingConvState;

    fn init_state(&self, _batch_size: usize, _sequence_length: usize) -> Self::State {
        StreamingConvState::default()
    }

    fn increment_step(&self, state: &mut Self::State, increment: usize) {
        state.step.increment(increment);
    }
}

impl StreamingModule for StreamingConvTranspose1d {
    type State = StreamingConvState;

    fn init_state(&self, _batch_size: usize, _sequence_length: usize) -> Self::State {
        StreamingConvState::default()
    }

    fn increment_step(&self, state: &mut Self::State, increment: usize) {
        state.step.increment(increment);
    }
}

#[cfg(test)]
mod tests {
    use super::{
        PaddingMode,
        StreamingConv1d,
        StreamingConv1dOp,
        StreamingConvConfig,
        StreamingConvKernel,
        StreamingConvTranspose1dOp,
        StreamingModule,
    };

    #[test]
    fn streaming_conv_state_increments() {
        let module = StreamingConv1d::default();
        let mut state = module.init_state(1, 0);
        module.increment_step(&mut state, 3);
        assert_eq!(state.step.index, 3);
    }

    #[test]
    fn streaming_conv_forward_accumulates_history() {
        let config = StreamingConvConfig {
            kernel_size: 3,
            stride: 1,
            dilation: 1,
            padding: 0,
            pad_mode: PaddingMode::Constant,
        };
        let kernel =
            StreamingConvKernel::new(vec![vec![1.0], vec![0.0], vec![-1.0]], Some(vec![0.0]))
                .unwrap();
        let op = StreamingConv1dOp::new(config, kernel);
        let mut state = StreamingConv1d::default().init_state(1, 0);

        let first = op.forward(&mut state, &[vec![1.0], vec![2.0], vec![3.0]]);
        assert_eq!(first, vec![vec![1.0], vec![2.0], vec![2.0]]);

        let second = op.forward(&mut state, &[vec![4.0]]);
        assert_eq!(second, vec![vec![2.0]]);
    }

    #[test]
    fn streaming_conv_transpose_emits_with_tail() {
        let config = StreamingConvConfig {
            kernel_size: 2,
            stride: 1,
            dilation: 1,
            padding: 0,
            pad_mode: PaddingMode::Constant,
        };
        let kernel =
            StreamingConvKernel::new(vec![vec![1.0], vec![1.0]], Some(vec![0.5])).unwrap();
        let op = StreamingConvTranspose1dOp::new(config, kernel);
        let mut state = StreamingConv1d::default().init_state(1, 0);

        let first = op.forward(&mut state, &[vec![1.0]]).unwrap();
        assert_eq!(first, vec![vec![1.5]]);

        let second = op.forward(&mut state, &[vec![2.0]]).unwrap();
        assert_eq!(second, vec![vec![3.5]]);
    }

    #[test]
    fn streaming_conv_multi_channel() {
        let config = StreamingConvConfig {
            kernel_size: 2,
            stride: 1,
            dilation: 1,
            padding: 0,
            pad_mode: PaddingMode::Constant,
        };
        let kernel = StreamingConvKernel::new(
            vec![vec![1.0, -1.0], vec![0.5, 0.5]],
            Some(vec![0.0, 0.0]),
        )
        .unwrap();
        let op = StreamingConv1dOp::new(config, kernel);
        let mut state = StreamingConv1d::default().init_state(1, 0);

        let output = op.forward(&mut state, &[vec![2.0, 4.0], vec![1.0, 3.0]]);
        assert_eq!(output, vec![vec![2.0, -4.0], vec![2.0, -1.0]]);
    }

    #[test]
    fn streaming_conv_replicate_padding() {
        let config = StreamingConvConfig {
            kernel_size: 2,
            stride: 1,
            dilation: 1,
            padding: 1,
            pad_mode: PaddingMode::Replicate,
        };
        let kernel =
            StreamingConvKernel::new(vec![vec![1.0], vec![1.0]], Some(vec![0.0])).unwrap();
        let op = StreamingConv1dOp::new(config, kernel);
        let mut state = StreamingConv1d::default().init_state(1, 0);

        let output = op.forward(&mut state, &[vec![2.0], vec![3.0]]);
        assert_eq!(output, vec![vec![4.0], vec![4.0]]);
    }

    #[test]
    fn streaming_conv_transpose_stride_padding() {
        let config = StreamingConvConfig {
            kernel_size: 3,
            stride: 2,
            dilation: 1,
            padding: 1,
            pad_mode: PaddingMode::Constant,
        };
        let kernel =
            StreamingConvKernel::new(vec![vec![1.0], vec![1.0], vec![1.0]], Some(vec![0.0]))
                .unwrap();
        let op = StreamingConvTranspose1dOp::new(config, kernel);
        let mut state = StreamingConv1d::default().init_state(1, 0);

        let output = op.forward(&mut state, &[vec![1.0], vec![1.0]]).unwrap();
        assert_eq!(output, vec![vec![1.0]]);

        let flushed = op.forward(&mut state, &[]).unwrap();
        assert_eq!(flushed, vec![vec![1.0], vec![2.0]]);
    }
}
