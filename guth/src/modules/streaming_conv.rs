use crate::state::{StreamStep, StreamingModule};
use anyhow::Result;

#[derive(Debug, Default)]
pub struct StreamingConv1d;

#[derive(Debug, Default)]
pub struct StreamingConvTranspose1d;

#[derive(Debug, Clone)]
pub struct StreamingConvState {
    pub step: StreamStep,
    pub history: Vec<f32>,
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
}

impl Default for StreamingConvConfig {
    fn default() -> Self {
        Self {
            kernel_size: 1,
            stride: 1,
            dilation: 1,
            padding: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StreamingConvKernel {
    pub weights: Vec<f32>,
    pub bias: Option<f32>,
}

impl StreamingConvKernel {
    pub fn new(weights: Vec<f32>, bias: Option<f32>) -> Result<Self> {
        if weights.is_empty() {
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

    pub fn forward(&self, state: &mut StreamingConvState, input: &[f32]) -> Vec<f32> {
        if input.is_empty() {
            return Vec::new();
        }

        let k = self.config.kernel_size;
        let dilation = self.config.dilation.max(1);
        let stride = self.config.stride.max(1);
        let padding = self.config.padding;

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

            let mut acc = 0.0f32;
            for tap in 0..k {
                let offset = tap * dilation;
                let target_idx = position as isize - offset as isize;
                let target_idx = target_idx - padding as isize;
                let sample = if target_idx < 0 || target_idx as usize >= total_len {
                    0.0
                } else {
                    extended[target_idx as usize]
                };
                let weight = self.kernel.weights[tap];
                acc += weight * sample;
            }
            if let Some(bias) = self.kernel.bias {
                acc += bias;
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

    pub fn forward(&self, state: &mut StreamingConvState, input: &[f32]) -> Result<Vec<f32>> {
        if input.is_empty() {
            return Ok(Vec::new());
        }

        if self.config.stride != 1 || self.config.dilation != 1 || self.config.padding != 0 {
            anyhow::bail!("transpose conv only supports stride=1, dilation=1, padding=0");
        }

        let k = self.config.kernel_size;
        let mut output = vec![0.0f32; input.len() + k - 1];

        for (i, sample) in input.iter().enumerate() {
            for (tap, weight) in self.kernel.weights.iter().enumerate() {
                output[i + tap] += sample * weight;
            }
        }

        if !state.history.is_empty() {
            for (idx, value) in state.history.iter().enumerate() {
                if idx < output.len() {
                    output[idx] += value;
                }
            }
        }

        let tail_len = k - 1;
        let emit_len = output.len().saturating_sub(tail_len);
        let mut emit = output[..emit_len].to_vec();
        let tail = output[emit_len..].to_vec();
        state.history = tail;

        if let Some(bias) = self.kernel.bias {
            for value in &mut emit {
                *value += bias;
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
        };
        let kernel = StreamingConvKernel::new(vec![1.0, 0.0, -1.0], Some(0.0)).unwrap();
        let op = StreamingConv1dOp::new(config, kernel);
        let mut state = StreamingConv1d::default().init_state(1, 0);

        let first = op.forward(&mut state, &[1.0, 2.0, 3.0]);
        assert_eq!(first, vec![1.0, 2.0, 2.0]);

        let second = op.forward(&mut state, &[4.0]);
        assert_eq!(second, vec![2.0]);
    }

    #[test]
    fn streaming_conv_transpose_emits_with_tail() {
        let config = StreamingConvConfig {
            kernel_size: 2,
            stride: 1,
            dilation: 1,
            padding: 0,
        };
        let kernel = StreamingConvKernel::new(vec![1.0, 1.0], Some(0.5)).unwrap();
        let op = StreamingConvTranspose1dOp::new(config, kernel);
        let mut state = StreamingConv1d::default().init_state(1, 0);

        let first = op.forward(&mut state, &[1.0]).unwrap();
        assert_eq!(first, vec![1.5]);

        let second = op.forward(&mut state, &[2.0]).unwrap();
        assert_eq!(second, vec![3.5]);
    }
}
