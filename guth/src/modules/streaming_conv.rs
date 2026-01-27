use crate::state::{StreamStep, StreamingModule};

#[derive(Debug, Default)]
pub struct StreamingConv1d;

#[derive(Debug, Default)]
pub struct StreamingConvTranspose1d;

#[derive(Debug, Clone)]
pub struct StreamingConvState {
    pub step: StreamStep,
}

impl Default for StreamingConvState {
    fn default() -> Self {
        Self {
            step: StreamStep::new(),
        }
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
