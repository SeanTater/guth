use crate::state::{StreamStep, StreamingModule};

#[derive(Debug, Default)]
pub struct StreamingMha;

#[derive(Debug, Clone)]
pub struct StreamingMhaState {
    pub step: StreamStep,
}

impl Default for StreamingMhaState {
    fn default() -> Self {
        Self {
            step: StreamStep::new(),
        }
    }
}

impl StreamingModule for StreamingMha {
    type State = StreamingMhaState;

    fn init_state(&self, _batch_size: usize, _sequence_length: usize) -> Self::State {
        StreamingMhaState::default()
    }

    fn increment_step(&self, state: &mut Self::State, increment: usize) {
        state.step.increment(increment);
    }
}

#[cfg(test)]
mod tests {
    use super::{StreamingMha, StreamingModule};

    #[test]
    fn streaming_mha_state_increments() {
        let module = StreamingMha::default();
        let mut state = module.init_state(1, 0);
        module.increment_step(&mut state, 2);
        assert_eq!(state.step.index, 2);
    }
}
