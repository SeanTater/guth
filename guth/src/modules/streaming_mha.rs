use crate::state::{StreamStep, StreamingModule};

#[derive(Debug, Default)]
pub struct StreamingMha;

#[derive(Debug, Clone)]
pub struct StreamingMhaState {
    pub step: StreamStep,
    pub cached_tokens: usize,
}

impl Default for StreamingMhaState {
    fn default() -> Self {
        Self {
            step: StreamStep::new(),
            cached_tokens: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StreamingMhaConfig {
    pub max_cache_tokens: usize,
}

impl Default for StreamingMhaConfig {
    fn default() -> Self {
        Self { max_cache_tokens: 0 }
    }
}

impl StreamingMha {
    pub fn append_tokens(state: &mut StreamingMhaState, tokens: usize) {
        state.cached_tokens = state.cached_tokens.saturating_add(tokens);
        state.step.increment(tokens);
    }

    pub fn causal_mask(query_len: usize, key_len: usize) -> Vec<Vec<bool>> {
        let mut mask = vec![vec![false; key_len]; query_len];
        for q in 0..query_len {
            for k in 0..key_len {
                mask[q][k] = k <= q;
            }
        }
        mask
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

    #[test]
    fn streaming_mha_append_updates_cache() {
        let mut state = StreamingMha::default().init_state(1, 0);
        StreamingMha::append_tokens(&mut state, 3);
        assert_eq!(state.cached_tokens, 3);
        assert_eq!(state.step.index, 3);
    }

    #[test]
    fn causal_mask_allows_past_and_self() {
        let mask = StreamingMha::causal_mask(3, 3);
        assert!(mask[0][0]);
        assert!(!mask[0][1]);
        assert!(mask[2][0]);
        assert!(mask[2][2]);
    }
}
