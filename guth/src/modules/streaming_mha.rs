use crate::state::{StreamStep, StreamingModule};

#[derive(Debug, Default)]
pub struct StreamingMha;

#[derive(Debug, Clone)]
pub struct StreamingMhaState {
    pub step: StreamStep,
    pub cached_tokens: usize,
    pub keys: Vec<Vec<f32>>,
    pub values: Vec<Vec<f32>>,
}

impl Default for StreamingMhaState {
    fn default() -> Self {
        Self {
            step: StreamStep::new(),
            cached_tokens: 0,
            keys: Vec::new(),
            values: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StreamingMhaConfig {
    pub max_cache_tokens: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub context: Option<usize>,
    pub causal: bool,
}

impl Default for StreamingMhaConfig {
    fn default() -> Self {
        Self {
            max_cache_tokens: 0,
            num_heads: 1,
            head_dim: 1,
            context: None,
            causal: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StreamingMhaOp {
    pub config: StreamingMhaConfig,
}

impl StreamingMhaOp {
    pub fn new(config: StreamingMhaConfig) -> Self {
        Self { config }
    }

    pub fn append_kv(&self, state: &mut StreamingMhaState, keys: &[Vec<f32>], values: &[Vec<f32>]) {
        state.keys.extend_from_slice(keys);
        state.values.extend_from_slice(values);
        state.cached_tokens = state.keys.len();
        if self.config.max_cache_tokens > 0 && state.keys.len() > self.config.max_cache_tokens {
            let overflow = state.keys.len() - self.config.max_cache_tokens;
            state.keys.drain(0..overflow);
            state.values.drain(0..overflow);
        }
        state.cached_tokens = state.keys.len();
        state.step.increment(keys.len());
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

    pub fn windowed_mask(query_len: usize, key_len: usize, context: usize) -> Vec<Vec<bool>> {
        let mut mask = vec![vec![false; key_len]; query_len];
        for q in 0..query_len {
            let start = q.saturating_sub(context.saturating_sub(1));
            for k in start..=q.min(key_len.saturating_sub(1)) {
                mask[q][k] = true;
            }
        }
        mask
    }

    pub fn apply_rope(vec: &mut [f32], head_dim: usize, position: usize, base: f32) {
        let half = head_dim / 2;
        for i in 0..half {
            let freq = base.powf(-2.0 * i as f32 / head_dim as f32);
            let angle = position as f32 * freq;
            let (sin, cos) = angle.sin_cos();
            let x1 = vec[i];
            let x2 = vec[i + half];
            vec[i] = x1 * cos - x2 * sin;
            vec[i + half] = x1 * sin + x2 * cos;
        }
    }

    pub fn attention(&self, state: &StreamingMhaState, queries: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let keys = &state.keys;
        let values = &state.values;
        let key_len = keys.len();
        if key_len == 0 {
            return vec![vec![0.0; self.config.num_heads * self.config.head_dim]; queries.len()];
        }

        let mask = if let Some(context) = self.config.context {
            Self::windowed_mask(queries.len(), key_len, context)
        } else if self.config.causal {
            Self::causal_mask(queries.len(), key_len)
        } else {
            vec![vec![true; key_len]; queries.len()]
        };

        let mut outputs = Vec::with_capacity(queries.len());
        for (q_idx, query) in queries.iter().enumerate() {
            let mut out = vec![0.0f32; self.config.num_heads * self.config.head_dim];
            for head in 0..self.config.num_heads {
                let q_start = head * self.config.head_dim;
                let q_slice = &query[q_start..q_start + self.config.head_dim];
                let mut scores = Vec::with_capacity(key_len);
                for (k_idx, key) in keys.iter().enumerate() {
                    let k_slice = &key[q_start..q_start + self.config.head_dim];
                    let mut score = 0.0f32;
                    for i in 0..self.config.head_dim {
                        score += q_slice[i] * k_slice[i];
                    }
                    scores.push(if mask[q_idx][k_idx] { score } else { f32::NEG_INFINITY });
                }
                let max_score = scores
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                let mut exp_sum = 0.0f32;
                let mut weights = vec![0.0f32; key_len];
                for (i, score) in scores.iter().enumerate() {
                    if *score == f32::NEG_INFINITY {
                        continue;
                    }
                    let value = (*score - max_score).exp();
                    weights[i] = value;
                    exp_sum += value;
                }
                if exp_sum == 0.0 {
                    continue;
                }
                for i in 0..key_len {
                    weights[i] /= exp_sum;
                }
                for (k_idx, value) in values.iter().enumerate() {
                    let v_slice = &value[q_start..q_start + self.config.head_dim];
                    let weight = weights[k_idx];
                    for i in 0..self.config.head_dim {
                        out[q_start + i] += weight * v_slice[i];
                    }
                }
            }
            outputs.push(out);
        }
        outputs
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
    use super::{StreamingMha, StreamingMhaConfig, StreamingMhaOp, StreamingModule};

    #[test]
    fn streaming_mha_state_increments() {
        let module = StreamingMha::default();
        let mut state = module.init_state(1, 0);
        module.increment_step(&mut state, 2);
        assert_eq!(state.step.index, 2);
    }

    #[test]
    fn streaming_mha_append_updates_cache() {
        let op = StreamingMhaOp::new(StreamingMhaConfig::default());
        let mut state = StreamingMha::default().init_state(1, 0);
        let keys = vec![vec![1.0], vec![2.0], vec![3.0]];
        let values = vec![vec![1.0], vec![2.0], vec![3.0]];
        op.append_kv(&mut state, &keys, &values);
        assert_eq!(state.cached_tokens, 3);
        assert_eq!(state.step.index, 3);
    }

    #[test]
    fn causal_mask_allows_past_and_self() {
        let mask = StreamingMhaOp::causal_mask(3, 3);
        assert!(mask[0][0]);
        assert!(!mask[0][1]);
        assert!(mask[2][0]);
        assert!(mask[2][2]);
    }

    #[test]
    fn windowed_mask_limits_context() {
        let mask = StreamingMhaOp::windowed_mask(3, 3, 2);
        assert!(!mask[0][2]);
        assert!(mask[2][1]);
    }

    #[test]
    fn rope_rotation_changes_vector() {
        let mut vec = vec![1.0f32, 0.0, 0.0, 1.0];
        StreamingMhaOp::apply_rope(&mut vec, 4, 1, 10000.0);
        assert_ne!(vec[0], 1.0);
        assert_ne!(vec[3], 1.0);
    }

    #[test]
    fn attention_uses_cached_values() {
        let mut state = StreamingMha::default().init_state(1, 0);
        let config = StreamingMhaConfig {
            max_cache_tokens: 0,
            num_heads: 1,
            head_dim: 2,
            context: None,
            causal: false,
        };
        let op = StreamingMhaOp::new(config);
        let keys = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let values = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        op.append_kv(&mut state, &keys, &values);

        let output = op.attention(&state, &[vec![1.0, 0.0]]);
        assert!((output[0][0] - 0.73).abs() < 0.05);
    }
}
