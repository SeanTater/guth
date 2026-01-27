use crate::state::{StreamStep, StreamingModule};
use burn::tensor::backend::Backend;
use burn::tensor::module::attention;
use burn::tensor::{Bool, Int, Tensor};
use burn_nn::{RotaryEncoding, RotaryEncodingConfig};

#[derive(Debug, Default)]
pub struct StreamingMha;

#[derive(Debug, Clone)]
pub struct StreamingMhaState<B: Backend> {
    pub step: StreamStep,
    pub cached_tokens: usize,
    pub keys: Option<Tensor<B, 4>>,
    pub values: Option<Tensor<B, 4>>,
}

impl<B: Backend> Default for StreamingMhaState<B> {
    fn default() -> Self {
        Self {
            step: StreamStep::new(),
            cached_tokens: 0,
            keys: None,
            values: None,
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
    pub rope_max_seq: Option<usize>,
    pub rope_theta: f32,
}

impl Default for StreamingMhaConfig {
    fn default() -> Self {
        Self {
            max_cache_tokens: 0,
            num_heads: 1,
            head_dim: 1,
            context: None,
            causal: true,
            rope_max_seq: None,
            rope_theta: 10000.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StreamingMhaOp<B: Backend> {
    pub config: StreamingMhaConfig,
    pub rope: Option<RotaryEncoding<B>>,
}

impl<B: Backend> StreamingMhaOp<B> {
    pub fn new(config: StreamingMhaConfig, device: &B::Device) -> Self {
        let rope = config.rope_max_seq.map(|max_seq| {
            RotaryEncodingConfig::new(max_seq, config.head_dim)
                .with_theta(config.rope_theta)
                .init::<B>(device)
        });
        Self { config, rope }
    }

    pub fn append_kv(&self, state: &mut StreamingMhaState<B>, keys: Tensor<B, 4>, values: Tensor<B, 4>) {
        let added = keys.dims()[2];
        let combined_keys = match state.keys.take() {
            Some(existing) => Tensor::cat(vec![existing, keys], 2),
            None => keys,
        };
        let combined_values = match state.values.take() {
            Some(existing) => Tensor::cat(vec![existing, values], 2),
            None => values,
        };

        let mut keys = combined_keys;
        let mut values = combined_values;
        let total = keys.dims()[2];
        if self.config.max_cache_tokens > 0 && total > self.config.max_cache_tokens {
            let start = total - self.config.max_cache_tokens;
            keys = keys.narrow(2, start, self.config.max_cache_tokens);
            values = values.narrow(2, start, self.config.max_cache_tokens);
        }

        state.cached_tokens = keys.dims()[2];
        state.step.increment(added);
        state.keys = Some(keys);
        state.values = Some(values);
    }

    pub fn apply_rope(&self, tensor: Tensor<B, 4>, start: usize) -> Tensor<B, 4> {
        match &self.rope {
            Some(rope) => rope.apply(tensor, start),
            None => tensor,
        }
    }

    pub fn attention(&self, state: &StreamingMhaState<B>, queries: Tensor<B, 4>) -> Tensor<B, 4> {
        let keys = state.keys.clone().expect("keys must be set");
        let values = state.values.clone().expect("values must be set");
        let batch = queries.dims()[0];
        let heads = queries.dims()[1];
        let q_len = queries.dims()[2];
        let k_len = keys.dims()[2];
        let q_start = state.step.index.saturating_sub(q_len);
        let k_start = state.step.index.saturating_sub(state.cached_tokens);
        let mask = self.build_mask(
            batch,
            heads,
            q_len,
            q_start,
            k_len,
            k_start,
            &queries.device(),
        );
        attention(queries, keys, values, mask)
    }

    fn build_mask(
        &self,
        batch: usize,
        heads: usize,
        q_len: usize,
        q_start: usize,
        k_len: usize,
        k_start: usize,
        device: &B::Device,
    ) -> Option<Tensor<B, 4, Bool>> {
        if !self.config.causal && self.config.context.is_none() {
            return None;
        }

        let q: Tensor<B, 2, Int> =
            Tensor::<B, 1, Int>::arange(q_start as i64..(q_start + q_len) as i64, device)
                .unsqueeze_dim::<2>(1)
                .repeat_dim(1, k_len);
        let k: Tensor<B, 2, Int> =
            Tensor::<B, 1, Int>::arange(k_start as i64..(k_start + k_len) as i64, device)
                .unsqueeze_dim::<2>(0)
                .repeat_dim(0, q_len);
        let mut mask: Tensor<B, 2, Bool> = k.clone().greater(q.clone());

        if let Some(context) = self.config.context {
            let lower = q.sub_scalar((context.saturating_sub(1)) as i64);
            let lower_mask: Tensor<B, 2, Bool> = lower.greater(k);
            mask = mask.bool_or(lower_mask);
        }

        let mask = mask.reshape([1, 1, q_len, k_len]); 
        let mask = mask.repeat_dim(0, batch).repeat_dim(1, heads);
        Some(mask)
    }
}

impl<B: Backend> StreamingModule<B> for StreamingMha {
    type State = StreamingMhaState<B>;

    fn init_state(&self, _batch_size: usize, _sequence_length: usize) -> Self::State {
        StreamingMhaState::default()
    }

    fn increment_step(&self, state: &mut Self::State, increment: usize) {
        state.step.increment(increment);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use burn_ndarray::{NdArray, NdArrayDevice};

    type TestBackend = NdArray<f32>;

    #[test]
    fn streaming_mha_append_updates_cache() {
        let device = NdArrayDevice::default();
        let config = StreamingMhaConfig {
            num_heads: 1,
            head_dim: 2,
            ..Default::default()
        };
        let op = StreamingMhaOp::<TestBackend>::new(config, &device);
        let mut state = StreamingMha::default().init_state(1, 0);

        let keys = Tensor::<TestBackend, 4>::from_floats([[[[1.0, 0.0], [0.0, 1.0]]]], &device);
        let values = Tensor::<TestBackend, 4>::from_floats([[[[1.0, 0.0], [0.0, 1.0]]]], &device);
        op.append_kv(&mut state, keys, values);
        assert_eq!(state.cached_tokens, 2);
    }

    #[test]
    fn attention_uses_cached_values() {
        let device = NdArrayDevice::default();
        let config = StreamingMhaConfig {
            num_heads: 1,
            head_dim: 2,
            causal: false,
            ..Default::default()
        };
        let op = StreamingMhaOp::<TestBackend>::new(config, &device);
        let mut state = StreamingMha::default().init_state(1, 0);

        let keys = Tensor::<TestBackend, 4>::from_floats([[[[1.0, 0.0], [0.0, 1.0]]]], &device);
        let values = Tensor::<TestBackend, 4>::from_floats([[[[1.0, 0.0], [0.0, 1.0]]]], &device);
        op.append_kv(&mut state, keys, values);

        let queries = Tensor::<TestBackend, 4>::from_floats([[[[1.0, 0.0]]]], &device);
        let output = op.attention(&state, queries).to_data();
        let value = output.as_slice::<f32>().unwrap()[0];
        assert!(value > 0.6 && value < 0.8);
    }

    #[test]
    fn rope_rotation_changes_vector() {
        let device = NdArrayDevice::default();
        let config = StreamingMhaConfig {
            num_heads: 1,
            head_dim: 4,
            rope_max_seq: Some(8),
            ..Default::default()
        };
        let op = StreamingMhaOp::<TestBackend>::new(config, &device);
        let input = Tensor::<TestBackend, 4>::from_floats([[[[1.0, 0.0, 0.0, 1.0]]]], &device);
        let output = op.apply_rope(input, 1).to_data();
        assert_ne!(output, TensorData::from([[[[1.0, 0.0, 0.0, 1.0]]]]));
    }
}
