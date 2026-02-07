//! Streaming multi-head attention with RoPE support.
//!
//! This module caches key/value tensors so attention can be computed over
//! incremental chunks while preserving causality and optional context windows.

use crate::perf::{self, Metric};
use crate::state::StreamingModule;
use burn::tensor::{activation::softmax, backend::Backend, Bool, Int, Tensor};
use std::marker::PhantomData;

/// Marker type used to implement the [`StreamingModule`] trait.
#[derive(Debug, Default)]
pub struct StreamingMha;

/// Streaming attention state (cache + position).
#[derive(Debug, Clone)]
pub struct StreamingMhaState<B: Backend> {
    /// Current streaming step.
    pub step: usize,
    /// Number of cached key/value positions.
    pub cached_tokens: usize,
    /// Cached keys `[batch, heads, seq, dim]`.
    pub keys: Option<Tensor<B, 4>>,
    /// Cached values `[batch, heads, seq, dim]`.
    pub values: Option<Tensor<B, 4>>,
}

impl<B: Backend> Default for StreamingMhaState<B> {
    /// Create an empty attention cache with step counter at zero.
    fn default() -> Self {
        Self {
            step: 0,
            cached_tokens: 0,
            keys: None,
            values: None,
        }
    }
}

/// Configuration for streaming attention.
#[derive(Debug, Clone)]
pub struct StreamingMhaConfig {
    /// Maximum number of cached tokens (0 = unlimited).
    pub max_cache_tokens: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Optional sliding context window.
    pub context: Option<usize>,
    /// Enforce causal masking.
    pub causal: bool,
    /// Optional RoPE max sequence length clamp.
    pub rope_max_seq: Option<usize>,
    /// RoPE base period.
    pub rope_theta: f32,
}

impl Default for StreamingMhaConfig {
    /// Default to a minimal 1-head, causal attention config.
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

/// Streaming attention operator that owns configuration only.
#[derive(Debug, Clone)]
pub struct StreamingMhaOp<B: Backend> {
    /// Attention configuration.
    pub config: StreamingMhaConfig,
    _backend: PhantomData<B>,
}

impl<B: Backend> StreamingMhaOp<B> {
    /// Create a new streaming attention op.
    pub fn new(config: StreamingMhaConfig, device: &B::Device) -> Self {
        let _ = device;
        Self {
            config,
            _backend: PhantomData,
        }
    }

    /// Append key/value tensors to the cache and update the step counter.
    pub fn append_kv(
        &self,
        state: &mut StreamingMhaState<B>,
        keys: Tensor<B, 4>,
        values: Tensor<B, 4>,
    ) {
        let _span = perf::span(Metric::MhaAppendKv);
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
        state.step = state.step.saturating_add(added);
        state.keys = Some(keys);
        state.values = Some(values);
    }

    /// Apply rotary position embeddings (RoPE) to a tensor chunk.
    pub fn apply_rope(&self, tensor: Tensor<B, 4>, start: usize) -> Tensor<B, 4> {
        let _span = perf::span(Metric::MhaApplyRope);
        let [batch, heads, seq, dim] = tensor.dims();
        if dim < 2 {
            return tensor;
        }
        let rot_dim = dim - (dim % 2);
        let (rot, tail) = if rot_dim == dim {
            (tensor, None)
        } else {
            (
                tensor.clone().narrow(3, 0, rot_dim),
                Some(tensor.narrow(3, rot_dim, dim - rot_dim)),
            )
        };

        let half = rot_dim / 2;
        let device = rot.device();

        let scale = -self.config.rope_theta.ln() * 2.0 / rot_dim as f32;
        let ds = Tensor::<B, 1, Int>::arange(0..half as i64, &device).float();
        let freqs = ds.mul_scalar(scale).exp();

        let mut ts = Tensor::<B, 1, Int>::arange(0..seq as i64, &device)
            .float()
            .add_scalar(start as f32);
        if let Some(max_seq) = self.config.rope_max_seq {
            let max = max_seq.saturating_sub(1) as f32;
            ts = ts.clamp(0.0, max);
        }
        let angles = ts.unsqueeze_dim::<2>(1).mul(freqs.unsqueeze_dim::<2>(0));
        let rotr = angles.clone().cos();
        let roti = angles.sin();

        let rotr = rotr
            .unsqueeze_dim::<3>(0)
            .unsqueeze_dim::<4>(2)
            .repeat_dim(0, batch)
            .repeat_dim(2, heads);
        let roti = roti
            .unsqueeze_dim::<3>(0)
            .unsqueeze_dim::<4>(2)
            .repeat_dim(0, batch)
            .repeat_dim(2, heads);

        let reshaped = rot.swap_dims(1, 2).reshape([batch, seq, heads, half, 2]);
        let real = reshaped
            .clone()
            .narrow(4, 0, 1)
            .reshape([batch, seq, heads, half]);
        let imag = reshaped
            .clone()
            .narrow(4, 1, 1)
            .reshape([batch, seq, heads, half]);

        let rot_real = real
            .clone()
            .mul(rotr.clone())
            .sub(imag.clone().mul(roti.clone()));
        let rot_imag = real.mul(roti).add(imag.mul(rotr));

        let rotated = Tensor::cat(
            vec![
                rot_real.unsqueeze_dim::<5>(4),
                rot_imag.unsqueeze_dim::<5>(4),
            ],
            4,
        )
        .reshape([batch, seq, heads, rot_dim])
        .swap_dims(1, 2);

        match tail {
            Some(tail) => Tensor::cat(vec![rotated, tail], 3),
            None => rotated,
        }
    }

    /// Compute attention over cached keys/values for the given queries.
    pub fn attention(&self, state: &StreamingMhaState<B>, queries: Tensor<B, 4>) -> Tensor<B, 4> {
        let _span = perf::span(Metric::MhaAttention);
        let keys = state.keys.clone().expect("keys must be set");
        let values = state.values.clone().expect("values must be set");
        let batch = queries.dims()[0];
        let heads = queries.dims()[1];
        let q_len = queries.dims()[2];
        let k_len = keys.dims()[2];

        let q_start = state.step.saturating_sub(q_len);
        let k_start = state.step.saturating_sub(state.cached_tokens);
        let mask = self.build_mask(
            batch,
            heads,
            q_len,
            q_start,
            k_len,
            k_start,
            &queries.device(),
        );

        let scale = (self.config.head_dim as f32).sqrt();
        let scores = queries
            .matmul(keys.clone().swap_dims(2, 3))
            .div_scalar(scale);
        let scores = if let Some(mask) = mask {
            let neg = Tensor::<B, 4>::full(scores.dims(), -1.0e9, &scores.device());
            scores.mask_where(mask, neg)
        } else {
            scores
        };
        let weights = softmax(scores, 3);
        weights.matmul(values)
    }

    #[allow(clippy::too_many_arguments)]
    /// Build a causal/context mask for attention.
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
        let _span = perf::span(Metric::MhaBuildMask);

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

    /// Initialize an empty attention cache.
    fn init_state(&self, _batch_size: usize, _sequence_length: usize) -> Self::State {
        StreamingMhaState::default()
    }

    /// Increment the internal step counter.
    fn increment_step(&self, state: &mut Self::State, increment: usize) {
        state.step = state.step.saturating_add(increment);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData as BurnTensorData;
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
        let mut state = StreamingMha.init_state(1, 0);

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
        let mut state = StreamingMha.init_state(1, 0);

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
        assert_ne!(output, BurnTensorData::from([[[[1.0, 0.0, 0.0, 1.0]]]]));
    }
}
