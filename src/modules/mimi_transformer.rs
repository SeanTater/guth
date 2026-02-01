//! Transformer blocks used inside the Mimi codec.
//!
//! Mimi uses a small transformer to model temporal relationships in the latent
//! frame sequence. This module implements those layers with streaming support.

use crate::modules::layer_scale::{LayerScale, LayerScaleConfig};
use crate::modules::linear::{
    apply_layer_norm_3d as apply_layer_norm, apply_linear_3d as apply_linear, merge_heads,
    split_qkv,
};
use burn::tensor::{activation::gelu, backend::Backend, module::attention, Bool, Int, Tensor};
use burn_nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};

/// Configuration for the Mimi transformer stack.
#[derive(Debug, Clone)]
pub struct MimiTransformerConfig {
    /// Model width.
    pub d_model: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// LayerScale init value.
    pub layer_scale: f32,
    /// Attention context window.
    pub context: usize,
    /// RoPE max period.
    pub max_period: f32,
    /// Feed-forward hidden size.
    pub dim_feedforward: usize,
}

/// Configuration for a projected Mimi transformer (input/output projections).
#[derive(Debug, Clone)]
pub struct MimiProjectedTransformerConfig {
    /// Input dimension before projection.
    pub input_dim: usize,
    /// Output projection dimensions.
    pub output_dims: Vec<usize>,
    /// Core transformer config.
    pub transformer: MimiTransformerConfig,
}

/// Self-attention module with RoPE and context window.
#[derive(Debug, Clone)]
pub struct MimiSelfAttention<B: Backend> {
    /// Combined QKV projection.
    pub in_proj: Linear<B>,
    /// Output projection.
    pub out_proj: Linear<B>,
    /// Number of heads.
    pub num_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Context window length.
    pub context: usize,
    /// RoPE max period.
    pub max_period: f32,
}

/// Streaming state for Mimi self-attention.
#[derive(Debug, Clone)]
pub struct MimiSelfAttentionState<B: Backend> {
    /// Current offset for RoPE positions.
    pub offset: Tensor<B, 1, Int>,
    /// Cached keys.
    pub keys: Option<Tensor<B, 4>>,
    /// Cached values.
    pub values: Option<Tensor<B, 4>>,
}

/// Single transformer layer used by Mimi.
#[derive(Debug, Clone)]
pub struct MimiTransformerLayer<B: Backend> {
    pub self_attn: MimiSelfAttention<B>,
    pub norm1: LayerNorm<B>,
    pub norm2: LayerNorm<B>,
    pub linear1: Linear<B>,
    pub linear2: Linear<B>,
    pub layer_scale_1: Option<LayerScale<B>>,
    pub layer_scale_2: Option<LayerScale<B>>,
}

/// Streaming state for a Mimi transformer layer.
#[derive(Debug, Clone)]
pub struct MimiTransformerLayerState<B: Backend> {
    pub self_attn: MimiSelfAttentionState<B>,
}

/// Streaming state for a full Mimi transformer stack.
#[derive(Debug, Clone)]
pub struct MimiTransformerState<B: Backend> {
    pub layers: Vec<MimiTransformerLayerState<B>>,
}

/// Transformer stack for Mimi latent sequences.
#[derive(Debug, Clone)]
pub struct MimiTransformer<B: Backend> {
    pub layers: Vec<MimiTransformerLayer<B>>,
}

/// Output projection selection for projected transformers.
#[derive(Debug, Clone)]
pub enum ProjectedOutput<B: Backend> {
    /// Pass-through output (no projection).
    Identity,
    /// Linear projection to a different dimension.
    Linear(Linear<B>),
}

/// Transformer with optional input and multiple output projections.
#[derive(Debug, Clone)]
pub struct MimiProjectedTransformer<B: Backend> {
    pub input_proj: Option<Linear<B>>,
    pub output_projs: Vec<ProjectedOutput<B>>,
    pub transformer: MimiTransformer<B>,
}

/// Compute absolute positions from an offset vector.
fn positions_from_offset<B: Backend>(
    offset: Tensor<B, 1, Int>,
    length: usize,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let batch = offset.dims()[0];
    let positions = Tensor::<B, 1, Int>::arange(0..length as i64, device).unsqueeze_dim::<2>(0);
    let positions = positions.repeat_dim(0, batch);
    let offset = offset.unsqueeze_dim::<2>(1).repeat_dim(1, length);
    positions.add(offset)
}

/// Apply rotary position embeddings to query/key tensors.
fn apply_rope<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    offset: Tensor<B, 1, Int>,
    max_period: f32,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let [batch, seq, heads, dim] = q.dims();
    let half = dim / 2;
    let device = q.device();

    let scale = -max_period.ln() * 2.0 / dim as f32;
    let ds = Tensor::<B, 1, Int>::arange(0..half as i64, &device).float();
    let freqs = ds.mul_scalar(scale).exp();

    let ts = Tensor::<B, 1, Int>::arange(0..seq as i64, &device)
        .float()
        .unsqueeze_dim::<2>(0);
    let ts = ts.repeat_dim(0, batch);
    let offset = offset.float().unsqueeze_dim::<2>(1).repeat_dim(1, seq);
    let angles = ts
        .add(offset)
        .unsqueeze_dim::<3>(2)
        .mul(freqs.unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0));

    let rotr = angles.clone().cos();
    let roti = angles.sin();
    let rotr = rotr.unsqueeze_dim::<4>(2).repeat_dim(2, heads);
    let roti = roti.unsqueeze_dim::<4>(2).repeat_dim(2, heads);

    let q = q.reshape([batch, seq, heads, half, 2]);
    let k = k.reshape([batch, seq, heads, half, 2]);

    let qr = q.clone().narrow(4, 0, 1).reshape([batch, seq, heads, half]);
    let qi = q.clone().narrow(4, 1, 1).reshape([batch, seq, heads, half]);
    let kr = k.clone().narrow(4, 0, 1).reshape([batch, seq, heads, half]);
    let ki = k.clone().narrow(4, 1, 1).reshape([batch, seq, heads, half]);

    let qor = qr
        .clone()
        .mul(rotr.clone())
        .sub(qi.clone().mul(roti.clone()));
    let qoi = qr.mul(roti.clone()).add(qi.mul(rotr.clone()));
    let kor = kr
        .clone()
        .mul(rotr.clone())
        .sub(ki.clone().mul(roti.clone()));
    let koi = kr.mul(roti).add(ki.mul(rotr));

    let qo = Tensor::cat(
        vec![qor.unsqueeze_dim::<5>(4), qoi.unsqueeze_dim::<5>(4)],
        4,
    )
    .reshape([batch, seq, heads, dim]);
    let ko = Tensor::cat(
        vec![kor.unsqueeze_dim::<5>(4), koi.unsqueeze_dim::<5>(4)],
        4,
    )
    .reshape([batch, seq, heads, dim]);

    (qo, ko)
}

impl<B: Backend + 'static> MimiSelfAttention<B> {
    /// Initialize a fresh streaming attention state.
    fn init_state(&self, batch_size: usize, device: &B::Device) -> MimiSelfAttentionState<B> {
        MimiSelfAttentionState {
            offset: Tensor::<B, 1, Int>::zeros([batch_size], device),
            keys: None,
            values: None,
        }
    }

    /// Advance the internal offset by `increment`.
    fn increment_step(&self, state: &mut MimiSelfAttentionState<B>, increment: usize) {
        state.offset = state.offset.clone().add_scalar(increment as i64);
    }

    /// Run attention over an input chunk and update the cache.
    fn forward(&self, input: Tensor<B, 3>, state: &mut MimiSelfAttentionState<B>) -> Tensor<B, 3> {
        let projected = apply_linear(&self.in_proj, input);

        let (q, k, v) = split_qkv(projected, self.num_heads, self.head_dim);

        let (q_rot, k_rot) = apply_rope(
            q.swap_dims(1, 2),
            k.swap_dims(1, 2),
            state.offset.clone(),
            self.max_period,
        );

        let q = q_rot.swap_dims(1, 2);
        let k = k_rot.swap_dims(1, 2);

        // Track whether we had cached keys (for context trimming decision)
        let had_cached_keys = state.keys.is_some();

        let mut keys = match state.keys.take() {
            Some(existing) => Tensor::cat(vec![existing, k], 2),
            None => k,
        };
        let mut values = match state.values.take() {
            Some(existing) => Tensor::cat(vec![existing, v], 2),
            None => v,
        };

        // Only trim to context window when we have accumulated cached keys.
        // On the first forward (batch processing), we need all keys for proper
        // causal attention within the sequence.
        let total_k = keys.dims()[2];
        if had_cached_keys && self.context > 0 && total_k > self.context {
            let start = total_k - self.context;
            keys = keys.narrow(2, start, self.context);
            values = values.narrow(2, start, self.context);
        }

        let k_len = keys.dims()[2];
        let q_len = q.dims()[2];
        let pos_q = positions_from_offset(state.offset.clone(), q_len, &keys.device());
        let start_k = state
            .offset
            .clone()
            .add_scalar(q_len as i64)
            .sub_scalar(k_len as i64);
        let pos_k = positions_from_offset(start_k, k_len, &keys.device());

        let pos_k_valid = pos_k.clone().greater_elem(-1);
        let pos_k_valid = pos_k_valid.unsqueeze_dim::<3>(1).repeat_dim(1, q_len);
        let delta = pos_q.unsqueeze_dim::<3>(2).sub(pos_k.unsqueeze_dim::<3>(1));
        let delta_nonneg = delta.clone().greater_elem(-1);
        let mut allowed: Tensor<B, 3, Bool> = pos_k_valid.bool_and(delta_nonneg);
        if self.context > 0 {
            let too_far = delta.greater_elem((self.context.saturating_sub(1)) as i64);
            allowed = allowed.bool_and(too_far.bool_not());
        }

        let heads = q.dims()[1];
        let allowed = allowed.unsqueeze_dim::<4>(1).repeat_dim(1, heads);
        let mask = allowed.bool_not();

        let attended = attention(q, keys.clone(), values.clone(), Some(mask));
        state.keys = Some(keys);
        state.values = Some(values);

        apply_linear(&self.out_proj, merge_heads(attended))
    }
}

impl<B: Backend + 'static> MimiTransformerLayer<B> {
    /// Construct a transformer layer from config.
    pub fn new(config: &MimiTransformerConfig, device: &B::Device) -> Self {
        let head_dim = config.d_model / config.num_heads;
        assert!(head_dim.is_multiple_of(2), "RoPE head_dim must be even");

        let in_proj = LinearConfig::new(config.d_model, config.d_model * 3)
            .with_bias(false)
            .init::<B>(device);
        let out_proj = LinearConfig::new(config.d_model, config.d_model)
            .with_bias(false)
            .init::<B>(device);
        let self_attn = MimiSelfAttention {
            in_proj,
            out_proj,
            num_heads: config.num_heads,
            head_dim,
            context: config.context,
            max_period: config.max_period,
        };

        let norm1 = LayerNormConfig::new(config.d_model)
            .with_epsilon(1e-5)
            .init::<B>(device);
        let norm2 = LayerNormConfig::new(config.d_model)
            .with_epsilon(1e-5)
            .init::<B>(device);

        let linear1 = LinearConfig::new(config.d_model, config.dim_feedforward)
            .with_bias(false)
            .init::<B>(device);
        let linear2 = LinearConfig::new(config.dim_feedforward, config.d_model)
            .with_bias(false)
            .init::<B>(device);

        let layer_scale_1 =
            Some(LayerScaleConfig::new(config.d_model, config.layer_scale).init(device));
        let layer_scale_2 =
            Some(LayerScaleConfig::new(config.d_model, config.layer_scale).init(device));

        Self {
            self_attn,
            norm1,
            norm2,
            linear1,
            linear2,
            layer_scale_1,
            layer_scale_2,
        }
    }

    /// Initialize streaming state for this layer.
    pub fn init_state(
        &self,
        batch_size: usize,
        device: &B::Device,
    ) -> MimiTransformerLayerState<B> {
        MimiTransformerLayerState {
            self_attn: self.self_attn.init_state(batch_size, device),
        }
    }

    /// Increment the attention position by `increment`.
    pub fn increment_step(&self, state: &mut MimiTransformerLayerState<B>, increment: usize) {
        self.self_attn
            .increment_step(&mut state.self_attn, increment);
    }

    /// Forward pass for a single transformer layer.
    pub fn forward(
        &self,
        input: Tensor<B, 3>,
        state: &mut MimiTransformerLayerState<B>,
    ) -> Tensor<B, 3> {
        let residual = input.clone();
        let normalized = apply_layer_norm(&self.norm1, input);
        let update = self.self_attn.forward(normalized, &mut state.self_attn);
        let update = match &self.layer_scale_1 {
            Some(scale) => scale.apply(update),
            None => update,
        };
        let hidden = residual.add(update);

        let ffn_input = apply_layer_norm(&self.norm2, hidden.clone());
        let ffn = apply_linear(&self.linear1, ffn_input);
        let ffn = gelu(ffn);
        let ffn = apply_linear(&self.linear2, ffn);
        let ffn = match &self.layer_scale_2 {
            Some(scale) => scale.apply(ffn),
            None => ffn,
        };

        hidden.add(ffn)
    }
}

impl<B: Backend + 'static> MimiTransformer<B> {
    /// Construct a transformer stack from config.
    pub fn new(config: MimiTransformerConfig, device: &B::Device) -> Self {
        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            layers.push(MimiTransformerLayer::new(&config, device));
        }
        Self { layers }
    }

    /// Initialize streaming state for all layers.
    pub fn init_state(&self, batch_size: usize, device: &B::Device) -> MimiTransformerState<B> {
        let layers = self
            .layers
            .iter()
            .map(|layer| layer.init_state(batch_size, device))
            .collect();
        MimiTransformerState { layers }
    }

    /// Increment all layer positions.
    pub fn increment_step(&self, state: &mut MimiTransformerState<B>, increment: usize) {
        for (layer, layer_state) in self.layers.iter().zip(state.layers.iter_mut()) {
            layer.increment_step(layer_state, increment);
        }
    }

    /// Forward pass through the transformer stack.
    pub fn forward(
        &self,
        mut input: Tensor<B, 3>,
        state: &mut MimiTransformerState<B>,
    ) -> Tensor<B, 3> {
        for (layer, layer_state) in self.layers.iter().zip(state.layers.iter_mut()) {
            input = layer.forward(input, layer_state);
        }
        input
    }
}

impl<B: Backend + 'static> MimiProjectedTransformer<B> {
    /// Construct a projected transformer from config.
    pub fn new(config: MimiProjectedTransformerConfig, device: &B::Device) -> Self {
        let transformer = MimiTransformer::new(config.transformer.clone(), device);
        let input_proj = if config.input_dim == config.transformer.d_model {
            None
        } else {
            Some(
                LinearConfig::new(config.input_dim, config.transformer.d_model)
                    .with_bias(false)
                    .init::<B>(device),
            )
        };

        let mut output_projs = Vec::with_capacity(config.output_dims.len());
        for output_dim in config.output_dims.iter().copied() {
            if output_dim == config.transformer.d_model {
                output_projs.push(ProjectedOutput::Identity);
            } else {
                let proj = LinearConfig::new(config.transformer.d_model, output_dim)
                    .with_bias(false)
                    .init::<B>(device);
                output_projs.push(ProjectedOutput::Linear(proj));
            }
        }

        Self {
            input_proj,
            output_projs,
            transformer,
        }
    }

    /// Initialize transformer state for streaming.
    pub fn init_state(&self, batch_size: usize, device: &B::Device) -> MimiTransformerState<B> {
        self.transformer.init_state(batch_size, device)
    }

    /// Increment the transformer position by `increment`.
    pub fn increment_step(&self, state: &mut MimiTransformerState<B>, increment: usize) {
        self.transformer.increment_step(state, increment);
    }

    /// Forward pass with input/output projections.
    pub fn forward(
        &self,
        input: Tensor<B, 3>,
        state: &mut MimiTransformerState<B>,
    ) -> Vec<Tensor<B, 3>> {
        let mut hidden = input.swap_dims(1, 2);
        if let Some(proj) = &self.input_proj {
            hidden = apply_linear(proj, hidden);
        }
        let hidden = self.transformer.forward(hidden, state);

        let mut outputs = Vec::with_capacity(self.output_projs.len());
        for proj in &self.output_projs {
            let out = match proj {
                ProjectedOutput::Identity => hidden.clone(),
                ProjectedOutput::Linear(linear) => apply_linear(linear, hidden.clone()),
            };
            outputs.push(out.swap_dims(1, 2));
        }
        outputs
    }
}
