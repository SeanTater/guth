use crate::modules::layer_scale::LayerScale;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use burn_nn::{LayerNorm, Linear};

#[derive(Debug, Clone)]
pub struct MimiTransformerConfig {
    pub d_model: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub layer_scale: f32,
    pub context: usize,
    pub max_period: f32,
    pub dim_feedforward: usize,
}

#[derive(Debug, Clone)]
pub struct MimiProjectedTransformerConfig {
    pub input_dim: usize,
    pub output_dims: Vec<usize>,
    pub transformer: MimiTransformerConfig,
}

#[derive(Debug, Clone)]
pub struct MimiSelfAttention<B: Backend> {
    pub in_proj: Linear<B>,
    pub out_proj: Linear<B>,
    pub num_heads: usize,
    pub head_dim: usize,
    pub context: usize,
    pub max_period: f32,
}

#[derive(Debug, Clone)]
pub struct MimiSelfAttentionState<B: Backend> {
    _phantom: std::marker::PhantomData<B>,
}

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

#[derive(Debug, Clone)]
pub struct MimiTransformerLayerState<B: Backend> {
    pub self_attn: MimiSelfAttentionState<B>,
}

#[derive(Debug, Clone)]
pub struct MimiTransformerState<B: Backend> {
    pub layers: Vec<MimiTransformerLayerState<B>>,
}

#[derive(Debug, Clone)]
pub struct MimiTransformer<B: Backend> {
    pub layers: Vec<MimiTransformerLayer<B>>,
}

impl<B: Backend> MimiTransformer<B> {
    pub fn new(_config: MimiTransformerConfig, _device: &B::Device) -> Self {
        todo!("MimiTransformer new not implemented")
    }

    pub fn init_state(&self, _batch_size: usize, _sequence_length: usize) -> MimiTransformerState<B> {
        todo!("MimiTransformer init_state not implemented")
    }

    pub fn forward(&self, _input: Tensor<B, 3>, _state: &mut MimiTransformerState<B>) -> Tensor<B, 3> {
        todo!("MimiTransformer forward not implemented")
    }
}

#[derive(Debug, Clone)]
pub enum ProjectedOutput<B: Backend> {
    Identity,
    Linear(Linear<B>),
}

#[derive(Debug, Clone)]
pub struct MimiProjectedTransformer<B: Backend> {
    pub input_proj: Option<Linear<B>>,
    pub output_projs: Vec<ProjectedOutput<B>>,
    pub transformer: MimiTransformer<B>,
}

impl<B: Backend> MimiProjectedTransformer<B> {
    pub fn new(_config: MimiProjectedTransformerConfig, _device: &B::Device) -> Self {
        todo!("MimiProjectedTransformer new not implemented")
    }

    pub fn init_state(&self, _batch_size: usize, _sequence_length: usize) -> MimiTransformerState<B> {
        todo!("MimiProjectedTransformer init_state not implemented")
    }

    pub fn forward(&self, _input: Tensor<B, 3>, _state: &mut MimiTransformerState<B>) -> Vec<Tensor<B, 3>> {
        todo!("MimiProjectedTransformer forward not implemented")
    }
}
