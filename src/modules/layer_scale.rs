//! LayerScale utility used by transformer blocks.
//!
//! LayerScale applies a learned, per-channel scale to residual updates to help
//! stabilize deep transformer training and inference.

use burn::tensor::{backend::Backend, Tensor, TensorData as BurnTensorData};

/// Configuration for creating a [`LayerScale`] module.
#[derive(Debug, Clone)]
pub struct LayerScaleConfig {
    /// Feature dimension of the scale vector.
    pub dim: usize,
    /// Initial scale value (often a small number).
    pub init_value: f32,
}

impl LayerScaleConfig {
    /// Create a new config with the given dimension and init value.
    pub fn new(dim: usize, init_value: f32) -> Self {
        Self { dim, init_value }
    }

    /// Initialize a LayerScale module on the given device.
    pub fn init<B: Backend>(&self, device: &B::Device) -> LayerScale<B> {
        let values = vec![self.init_value; self.dim];
        let data = BurnTensorData::new(values, [self.dim]);
        let scale = Tensor::<B, 1>::from_data(data, device);
        LayerScale { scale }
    }
}

/// Learnable per-channel scale used in residual branches.
#[derive(Debug, Clone)]
pub struct LayerScale<B: Backend> {
    /// Scale vector with shape `[dim]`.
    pub scale: Tensor<B, 1>,
}

impl<B: Backend> LayerScale<B> {
    /// Apply the scale to a `[batch, seq, dim]` tensor.
    pub fn apply(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq, dim] = input.dims();
        let scale = self.scale.clone().reshape([1, 1, dim]);
        input.mul(scale).reshape([batch, seq, dim])
    }
}
