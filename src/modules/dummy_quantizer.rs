//! Simple quantizer placeholder used by Mimi in this Rust port.
//!
//! The original model uses a learned VQ module. For inference, a depthwise
//! convolution can stand in to match the expected tensor shapes.

use burn::tensor::{backend::Backend, module::conv1d, ops::ConvOptions, Tensor};

/// Minimal quantizer implemented as a 1x1 convolution.
#[derive(Debug, Clone)]
pub struct DummyQuantizer<B: Backend> {
    /// Convolution weights with shape `[out, in, 1]`.
    pub weight: Tensor<B, 3>,
}

impl<B: Backend> DummyQuantizer<B> {
    /// Create a new dummy quantizer from weights.
    pub fn new(weight: Tensor<B, 3>) -> Self {
        Self { weight }
    }

    /// Apply the quantizer convolution to input latents.
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        conv1d(
            input,
            self.weight.clone(),
            None,
            ConvOptions::new([1], [0], [1], 1),
        )
    }
}
