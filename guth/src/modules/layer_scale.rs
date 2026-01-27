use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

#[derive(Debug, Clone)]
pub struct LayerScaleConfig {
    pub dim: usize,
    pub init_value: f32,
}

impl LayerScaleConfig {
    pub fn new(dim: usize, init_value: f32) -> Self {
        Self { dim, init_value }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> LayerScale<B> {
        let values = vec![self.init_value; self.dim];
        let data = TensorData::new(values, [self.dim]);
        let scale = Tensor::<B, 1>::from_data(data, device);
        LayerScale { scale }
    }
}

#[derive(Debug, Clone)]
pub struct LayerScale<B: Backend> {
    pub scale: Tensor<B, 1>,
}

impl<B: Backend> LayerScale<B> {
    pub fn apply(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq, dim] = input.dims();
        let scale = self.scale.clone().reshape([1, 1, dim]);
        input.mul(scale).reshape([batch, seq, dim])
    }
}
