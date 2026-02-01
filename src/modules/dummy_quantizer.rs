use burn::tensor::backend::Backend;
use burn::tensor::module::conv1d;
use burn::tensor::ops::ConvOptions;
use burn::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct DummyQuantizer<B: Backend> {
    pub weight: Tensor<B, 3>,
}

impl<B: Backend> DummyQuantizer<B> {
    pub fn new(weight: Tensor<B, 3>) -> Self {
        Self { weight }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        conv1d(
            input,
            self.weight.clone(),
            None,
            ConvOptions::new([1], [0], [1], 1),
        )
    }
}
