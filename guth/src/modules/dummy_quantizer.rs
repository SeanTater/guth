use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct DummyQuantizer<B: Backend> {
    pub weight: Tensor<B, 3>,
}

impl<B: Backend> DummyQuantizer<B> {
    pub fn new(weight: Tensor<B, 3>) -> Self {
        Self { weight }
    }

    pub fn forward(&self, _input: Tensor<B, 3>) -> Tensor<B, 3> {
        todo!("DummyQuantizer forward not implemented")
    }
}
