use burn::tensor::{Tensor, TensorData};
use burn::tensor::backend::Backend;
use burn_nn::Linear;

#[cfg(feature = "backend-cpu")]
use std::any::TypeId;

#[cfg(feature = "backend-cpu")]
fn is_cpu_backend<B: Backend + 'static>() -> bool {
    TypeId::of::<B>() == TypeId::of::<burn_cpu::Cpu>()
}

#[cfg(feature = "backend-cpu")]
fn apply_linear_cpu<B: Backend>(linear: &Linear<B>, input: Tensor<B, 2>) -> Tensor<B, 2> {
    let device = input.device();
    let input_data = input.to_data();
    let weight_data = linear.weight.val().to_data();
    let bias_data = linear.bias.as_ref().map(|b| b.val().to_data());

    let shape = &input_data.shape;
    let weight_shape = &weight_data.shape;
    let batch = shape[0];
    let in_dim = shape[1];
    let out_dim = weight_shape[1];

    if batch == 0 || in_dim == 0 {
        return Tensor::from_data(TensorData::new(Vec::<f32>::new(), [batch, out_dim]), &device);
    }

    let input_vals = input_data
        .as_slice::<f32>()
        .expect("cpu linear expects f32 input");
    let weight_vals = weight_data
        .as_slice::<f32>()
        .expect("cpu linear expects f32 weights");
    let bias_vals = bias_data
        .as_ref()
        .map(|data| data.as_slice::<f32>().expect("cpu linear expects f32 bias"));

    let mut out = vec![0.0f32; batch * out_dim];
    for b in 0..batch {
        let input_row = &input_vals[b * in_dim..(b + 1) * in_dim];
        for o in 0..out_dim {
            let mut acc = 0.0f32;
            for i in 0..in_dim {
                acc += input_row[i] * weight_vals[i * out_dim + o];
            }
            if let Some(bias) = bias_vals {
                acc += bias[o];
            }
            out[b * out_dim + o] = acc;
        }
    }

    Tensor::from_data(TensorData::new(out, [batch, out_dim]), &device)
}

pub fn apply_linear_2d<B: Backend + 'static>(linear: &Linear<B>, input: Tensor<B, 2>) -> Tensor<B, 2> {
    #[cfg(feature = "backend-cpu")]
    if is_cpu_backend::<B>() {
        return apply_linear_cpu(linear, input);
    }

    linear.forward(input)
}

pub fn apply_linear_3d<B: Backend + 'static>(linear: &Linear<B>, input: Tensor<B, 3>) -> Tensor<B, 3> {
    let [batch, seq, in_dim] = input.dims();
    let [_in_dim, out_dim] = linear.weight.shape().dims::<2>();
    let device = input.device();

    if batch == 0 {
        return Tensor::from_data(TensorData::new(Vec::<f32>::new(), [batch, 0, out_dim]), &device);
    }
    if seq == 0 || in_dim == 0 {
        return Tensor::from_data(TensorData::new(Vec::<f32>::new(), [batch, 0, out_dim]), &device);
    }

    let reshaped = input.reshape([batch * seq, in_dim]);
    let projected = apply_linear_2d(linear, reshaped);
    projected.reshape([batch, seq, out_dim])
}
