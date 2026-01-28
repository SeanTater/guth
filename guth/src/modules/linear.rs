use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use burn_nn::Linear;

/// Applies a linear layer to a 3D tensor.
///
/// Burn's `Linear::forward` natively supports tensors of any rank >= 2, so in
/// principle we could just call `linear.forward(input)`. However, `burn-ndarray`
/// 0.20.1 panics with a divide-by-zero in its matmul implementation when the
/// sequence dimension is zero (e.g. shape `[batch, 0, dim]`).
///
/// This helper works around the bug by constructing an empty output tensor
/// directly when `seq == 0`. Once the upstream bug is fixed, this function can
/// be replaced with a direct `linear.forward(input)` call.
///
/// Tracking: https://github.com/tracel-ai/burn/issues — search for zero-length matmul
pub fn apply_linear_3d<B: Backend>(linear: &Linear<B>, input: Tensor<B, 3>) -> Tensor<B, 3> {
    let [batch, seq, _in_dim] = input.dims();
    if seq == 0 {
        // Workaround: construct empty output with correct shape
        let [_in_dim, out_dim] = linear.weight.shape().dims::<2>();
        let device = input.device();
        return Tensor::from_data(
            TensorData::new(Vec::<f32>::new(), [batch, 0, out_dim]),
            &device,
        );
    }
    linear.forward(input)
}
