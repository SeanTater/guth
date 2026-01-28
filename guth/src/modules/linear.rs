use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use burn_nn::{LayerNorm, Linear};

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

/// Applies layer normalization to a 3D tensor.
///
/// Burn's `LayerNorm::forward` natively supports tensors of any rank, so in
/// principle we could just call `norm.forward(input)`. However, `burn-ndarray`
/// 0.20.1 has issues with zero-length dimensions in various operations.
///
/// This helper returns the input unchanged if any dimension is zero (nothing
/// to normalize, and avoids potential panics). Once the upstream bugs are
/// fixed, this function can be replaced with a direct `norm.forward(input)` call.
pub fn apply_layer_norm_3d<B: Backend>(norm: &LayerNorm<B>, input: Tensor<B, 3>) -> Tensor<B, 3> {
    let [batch, seq, dim] = input.dims();
    if batch == 0 || seq == 0 || dim == 0 {
        return input;
    }
    norm.forward(input)
}

/// Splits a combined QKV projection into separate query, key, and value tensors.
///
/// Takes a tensor of shape `[batch, seq, 3 * num_heads * head_dim]` (the output of
/// a combined Q/K/V linear projection) and splits it into three tensors, each of
/// shape `[batch, num_heads, seq, head_dim]`.
///
/// This is a common pattern in transformer attention where Q, K, V are computed
/// with a single linear layer for efficiency.
pub fn split_qkv<B: Backend>(
    qkv: Tensor<B, 3>,
    num_heads: usize,
    head_dim: usize,
) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
    let [batch, seq, _] = qkv.dims();
    let qkv = qkv.reshape([batch, seq, 3, num_heads, head_dim]);
    let q = qkv
        .clone()
        .narrow(2, 0, 1)
        .reshape([batch, seq, num_heads, head_dim])
        .swap_dims(1, 2);
    let k = qkv
        .clone()
        .narrow(2, 1, 1)
        .reshape([batch, seq, num_heads, head_dim])
        .swap_dims(1, 2);
    let v = qkv
        .narrow(2, 2, 1)
        .reshape([batch, seq, num_heads, head_dim])
        .swap_dims(1, 2);
    (q, k, v)
}

/// Merges attention heads back into a single tensor.
///
/// Takes a tensor of shape `[batch, num_heads, seq, head_dim]` and reshapes it
/// to `[batch, seq, num_heads * head_dim]` for the output projection.
pub fn merge_heads<B: Backend>(input: Tensor<B, 4>) -> Tensor<B, 3> {
    let [batch, heads, seq, dim] = input.dims();
    input.swap_dims(1, 2).reshape([batch, seq, heads * dim])
}
