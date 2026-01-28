use crate::modules::layer_scale::{LayerScale, LayerScaleConfig};
use crate::modules::linear::apply_linear_3d;
use crate::modules::streaming_mha::{StreamingMha, StreamingMhaConfig, StreamingMhaOp, StreamingMhaState};
use crate::state::StreamingModule;
use burn::tensor::activation::gelu;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use burn_nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};

#[derive(Debug, Clone)]
pub struct StreamingTransformerLayerConfig {
    pub d_model: usize,
    pub num_heads: usize,
    pub ffn_dim: usize,
    pub causal: bool,
    pub context: Option<usize>,
    pub rope_max_seq: Option<usize>,
    pub rope_theta: f32,
    pub layer_scale: Option<f32>,
}

impl Default for StreamingTransformerLayerConfig {
    fn default() -> Self {
        Self {
            d_model: 512,
            num_heads: 8,
            ffn_dim: 2048,
            causal: true,
            context: None,
            rope_max_seq: None,
            rope_theta: 10000.0,
            layer_scale: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StreamingTransformerLayer<B: Backend> {
    pub num_heads: usize,
    pub head_dim: usize,
    pub qkv: Linear<B>,
    pub out_proj: Linear<B>,
    pub norm1: LayerNorm<B>,
    pub norm2: LayerNorm<B>,
    pub ffn_in: Linear<B>,
    pub ffn_out: Linear<B>,
    pub layer_scale_1: Option<LayerScale<B>>,
    pub layer_scale_2: Option<LayerScale<B>>,
    pub mha: StreamingMhaOp<B>,
}

#[derive(Debug, Clone)]
pub struct StreamingTransformerLayerState<B: Backend> {
    pub mha: StreamingMhaState<B>,
}

impl<B: Backend + 'static> StreamingTransformerLayer<B> {
    pub fn new(config: StreamingTransformerLayerConfig, device: &B::Device) -> Self {
        let head_dim = config.d_model / config.num_heads;
        let qkv = LinearConfig::new(config.d_model, config.d_model * 3)
            .with_bias(false)
            .init::<B>(device);
        let out_proj = LinearConfig::new(config.d_model, config.d_model)
            .with_bias(false)
            .init::<B>(device);
        let norm1 = LayerNormConfig::new(config.d_model).init::<B>(device);
        let norm2 = LayerNormConfig::new(config.d_model).init::<B>(device);
        let ffn_in = LinearConfig::new(config.d_model, config.ffn_dim)
            .with_bias(false)
            .init::<B>(device);
        let ffn_out = LinearConfig::new(config.ffn_dim, config.d_model)
            .with_bias(false)
            .init::<B>(device);

        let layer_scale_1 = config
            .layer_scale
            .map(|value| LayerScaleConfig::new(config.d_model, value).init::<B>(device));
        let layer_scale_2 = config
            .layer_scale
            .map(|value| LayerScaleConfig::new(config.d_model, value).init::<B>(device));

        let mha_config = StreamingMhaConfig {
            max_cache_tokens: 0,
            num_heads: config.num_heads,
            head_dim,
            context: config.context,
            causal: config.causal,
            rope_max_seq: config.rope_max_seq,
            rope_theta: config.rope_theta,
        };
        let mha = StreamingMhaOp::new(mha_config, device);

        Self {
            num_heads: config.num_heads,
            head_dim,
            qkv,
            out_proj,
            norm1,
            norm2,
            ffn_in,
            ffn_out,
            layer_scale_1,
            layer_scale_2,
            mha,
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>, state: &mut StreamingTransformerLayerState<B>) -> Tensor<B, 3> {
        let residual = input.clone();
        let normalized = self.apply_layer_norm(&self.norm1, input);
        let qkv = self.apply_linear(&self.qkv, normalized);
        let (queries, keys, values) = self.split_qkv(qkv);

        let start = state.mha.step.index;
        let queries = self.mha.apply_rope(queries, start);
        let keys = self.mha.apply_rope(keys, start);

        self.mha.append_kv(&mut state.mha, keys, values);
        let attn = self.mha.attention(&state.mha, queries);
        let attn = self.merge_heads(attn);
        let attn = self.apply_linear(&self.out_proj, attn);
        let attn = match &self.layer_scale_1 {
            Some(scale) => scale.apply(attn),
            None => attn,
        };
        let hidden = residual.add(attn);

        let ffn_input = self.apply_layer_norm(&self.norm2, hidden.clone());
        let ffn = self.apply_linear(&self.ffn_in, ffn_input);
        let ffn = gelu(ffn);
        let ffn = self.apply_linear(&self.ffn_out, ffn);
        let ffn = match &self.layer_scale_2 {
            Some(scale) => scale.apply(ffn),
            None => ffn,
        };

        hidden.add(ffn)
    }

    fn apply_linear(&self, linear: &Linear<B>, input: Tensor<B, 3>) -> Tensor<B, 3> {
        apply_linear_3d(linear, input)
    }

    fn apply_layer_norm(&self, norm: &LayerNorm<B>, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq, dim] = input.dims();
        if batch == 0 || seq == 0 || dim == 0 {
            return input;
        }
        let flat = input.reshape([batch * seq, dim]);
        let output = norm.forward(flat);
        output.reshape([batch, seq, dim])
    }

    fn split_qkv(&self, qkv: Tensor<B, 3>) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let [batch, seq, _] = qkv.dims();
        let qkv = qkv.reshape([batch, seq, 3, self.num_heads, self.head_dim]);
        let q = qkv
            .clone()
            .narrow(2, 0, 1)
            .reshape([batch, seq, self.num_heads, self.head_dim]);
        let k = qkv
            .clone()
            .narrow(2, 1, 1)
            .reshape([batch, seq, self.num_heads, self.head_dim]);
        let v = qkv
            .narrow(2, 2, 1)
            .reshape([batch, seq, self.num_heads, self.head_dim]);
        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);
        (q, k, v)
    }

    fn merge_heads(&self, input: Tensor<B, 4>) -> Tensor<B, 3> {
        let [batch, heads, seq, dim] = input.dims();
        let merged = input.swap_dims(1, 2).reshape([batch, seq, heads * dim]);
        merged
    }
}

impl<B: Backend + 'static> StreamingModule<B> for StreamingTransformerLayer<B> {
    type State = StreamingTransformerLayerState<B>;

    fn init_state(&self, _batch_size: usize, _sequence_length: usize) -> Self::State {
        StreamingTransformerLayerState {
            mha: StreamingMha::default().init_state(0, 0),
        }
    }

    fn increment_step(&self, state: &mut Self::State, increment: usize) {
        state.mha.step.increment(increment);
    }
}

#[derive(Debug, Clone)]
pub struct StreamingTransformerConfig {
    pub num_layers: usize,
    pub layer: StreamingTransformerLayerConfig,
}

#[derive(Debug, Clone)]
pub struct StreamingTransformer<B: Backend> {
    pub layers: Vec<StreamingTransformerLayer<B>>,
}

#[derive(Debug, Clone)]
pub struct StreamingTransformerState<B: Backend> {
    pub layers: Vec<StreamingTransformerLayerState<B>>,
}

impl<B: Backend + 'static> StreamingTransformer<B> {
    pub fn new(config: StreamingTransformerConfig, device: &B::Device) -> Self {
        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            layers.push(StreamingTransformerLayer::new(config.layer.clone(), device));
        }
        Self { layers }
    }

    pub fn forward(&self, mut input: Tensor<B, 3>, state: &mut StreamingTransformerState<B>) -> Tensor<B, 3> {
        for (layer, layer_state) in self.layers.iter().zip(state.layers.iter_mut()) {
            input = layer.forward(input, layer_state);
        }
        input
    }
}

impl<B: Backend + 'static> StreamingModule<B> for StreamingTransformer<B> {
    type State = StreamingTransformerState<B>;

    fn init_state(&self, batch_size: usize, sequence_length: usize) -> Self::State {
        let layers = self
            .layers
            .iter()
            .map(|layer| layer.init_state(batch_size, sequence_length))
            .collect();
        StreamingTransformerState { layers }
    }

    fn increment_step(&self, state: &mut Self::State, increment: usize) {
        for (layer, layer_state) in self.layers.iter().zip(state.layers.iter_mut()) {
            layer.increment_step(layer_state, increment);
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProjectedTransformerConfig {
    pub input_dim: usize,
    pub model_dim: usize,
    pub output_dim: usize,
    pub transformer: StreamingTransformerConfig,
}

#[derive(Debug, Clone)]
pub struct ProjectedTransformer<B: Backend> {
    input_proj: Linear<B>,
    output_proj: Linear<B>,
    transformer: StreamingTransformer<B>,
}

impl<B: Backend> ProjectedTransformer<B> {
    pub fn new(config: ProjectedTransformerConfig, device: &B::Device) -> Self {
        let input_proj = LinearConfig::new(config.input_dim, config.model_dim).init::<B>(device);
        let output_proj = LinearConfig::new(config.model_dim, config.output_dim).init::<B>(device);
        let transformer = StreamingTransformer::new(config.transformer, device);
        Self {
            input_proj,
            output_proj,
            transformer,
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>, state: &mut StreamingTransformerState<B>) -> Tensor<B, 3> {
        let transposed = input.swap_dims(1, 2);
        let projected = apply_linear_3d(&self.input_proj, transposed);
        let output = self.transformer.forward(projected, state);
        let output = apply_linear_3d(&self.output_proj, output);
        output.swap_dims(1, 2)
    }
}

impl<B: Backend> StreamingModule<B> for ProjectedTransformer<B> {
    type State = StreamingTransformerState<B>;

    fn init_state(&self, batch_size: usize, sequence_length: usize) -> Self::State {
        self.transformer.init_state(batch_size, sequence_length)
    }

    fn increment_step(&self, state: &mut Self::State, increment: usize) {
        self.transformer.increment_step(state, increment);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::module::Param;
    use burn::tensor::TensorData;
    use burn_ndarray::{NdArray, NdArrayDevice};
    use serde::Deserialize;
    use std::fs;
    use std::path::PathBuf;

    type TestBackend = NdArray<f32>;

    fn make_input(device: &NdArrayDevice, batch: usize, seq: usize, dim: usize) -> Tensor<TestBackend, 3> {
        let mut data = Vec::with_capacity(batch * seq * dim);
        for b in 0..batch {
            for t in 0..seq {
                for d in 0..dim {
                    let value = (b * 100 + t * 10 + d) as f32 / 100.0;
                    data.push(value);
                }
            }
        }
        let data = TensorData::new(data, [batch, seq, dim]);
        Tensor::from_data(data, device)
    }

    fn max_abs_diff(a: Tensor<TestBackend, 3>, b: Tensor<TestBackend, 3>) -> f32 {
        let a = a.to_data();
        let b = b.to_data();
        let a_values = a.as_slice::<f32>().unwrap();
        let b_values = b.as_slice::<f32>().unwrap();
        a_values
            .iter()
            .zip(b_values.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, |acc, v| acc.max(v))
    }

    #[derive(Debug, Deserialize)]
    struct TransformerFixture {
        config: TransformerFixtureConfig,
        weights: TransformerWeights,
        input: Vec<Vec<Vec<f32>>>,
        output: Vec<Vec<Vec<f32>>>,
    }

    #[derive(Debug, Deserialize)]
    struct TransformerFixtureConfig {
        d_model: usize,
        num_heads: usize,
        ffn_dim: usize,
        causal: bool,
    }

    #[derive(Debug, Deserialize)]
    struct TransformerWeights {
        norm1: NormWeights,
        norm2: NormWeights,
        qkv: LinearWeights,
        out_proj: LinearWeights,
        ffn_in: LinearWeights,
        ffn_out: LinearWeights,
    }

    #[derive(Debug, Deserialize)]
    struct LinearWeights {
        weight: Vec<Vec<f32>>,
        bias: Vec<f32>,
    }

    #[derive(Debug, Deserialize)]
    struct NormWeights {
        gamma: Vec<f32>,
        beta: Vec<f32>,
    }

    fn fixture_path(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join(name)
    }

    fn load_fixture(name: &str) -> TransformerFixture {
        let data = fs::read_to_string(fixture_path(name)).expect("fixture read");
        serde_json::from_str(&data).expect("fixture parse")
    }

    fn tensor3(device: &NdArrayDevice, data: Vec<Vec<Vec<f32>>>) -> Tensor<TestBackend, 3> {
        let b = data.len();
        let t = data[0].len();
        let d = data[0][0].len();
        let flat: Vec<f32> = data.into_iter().flatten().flatten().collect();
        let td = TensorData::new(flat, [b, t, d]);
        Tensor::from_data(td, device)
    }

    fn apply_linear(linear: &mut Linear<TestBackend>, weights: &LinearWeights, device: &NdArrayDevice) {
        let rows = weights.weight.len();
        let cols = weights.weight.first().map(|row| row.len()).unwrap_or(0);
        let flat: Vec<f32> = weights.weight.clone().into_iter().flatten().collect();
        let weight = Tensor::from_data(TensorData::new(flat, [rows, cols]), device);
        linear.weight = Param::from_tensor(weight);
        let bias = Tensor::from_data(TensorData::new(weights.bias.clone(), [weights.bias.len()]), device);
        linear.bias = Some(Param::from_tensor(bias));
    }

    fn apply_norm(norm: &mut LayerNorm<TestBackend>, weights: &NormWeights, device: &NdArrayDevice) {
        let gamma = Tensor::from_data(TensorData::new(weights.gamma.clone(), [weights.gamma.len()]), device);
        let beta = Tensor::from_data(TensorData::new(weights.beta.clone(), [weights.beta.len()]), device);
        norm.gamma = Param::from_tensor(gamma);
        norm.beta = Some(Param::from_tensor(beta));
    }

    #[test]
    fn layer_streaming_matches_batch() {
        let device = NdArrayDevice::default();
        let config = StreamingTransformerLayerConfig {
            d_model: 8,
            num_heads: 2,
            ffn_dim: 16,
            causal: true,
            ..Default::default()
        };
        let layer = StreamingTransformerLayer::<TestBackend>::new(config, &device);

        let input = make_input(&device, 1, 4, 8);
        let mut state_batch = layer.init_state(1, 4);
        let batch_output = layer.forward(input.clone(), &mut state_batch);

        let mut state_stream = layer.init_state(1, 4);
        let chunk1 = input.clone().narrow(1, 0, 2);
        let chunk2 = input.narrow(1, 2, 2);
        let out1 = layer.forward(chunk1, &mut state_stream);
        let out2 = layer.forward(chunk2, &mut state_stream);
        let stream_output = Tensor::cat(vec![out1, out2], 1);

        let diff = max_abs_diff(batch_output, stream_output);
        assert!(diff < 1e-4, "max diff {diff}");
    }

    #[test]
    fn transformer_stack_streaming_matches_batch() {
        let device = NdArrayDevice::default();
        let layer_config = StreamingTransformerLayerConfig {
            d_model: 8,
            num_heads: 2,
            ffn_dim: 16,
            causal: true,
            ..Default::default()
        };
        let config = StreamingTransformerConfig {
            num_layers: 2,
            layer: layer_config,
        };
        let transformer = StreamingTransformer::<TestBackend>::new(config, &device);

        let input = make_input(&device, 1, 6, 8);
        let mut state_batch = transformer.init_state(1, 6);
        let batch_output = transformer.forward(input.clone(), &mut state_batch);

        let mut state_stream = transformer.init_state(1, 6);
        let chunk1 = input.clone().narrow(1, 0, 2);
        let chunk2 = input.clone().narrow(1, 2, 2);
        let chunk3 = input.narrow(1, 4, 2);
        let out1 = transformer.forward(chunk1, &mut state_stream);
        let out2 = transformer.forward(chunk2, &mut state_stream);
        let out3 = transformer.forward(chunk3, &mut state_stream);
        let stream_output = Tensor::cat(vec![out1, out2, out3], 1);

        let diff = max_abs_diff(batch_output, stream_output);
        assert!(diff < 1e-4, "max diff {diff}");
    }

    #[test]
    fn projected_transformer_transposes_and_streams() {
        let device = NdArrayDevice::default();
        let layer_config = StreamingTransformerLayerConfig {
            d_model: 6,
            num_heads: 2,
            ffn_dim: 12,
            causal: true,
            ..Default::default()
        };
        let transformer = StreamingTransformerConfig {
            num_layers: 1,
            layer: layer_config,
        };
        let config = ProjectedTransformerConfig {
            input_dim: 4,
            model_dim: 6,
            output_dim: 4,
            transformer,
        };
        let projected = ProjectedTransformer::<TestBackend>::new(config, &device);

        let input = make_input(&device, 1, 5, 4).swap_dims(1, 2);
        let mut state_batch = projected.init_state(1, 5);
        let batch_output = projected.forward(input.clone(), &mut state_batch);

        let mut state_stream = projected.init_state(1, 5);
        let chunk1 = input.clone().narrow(2, 0, 2);
        let chunk2 = input.clone().narrow(2, 2, 2);
        let chunk3 = input.narrow(2, 4, 1);
        let out1 = projected.forward(chunk1, &mut state_stream);
        let out2 = projected.forward(chunk2, &mut state_stream);
        let out3 = projected.forward(chunk3, &mut state_stream);
        let stream_output = Tensor::cat(vec![out1, out2, out3], 2);

        let diff = max_abs_diff(batch_output.swap_dims(1, 2), stream_output.swap_dims(1, 2));
        assert!(diff < 1e-4, "max diff {diff}");
    }

    #[test]
    fn transformer_layer_matches_fixture() {
        let device = NdArrayDevice::default();
        let fixture = load_fixture("transformer_layer.json");
        let config = StreamingTransformerLayerConfig {
            d_model: fixture.config.d_model,
            num_heads: fixture.config.num_heads,
            ffn_dim: fixture.config.ffn_dim,
            causal: fixture.config.causal,
            ..Default::default()
        };
        let mut layer = StreamingTransformerLayer::<TestBackend>::new(config, &device);

        apply_norm(&mut layer.norm1, &fixture.weights.norm1, &device);
        apply_norm(&mut layer.norm2, &fixture.weights.norm2, &device);
        apply_linear(&mut layer.qkv, &fixture.weights.qkv, &device);
        apply_linear(&mut layer.out_proj, &fixture.weights.out_proj, &device);
        apply_linear(&mut layer.ffn_in, &fixture.weights.ffn_in, &device);
        apply_linear(&mut layer.ffn_out, &fixture.weights.ffn_out, &device);

        let input = tensor3(&device, fixture.input);
        let expected = tensor3(&device, fixture.output).to_data();
        let mut state = layer.init_state(1, input.dims()[1]);
        let output = layer.forward(input, &mut state).to_data();
        output.assert_approx_eq(&expected, burn::tensor::Tolerance::<f32>::absolute(1e-4));
    }
}
