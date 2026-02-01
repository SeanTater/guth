//! Flow network used by FlowLM to map noise to latent frames.
//!
//! This is a small MLP with adaptive layer normalization (AdaLN) conditioned on
//! text and time embeddings, similar to diffusion-style parameterizations.

use burn::{
    module::Param,
    tensor::{activation::silu, backend::Backend, Tensor, TensorData as BurnTensorData},
};
use burn_nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};

/// Apply shift/scale modulation used by AdaLN.
fn modulate<B: Backend>(x: Tensor<B, 2>, shift: Tensor<B, 2>, scale: Tensor<B, 2>) -> Tensor<B, 2> {
    let scale = scale.add_scalar(1.0);
    x.mul(scale).add(shift)
}

/// Configuration for timestep embeddings.
#[derive(Debug, Clone)]
pub struct TimestepEmbedderConfig {
    /// Hidden size of the embedder.
    pub hidden_size: usize,
    /// Size of the sinusoidal frequency embedding.
    pub frequency_embedding_size: usize,
    /// Max period for log-spaced frequencies.
    pub max_period: f32,
}

impl TimestepEmbedderConfig {
    /// Create a config with default frequency settings.
    pub fn new(hidden_size: usize) -> Self {
        Self {
            hidden_size,
            frequency_embedding_size: 256,
            max_period: 10000.0,
        }
    }
}

/// Configuration for RMS normalization.
#[derive(Debug, Clone)]
pub struct RmsNormConfig {
    /// Feature dimension.
    pub dim: usize,
    /// Numerical epsilon.
    pub epsilon: f32,
}

impl RmsNormConfig {
    /// Create a new RMSNorm config with default epsilon.
    pub fn new(dim: usize) -> Self {
        Self { dim, epsilon: 1e-5 }
    }

    /// Set a custom epsilon value.
    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Initialize an RMSNorm module on the given device.
    pub fn init<B: Backend>(&self, device: &B::Device) -> RmsNorm<B> {
        let values = vec![1.0_f32; self.dim];
        let gamma = Tensor::<B, 1>::from_data(BurnTensorData::new(values, [self.dim]), device);
        RmsNorm {
            gamma: Param::from_tensor(gamma),
            epsilon: self.epsilon,
        }
    }
}

/// RMS normalization with learnable scale.
#[derive(Debug, Clone)]
pub struct RmsNorm<B: Backend> {
    /// Scale parameter.
    pub gamma: Param<Tensor<B, 1>>,
    /// Numerical epsilon.
    pub epsilon: f32,
}

impl<B: Backend> RmsNorm<B> {
    /// Apply RMS normalization across the feature dimension.
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let dim = x.dims()[1];
        let mean = x.clone().mean_dim(1);
        let centered = x.clone().sub(mean);
        let var = centered
            .clone()
            .square()
            .sum_dim(1)
            .div_scalar((dim.saturating_sub(1)) as f32);
        let denom = var.add_scalar(self.epsilon).sqrt();
        let gamma = self.gamma.val().unsqueeze_dim::<2>(0);
        x.mul(gamma).div(denom)
    }
}

/// Sinusoidal timestep embedder with small MLP projection.
#[derive(Debug, Clone)]
pub struct TimestepEmbedder<B: Backend> {
    /// Precomputed frequencies.
    pub freqs: Tensor<B, 1>,
    /// Input projection.
    pub proj_in: Linear<B>,
    /// Output projection.
    pub proj_out: Linear<B>,
    /// RMS normalization.
    pub norm: RmsNorm<B>,
}

impl<B: Backend + 'static> TimestepEmbedder<B> {
    /// Create a new embedder from config.
    pub fn new(config: TimestepEmbedderConfig, device: &B::Device) -> Self {
        let half = config.frequency_embedding_size / 2;
        let mut values = Vec::with_capacity(half);
        for i in 0..half {
            let exponent = -config.max_period.ln() * (i as f32) / (half as f32);
            values.push(exponent.exp());
        }
        let freqs = Tensor::<B, 1>::from_data(BurnTensorData::new(values, [half]), device);

        let proj_in = LinearConfig::new(config.frequency_embedding_size, config.hidden_size)
            .init::<B>(device);
        let proj_out = LinearConfig::new(config.hidden_size, config.hidden_size).init::<B>(device);
        let norm = RmsNormConfig::new(config.hidden_size).init::<B>(device);
        Self {
            freqs,
            proj_in,
            proj_out,
            norm,
        }
    }

    /// Embed a batch of timestep scalars.
    pub fn forward(&self, t: Tensor<B, 2>) -> Tensor<B, 2> {
        let freqs = self.freqs.clone().unsqueeze_dim::<2>(0);
        let args = t.mul(freqs);
        let embedding = Tensor::cat(vec![args.clone().cos(), args.sin()], 1);
        let hidden = self.proj_in.forward(embedding);
        let hidden = silu(hidden);
        let hidden = self.proj_out.forward(hidden);
        self.norm.forward(hidden)
    }
}

/// Configuration for a residual MLP block.
#[derive(Debug, Clone)]
pub struct ResBlockConfig {
    /// Feature size.
    pub channels: usize,
}

impl ResBlockConfig {
    /// Create a config for a given channel size.
    pub fn new(channels: usize) -> Self {
        Self { channels }
    }
}

/// Residual MLP block with AdaLN-style modulation.
#[derive(Debug, Clone)]
pub struct ResBlock<B: Backend> {
    pub channels: usize,
    pub norm: LayerNorm<B>,
    pub mlp_in: Linear<B>,
    pub mlp_out: Linear<B>,
    pub mod_linear: Linear<B>,
}

impl<B: Backend + 'static> ResBlock<B> {
    /// Construct a residual block from config.
    pub fn new(config: ResBlockConfig, device: &B::Device) -> Self {
        let norm = LayerNormConfig::new(config.channels)
            .with_epsilon(1e-6)
            .init::<B>(device);
        let mlp_in = LinearConfig::new(config.channels, config.channels).init::<B>(device);
        let mlp_out = LinearConfig::new(config.channels, config.channels).init::<B>(device);
        let mod_linear = LinearConfig::new(config.channels, config.channels * 3).init::<B>(device);
        Self {
            channels: config.channels,
            norm,
            mlp_in,
            mlp_out,
            mod_linear,
        }
    }

    /// Forward pass with conditioning signal `y`.
    pub fn forward(&self, x: Tensor<B, 2>, y: Tensor<B, 2>) -> Tensor<B, 2> {
        let modulation = self.mod_linear.forward(silu(y));
        let shift = modulation.clone().narrow(1, 0, self.channels);
        let scale = modulation.clone().narrow(1, self.channels, self.channels);
        let gate = modulation.narrow(1, self.channels * 2, self.channels);

        let h = modulate(self.norm.forward(x.clone()), shift, scale);
        let h = self.mlp_in.forward(h);
        let h = silu(h);
        let h = self.mlp_out.forward(h);
        x.add(gate.mul(h))
    }
}

/// Configuration for the final AdaLN layer.
#[derive(Debug, Clone)]
pub struct FinalLayerConfig {
    /// Hidden size.
    pub model_channels: usize,
    /// Output size.
    pub out_channels: usize,
}

impl FinalLayerConfig {
    /// Create a config from sizes.
    pub fn new(model_channels: usize, out_channels: usize) -> Self {
        Self {
            model_channels,
            out_channels,
        }
    }
}

/// Final projection layer for the flow network.
#[derive(Debug, Clone)]
pub struct FinalLayer<B: Backend> {
    pub model_channels: usize,
    pub norm: LayerNorm<B>,
    pub linear: Linear<B>,
    pub mod_linear: Linear<B>,
}

impl<B: Backend + 'static> FinalLayer<B> {
    /// Construct the final layer from config.
    pub fn new(config: FinalLayerConfig, device: &B::Device) -> Self {
        let norm = LayerNormConfig::new(config.model_channels)
            .with_epsilon(1e-6)
            .with_bias(false)
            .init::<B>(device);
        let linear =
            LinearConfig::new(config.model_channels, config.out_channels).init::<B>(device);
        let mod_linear =
            LinearConfig::new(config.model_channels, config.model_channels * 2).init::<B>(device);
        Self {
            model_channels: config.model_channels,
            norm,
            linear,
            mod_linear,
        }
    }

    /// Forward pass with conditioning signal `y`.
    pub fn forward(&self, x: Tensor<B, 2>, y: Tensor<B, 2>) -> Tensor<B, 2> {
        let modulation = self.mod_linear.forward(silu(y));
        let shift = modulation.clone().narrow(1, 0, self.model_channels);
        let scale = modulation.narrow(1, self.model_channels, self.model_channels);
        let x = modulate(self.norm.forward(x), shift, scale);
        self.linear.forward(x)
    }
}

/// Configuration for the full SimpleMlpAdaLn network.
#[derive(Debug, Clone)]
pub struct SimpleMlpAdaLnConfig {
    /// Input latent dimension.
    pub in_channels: usize,
    /// Hidden model dimension.
    pub model_channels: usize,
    /// Output latent dimension.
    pub out_channels: usize,
    /// Conditioning dimension.
    pub cond_channels: usize,
    /// Number of residual blocks.
    pub num_res_blocks: usize,
    /// Number of time condition inputs.
    pub num_time_conds: usize,
    /// Timestep embedding configuration.
    pub time_embed: TimestepEmbedderConfig,
}

impl SimpleMlpAdaLnConfig {
    /// Create a config with default depth and time settings.
    pub fn new(
        in_channels: usize,
        model_channels: usize,
        out_channels: usize,
        cond_channels: usize,
    ) -> Self {
        Self {
            in_channels,
            model_channels,
            out_channels,
            cond_channels,
            num_res_blocks: 2,
            num_time_conds: 2,
            time_embed: TimestepEmbedderConfig::new(model_channels),
        }
    }
}

/// Small MLP used as the flow network inside FlowLM.
#[derive(Debug, Clone)]
pub struct SimpleMlpAdaLn<B: Backend> {
    pub num_time_conds: usize,
    pub time_embed: Vec<TimestepEmbedder<B>>,
    pub cond_embed: Linear<B>,
    pub input_proj: Linear<B>,
    pub res_blocks: Vec<ResBlock<B>>,
    pub final_layer: FinalLayer<B>,
}

impl<B: Backend + 'static> SimpleMlpAdaLn<B> {
    /// Construct the flow network from config.
    pub fn new(config: SimpleMlpAdaLnConfig, device: &B::Device) -> Self {
        assert!(config.num_time_conds != 1, "num_time_conds must not be 1");
        let mut time_embed = Vec::with_capacity(config.num_time_conds);
        for _ in 0..config.num_time_conds {
            time_embed.push(TimestepEmbedder::new(config.time_embed.clone(), device));
        }

        let cond_embed =
            LinearConfig::new(config.cond_channels, config.model_channels).init::<B>(device);
        let input_proj =
            LinearConfig::new(config.in_channels, config.model_channels).init::<B>(device);

        let mut res_blocks = Vec::with_capacity(config.num_res_blocks);
        for _ in 0..config.num_res_blocks {
            res_blocks.push(ResBlock::new(
                ResBlockConfig::new(config.model_channels),
                device,
            ));
        }

        let final_layer = FinalLayer::new(
            FinalLayerConfig::new(config.model_channels, config.out_channels),
            device,
        );

        Self {
            num_time_conds: config.num_time_conds,
            time_embed,
            cond_embed,
            input_proj,
            res_blocks,
            final_layer,
        }
    }

    /// Forward pass for the flow network.
    pub fn forward(
        &self,
        c: Tensor<B, 2>,
        s: Tensor<B, 2>,
        t: Tensor<B, 2>,
        x: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let mut x = self.input_proj.forward(x);
        let ts = [s, t];
        assert_eq!(ts.len(), self.num_time_conds);

        let mut t_combined: Option<Tensor<B, 2>> = None;
        for (index, embedder) in self.time_embed.iter().enumerate() {
            let embedded = embedder.forward(ts[index].clone());
            t_combined = Some(match t_combined {
                Some(acc) => acc.add(embedded),
                None => embedded,
            });
        }
        let t_combined = t_combined
            .expect("time embeddings missing")
            .div_scalar(self.num_time_conds as f32);

        let cond = self.cond_embed.forward(c);
        let y = t_combined.add(cond);

        for block in &self.res_blocks {
            x = block.forward(x, y.clone());
        }

        self.final_layer.forward(x, y)
    }
}

/// Langevin-style decode loop for flow matching.
pub fn lsd_decode<B: Backend, F>(v_t: F, x_0: Tensor<B, 2>, num_steps: usize) -> Tensor<B, 2>
where
    F: Fn(Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) -> Tensor<B, 2>,
{
    let mut current = x_0.clone();
    let device = x_0.device();
    let batch = x_0.dims()[0];
    for i in 0..num_steps {
        let s = (i as f32) / (num_steps as f32);
        let t = (i as f32 + 1.0) / (num_steps as f32);
        let s_tensor = Tensor::<B, 2>::full([batch, 1], s, &device);
        let t_tensor = Tensor::<B, 2>::full([batch, 1], t, &device);
        let flow_dir = v_t(s_tensor, t_tensor, current.clone());
        current = current.add(flow_dir.div_scalar(num_steps as f32));
    }
    current
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{
        module::Param,
        tensor::{TensorData as BurnTensorData, Tolerance},
    };
    use burn_ndarray::{NdArray, NdArrayDevice};
    use serde::Deserialize;
    use std::fs;
    use std::path::PathBuf;

    type TestBackend = NdArray<f32>;

    /// Linear layer weights from a fixture file.
    #[derive(Debug, Deserialize)]
    struct LinearWeights {
        weight: Vec<Vec<f32>>,
        bias: Vec<f32>,
    }

    /// LayerNorm weights from a fixture file.
    #[derive(Debug, Deserialize)]
    struct NormWeights {
        weight: Vec<f32>,
        bias: Option<Vec<f32>>,
    }

    /// Timestep embedder weights from a fixture file.
    #[derive(Debug, Deserialize)]
    struct TimeEmbedWeights {
        proj_in: LinearWeights,
        proj_out: LinearWeights,
        rms_weight: Vec<f32>,
    }

    /// Residual block weights from a fixture file.
    #[derive(Debug, Deserialize)]
    struct ResBlockWeights {
        norm: NormWeights,
        mlp_in: LinearWeights,
        mlp_out: LinearWeights,
        modulation: LinearWeights,
    }

    /// Final layer weights from a fixture file.
    #[derive(Debug, Deserialize)]
    struct FinalLayerWeights {
        norm: NormWeights,
        linear: LinearWeights,
        modulation: LinearWeights,
    }

    /// Full flow net weights for a fixture.
    #[derive(Debug, Deserialize)]
    struct FlowNetWeights {
        input_proj: LinearWeights,
        cond_embed: LinearWeights,
        time_embed: Vec<TimeEmbedWeights>,
        res_blocks: Vec<ResBlockWeights>,
        final_layer: FinalLayerWeights,
    }

    /// Input tensors for a flow net fixture.
    #[derive(Debug, Deserialize)]
    struct FlowNetInputs {
        c: Vec<Vec<f32>>,
        s: Vec<Vec<f32>>,
        t: Vec<Vec<f32>>,
        x: Vec<Vec<f32>>,
        noise: Vec<Vec<f32>>,
    }

    /// Expected outputs for a flow net fixture.
    #[derive(Debug, Deserialize)]
    struct FlowNetExpected {
        flow_out: Vec<Vec<f32>>,
        lsd_out: Vec<Vec<f32>>,
    }

    /// Config values stored in a fixture file.
    #[derive(Debug, Deserialize)]
    struct FlowNetConfigData {
        in_channels: usize,
        model_channels: usize,
        out_channels: usize,
        cond_channels: usize,
        num_res_blocks: usize,
        num_time_conds: usize,
        frequency_embedding_size: usize,
        max_period: f32,
    }

    /// Full flow net test fixture (config, weights, inputs, outputs).
    #[derive(Debug, Deserialize)]
    struct FlowNetFixture {
        config: FlowNetConfigData,
        weights: FlowNetWeights,
        inputs: FlowNetInputs,
        expected: FlowNetExpected,
    }

    /// RMSNorm fixture data for regression tests.
    #[derive(Debug, Deserialize)]
    struct RmsNormFixture {
        eps: f32,
        input: Vec<Vec<f32>>,
        gamma: Vec<f32>,
        output: Vec<Vec<f32>>,
    }

    /// Resolve the flow net fixture file path.
    fn fixture_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("flow_net.json")
    }

    /// Load the flow net fixture JSON.
    fn load_fixture() -> FlowNetFixture {
        let data = fs::read_to_string(fixture_path()).expect("fixture missing");
        serde_json::from_str(&data).expect("fixture parse")
    }

    /// Load the RMSNorm fixture JSON.
    fn load_rmsnorm_fixture() -> RmsNormFixture {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("rmsnorm.json");
        let data = fs::read_to_string(path).expect("rmsnorm fixture read");
        serde_json::from_str(&data).expect("rmsnorm fixture parse")
    }

    /// Build a 2D tensor from nested vectors.
    fn tensor2(device: &NdArrayDevice, data: Vec<Vec<f32>>) -> Tensor<TestBackend, 2> {
        let rows = data.len();
        let cols = data.first().map(|row| row.len()).unwrap_or(0);
        let flat: Vec<f32> = data.into_iter().flatten().collect();
        let tensor_data = BurnTensorData::new(flat, [rows, cols]);
        Tensor::from_data(tensor_data, device)
    }

    /// Assign fixture weights to a linear layer.
    fn apply_linear(
        linear: &mut Linear<TestBackend>,
        weights: &LinearWeights,
        device: &NdArrayDevice,
    ) {
        let rows = weights.weight.len();
        let cols = weights.weight.first().map(|row| row.len()).unwrap_or(0);
        let flat: Vec<f32> = weights.weight.clone().into_iter().flatten().collect();
        let weight = Tensor::from_data(BurnTensorData::new(flat, [rows, cols]), device);
        linear.weight = Param::from_tensor(weight);
        let bias = Tensor::from_data(
            BurnTensorData::new(weights.bias.clone(), [weights.bias.len()]),
            device,
        );
        linear.bias = Some(Param::from_tensor(bias));
    }

    /// Assign fixture weights to a layer norm.
    fn apply_norm(
        norm: &mut LayerNorm<TestBackend>,
        weights: &NormWeights,
        device: &NdArrayDevice,
    ) {
        let gamma = Tensor::from_data(
            BurnTensorData::new(weights.weight.clone(), [weights.weight.len()]),
            device,
        );
        norm.gamma = Param::from_tensor(gamma);
        norm.beta = weights.bias.as_ref().map(|bias| {
            let beta = Tensor::from_data(BurnTensorData::new(bias.clone(), [bias.len()]), device);
            Param::from_tensor(beta)
        });
    }

    /// Assign fixture weights to an RMSNorm.
    fn apply_rms(norm: &mut RmsNorm<TestBackend>, weights: &[f32], device: &NdArrayDevice) {
        let gamma = Tensor::from_data(
            BurnTensorData::new(weights.to_vec(), [weights.len()]),
            device,
        );
        norm.gamma = Param::from_tensor(gamma);
    }

    #[test]
    fn flow_net_matches_fixture() {
        let fixture = load_fixture();
        let device = NdArrayDevice::default();

        let mut config = SimpleMlpAdaLnConfig::new(
            fixture.config.in_channels,
            fixture.config.model_channels,
            fixture.config.out_channels,
            fixture.config.cond_channels,
        );
        config.num_res_blocks = fixture.config.num_res_blocks;
        config.num_time_conds = fixture.config.num_time_conds;
        config.time_embed.frequency_embedding_size = fixture.config.frequency_embedding_size;
        config.time_embed.max_period = fixture.config.max_period;

        let mut model = SimpleMlpAdaLn::<TestBackend>::new(config, &device);
        apply_linear(&mut model.input_proj, &fixture.weights.input_proj, &device);
        apply_linear(&mut model.cond_embed, &fixture.weights.cond_embed, &device);

        for (embedder, weights) in model
            .time_embed
            .iter_mut()
            .zip(fixture.weights.time_embed.iter())
        {
            apply_linear(&mut embedder.proj_in, &weights.proj_in, &device);
            apply_linear(&mut embedder.proj_out, &weights.proj_out, &device);
            apply_rms(&mut embedder.norm, &weights.rms_weight, &device);
        }

        for (block, weights) in model
            .res_blocks
            .iter_mut()
            .zip(fixture.weights.res_blocks.iter())
        {
            apply_norm(&mut block.norm, &weights.norm, &device);
            apply_linear(&mut block.mlp_in, &weights.mlp_in, &device);
            apply_linear(&mut block.mlp_out, &weights.mlp_out, &device);
            apply_linear(&mut block.mod_linear, &weights.modulation, &device);
        }

        apply_norm(
            &mut model.final_layer.norm,
            &fixture.weights.final_layer.norm,
            &device,
        );
        apply_linear(
            &mut model.final_layer.linear,
            &fixture.weights.final_layer.linear,
            &device,
        );
        apply_linear(
            &mut model.final_layer.mod_linear,
            &fixture.weights.final_layer.modulation,
            &device,
        );

        let c = tensor2(&device, fixture.inputs.c);
        let s = tensor2(&device, fixture.inputs.s);
        let t = tensor2(&device, fixture.inputs.t);
        let x = tensor2(&device, fixture.inputs.x);
        let expected = tensor2(&device, fixture.expected.flow_out);

        let output = model.forward(c, s, t, x);
        output.to_data().assert_approx_eq(
            &expected.to_data(),
            Tolerance::<f32>::absolute(1e-4).set_relative(1e-4),
        );
    }

    #[test]
    fn lsd_decode_matches_fixture() {
        let fixture = load_fixture();
        let device = NdArrayDevice::default();

        let mut config = SimpleMlpAdaLnConfig::new(
            fixture.config.in_channels,
            fixture.config.model_channels,
            fixture.config.out_channels,
            fixture.config.cond_channels,
        );
        config.num_res_blocks = fixture.config.num_res_blocks;
        config.num_time_conds = fixture.config.num_time_conds;
        config.time_embed.frequency_embedding_size = fixture.config.frequency_embedding_size;
        config.time_embed.max_period = fixture.config.max_period;

        let mut model = SimpleMlpAdaLn::<TestBackend>::new(config, &device);
        apply_linear(&mut model.input_proj, &fixture.weights.input_proj, &device);
        apply_linear(&mut model.cond_embed, &fixture.weights.cond_embed, &device);

        for (embedder, weights) in model
            .time_embed
            .iter_mut()
            .zip(fixture.weights.time_embed.iter())
        {
            apply_linear(&mut embedder.proj_in, &weights.proj_in, &device);
            apply_linear(&mut embedder.proj_out, &weights.proj_out, &device);
            apply_rms(&mut embedder.norm, &weights.rms_weight, &device);
        }

        for (block, weights) in model
            .res_blocks
            .iter_mut()
            .zip(fixture.weights.res_blocks.iter())
        {
            apply_norm(&mut block.norm, &weights.norm, &device);
            apply_linear(&mut block.mlp_in, &weights.mlp_in, &device);
            apply_linear(&mut block.mlp_out, &weights.mlp_out, &device);
            apply_linear(&mut block.mod_linear, &weights.modulation, &device);
        }

        apply_norm(
            &mut model.final_layer.norm,
            &fixture.weights.final_layer.norm,
            &device,
        );
        apply_linear(
            &mut model.final_layer.linear,
            &fixture.weights.final_layer.linear,
            &device,
        );
        apply_linear(
            &mut model.final_layer.mod_linear,
            &fixture.weights.final_layer.modulation,
            &device,
        );

        let c = tensor2(&device, fixture.inputs.c);
        let noise = tensor2(&device, fixture.inputs.noise);
        let expected = tensor2(&device, fixture.expected.lsd_out);

        let output = lsd_decode(|s, t, x| model.forward(c.clone(), s, t, x), noise, 3);
        output.to_data().assert_approx_eq(
            &expected.to_data(),
            Tolerance::<f32>::absolute(1e-4).set_relative(1e-4),
        );
    }

    #[test]
    fn rmsnorm_matches_fixture() {
        let fixture = load_rmsnorm_fixture();
        let device = NdArrayDevice::default();

        let mut norm = RmsNormConfig::new(fixture.gamma.len())
            .with_epsilon(fixture.eps)
            .init::<TestBackend>(&device);

        let gamma = Tensor::from_data(
            BurnTensorData::new(fixture.gamma.clone(), [fixture.gamma.len()]),
            &device,
        );
        norm.gamma = Param::from_tensor(gamma);

        let input = tensor2(&device, fixture.input);
        let expected = tensor2(&device, fixture.output).to_data();
        let output = norm.forward(input).to_data();
        output.assert_approx_eq(
            &expected,
            Tolerance::<f32>::absolute(1e-4).set_relative(1e-4),
        );
    }
}
