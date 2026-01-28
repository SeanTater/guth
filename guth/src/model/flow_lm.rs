use crate::conditioner::text::LutConditioner;
use crate::config::FlowLmConfig;
use crate::download::download_if_necessary;
use crate::modules::flow_net::{lsd_decode, SimpleMlpAdaLn};
use crate::modules::linear::apply_linear_3d as apply_linear_3d_util;
use crate::modules::transformer::{StreamingTransformer, StreamingTransformerState};
use crate::state::StreamingModule;
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Distribution, ElementConversion, Tensor, TensorData as BurnTensorData};
use burn_nn::{LayerNorm, Linear};
use std::collections::HashMap;
use safetensors::Dtype;

#[derive(Debug)]
pub struct FlowLmState<B: Backend> {
    pub transformer: StreamingTransformerState<B>,
}

#[derive(Debug)]
pub struct FlowLmModel<B: Backend> {
    pub conditioner: LutConditioner<B>,
    pub flow_net: SimpleMlpAdaLn<B>,
    pub transformer: StreamingTransformer<B>,
    pub input_linear: Linear<B>,
    pub out_norm: LayerNorm<B>,
    pub out_eos: Linear<B>,
    pub emb_std: Tensor<B, 1>,
    pub emb_mean: Tensor<B, 1>,
    pub bos_emb: Tensor<B, 1>,
    pub dim: usize,
    pub ldim: usize,
}

impl<B: Backend + 'static> FlowLmModel<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        conditioner: LutConditioner<B>,
        flow_net: SimpleMlpAdaLn<B>,
        transformer: StreamingTransformer<B>,
        input_linear: Linear<B>,
        out_norm: LayerNorm<B>,
        out_eos: Linear<B>,
        emb_std: Tensor<B, 1>,
        emb_mean: Tensor<B, 1>,
        bos_emb: Tensor<B, 1>,
        dim: usize,
        ldim: usize,
    ) -> Self {
        Self {
            conditioner,
            flow_net,
            transformer,
            input_linear,
            out_norm,
            out_eos,
            emb_std,
            emb_mean,
            bos_emb,
            dim,
            ldim,
        }
    }

    pub fn init_state(&self, batch_size: usize, sequence_length: usize) -> FlowLmState<B> {
        let transformer = self.transformer.init_state(batch_size, sequence_length);
        FlowLmState { transformer }
    }

    pub fn from_config(
        config: &FlowLmConfig,
        latent_dim: usize,
        device: &B::Device,
    ) -> anyhow::Result<Self> {
        let tokenizer_path = download_if_necessary(&config.lookup_table.tokenizer_path)?;
        let conditioner = LutConditioner::new(
            config.lookup_table.n_bins as usize,
            tokenizer_path,
            config.lookup_table.dim as usize,
            config.transformer.d_model as usize,
            device,
        )?;

        let mut flow_config = crate::modules::flow_net::SimpleMlpAdaLnConfig::new(
            latent_dim,
            config.flow.dim as usize,
            latent_dim,
            config.transformer.d_model as usize,
        );
        flow_config.num_res_blocks = config.flow.depth as usize;
        flow_config.num_time_conds = 2;

        let flow_net = SimpleMlpAdaLn::new(flow_config, device);

        let ffn_dim = (config.transformer.d_model * config.transformer.hidden_scale) as usize;
        let transformer = StreamingTransformer::new(
            crate::modules::transformer::StreamingTransformerConfig {
                num_layers: config.transformer.num_layers as usize,
                layer: crate::modules::transformer::StreamingTransformerLayerConfig {
                    d_model: config.transformer.d_model as usize,
                    num_heads: config.transformer.num_heads as usize,
                    ffn_dim,
                    causal: true,
                    context: None,
                    rope_max_seq: None,
                    rope_theta: config.transformer.max_period as f32,
                    layer_scale: None,
                },
            },
            device,
        );

        let input_linear = burn_nn::LinearConfig::new(latent_dim, config.transformer.d_model as usize)
            .with_bias(false)
            .init::<B>(device);
        let out_norm = burn_nn::LayerNormConfig::new(config.transformer.d_model as usize)
            .with_epsilon(1e-5)
            .init::<B>(device);
        let out_eos = burn_nn::LinearConfig::new(config.transformer.d_model as usize, 1)
            .init::<B>(device);

        let emb_std = Tensor::<B, 1>::ones([latent_dim], device);
        let emb_mean = Tensor::<B, 1>::zeros([latent_dim], device);
        let bos_emb = Tensor::<B, 1>::random([latent_dim], Distribution::Normal(0.0, 1.0), device);

        Ok(Self::new(
            conditioner,
            flow_net,
            transformer,
            input_linear,
            out_norm,
            out_eos,
            emb_std,
            emb_mean,
            bos_emb,
            config.transformer.d_model as usize,
            latent_dim,
        ))
    }

    pub fn forward(
        &self,
        sequence: Tensor<B, 3>,
        text_embeddings: Tensor<B, 3>,
        state: &mut FlowLmState<B>,
        lsd_decode_steps: usize,
        temp: f32,
        noise_clamp: Option<f32>,
        eos_threshold: f32,
    ) -> (Tensor<B, 2>, Tensor<B, 2, Bool>) {
        let [batch, seq_len, _] = sequence.dims();
        let device = sequence.device();
        let sequence = if seq_len == 0 {
            sequence
        } else {
            let bos = self
                .bos_emb
                .clone()
                .reshape([1, 1, self.ldim])
                .repeat_dim(0, batch)
                .repeat_dim(1, seq_len);
            let nan_mask = sequence.clone().is_nan();
            sequence.mask_where(nan_mask, bos)
        };

        let input_ = apply_linear_3d(&self.input_linear, sequence.clone());
        let transformer_out = self.backbone(input_, text_embeddings, seq_len, &mut state.transformer);
        let last_index = if seq_len == 0 {
            transformer_out.dims()[1].saturating_sub(1)
        } else {
            seq_len - 1
        };
        let last = transformer_out.narrow(1, last_index, 1).reshape([batch, self.dim]);
        let eos_logits = self.out_eos.forward(last.clone());
        let eos = eos_logits.greater_elem(eos_threshold);

        assert!(lsd_decode_steps > 0, "lsd_decode_steps must be > 0");
        let noise = make_noise::<B>([batch, self.ldim], temp, noise_clamp, &device);
        let conditioned = last;
        let output = lsd_decode(
            |s, t, x| self.flow_net.forward(conditioned.clone(), s, t, x),
            noise,
            lsd_decode_steps,
        );

        (output, eos)
    }

    pub fn sample_next_latent(
        &self,
        sequence: Tensor<B, 3>,
        text_embeddings: Tensor<B, 3>,
        state: &mut FlowLmState<B>,
        lsd_decode_steps: usize,
        temp: f32,
        noise_clamp: Option<f32>,
        eos_threshold: f32,
    ) -> (Tensor<B, 2>, Tensor<B, 2, Bool>) {
        self.forward(
            sequence,
            text_embeddings,
            state,
            lsd_decode_steps,
            temp,
            noise_clamp,
            eos_threshold,
        )
    }

    fn backbone(
        &self,
        input: Tensor<B, 3>,
        text_embeddings: Tensor<B, 3>,
        sequence_len: usize,
        state: &mut StreamingTransformerState<B>,
    ) -> Tensor<B, 3> {
        let text_len = text_embeddings.dims()[1];
        let input_len = input.dims()[1];
        let combined = if text_len == 0 {
            input
        } else if input_len == 0 {
            text_embeddings
        } else {
            Tensor::cat(vec![text_embeddings, input], 1)
        };
        let mut output = self.transformer.forward(combined, state);
        output = apply_layer_norm_3d(&self.out_norm, output);
        if sequence_len == 0 {
            return output;
        }
        let total_len = output.dims()[1];
        output.narrow(1, total_len - sequence_len, sequence_len)
    }

    pub fn load_state_dict(
        &mut self,
        state: &HashMap<String, crate::weights::TensorData>,
        device: &B::Device,
    ) -> anyhow::Result<()> {
        for (name, tensor) in state {
            if name == "bos_emb" {
                self.bos_emb = tensor1_from_data(tensor, device)?;
                continue;
            }
            if name == "emb_std" {
                self.emb_std = tensor1_from_data(tensor, device)?;
                continue;
            }
            if name == "emb_mean" {
                self.emb_mean = tensor1_from_data(tensor, device)?;
                continue;
            }
            if name == "conditioner.embed.weight" {
                let weight = tensor2_from_data(tensor, device)?;
                self.conditioner.embed.weight = burn::module::Param::from_tensor(weight);
                continue;
            }
            if name == "input_linear.weight" {
                let weight = tensor2_from_data(tensor, device)?;
                self.input_linear.weight = burn::module::Param::from_tensor(weight.transpose());
                continue;
            }
            if name == "out_norm.weight" {
                let weight = tensor1_from_data(tensor, device)?;
                self.out_norm.gamma = burn::module::Param::from_tensor(weight);
                continue;
            }
            if name == "out_norm.bias" {
                let bias = tensor1_from_data(tensor, device)?;
                self.out_norm.beta = Some(burn::module::Param::from_tensor(bias));
                continue;
            }
            if name == "out_eos.weight" {
                let weight = tensor2_from_data(tensor, device)?;
                self.out_eos.weight = burn::module::Param::from_tensor(weight.transpose());
                continue;
            }
            if name == "out_eos.bias" {
                let bias = tensor1_from_data(tensor, device)?;
                self.out_eos.bias = Some(burn::module::Param::from_tensor(bias));
                continue;
            }

            if let Some(rest) = name.strip_prefix("transformer.layers.") {
                apply_transformer_weight(&mut self.transformer, rest, tensor, device)?;
                continue;
            }

            if let Some(rest) = name.strip_prefix("flow_net.") {
                apply_flow_net_weight(&mut self.flow_net, rest, tensor, device)?;
                continue;
            }
        }

        Ok(())
    }
}

fn apply_linear_3d<B: Backend + 'static>(linear: &Linear<B>, input: Tensor<B, 3>) -> Tensor<B, 3> {
    apply_linear_3d_util(linear, input)
}

fn apply_layer_norm_3d<B: Backend>(norm: &LayerNorm<B>, input: Tensor<B, 3>) -> Tensor<B, 3> {
    let [batch, seq, dim] = input.dims();
    if seq == 0 {
        return input;
    }
    let flat = input.reshape([batch * seq, dim]);
    let output = norm.forward(flat);
    output.reshape([batch, seq, dim])
}

fn tensor_f32(tensor: &crate::weights::TensorData) -> anyhow::Result<Vec<f32>> {
    match tensor.dtype {
        Dtype::F32 => {
            let mut values = Vec::with_capacity(tensor.data.len() / 4);
            for chunk in tensor.data.chunks_exact(4) {
                values.push(f32::from_le_bytes(chunk.try_into().unwrap()));
            }
            Ok(values)
        }
        Dtype::BF16 => {
            let mut values = Vec::with_capacity(tensor.data.len() / 2);
            for chunk in tensor.data.chunks_exact(2) {
                let bits = u16::from_le_bytes(chunk.try_into().unwrap()) as u32;
                values.push(f32::from_bits(bits << 16));
            }
            Ok(values)
        }
        _ => anyhow::bail!("Unsupported dtype {:?}", tensor.dtype),
    }
}

fn tensor1_from_data<B: Backend>(
    tensor: &crate::weights::TensorData,
    device: &B::Device,
) -> anyhow::Result<Tensor<B, 1>> {
    let shape: [usize; 1] = tensor.shape.clone().try_into().map_err(|_| {
        anyhow::anyhow!("Expected 1D tensor, got shape {:?}", tensor.shape)
    })?;
    let values = tensor_f32(tensor)?;
    Ok(Tensor::from_data(BurnTensorData::new(values, shape), device))
}

fn tensor2_from_data<B: Backend>(
    tensor: &crate::weights::TensorData,
    device: &B::Device,
) -> anyhow::Result<Tensor<B, 2>> {
    let shape: [usize; 2] = tensor.shape.clone().try_into().map_err(|_| {
        anyhow::anyhow!("Expected 2D tensor, got shape {:?}", tensor.shape)
    })?;
    let values = tensor_f32(tensor)?;
    Ok(Tensor::from_data(BurnTensorData::new(values, shape), device))
}

fn apply_transformer_weight<B: Backend>(
    transformer: &mut StreamingTransformer<B>,
    rest: &str,
    tensor: &crate::weights::TensorData,
    device: &B::Device,
) -> anyhow::Result<()> {
    let mut parts = rest.split('.');
    let layer_idx: usize = parts
        .next()
        .ok_or_else(|| anyhow::anyhow!("missing layer index"))?
        .parse()?;
    let layer = transformer
        .layers
        .get_mut(layer_idx)
        .ok_or_else(|| anyhow::anyhow!("invalid layer index {layer_idx}"))?;
    let tail: Vec<&str> = parts.collect();
    match tail.as_slice() {
        ["self_attn", "in_proj", "weight"] => {
            let weight = tensor2_from_data(tensor, device)?;
            layer.qkv.weight = burn::module::Param::from_tensor(weight.transpose());
        }
        ["self_attn", "out_proj", "weight"] => {
            let weight = tensor2_from_data(tensor, device)?;
            layer.out_proj.weight = burn::module::Param::from_tensor(weight.transpose());
        }
        ["norm1", "weight"] => {
            let weight = tensor1_from_data(tensor, device)?;
            layer.norm1.gamma = burn::module::Param::from_tensor(weight);
        }
        ["norm1", "bias"] => {
            let bias = tensor1_from_data(tensor, device)?;
            layer.norm1.beta = Some(burn::module::Param::from_tensor(bias));
        }
        ["norm2", "weight"] => {
            let weight = tensor1_from_data(tensor, device)?;
            layer.norm2.gamma = burn::module::Param::from_tensor(weight);
        }
        ["norm2", "bias"] => {
            let bias = tensor1_from_data(tensor, device)?;
            layer.norm2.beta = Some(burn::module::Param::from_tensor(bias));
        }
        ["linear1", "weight"] => {
            let weight = tensor2_from_data(tensor, device)?;
            layer.ffn_in.weight = burn::module::Param::from_tensor(weight.transpose());
        }
        ["linear2", "weight"] => {
            let weight = tensor2_from_data(tensor, device)?;
            layer.ffn_out.weight = burn::module::Param::from_tensor(weight.transpose());
        }
        _ => {}
    }
    Ok(())
}

fn apply_flow_net_weight<B: Backend>(
    flow_net: &mut SimpleMlpAdaLn<B>,
    rest: &str,
    tensor: &crate::weights::TensorData,
    device: &B::Device,
) -> anyhow::Result<()> {
    if rest == "cond_embed.weight" {
        let weight = tensor2_from_data(tensor, device)?;
        flow_net.cond_embed.weight = burn::module::Param::from_tensor(weight.transpose());
        return Ok(());
    }
    if rest == "cond_embed.bias" {
        let bias = tensor1_from_data(tensor, device)?;
        flow_net.cond_embed.bias = Some(burn::module::Param::from_tensor(bias));
        return Ok(());
    }
    if rest == "input_proj.weight" {
        let weight = tensor2_from_data(tensor, device)?;
        flow_net.input_proj.weight = burn::module::Param::from_tensor(weight.transpose());
        return Ok(());
    }
    if rest == "input_proj.bias" {
        let bias = tensor1_from_data(tensor, device)?;
        flow_net.input_proj.bias = Some(burn::module::Param::from_tensor(bias));
        return Ok(());
    }
    if let Some(rest) = rest.strip_prefix("time_embed.") {
        let mut parts = rest.split('.');
        let idx: usize = parts
            .next()
            .ok_or_else(|| anyhow::anyhow!("missing time embed index"))?
            .parse()?;
        let embedder = flow_net
            .time_embed
            .get_mut(idx)
            .ok_or_else(|| anyhow::anyhow!("invalid time embed index {idx}"))?;
        let tail: Vec<&str> = parts.collect();
        match tail.as_slice() {
            ["freqs"] => {
                embedder.freqs = tensor1_from_data(tensor, device)?;
            }
            ["mlp", "0", "weight"] => {
                let weight = tensor2_from_data(tensor, device)?;
                embedder.proj_in.weight = burn::module::Param::from_tensor(weight.transpose());
            }
            ["mlp", "0", "bias"] => {
                let bias = tensor1_from_data(tensor, device)?;
                embedder.proj_in.bias = Some(burn::module::Param::from_tensor(bias));
            }
            ["mlp", "2", "weight"] => {
                let weight = tensor2_from_data(tensor, device)?;
                embedder.proj_out.weight = burn::module::Param::from_tensor(weight.transpose());
            }
            ["mlp", "2", "bias"] => {
                let bias = tensor1_from_data(tensor, device)?;
                embedder.proj_out.bias = Some(burn::module::Param::from_tensor(bias));
            }
            ["mlp", "3", "alpha"] => {
                let weight = tensor1_from_data(tensor, device)?;
                embedder.norm.gamma = burn::module::Param::from_tensor(weight);
            }
            _ => {}
        }
        return Ok(());
    }
    if let Some(rest) = rest.strip_prefix("res_blocks.") {
        let mut parts = rest.split('.');
        let idx: usize = parts
            .next()
            .ok_or_else(|| anyhow::anyhow!("missing resblock index"))?
            .parse()?;
        let block = flow_net
            .res_blocks
            .get_mut(idx)
            .ok_or_else(|| anyhow::anyhow!("invalid resblock index {idx}"))?;
        let tail: Vec<&str> = parts.collect();
        match tail.as_slice() {
            ["in_ln", "weight"] => {
                let weight = tensor1_from_data(tensor, device)?;
                block.norm.gamma = burn::module::Param::from_tensor(weight);
            }
            ["in_ln", "bias"] => {
                let bias = tensor1_from_data(tensor, device)?;
                block.norm.beta = Some(burn::module::Param::from_tensor(bias));
            }
            ["mlp", "0", "weight"] => {
                let weight = tensor2_from_data(tensor, device)?;
                block.mlp_in.weight = burn::module::Param::from_tensor(weight.transpose());
            }
            ["mlp", "0", "bias"] => {
                let bias = tensor1_from_data(tensor, device)?;
                block.mlp_in.bias = Some(burn::module::Param::from_tensor(bias));
            }
            ["mlp", "2", "weight"] => {
                let weight = tensor2_from_data(tensor, device)?;
                block.mlp_out.weight = burn::module::Param::from_tensor(weight.transpose());
            }
            ["mlp", "2", "bias"] => {
                let bias = tensor1_from_data(tensor, device)?;
                block.mlp_out.bias = Some(burn::module::Param::from_tensor(bias));
            }
            ["adaLN_modulation", "1", "weight"] => {
                let weight = tensor2_from_data(tensor, device)?;
                block.mod_linear.weight = burn::module::Param::from_tensor(weight.transpose());
            }
            ["adaLN_modulation", "1", "bias"] => {
                let bias = tensor1_from_data(tensor, device)?;
                block.mod_linear.bias = Some(burn::module::Param::from_tensor(bias));
            }
            _ => {}
        }
        return Ok(());
    }
    if let Some(rest) = rest.strip_prefix("final_layer.") {
        let tail: Vec<&str> = rest.split('.').collect();
        match tail.as_slice() {
            ["linear", "weight"] => {
                let weight = tensor2_from_data(tensor, device)?;
                flow_net.final_layer.linear.weight =
                    burn::module::Param::from_tensor(weight.transpose());
            }
            ["linear", "bias"] => {
                let bias = tensor1_from_data(tensor, device)?;
                flow_net.final_layer.linear.bias = Some(burn::module::Param::from_tensor(bias));
            }
            ["adaLN_modulation", "1", "weight"] => {
                let weight = tensor2_from_data(tensor, device)?;
                flow_net.final_layer.mod_linear.weight =
                    burn::module::Param::from_tensor(weight.transpose());
            }
            ["adaLN_modulation", "1", "bias"] => {
                let bias = tensor1_from_data(tensor, device)?;
                flow_net.final_layer.mod_linear.bias =
                    Some(burn::module::Param::from_tensor(bias));
            }
            _ => {}
        }
    }
    Ok(())
}

fn make_noise<B: Backend>(
    shape: [usize; 2],
    temp: f32,
    noise_clamp: Option<f32>,
    device: &B::Device,
) -> Tensor<B, 2> {
    if temp == 0.0 {
        return Tensor::<B, 2>::zeros(shape, device);
    }
    let std = (temp as f64).sqrt();
    if let Some(clamp) = noise_clamp {
        return truncated_normal(shape, std, clamp, device);
    }
    Tensor::<B, 2>::random(shape, Distribution::Normal(0.0, std), device)
}

fn truncated_normal<B: Backend>(
    shape: [usize; 2],
    std: f64,
    clamp: f32,
    device: &B::Device,
) -> Tensor<B, 2> {
    let mut noise = Tensor::<B, 2>::random(shape, Distribution::Normal(0.0, std), device);
    let mut mask = out_of_bounds(noise.clone(), clamp);
    let mut remaining: bool = mask.clone().any().into_scalar().elem();
    let mut attempts = 0;
    while remaining && attempts < 10 {
        let resample = Tensor::<B, 2>::random(shape, Distribution::Normal(0.0, std), device);
        noise = noise.mask_where(mask, resample);
        mask = out_of_bounds(noise.clone(), clamp);
        remaining = mask.clone().any().into_scalar().elem();
        attempts += 1;
    }
    if remaining {
        noise = noise.clamp(-clamp, clamp);
    }
    noise
}

fn out_of_bounds<B: Backend>(noise: Tensor<B, 2>, clamp: f32) -> Tensor<B, 2, Bool> {
    noise
        .clone()
        .lower_elem(-clamp)
        .bool_or(noise.greater_elem(clamp))
}

#[cfg(test)]
mod tests {
    use super::{apply_linear_3d, make_noise, tensor_f32};
    use burn_nn::LinearConfig;
    use burn_ndarray::{NdArray, NdArrayDevice};
    use safetensors::Dtype;

    #[test]
    fn truncated_noise_respects_clamp() {
        let device = NdArrayDevice::default();
        let noise = make_noise::<NdArray<f32>>([8, 8], 1.0, Some(0.5), &device);
        let data = noise.to_data();
        let values = data.as_slice::<f32>().expect("noise values");
        assert!(values.iter().all(|v| v.abs() <= 0.5 + 1e-6));
    }

    #[test]
    fn tensor_f32_decodes_bf16() {
        let values = [0.0_f32, 1.2345_f32, -2.75_f32];
        let mut data = Vec::new();
        let mut expected = Vec::new();
        for value in values {
            let bits = value.to_bits();
            let bf16 = (bits >> 16) as u16;
            data.extend_from_slice(&bf16.to_le_bytes());
            expected.push(f32::from_bits((bf16 as u32) << 16));
        }
        let tensor = crate::weights::TensorData {
            dtype: Dtype::BF16,
            shape: vec![expected.len()],
            data,
        };
        let decoded = tensor_f32(&tensor).expect("decode bf16");
        assert_eq!(decoded, expected);
    }

    #[test]
    fn apply_linear_3d_empty_sequence_returns_empty() {
        let device = NdArrayDevice::default();
        let linear = LinearConfig::new(4, 6).init::<NdArray<f32>>(&device);
        let input = burn::tensor::Tensor::<NdArray<f32>, 3>::zeros([1, 0, 4], &device);
        let output = apply_linear_3d(&linear, input);
        assert_eq!(output.dims(), [1, 0, 6]);
    }
}
