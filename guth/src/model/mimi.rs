use crate::modules::dummy_quantizer::DummyQuantizer;
use crate::modules::mimi_transformer::{
    MimiProjectedTransformer, MimiTransformerState, ProjectedOutput,
};
use crate::modules::resample::{
    ConvDownsample1d, ConvDownsample1dState, ConvTrUpsample1d, ConvTrUpsample1dState,
};
use crate::modules::seanet::{SeanetDecoder, SeanetEncoder, SeanetLayer, SeanetState};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData as BurnTensorData};
use burn::module::Param;
use safetensors::Dtype;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct MimiState<B: Backend> {
    pub encoder: SeanetState<B>,
    pub decoder: SeanetState<B>,
    pub encoder_transformer: MimiTransformerState<B>,
    pub decoder_transformer: MimiTransformerState<B>,
    pub downsample: Option<ConvDownsample1dState<B>>,
    pub upsample: Option<ConvTrUpsample1dState<B>>,
}

#[derive(Debug, Clone)]
pub struct MimiModel<B: Backend> {
    pub encoder: SeanetEncoder<B>,
    pub decoder: SeanetDecoder<B>,
    pub encoder_transformer: MimiProjectedTransformer<B>,
    pub decoder_transformer: MimiProjectedTransformer<B>,
    pub quantizer: DummyQuantizer<B>,
    pub frame_rate: f32,
    pub encoder_frame_rate: f32,
    pub sample_rate: usize,
    pub channels: usize,
    pub dimension: usize,
    pub downsample: Option<ConvDownsample1d<B>>,
    pub upsample: Option<ConvTrUpsample1d<B>>,
}

impl<B: Backend> MimiModel<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        encoder: SeanetEncoder<B>,
        decoder: SeanetDecoder<B>,
        encoder_transformer: MimiProjectedTransformer<B>,
        decoder_transformer: MimiProjectedTransformer<B>,
        quantizer: DummyQuantizer<B>,
        frame_rate: f32,
        encoder_frame_rate: f32,
        sample_rate: usize,
        channels: usize,
        dimension: usize,
        downsample: Option<ConvDownsample1d<B>>,
        upsample: Option<ConvTrUpsample1d<B>>,
    ) -> Self {
        if (encoder_frame_rate - frame_rate).abs() > f32::EPSILON {
            assert!(
                downsample.is_some() && upsample.is_some(),
                "resample weights required when encoder_frame_rate differs from frame_rate"
            );
        }
        Self {
            encoder,
            decoder,
            encoder_transformer,
            decoder_transformer,
            quantizer,
            frame_rate,
            encoder_frame_rate,
            sample_rate,
            channels,
            dimension,
            downsample,
            upsample,
        }
    }

    pub fn frame_size(&self) -> usize {
        (self.sample_rate as f32 / self.frame_rate) as usize
    }

    pub fn init_state(&self, batch_size: usize, _sequence_length: usize, device: &B::Device) -> MimiState<B> {
        let encoder = self.encoder.init_state(batch_size);
        let decoder = self.decoder.init_state(batch_size);
        let encoder_transformer = self.encoder_transformer.init_state(batch_size, device);
        let decoder_transformer = self.decoder_transformer.init_state(batch_size, device);
        let downsample = self
            .downsample
            .as_ref()
            .map(|downsample| downsample.init_state(batch_size, device));
        let upsample = self
            .upsample
            .as_ref()
            .map(|upsample| upsample.init_state(batch_size, device));

        MimiState {
            encoder,
            decoder,
            encoder_transformer,
            decoder_transformer,
            downsample,
            upsample,
        }
    }

    pub fn encode_to_latent(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, channels, length] = input.dims();
        assert_eq!(channels, self.channels);

        let frame_size = self.frame_size();
        let remainder = length % frame_size;
        let input = if remainder == 0 {
            input
        } else {
            let pad = frame_size - remainder;
            let device = input.device();
            let padding = Tensor::zeros([batch, channels, pad], &device);
            Tensor::cat(vec![input, padding], 2)
        };

        let mut encoder_state = self.encoder.init_state(batch);
        let mut transformer_state = self.encoder_transformer.init_state(batch, &input.device());
        let mut downsample_state = self
            .downsample
            .as_ref()
            .map(|downsample| downsample.init_state(batch, &input.device()));

        let emb = self
            .encoder
            .forward(input, &mut encoder_state)
            .expect("seanet encoder");
        let mut outputs = self.encoder_transformer.forward(emb, &mut transformer_state);
        let emb = outputs.remove(0);
        self.to_framerate(emb, downsample_state.as_mut())
    }

    pub fn decode_from_latent(&self, latent: Tensor<B, 3>, state: &mut MimiState<B>) -> Tensor<B, 3> {
        let emb = self.to_encoder_framerate(latent, state.upsample.as_mut());
        let mut outputs = self
            .decoder_transformer
            .forward(emb, &mut state.decoder_transformer);
        let emb = outputs.remove(0);
        self.decoder
            .forward(emb, &mut state.decoder)
            .expect("seanet decoder")
    }

    pub fn increment_step(&self, state: &mut MimiState<B>, increment: usize) {
        self.encoder_transformer
            .increment_step(&mut state.encoder_transformer, increment);
        self.decoder_transformer
            .increment_step(&mut state.decoder_transformer, increment);
    }

    fn to_framerate(&self, input: Tensor<B, 3>, state: Option<&mut ConvDownsample1dState<B>>) -> Tensor<B, 3> {
        match (&self.downsample, state) {
            (Some(downsample), Some(state)) => downsample.forward(input, state),
            _ => input,
        }
    }

    fn to_encoder_framerate(
        &self,
        input: Tensor<B, 3>,
        state: Option<&mut ConvTrUpsample1dState<B>>,
    ) -> Tensor<B, 3> {
        match (&self.upsample, state) {
            (Some(upsample), Some(state)) => upsample.forward(input, state).expect("upsample"),
            _ => input,
        }
    }

    pub fn load_state_dict(
        &mut self,
        state: &HashMap<String, crate::weights::TensorData>,
        device: &B::Device,
    ) -> anyhow::Result<()> {
        for (name, tensor) in state {
            if name == "quantizer.weight" {
                self.quantizer.weight = tensor3_from_data(tensor, device)?;
                continue;
            }
            if let Some(rest) = name.strip_prefix("encoder.layers.") {
                apply_seanet_weight(&mut self.encoder.layers, rest, tensor, device)?;
                continue;
            }
            if let Some(rest) = name.strip_prefix("decoder.layers.") {
                apply_seanet_weight(&mut self.decoder.layers, rest, tensor, device)?;
                continue;
            }
            if let Some(rest) = name.strip_prefix("encoder_transformer.") {
                apply_projected_transformer_weight(&mut self.encoder_transformer, rest, tensor, device)?;
                continue;
            }
            if let Some(rest) = name.strip_prefix("decoder_transformer.") {
                apply_projected_transformer_weight(&mut self.decoder_transformer, rest, tensor, device)?;
                continue;
            }
            if let Some(rest) = name.strip_prefix("downsample.") {
                if let Some(downsample) = self.downsample.as_mut() {
                    apply_downsample_weight(downsample, rest, tensor, device)?;
                }
                continue;
            }
            if let Some(rest) = name.strip_prefix("upsample.") {
                if let Some(upsample) = self.upsample.as_mut() {
                    apply_upsample_weight(upsample, rest, tensor, device)?;
                }
            }
        }

        Ok(())
    }
}

fn tensor_f32(tensor: &crate::weights::TensorData) -> anyhow::Result<Vec<f32>> {
    if tensor.dtype != Dtype::F32 {
        anyhow::bail!("Unsupported dtype {:?}", tensor.dtype);
    }
    let mut values = Vec::with_capacity(tensor.data.len() / 4);
    for chunk in tensor.data.chunks_exact(4) {
        values.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(values)
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

fn tensor3_from_data<B: Backend>(
    tensor: &crate::weights::TensorData,
    device: &B::Device,
) -> anyhow::Result<Tensor<B, 3>> {
    let shape: [usize; 3] = tensor.shape.clone().try_into().map_err(|_| {
        anyhow::anyhow!("Expected 3D tensor, got shape {:?}", tensor.shape)
    })?;
    let values = tensor_f32(tensor)?;
    Ok(Tensor::from_data(BurnTensorData::new(values, shape), device))
}

fn apply_seanet_weight<B: Backend>(
    layers: &mut [SeanetLayer<B>],
    rest: &str,
    tensor: &crate::weights::TensorData,
    device: &B::Device,
) -> anyhow::Result<()> {
    let mut parts = rest.split('.');
    let layer_idx: usize = parts
        .next()
        .ok_or_else(|| anyhow::anyhow!("missing layer index"))?
        .parse()?;
    let layer = layers
        .get_mut(layer_idx)
        .ok_or_else(|| anyhow::anyhow!("invalid seanet layer index {layer_idx}"))?;
    let tail: Vec<&str> = parts.collect();
    match (layer, tail.as_slice()) {
        (SeanetLayer::Conv1d(conv), ["conv", "weight"]) => {
            conv.weight = tensor3_from_data(tensor, device)?;
        }
        (SeanetLayer::Conv1d(conv), ["conv", "bias"]) => {
            conv.bias = Some(tensor1_from_data(tensor, device)?);
        }
        (SeanetLayer::ConvTranspose1d(conv), ["conv_transpose", "weight"]) => {
            conv.weight = tensor3_from_data(tensor, device)?;
        }
        (SeanetLayer::ConvTranspose1d(conv), ["conv_transpose", "bias"]) => {
            conv.bias = Some(tensor1_from_data(tensor, device)?);
        }
        (SeanetLayer::ResBlock(block), ["resblock", conv_idx, "weight"]) => {
            let idx: usize = conv_idx.parse()?;
            let conv = block
                .convs
                .get_mut(idx)
                .ok_or_else(|| anyhow::anyhow!("invalid resblock conv index {idx}"))?;
            conv.weight = tensor3_from_data(tensor, device)?;
        }
        (SeanetLayer::ResBlock(block), ["resblock", conv_idx, "bias"]) => {
            let idx: usize = conv_idx.parse()?;
            let conv = block
                .convs
                .get_mut(idx)
                .ok_or_else(|| anyhow::anyhow!("invalid resblock conv index {idx}"))?;
            conv.bias = Some(tensor1_from_data(tensor, device)?);
        }
        _ => {}
    }
    Ok(())
}

fn apply_projected_transformer_weight<B: Backend>(
    transformer: &mut MimiProjectedTransformer<B>,
    rest: &str,
    tensor: &crate::weights::TensorData,
    device: &B::Device,
) -> anyhow::Result<()> {
    if rest == "input_proj.weight" {
        if let Some(proj) = transformer.input_proj.as_mut() {
            set_linear_weight(proj, tensor, device)?;
        }
        return Ok(());
    }
    if let Some(rest) = rest.strip_prefix("output_projs.") {
        let mut parts = rest.split('.');
        let idx: usize = parts
            .next()
            .ok_or_else(|| anyhow::anyhow!("missing output proj index"))?
            .parse()?;
        let tail: Vec<&str> = parts.collect();
        if let (Some(ProjectedOutput::Linear(linear)), ["weight"]) =
            (transformer.output_projs.get_mut(idx), tail.as_slice())
        {
            set_linear_weight(linear, tensor, device)?;
        }
        return Ok(());
    }
    if let Some(rest) = rest.strip_prefix("layers.") {
        let mut parts = rest.split('.');
        let idx: usize = parts
            .next()
            .ok_or_else(|| anyhow::anyhow!("missing layer index"))?
            .parse()?;
        let layer = transformer
            .transformer
            .layers
            .get_mut(idx)
            .ok_or_else(|| anyhow::anyhow!("invalid transformer layer {idx}"))?;
        let tail: Vec<&str> = parts.collect();
        match tail.as_slice() {
            ["self_attn", "in_proj", "weight"] => {
                set_linear_weight(&mut layer.self_attn.in_proj, tensor, device)?;
            }
            ["self_attn", "out_proj", "weight"] => {
                set_linear_weight(&mut layer.self_attn.out_proj, tensor, device)?;
            }
            ["norm1", "weight"] => {
                let weight = tensor1_from_data(tensor, device)?;
                layer.norm1.gamma = Param::from_tensor(weight);
            }
            ["norm1", "bias"] => {
                let bias = tensor1_from_data(tensor, device)?;
                layer.norm1.beta = Some(Param::from_tensor(bias));
            }
            ["norm2", "weight"] => {
                let weight = tensor1_from_data(tensor, device)?;
                layer.norm2.gamma = Param::from_tensor(weight);
            }
            ["norm2", "bias"] => {
                let bias = tensor1_from_data(tensor, device)?;
                layer.norm2.beta = Some(Param::from_tensor(bias));
            }
            ["linear1", "weight"] => {
                set_linear_weight(&mut layer.linear1, tensor, device)?;
            }
            ["linear2", "weight"] => {
                set_linear_weight(&mut layer.linear2, tensor, device)?;
            }
            ["layer_scale_1", "scale"] => {
                if let Some(scale) = layer.layer_scale_1.as_mut() {
                    scale.scale = tensor1_from_data(tensor, device)?;
                }
            }
            ["layer_scale_2", "scale"] => {
                if let Some(scale) = layer.layer_scale_2.as_mut() {
                    scale.scale = tensor1_from_data(tensor, device)?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn set_linear_weight<B: Backend>(
    linear: &mut burn_nn::Linear<B>,
    tensor: &crate::weights::TensorData,
    device: &B::Device,
) -> anyhow::Result<()> {
    let weight = tensor2_from_data(tensor, device)?;
    linear.weight = Param::from_tensor(weight.transpose());
    Ok(())
}

fn apply_downsample_weight<B: Backend>(
    downsample: &mut ConvDownsample1d<B>,
    rest: &str,
    tensor: &crate::weights::TensorData,
    device: &B::Device,
) -> anyhow::Result<()> {
    if rest == "conv.weight" {
        downsample.conv.weight = tensor3_from_data(tensor, device)?;
    }
    Ok(())
}

fn apply_upsample_weight<B: Backend>(
    upsample: &mut ConvTrUpsample1d<B>,
    rest: &str,
    tensor: &crate::weights::TensorData,
    device: &B::Device,
) -> anyhow::Result<()> {
    if rest == "conv.weight" {
        upsample.conv.weight = tensor3_from_data(tensor, device)?;
    }
    Ok(())
}
