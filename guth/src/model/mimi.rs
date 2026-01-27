use crate::modules::dummy_quantizer::DummyQuantizer;
use crate::modules::mimi_transformer::{MimiProjectedTransformer, MimiTransformerState};
use crate::modules::resample::{ConvDownsample1d, ConvDownsample1dState, ConvTrUpsample1d, ConvTrUpsample1dState};
use crate::modules::seanet::{SeanetDecoder, SeanetEncoder, SeanetState};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

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
}
