use crate::modules::dummy_quantizer::DummyQuantizer;
use crate::modules::mimi_transformer::{MimiProjectedTransformer, MimiTransformerState};
use crate::modules::seanet::{SeanetDecoder, SeanetEncoder, SeanetState};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct MimiState<B: Backend> {
    pub encoder: SeanetState<B>,
    pub decoder: SeanetState<B>,
    pub encoder_transformer: MimiTransformerState<B>,
    pub decoder_transformer: MimiTransformerState<B>,
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
}

impl<B: Backend> MimiModel<B> {
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
    ) -> Self {
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
        }
    }

    pub fn init_state(&self, _batch_size: usize, _sequence_length: usize) -> MimiState<B> {
        todo!("MimiModel init_state not implemented")
    }

    pub fn encode_to_latent(&self, _input: Tensor<B, 3>) -> Tensor<B, 3> {
        todo!("MimiModel encode_to_latent not implemented")
    }

    pub fn decode_from_latent(&self, _latent: Tensor<B, 3>, _state: &mut MimiState<B>) -> Tensor<B, 3> {
        todo!("MimiModel decode_from_latent not implemented")
    }

    pub fn increment_step(&self, _state: &mut MimiState<B>, _increment: usize) {
        todo!("MimiModel increment_step not implemented")
    }
}
