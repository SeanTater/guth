use crate::conditioner::text::LutConditioner;
use crate::modules::flow_net::{lsd_decode, SimpleMlpAdaLn};
use crate::modules::transformer::{StreamingTransformer, StreamingTransformerState};
use crate::state::StreamingModule;
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Distribution, Tensor};
use burn_nn::{LayerNorm, Linear};

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

impl<B: Backend> FlowLmModel<B> {
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
        let bos = self
            .bos_emb
            .clone()
            .reshape([1, 1, self.ldim])
            .repeat_dim(0, batch)
            .repeat_dim(1, seq_len);
        let nan_mask = sequence.clone().is_nan();
        let sequence = sequence.mask_where(nan_mask, bos);

        let input_ = apply_linear_3d(&self.input_linear, sequence.clone());
        let transformer_out = self.backbone(input_, text_embeddings, seq_len, &mut state.transformer);
        let last = transformer_out.narrow(1, seq_len - 1, 1).reshape([batch, self.dim]);
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
        let combined = Tensor::cat(vec![text_embeddings, input], 1);
        let mut output = self.transformer.forward(combined, state);
        output = apply_layer_norm_3d(&self.out_norm, output);
        let total_len = output.dims()[1];
        output.narrow(1, total_len - sequence_len, sequence_len)
    }
}

fn apply_linear_3d<B: Backend>(linear: &Linear<B>, input: Tensor<B, 3>) -> Tensor<B, 3> {
    let [batch, seq, dim] = input.dims();
    let flat = input.reshape([batch * seq, dim]);
    let output = linear.forward(flat);
    let out_dim = output.dims()[1];
    output.reshape([batch, seq, out_dim])
}

fn apply_layer_norm_3d<B: Backend>(norm: &LayerNorm<B>, input: Tensor<B, 3>) -> Tensor<B, 3> {
    let [batch, seq, dim] = input.dims();
    let flat = input.reshape([batch * seq, dim]);
    let output = norm.forward(flat);
    output.reshape([batch, seq, dim])
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
    let mut noise = Tensor::<B, 2>::random(shape, Distribution::Normal(0.0, std), device);
    if let Some(clamp) = noise_clamp {
        noise = noise.clamp(-clamp, clamp);
    }
    noise
}
