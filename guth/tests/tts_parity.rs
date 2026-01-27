use burn::tensor::{Int, Tensor, TensorData, Tolerance};
use burn_ndarray::{NdArray, NdArrayDevice};
use serde::Deserialize;

use guth::conditioner::text::LutConditioner;
use guth::model::flow_lm::FlowLmModel;
use guth::model::mimi::MimiModel;
use guth::model::tts::TtsModel;
use guth::modules::dummy_quantizer::DummyQuantizer;
use guth::modules::flow_net::{SimpleMlpAdaLn, SimpleMlpAdaLnConfig};
use guth::modules::mimi_transformer::{
    MimiProjectedTransformer, MimiProjectedTransformerConfig, MimiTransformerConfig,
};
use guth::modules::seanet::{SeanetDecoder, SeanetEncoder, SeanetLayer, SeanetResnetBlock};
use guth::modules::streaming_conv::{PaddingMode, StreamingConv1dOp, StreamingConvConfig, StreamingConvTranspose1dOp};
use guth::modules::transformer::{
    StreamingTransformer, StreamingTransformerConfig, StreamingTransformerLayerConfig,
};
use guth::weights::{load_flow_lm_state_dict, load_mimi_state_dict};

const FIXTURE_DIR: &str = "tests/fixtures";
type TestBackend = NdArray<f32>;

#[derive(Debug, Deserialize)]
struct TtsFixture {
    config: FlowLmConfig,
    flow_net: FlowNetFixture,
    conditioner: ConditionerFixture,
    generation: GenerationFixture,
    mimi_config: MimiConfigFixture,
    latents: Vec<Vec<Vec<f32>>>,
    eos: Vec<Vec<bool>>,
    audio_full: Vec<Vec<Vec<f32>>>,
}

#[derive(Debug, Deserialize)]
struct FlowLmConfig {
    ldim: usize,
    dim: usize,
    num_heads: usize,
    num_layers: usize,
    ffn_dim: usize,
    max_period: f32,
}

#[derive(Debug, Deserialize)]
struct ConditionerFixture {
    n_bins: usize,
    tokens: Vec<Vec<i64>>,
}

#[derive(Debug, Deserialize)]
struct GenerationFixture {
    max_gen_len: usize,
    frames_after_eos: usize,
    temp: f32,
    lsd_decode_steps: usize,
    noise_clamp: Option<f32>,
    eos_threshold: f32,
}

#[derive(Debug, Deserialize)]
struct FlowNetFixture {
    config: FlowNetConfig,
}

#[derive(Debug, Deserialize)]
struct FlowNetConfig {
    in_channels: usize,
    model_channels: usize,
    out_channels: usize,
    cond_channels: usize,
    num_res_blocks: usize,
    num_time_conds: usize,
    frequency_embedding_size: usize,
    max_period: f32,
}

#[derive(Debug, Deserialize)]
struct MimiConfigFixture {
    sample_rate: usize,
    frame_rate: f32,
    encoder_frame_rate: f32,
    channels: usize,
    dimension: usize,
    transformer: MimiTransformerFixtureConfig,
}

#[derive(Debug, Deserialize)]
struct MimiTransformerFixtureConfig {
    input_dimension: usize,
    output_dimensions: Vec<usize>,
    d_model: usize,
    num_heads: usize,
    num_layers: usize,
    layer_scale: f32,
    context: usize,
    max_period: f32,
    dim_feedforward: usize,
}

#[derive(Deserialize)]
struct SeanetFixture {
    layers: Vec<LayerFixture>,
}

#[derive(Deserialize)]
#[serde(tag = "kind")]
enum LayerFixture {
    #[serde(rename = "conv1d")]
    Conv1d {
        config: ConvConfigFixture,
        weight: Vec<Vec<Vec<f32>>>,
        bias: Vec<f32>,
    },
    #[serde(rename = "conv_transpose")]
    ConvTranspose {
        config: ConvConfigFixture,
        weight: Vec<Vec<Vec<f32>>>,
        bias: Vec<f32>,
    },
    #[serde(rename = "elu")]
    Elu,
    #[serde(rename = "resblock")]
    ResBlock { convs: Vec<ConvLayerFixture> },
}

#[derive(Deserialize)]
struct ConvLayerFixture {
    config: ConvConfigFixture,
    weight: Vec<Vec<Vec<f32>>>,
    bias: Vec<f32>,
}

#[derive(Deserialize)]
struct ConvConfigFixture {
    kernel_size: usize,
    stride: usize,
    dilation: usize,
    padding: usize,
    pad_mode: Option<String>,
}

fn read_fixture<T: for<'de> Deserialize<'de>>(name: &str) -> T {
    let path = format!("{FIXTURE_DIR}/{name}");
    let data = std::fs::read_to_string(path).expect("fixture read");
    serde_json::from_str(&data).expect("fixture parse")
}

fn tensor2_int(device: &NdArrayDevice, data: Vec<Vec<i64>>) -> Tensor<TestBackend, 2, Int> {
    let rows = data.len();
    let cols = data.first().map(|row| row.len()).unwrap_or(0);
    let flat: Vec<i64> = data.into_iter().flatten().collect();
    let td = TensorData::new(flat, [rows, cols]);
    Tensor::from_data(td, device)
}

fn tensor3(device: &NdArrayDevice, data: Vec<Vec<Vec<f32>>>) -> Tensor<TestBackend, 3> {
    let b = data.len();
    let c = data[0].len();
    let t = data[0][0].len();
    let mut flat = Vec::with_capacity(b * c * t);
    for bb in data {
        for cc in bb {
            for val in cc {
                flat.push(val);
            }
        }
    }
    let td = TensorData::new(flat, [b, c, t]);
    Tensor::from_data(td, device)
}

fn tensor2_from_state_data(
    data: &guth::weights::TensorData,
    device: &NdArrayDevice,
) -> Tensor<TestBackend, 2> {
    let shape = [data.shape[0], data.shape[1]];
    let mut values = Vec::with_capacity(data.data.len() / 4);
    for chunk in data.data.chunks_exact(4) {
        values.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    Tensor::from_data(TensorData::new(values, shape), device)
}

fn build_flow_net(device: &NdArrayDevice, fixture: FlowNetFixture) -> SimpleMlpAdaLn<TestBackend> {
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

    SimpleMlpAdaLn::<TestBackend>::new(config, device)
}

fn build_conv(
    config: &ConvConfigFixture,
    weight: Vec<Vec<Vec<f32>>>,
    bias: Vec<f32>,
    device: &NdArrayDevice,
) -> StreamingConv1dOp<TestBackend> {
    let pad_mode = match config.pad_mode.as_deref() {
        Some("replicate") => PaddingMode::Replicate,
        _ => PaddingMode::Constant,
    };
    let conv_config = StreamingConvConfig {
        kernel_size: config.kernel_size,
        stride: config.stride,
        dilation: config.dilation,
        padding: config.padding,
        pad_mode,
        groups: 1,
    };
    let out_channels = weight.len();
    let in_channels = weight[0].len();
    let kernel = weight[0][0].len();
    let weight = Tensor::<TestBackend, 3>::zeros([out_channels, in_channels, kernel], device);
    let bias = Tensor::<TestBackend, 1>::zeros([bias.len()], device);
    StreamingConv1dOp::new(conv_config, weight, Some(bias))
}

fn build_convtr(
    config: &ConvConfigFixture,
    weight: Vec<Vec<Vec<f32>>>,
    bias: Vec<f32>,
    device: &NdArrayDevice,
) -> StreamingConvTranspose1dOp<TestBackend> {
    let conv_config = StreamingConvConfig {
        kernel_size: config.kernel_size,
        stride: config.stride,
        dilation: config.dilation,
        padding: config.padding,
        pad_mode: PaddingMode::Constant,
        groups: 1,
    };
    let out_channels = weight.len();
    let in_channels = weight[0].len();
    let kernel = weight[0][0].len();
    let weight = Tensor::<TestBackend, 3>::zeros([out_channels, in_channels, kernel], device);
    let bias = Tensor::<TestBackend, 1>::zeros([bias.len()], device);
    StreamingConvTranspose1dOp::new(conv_config, weight, Some(bias))
}

fn build_seanet_encoder(device: &NdArrayDevice) -> SeanetEncoder<TestBackend> {
    let fixture: SeanetFixture = read_fixture("seanet_encoder.json");
    let mut layers = Vec::new();
    for layer in fixture.layers {
        match layer {
            LayerFixture::Conv1d { config, weight, bias } => {
                layers.push(SeanetLayer::Conv1d(build_conv(&config, weight, bias, device)));
            }
            LayerFixture::Elu => layers.push(SeanetLayer::Elu),
            LayerFixture::ResBlock { convs } => {
                let mut block_layers = Vec::new();
                for conv in convs {
                    block_layers.push(build_conv(&conv.config, conv.weight, conv.bias, device));
                }
                layers.push(SeanetLayer::ResBlock(SeanetResnetBlock::new(block_layers)));
            }
            _ => unreachable!("unexpected layer in encoder"),
        }
    }
    SeanetEncoder::new(layers)
}

fn build_seanet_decoder(device: &NdArrayDevice) -> SeanetDecoder<TestBackend> {
    let fixture: SeanetFixture = read_fixture("seanet_decoder.json");
    let mut layers = Vec::new();
    for layer in fixture.layers {
        match layer {
            LayerFixture::ConvTranspose { config, weight, bias } => layers.push(SeanetLayer::ConvTranspose1d(
                build_convtr(&config, weight, bias, device),
            )),
            LayerFixture::Conv1d { config, weight, bias } => {
                layers.push(SeanetLayer::Conv1d(build_conv(&config, weight, bias, device)));
            }
            LayerFixture::Elu => layers.push(SeanetLayer::Elu),
            LayerFixture::ResBlock { convs } => {
                let mut block_layers = Vec::new();
                for conv in convs {
                    block_layers.push(build_conv(&conv.config, conv.weight, conv.bias, device));
                }
                layers.push(SeanetLayer::ResBlock(SeanetResnetBlock::new(block_layers)));
            }
        }
    }
    SeanetDecoder::new(layers)
}

fn build_projected_transformer(
    config: &MimiTransformerFixtureConfig,
    device: &NdArrayDevice,
) -> MimiProjectedTransformer<TestBackend> {
    let transformer_config = MimiTransformerConfig {
        d_model: config.d_model,
        num_heads: config.num_heads,
        num_layers: config.num_layers,
        layer_scale: config.layer_scale,
        context: config.context,
        max_period: config.max_period,
        dim_feedforward: config.dim_feedforward,
    };

    let projected_config = MimiProjectedTransformerConfig {
        input_dim: config.input_dimension,
        output_dims: config.output_dimensions.clone(),
        transformer: transformer_config,
    };
    MimiProjectedTransformer::<TestBackend>::new(projected_config, device)
}

#[test]
fn tts_model_matches_fixture() {
    let device = NdArrayDevice::default();
    let fixture: TtsFixture = read_fixture("tts_model.json");

    let embed = burn_nn::EmbeddingConfig::new(
        fixture.conditioner.n_bins,
        fixture.config.dim,
    )
    .init::<TestBackend>(&device);
    let conditioner = LutConditioner {
        tokenizer: None,
        embed,
        dim: fixture.config.dim,
        output_dim: fixture.config.dim,
    };

    let transformer_config = StreamingTransformerConfig {
        num_layers: fixture.config.num_layers,
        layer: StreamingTransformerLayerConfig {
            d_model: fixture.config.dim,
            num_heads: fixture.config.num_heads,
            ffn_dim: fixture.config.ffn_dim,
            causal: true,
            context: None,
            rope_max_seq: Some(64),
            rope_theta: fixture.config.max_period,
            layer_scale: None,
        },
    };
    let transformer = StreamingTransformer::<TestBackend>::new(transformer_config, &device);

    let flow_net = build_flow_net(&device, fixture.flow_net);

    let input_linear = burn_nn::LinearConfig::new(fixture.config.ldim, fixture.config.dim)
        .with_bias(false)
        .init::<TestBackend>(&device);

    let out_norm = burn_nn::LayerNormConfig::new(fixture.config.dim)
        .init::<TestBackend>(&device);

    let out_eos = burn_nn::LinearConfig::new(fixture.config.dim, 1)
        .init::<TestBackend>(&device);
    let emb_mean = Tensor::<TestBackend, 1>::zeros([fixture.config.ldim], &device);
    let emb_std = Tensor::<TestBackend, 1>::zeros([fixture.config.ldim], &device);
    let bos_emb = Tensor::<TestBackend, 1>::zeros([fixture.config.ldim], &device);

    let mut flow_lm = FlowLmModel::new(
        conditioner,
        flow_net,
        transformer,
        input_linear,
        out_norm,
        out_eos,
        emb_std,
        emb_mean,
        bos_emb,
        fixture.config.dim,
        fixture.config.ldim,
    );
    let flow_state = load_flow_lm_state_dict("tests/fixtures/tts_flow_lm_state.safetensors")
        .expect("load flow lm state dict");
    flow_lm
        .load_state_dict(&flow_state, &device)
        .expect("apply flow lm state dict");

    let encoder = build_seanet_encoder(&device);
    let decoder = build_seanet_decoder(&device);
    let encoder_transformer =
        build_projected_transformer(&fixture.mimi_config.transformer, &device);
    let decoder_transformer =
        build_projected_transformer(&fixture.mimi_config.transformer, &device);
    let quantizer_weight = Tensor::<TestBackend, 3>::zeros([1, 1, 1], &device);
    let quantizer = DummyQuantizer::new(quantizer_weight);

    let mut mimi = MimiModel::new(
        encoder,
        decoder,
        encoder_transformer,
        decoder_transformer,
        quantizer,
        fixture.mimi_config.frame_rate,
        fixture.mimi_config.encoder_frame_rate,
        fixture.mimi_config.sample_rate,
        fixture.mimi_config.channels,
        fixture.mimi_config.dimension,
        None,
        None,
    );
    let mimi_state = load_mimi_state_dict("tests/fixtures/tts_mimi_state.safetensors")
        .expect("load mimi state dict");
    mimi
        .load_state_dict(&mimi_state, &device)
        .expect("apply mimi state dict");

    let speaker_proj_data = flow_state
        .get("speaker_proj_weight")
        .expect("speaker proj weight");
    let speaker_proj_weight = tensor2_from_state_data(speaker_proj_data, &device);
    let tts = TtsModel::new(
        flow_lm,
        mimi,
        speaker_proj_weight,
        fixture.generation.temp,
        fixture.generation.lsd_decode_steps,
        fixture.generation.noise_clamp,
        fixture.generation.eos_threshold,
    );

    let tokens = tensor2_int(&device, fixture.conditioner.tokens);
    let flow_len = tokens.dims()[1] + fixture.generation.max_gen_len + 1;
    let mut state = tts.init_state(1, flow_len, fixture.generation.max_gen_len, &device);

    let (latents, eos, audio) = tts.generate_audio_from_tokens(
        tokens,
        &mut state,
        fixture.generation.max_gen_len,
        fixture.generation.frames_after_eos,
    );

    latents.to_data().assert_approx_eq(
        &tensor3(&device, fixture.latents).to_data(),
        Tolerance::<f32>::absolute(1e-4).set_relative(1e-4),
    );
    let eos_data = eos.to_data();
    let eos_slice = eos_data.as_slice::<bool>().expect("eos slice");
    let expected: Vec<bool> = fixture.eos.into_iter().flatten().collect();
    assert_eq!(eos_slice, expected.as_slice());

    audio.to_data().assert_approx_eq(
        &tensor3(&device, fixture.audio_full).to_data(),
        Tolerance::<f32>::absolute(1e-4).set_relative(1e-4),
    );
}
