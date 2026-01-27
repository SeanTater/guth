import json
import math
import pathlib
from typing import Any, Iterable

import torch
import safetensors.torch

from pocket_tts.conditioners.base import TokenizedText
from pocket_tts.conditioners.text import LUTConditioner
from pocket_tts.models.flow_lm import FlowLMModel
from pocket_tts.models.mimi import MimiModel
from pocket_tts.modules.dummy_quantizer import DummyQuantizer
from pocket_tts.modules.mimi_transformer import ProjectedTransformer, StreamingTransformer
from pocket_tts.modules.seanet import SEANetDecoder, SEANetEncoder, SEANetResnetBlock
from pocket_tts.modules.conv import StreamingConv1d, StreamingConvTranspose1d
from pocket_tts.modules.stateful_module import increment_steps, init_states
from pocket_tts.modules.mlp import SimpleMLPAdaLN

ROOT = pathlib.Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "guth" / "tests" / "fixtures"


def to_list(tensor: torch.Tensor):
    return tensor.detach().cpu().tolist()


def apply_linear(module: torch.nn.Linear, weight, bias=None):
    module.weight.data = torch.tensor(weight, dtype=module.weight.dtype).t()
    if bias is not None:
        module.bias.data = torch.tensor(bias, dtype=module.weight.dtype)


def apply_layer_norm(module: torch.nn.LayerNorm, weight, bias):
    module.weight.data = torch.tensor(weight, dtype=module.weight.dtype)
    module.bias.data = torch.tensor(bias, dtype=module.weight.dtype)


def serialize_flow_net(flow_net: SimpleMLPAdaLN):
    def transpose(weight):
        return [list(row) for row in zip(*weight)]

    time_embed = []
    for embedder in flow_net.time_embed:
        time_embed.append(
            {
                "proj_in": {
                    "weight": transpose(to_list(embedder.mlp[0].weight)),
                    "bias": to_list(embedder.mlp[0].bias),
                },
                "proj_out": {
                    "weight": transpose(to_list(embedder.mlp[2].weight)),
                    "bias": to_list(embedder.mlp[2].bias),
                },
                "rms_weight": to_list(embedder.mlp[3].alpha),
            }
        )

    res_blocks = []
    for block in flow_net.res_blocks:
        res_blocks.append(
            {
                "norm": {
                    "weight": to_list(block.in_ln.weight),
                    "bias": to_list(block.in_ln.bias),
                },
                "mlp_in": {
                    "weight": transpose(to_list(block.mlp[0].weight)),
                    "bias": to_list(block.mlp[0].bias),
                },
                "mlp_out": {
                    "weight": transpose(to_list(block.mlp[2].weight)),
                    "bias": to_list(block.mlp[2].bias),
                },
                "modulation": {
                    "weight": transpose(to_list(block.adaLN_modulation[1].weight)),
                    "bias": to_list(block.adaLN_modulation[1].bias),
                },
            }
        )

    return {
        "input_proj": {
            "weight": transpose(to_list(flow_net.input_proj.weight)),
            "bias": to_list(flow_net.input_proj.bias),
        },
        "cond_embed": {
            "weight": transpose(to_list(flow_net.cond_embed.weight)),
            "bias": to_list(flow_net.cond_embed.bias),
        },
        "time_embed": time_embed,
        "res_blocks": res_blocks,
        "final_layer": {
            "norm": {
                "weight": [1.0 for _ in range(flow_net.model_channels)],
                "bias": None,
            },
            "linear": {
                "weight": transpose(to_list(flow_net.final_layer.linear.weight)),
                "bias": to_list(flow_net.final_layer.linear.bias),
            },
            "modulation": {
                "weight": transpose(to_list(flow_net.final_layer.adaLN_modulation[1].weight)),
                "bias": to_list(flow_net.final_layer.adaLN_modulation[1].bias),
            },
        },
    }


def serialize_transformer(transformer):
    layers = []
    for layer in transformer.layers:
        layers.append(
            {
                "self_attn": {
                    "in_proj": {"weight": to_list(layer.self_attn.in_proj.weight)},
                    "out_proj": {"weight": to_list(layer.self_attn.out_proj.weight)},
                },
                "norm1": {
                    "gamma": to_list(layer.norm1.weight),
                    "beta": to_list(layer.norm1.bias),
                },
                "norm2": {
                    "gamma": to_list(layer.norm2.weight),
                    "beta": to_list(layer.norm2.bias),
                },
                "linear1": {"weight": to_list(layer.linear1.weight)},
                "linear2": {"weight": to_list(layer.linear2.weight)},
            }
        )
    return {"layers": layers}


def add_seanet_weights(state: dict[str, torch.Tensor], prefix: str, model: torch.nn.Module):
    for idx, layer in enumerate(model.model):
        if isinstance(layer, StreamingConv1d):
            state[f"{prefix}.layers.{idx}.conv.weight"] = layer.conv.weight
            state[f"{prefix}.layers.{idx}.conv.bias"] = layer.conv.bias
        elif isinstance(layer, StreamingConvTranspose1d):
            state[f"{prefix}.layers.{idx}.conv_transpose.weight"] = layer.convtr.weight
            state[f"{prefix}.layers.{idx}.conv_transpose.bias"] = layer.convtr.bias
        elif isinstance(layer, SEANetResnetBlock):
            convs = [block for block in layer.block if isinstance(block, StreamingConv1d)]
            for conv_idx, conv in enumerate(convs):
                state[f"{prefix}.layers.{idx}.resblock.{conv_idx}.weight"] = conv.conv.weight
                state[f"{prefix}.layers.{idx}.resblock.{conv_idx}.bias"] = conv.conv.bias


def add_projected_transformer_state(
    state: dict[str, torch.Tensor], prefix: str, transformer: ProjectedTransformer
):
    if transformer.input_proj is not None:
        state[f"{prefix}.input_proj.weight"] = transformer.input_proj.weight
    for idx, proj in enumerate(transformer.output_projs):
        if isinstance(proj, torch.nn.Identity):
            continue
        state[f"{prefix}.output_projs.{idx}.weight"] = proj.weight
    for idx, layer in enumerate(transformer.transformer.layers):
        state[f"{prefix}.layers.{idx}.self_attn.in_proj.weight"] = layer.self_attn.in_proj.weight
        state[f"{prefix}.layers.{idx}.self_attn.out_proj.weight"] = layer.self_attn.out_proj.weight
        state[f"{prefix}.layers.{idx}.norm1.weight"] = layer.norm1.weight
        state[f"{prefix}.layers.{idx}.norm1.bias"] = layer.norm1.bias
        state[f"{prefix}.layers.{idx}.norm2.weight"] = layer.norm2.weight
        state[f"{prefix}.layers.{idx}.norm2.bias"] = layer.norm2.bias
        state[f"{prefix}.layers.{idx}.linear1.weight"] = layer.linear1.weight
        state[f"{prefix}.layers.{idx}.linear2.weight"] = layer.linear2.weight
        if hasattr(layer, "layer_scale_1"):
            state[f"{prefix}.layers.{idx}.layer_scale_1.scale"] = layer.layer_scale_1.scale
            state[f"{prefix}.layers.{idx}.layer_scale_2.scale"] = layer.layer_scale_2.scale


def load_fixture(name: str):
    with (FIXTURES / name).open("r", encoding="utf-8") as f:
        return json.load(f)


def next_module_of_type(modules: Iterable[torch.nn.Module], cls):
    for module in modules:
        if isinstance(module, cls):
            return module
    raise ValueError(f"Module of type {cls} not found")


def apply_conv_weights(conv: torch.nn.Conv1d, fixture):
    conv.weight.data = torch.tensor(fixture["weight"], dtype=torch.float32)
    conv.bias.data = torch.tensor(fixture["bias"], dtype=torch.float32)


def apply_convtr_weights(convtr: torch.nn.ConvTranspose1d, fixture):
    convtr.weight.data = torch.tensor(fixture["weight"], dtype=torch.float32)
    convtr.bias.data = torch.tensor(fixture["bias"], dtype=torch.float32)


def load_seanet_from_fixture(fixture_name: str, kind: str):
    fixture = load_fixture(fixture_name)
    config = fixture["config"]
    if kind == "encoder":
        model = SEANetEncoder(**config)
        model_layers = iter(model.model)
        for layer in fixture["layers"]:
            if layer["kind"] == "conv1d":
                module = next_module_of_type(model_layers, StreamingConv1d)
                apply_conv_weights(module.conv, layer)
            elif layer["kind"] == "elu":
                continue
            elif layer["kind"] == "resblock":
                module = next_module_of_type(model_layers, SEANetResnetBlock)
                convs = [block for block in module.block if isinstance(block, StreamingConv1d)]
                for conv_layer, conv_fixture in zip(convs, layer["convs"], strict=True):
                    apply_conv_weights(conv_layer.conv, conv_fixture)
            else:
                raise ValueError(f"Unexpected encoder layer {layer['kind']}")
    else:
        model = SEANetDecoder(**config)
        model_layers = iter(model.model)
        for layer in fixture["layers"]:
            if layer["kind"] in ("convtranspose", "conv_transpose"):
                module = next_module_of_type(model_layers, StreamingConvTranspose1d)
                apply_convtr_weights(module.convtr, layer)
            elif layer["kind"] == "conv1d":
                module = next_module_of_type(model_layers, StreamingConv1d)
                apply_conv_weights(module.conv, layer)
            elif layer["kind"] == "elu":
                continue
            elif layer["kind"] == "resblock":
                module = next_module_of_type(model_layers, SEANetResnetBlock)
                convs = [block for block in module.block if isinstance(block, StreamingConv1d)]
                for conv_layer, conv_fixture in zip(convs, layer["convs"], strict=True):
                    apply_conv_weights(conv_layer.conv, conv_fixture)
            else:
                raise ValueError(f"Unexpected decoder layer {layer['kind']}")
    return model


def apply_projected_transformer_weights(model: ProjectedTransformer, fixture):
    if fixture["input_proj"]["kind"] == "linear":
        model.input_proj.weight.data = torch.tensor(
            fixture["input_proj"]["weight"], dtype=model.input_proj.weight.dtype
        )

    for proj, proj_fixture in zip(model.output_projs, fixture["output_projs"], strict=True):
        if proj_fixture["kind"] == "linear":
            proj.weight.data = torch.tensor(proj_fixture["weight"], dtype=proj.weight.dtype)

    for layer, layer_fixture in zip(model.transformer.layers, fixture["layers"], strict=True):
        layer.self_attn.in_proj.weight.data = torch.tensor(
            layer_fixture["self_attn"]["in_proj"]["weight"], dtype=layer.self_attn.in_proj.weight.dtype
        )
        layer.self_attn.out_proj.weight.data = torch.tensor(
            layer_fixture["self_attn"]["out_proj"]["weight"], dtype=layer.self_attn.out_proj.weight.dtype
        )
        apply_layer_norm(layer.norm1, layer_fixture["norm1"]["gamma"], layer_fixture["norm1"]["beta"])
        apply_layer_norm(layer.norm2, layer_fixture["norm2"]["gamma"], layer_fixture["norm2"]["beta"])
        layer.linear1.weight.data = torch.tensor(
            layer_fixture["linear1"]["weight"], dtype=layer.linear1.weight.dtype
        )
        layer.linear2.weight.data = torch.tensor(
            layer_fixture["linear2"]["weight"], dtype=layer.linear2.weight.dtype
        )
        if hasattr(layer, "layer_scale_1"):
            layer.layer_scale_1.scale.data = torch.tensor(
                layer_fixture["layer_scale_1"]["scale"], dtype=layer.layer_scale_1.scale.dtype
            )
            layer.layer_scale_2.scale.data = torch.tensor(
                layer_fixture["layer_scale_2"]["scale"], dtype=layer.layer_scale_2.scale.dtype
            )


def build_mimi_from_fixtures(quantizer_weight: torch.Tensor):
    mimi_fixture = load_fixture("mimi_model.json")

    encoder = load_seanet_from_fixture("seanet_encoder.json", "encoder")
    decoder = load_seanet_from_fixture("seanet_decoder.json", "decoder")

    transformer_config = mimi_fixture["config"]["transformer"]
    transformer_config["output_dimensions"] = tuple(transformer_config["output_dimensions"])
    encoder_transformer = ProjectedTransformer(**transformer_config)
    decoder_transformer = ProjectedTransformer(**transformer_config)
    apply_projected_transformer_weights(encoder_transformer, mimi_fixture["encoder_transformer"])
    apply_projected_transformer_weights(decoder_transformer, mimi_fixture["decoder_transformer"])

    quantizer = DummyQuantizer(dimension=4, output_dimension=4)
    quantizer.output_proj.weight.data = quantizer_weight

    return MimiModel(
        encoder=encoder,
        decoder=decoder,
        quantizer=quantizer,
        frame_rate=mimi_fixture["config"]["frame_rate"],
        encoder_frame_rate=mimi_fixture["config"]["encoder_frame_rate"],
        sample_rate=mimi_fixture["config"]["sample_rate"],
        channels=mimi_fixture["config"]["channels"],
        encoder_transformer=encoder_transformer,
        decoder_transformer=decoder_transformer,
    )


def run_flow_lm_and_increment(
    model: FlowLMModel,
    model_state: dict,
    lsd_decode_steps: int,
    temp: float,
    noise_clamp: float | None,
    eos_threshold: float,
    text_tokens: torch.Tensor | None = None,
    backbone_input_latents: torch.Tensor | None = None,
    audio_conditioning: torch.Tensor | None = None,
):
    if text_tokens is None:
        text_tokens = torch.zeros((1, 0), dtype=torch.long)
    if backbone_input_latents is None:
        backbone_input_latents = torch.empty((1, 0, model.ldim), dtype=model.dtype)
    if audio_conditioning is None:
        audio_conditioning = torch.empty((1, 0, model.dim), dtype=model.dtype)

    text_embeddings = model.conditioner(TokenizedText(text_tokens))
    text_embeddings = torch.cat([text_embeddings, audio_conditioning], dim=1)
    output, is_eos = model._sample_next_latent(
        backbone_input_latents,
        text_embeddings,
        model_state=model_state,
        lsd_decode_steps=lsd_decode_steps,
        temp=temp,
        noise_clamp=noise_clamp,
        eos_threshold=eos_threshold,
    )
    increment_by = (
        text_tokens.shape[1] + backbone_input_latents.shape[1] + audio_conditioning.shape[1]
    )
    increment_steps(model, model_state, increment=increment_by)
    return output[:, None, :], is_eos


def main() -> None:
    # Regenerate with: uv run --with torch guth/tests/fixtures/generate_tts_fixtures.py
    torch.manual_seed(31)

    mimi_fixture = load_fixture("mimi_model.json")

    ldim = 4
    dim = 4

    flow_net = SimpleMLPAdaLN(
        in_channels=ldim,
        model_channels=4,
        out_channels=ldim,
        cond_channels=dim,
        num_res_blocks=2,
        num_time_conds=2,
    )
    for embedder in flow_net.time_embed:
        embedder.frequency_embedding_size = 8
        half = 8 // 2
        freqs = torch.exp(-math.log(10.0) * torch.arange(start=0, end=half) / half)
        embedder.freqs.data = freqs
        embedder.mlp[0].weight.data = torch.randn(
            embedder.mlp[0].out_features, 8, dtype=embedder.mlp[0].weight.dtype
        )

    transformer = StreamingTransformer(
        d_model=dim,
        num_heads=2,
        num_layers=1,
        dim_feedforward=8,
        max_period=10000.0,
        kind="flow_lm",
    )

    # Set transformer weights deterministically.
    for layer in transformer.layers:
        torch.nn.init.uniform_(layer.self_attn.in_proj.weight, a=-0.1, b=0.1)
        torch.nn.init.uniform_(layer.self_attn.out_proj.weight, a=-0.1, b=0.1)
        torch.nn.init.uniform_(layer.linear1.weight, a=-0.1, b=0.1)
        torch.nn.init.uniform_(layer.linear2.weight, a=-0.1, b=0.1)
        torch.nn.init.uniform_(layer.norm1.weight, a=0.9, b=1.1)
        torch.nn.init.uniform_(layer.norm2.weight, a=0.9, b=1.1)
        torch.nn.init.uniform_(layer.norm1.bias, a=-0.05, b=0.05)
        torch.nn.init.uniform_(layer.norm2.bias, a=-0.05, b=0.05)

    n_bins = 48
    conditioner = LUTConditioner(
        n_bins=n_bins,
        tokenizer_path=str(FIXTURES / "tokenizer.model"),
        dim=dim,
        output_dim=dim,
    )
    torch.nn.init.uniform_(conditioner.embed.weight, a=-0.2, b=0.2)

    flow_lm = FlowLMModel(
        conditioner=conditioner,
        flow_net=flow_net,
        transformer=transformer,
        dim=dim,
        ldim=ldim,
        dtype=torch.float32,
    )

    torch.nn.init.uniform_(flow_lm.input_linear.weight, a=-0.1, b=0.1)
    torch.nn.init.uniform_(flow_lm.out_norm.weight, a=0.9, b=1.1)
    torch.nn.init.uniform_(flow_lm.out_norm.bias, a=-0.05, b=0.05)
    torch.nn.init.uniform_(flow_lm.out_eos.weight, a=-0.1, b=0.1)
    torch.nn.init.uniform_(flow_lm.out_eos.bias, a=-0.05, b=0.05)
    torch.nn.init.uniform_(flow_lm.bos_emb, a=-0.1, b=0.1)

    speaker_proj_weight = torch.empty((dim, ldim), dtype=torch.float32)
    torch.nn.init.uniform_(speaker_proj_weight, a=-0.1, b=0.1)

    text = "Hello."
    tokens = flow_lm.conditioner.prepare(text).tokens

    max_gen_len = 2
    frames_after_eos = 0
    temp = 0.0
    lsd_decode_steps = 2
    noise_clamp = None
    eos_threshold = 10.0

    sequence_length = tokens.shape[1] + max_gen_len + 1
    flow_state = init_states(flow_lm, batch_size=1, sequence_length=sequence_length)

    run_flow_lm_and_increment(
        flow_lm,
        flow_state,
        lsd_decode_steps,
        temp,
        noise_clamp,
        eos_threshold,
        text_tokens=tokens,
    )

    backbone_input = torch.full(
        (1, 1, ldim),
        fill_value=float("nan"),
        dtype=flow_lm.dtype,
    )
    latents = []
    eos_flags = []
    for _ in range(max_gen_len):
        next_latent, is_eos = run_flow_lm_and_increment(
            flow_lm,
            flow_state,
            lsd_decode_steps,
            temp,
            noise_clamp,
            eos_threshold,
            backbone_input_latents=backbone_input,
        )
        latents.append(next_latent)
        eos_flags.append(is_eos)
        backbone_input = next_latent

    quantizer_weight = torch.empty((4, 4, 1), dtype=torch.float32)
    torch.nn.init.uniform_(quantizer_weight, a=-0.2, b=0.2)
    mimi = build_mimi_from_fixtures(quantizer_weight)
    mimi_state = init_states(mimi, batch_size=1, sequence_length=max_gen_len)

    audio_chunks = []
    for latent in latents:
        mimi_decoding_input = latent * flow_lm.emb_std + flow_lm.emb_mean
        transposed = mimi_decoding_input.transpose(-1, -2)
        quantized = mimi.quantizer(transposed)
        audio = mimi.decode_from_latent(quantized, mimi_state)
        increment_steps(mimi, mimi_state, increment=1)
        audio_chunks.append(audio)

    audio_full = torch.cat(audio_chunks, dim=2)

    fixture = {
        "config": {
            "ldim": ldim,
            "dim": dim,
            "num_heads": 2,
            "num_layers": 1,
            "ffn_dim": 8,
            "max_period": 10000.0,
        },
        "flow_net": {
            "config": {
                "in_channels": ldim,
                "model_channels": 4,
                "out_channels": ldim,
                "cond_channels": dim,
                "num_res_blocks": 2,
                "num_time_conds": 2,
                "frequency_embedding_size": 8,
                "max_period": 10.0,
            },
        },
        "conditioner": {
            "n_bins": n_bins,
            "text": text,
            "tokens": to_list(tokens),
        },
        "generation": {
            "max_gen_len": max_gen_len,
            "frames_after_eos": frames_after_eos,
            "temp": temp,
            "lsd_decode_steps": lsd_decode_steps,
            "noise_clamp": noise_clamp,
            "eos_threshold": eos_threshold,
        },
        "mimi_config": mimi_fixture["config"],
        "latents": to_list(torch.cat(latents, dim=1)),
        "eos": to_list(torch.cat(eos_flags, dim=1)),
        "audio_full": to_list(audio_full),
    }

    with (FIXTURES / "tts_model.json").open("w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=2)

    flow_state = {key: value.contiguous() for key, value in flow_lm.state_dict().items()}
    flow_state["speaker_proj_weight"] = speaker_proj_weight
    safetensors.torch.save_file(flow_state, FIXTURES / "tts_flow_lm_state.safetensors")

    mimi_state = {}
    add_seanet_weights(mimi_state, "encoder", mimi.encoder)
    add_seanet_weights(mimi_state, "decoder", mimi.decoder)
    add_projected_transformer_state(mimi_state, "encoder_transformer", mimi.encoder_transformer)
    add_projected_transformer_state(mimi_state, "decoder_transformer", mimi.decoder_transformer)
    mimi_state["quantizer.weight"] = mimi.quantizer.output_proj.weight
    mimi_state = {key: value.contiguous() for key, value in mimi_state.items()}
    safetensors.torch.save_file(mimi_state, FIXTURES / "tts_mimi_state.safetensors")


if __name__ == "__main__":
    main()
