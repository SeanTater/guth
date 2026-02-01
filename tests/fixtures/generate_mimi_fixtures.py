import json
import pathlib
from typing import Iterable

import torch
import safetensors.torch

from pocket_tts.modules.dummy_quantizer import DummyQuantizer
from pocket_tts.modules.mimi_transformer import ProjectedTransformer
from pocket_tts.modules.seanet import SEANetDecoder, SEANetEncoder, SEANetResnetBlock
from pocket_tts.modules.conv import StreamingConv1d, StreamingConvTranspose1d
from pocket_tts.models.mimi import MimiModel
from pocket_tts.modules.stateful_module import init_states

ROOT = pathlib.Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "guth" / "tests" / "fixtures"


def to_list(tensor: torch.Tensor):
    return tensor.detach().cpu().tolist()


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


def serialize_projected_transformer(model: ProjectedTransformer):
    layers = []
    for layer in model.transformer.layers:
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
                "layer_scale_1": {"scale": to_list(layer.layer_scale_1.scale)},
                "layer_scale_2": {"scale": to_list(layer.layer_scale_2.scale)},
            }
        )

    output_projs = []
    for proj in model.output_projs:
        if isinstance(proj, torch.nn.Identity):
            output_projs.append({"kind": "identity"})
        else:
            output_projs.append({"kind": "linear", "weight": to_list(proj.weight)})

    return {
        "input_proj": {
            "kind": "identity" if model.input_proj is None else "linear",
            "weight": None if model.input_proj is None else to_list(model.input_proj.weight),
        },
        "output_projs": output_projs,
        "layers": layers,
    }


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


def add_projected_transformer_weights(
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
        state[f"{prefix}.layers.{idx}.layer_scale_1.scale"] = layer.layer_scale_1.scale
        state[f"{prefix}.layers.{idx}.layer_scale_2.scale"] = layer.layer_scale_2.scale


def main():
    torch.manual_seed(11)

    encoder = load_seanet_from_fixture("seanet_encoder.json", "encoder")
    decoder = load_seanet_from_fixture("seanet_decoder.json", "decoder")

    transformer_config = {
        "input_dimension": 4,
        "output_dimensions": (4,),
        "d_model": 8,
        "num_heads": 2,
        "num_layers": 1,
        "layer_scale": 0.1,
        "context": 4,
        "max_period": 10000.0,
        "dim_feedforward": 16,
    }

    encoder_transformer = ProjectedTransformer(**transformer_config)
    decoder_transformer = ProjectedTransformer(**transformer_config)

    quantizer = DummyQuantizer(dimension=4, output_dimension=3)

    mimi = MimiModel(
        encoder=encoder,
        decoder=decoder,
        quantizer=quantizer,
        frame_rate=2.0,
        encoder_frame_rate=2.0,
        sample_rate=8,
        channels=1,
        encoder_transformer=encoder_transformer,
        decoder_transformer=decoder_transformer,
    )

    audio_input = torch.randn(1, 1, 7)
    latent_output = mimi.encode_to_latent(audio_input)

    latent_input = torch.randn(1, 4, latent_output.shape[-1])
    mimi_state = init_states(mimi, batch_size=1, sequence_length=latent_input.shape[-1])
    audio_output = mimi.decode_from_latent(latent_input, mimi_state)

    fixture = {
        "config": {
            "sample_rate": 8,
            "frame_rate": 2.0,
            "encoder_frame_rate": 2.0,
            "channels": 1,
            "dimension": 4,
            "transformer": {
                "input_dimension": transformer_config["input_dimension"],
                "output_dimensions": list(transformer_config["output_dimensions"]),
                "d_model": transformer_config["d_model"],
                "num_heads": transformer_config["num_heads"],
                "num_layers": transformer_config["num_layers"],
                "layer_scale": transformer_config["layer_scale"],
                "context": transformer_config["context"],
                "max_period": transformer_config["max_period"],
                "dim_feedforward": transformer_config["dim_feedforward"],
            },
        },
        "encoder_transformer": serialize_projected_transformer(encoder_transformer),
        "decoder_transformer": serialize_projected_transformer(decoder_transformer),
        "audio_input": to_list(audio_input),
        "latent_output": to_list(latent_output),
        "latent_input": to_list(latent_input),
        "audio_output": to_list(audio_output),
    }

    with (FIXTURES / "mimi_model.json").open("w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=2)

    state = {}
    add_seanet_weights(state, "encoder", encoder)
    add_seanet_weights(state, "decoder", decoder)
    add_projected_transformer_weights(state, "encoder_transformer", encoder_transformer)
    add_projected_transformer_weights(state, "decoder_transformer", decoder_transformer)
    state["quantizer.weight"] = quantizer.output_proj.weight
    state = {key: value.contiguous() for key, value in state.items()}
    safetensors.torch.save_file(state, FIXTURES / "mimi_state.safetensors")

    quantizer_input = torch.randn(1, 4, 5)
    quantizer_output = quantizer(quantizer_input)
    quantizer_fixture = {
        "dimension": 4,
        "output_dimension": 3,
        "weight": to_list(quantizer.output_proj.weight),
        "input": to_list(quantizer_input),
        "output": to_list(quantizer_output),
    }
    with (FIXTURES / "dummy_quantizer.json").open("w", encoding="utf-8") as f:
        json.dump(quantizer_fixture, f, indent=2)


if __name__ == "__main__":
    main()
