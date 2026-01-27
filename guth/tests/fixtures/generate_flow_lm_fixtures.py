import json
import math
import pathlib
from typing import Any

import torch

from pocket_tts.models.flow_lm import FlowLMModel
from pocket_tts.modules.mimi_transformer import StreamingTransformer
from pocket_tts.modules.mlp import SimpleMLPAdaLN
from pocket_tts.conditioners.text import LUTConditioner
from pocket_tts.conditioners.base import TokenizedText
from pocket_tts.modules.stateful_module import init_states

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


def load_flow_net_weights(flow_net: SimpleMLPAdaLN, fixture: dict[str, Any]):
    weights = fixture["weights"]

    apply_linear(flow_net.input_proj, weights["input_proj"]["weight"], weights["input_proj"]["bias"])
    apply_linear(flow_net.cond_embed, weights["cond_embed"]["weight"], weights["cond_embed"]["bias"])

    for embedder, w in zip(flow_net.time_embed, weights["time_embed"], strict=True):
        apply_linear(embedder.mlp[0], w["proj_in"]["weight"], w["proj_in"]["bias"])
        apply_linear(embedder.mlp[2], w["proj_out"]["weight"], w["proj_out"]["bias"])
        embedder.mlp[3].alpha.data = torch.tensor(w["rms_weight"], dtype=embedder.mlp[3].alpha.dtype)

    for block, w in zip(flow_net.res_blocks, weights["res_blocks"], strict=True):
        apply_layer_norm(block.in_ln, w["norm"]["weight"], w["norm"]["bias"])
        apply_linear(block.mlp[0], w["mlp_in"]["weight"], w["mlp_in"]["bias"])
        apply_linear(block.mlp[2], w["mlp_out"]["weight"], w["mlp_out"]["bias"])
        apply_linear(block.adaLN_modulation[1], w["modulation"]["weight"], w["modulation"]["bias"])

    apply_linear(flow_net.final_layer.linear, weights["final_layer"]["linear"]["weight"], weights["final_layer"]["linear"]["bias"])
    apply_linear(
        flow_net.final_layer.adaLN_modulation[1],
        weights["final_layer"]["modulation"]["weight"],
        weights["final_layer"]["modulation"]["bias"],
    )


def main() -> None:
    # Regenerate with: uv run --with torch guth/tests/fixtures/generate_flow_lm_fixtures.py
    torch.manual_seed(7)

    # Use flow net fixture weights to align with existing Rust flow_net tests.
    with (FIXTURES / "flow_net.json").open("r", encoding="utf-8") as f:
        flow_net_fixture = json.load(f)

    flow_config = flow_net_fixture["config"]
    ldim = flow_config["in_channels"]
    dim = flow_config["cond_channels"]

    flow_net = SimpleMLPAdaLN(
        in_channels=flow_config["in_channels"],
        model_channels=flow_config["model_channels"],
        out_channels=flow_config["out_channels"],
        cond_channels=flow_config["cond_channels"],
        num_res_blocks=flow_config["num_res_blocks"],
        num_time_conds=flow_config["num_time_conds"],
    )
    for embedder in flow_net.time_embed:
        embedder.frequency_embedding_size = flow_config["frequency_embedding_size"]
        half = flow_config["frequency_embedding_size"] // 2
        freqs = torch.exp(
            -math.log(flow_config["max_period"]) * torch.arange(start=0, end=half) / half
        )
        embedder.freqs.data = freqs

    load_flow_net_weights(flow_net, flow_net_fixture)

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

    model = FlowLMModel(
        conditioner=conditioner,
        flow_net=flow_net,
        transformer=transformer,
        dim=dim,
        ldim=ldim,
        dtype=torch.float32,
    )

    torch.nn.init.uniform_(model.input_linear.weight, a=-0.1, b=0.1)
    torch.nn.init.uniform_(model.out_norm.weight, a=0.9, b=1.1)
    torch.nn.init.uniform_(model.out_norm.bias, a=-0.05, b=0.05)
    torch.nn.init.uniform_(model.out_eos.weight, a=-0.1, b=0.1)
    torch.nn.init.uniform_(model.out_eos.bias, a=-0.05, b=0.05)
    torch.nn.init.uniform_(model.bos_emb, a=-0.1, b=0.1)

    tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    text_embeddings = model.conditioner._get_condition(TokenizedText(tokens))

    sequence = torch.tensor(
        [[[float("nan"), 0.1, -0.2], [0.05, -0.1, 0.2]]], dtype=torch.float32
    )
    sequence_nan_mask = torch.isnan(sequence)
    sequence_clean = torch.nan_to_num(sequence, nan=0.0)

    sequence_length = sequence.shape[1] + text_embeddings.shape[1]
    model_state = init_states(model, batch_size=1, sequence_length=sequence_length)
    transformer_state = init_states(model, batch_size=1, sequence_length=sequence_length)

    sequence_with_bos = torch.where(torch.isnan(sequence), model.bos_emb, sequence)
    input_linear = model.input_linear(sequence_with_bos)
    transformer_input = torch.cat([text_embeddings, input_linear], dim=1)
    transformer_raw = model.transformer(transformer_input, transformer_state)
    transformer_normed = model.out_norm(transformer_raw)
    transformer_trimmed = transformer_normed[:, -sequence.shape[1] :]
    transformer_last = transformer_trimmed[:, -1]

    latent, eos = model(
        sequence=sequence,
        text_embeddings=text_embeddings,
        model_state=model_state,
        lsd_decode_steps=2,
        temp=0.0,
        noise_clamp=None,
        eos_threshold=0.0,
    )

    fixture = {
        "config": {
            "ldim": ldim,
            "dim": dim,
            "num_heads": 2,
            "num_layers": 1,
            "ffn_dim": 8,
            "max_period": 10000.0,
        },
        "conditioner": {
            "n_bins": n_bins,
            "embed_weight": to_list(conditioner.embed.weight),
            "tokens": to_list(tokens),
            "text_embeddings": to_list(text_embeddings),
        },
        "input_linear_weight": to_list(model.input_linear.weight),
        "bos_emb": to_list(model.bos_emb),
        "emb_mean": to_list(model.emb_mean),
        "emb_std": to_list(model.emb_std),
        "transformer": {
            "layers": [
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
                for layer in transformer.layers
            ]
        },
        "out_norm": {"gamma": to_list(model.out_norm.weight), "beta": to_list(model.out_norm.bias)},
        "out_eos": {"weight": to_list(model.out_eos.weight), "bias": to_list(model.out_eos.bias)},
        "sequence": to_list(sequence_clean),
        "sequence_nan_mask": to_list(sequence_nan_mask),
        "input_linear": to_list(input_linear),
        "transformer_input": to_list(transformer_input),
        "transformer_raw": to_list(transformer_raw),
        "transformer_normed": to_list(transformer_normed),
        "transformer_trimmed": to_list(transformer_trimmed),
        "transformer_last": to_list(transformer_last),
        "latent": to_list(latent),
        "eos": to_list(eos),
    }

    with (FIXTURES / "flow_lm_model.json").open("w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=2)


if __name__ == "__main__":
    main()
