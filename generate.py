"""Generate text from a trained checkpoint produced by train.py.

Usage:
    python generate.py --prompt "The lighthouse" --new-chars 300
"""

import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

from model import GPT
from utils import pick_device


def generate(
    model: GPT,
    context: torch.Tensor,
    new_chars: int,
    context_length: int,
    itos: dict,
    temperature: float = 1.0,
    seed: int | None = None,
) -> str:
    """Autoregressively sample `new_chars` characters following `context`.

    Each step: crop to the last `context_length` tokens, run the model, take the
    final position's logits, apply temperature + softmax, and sample the next token.
    """
    model.eval()
    generator = torch.Generator(device=context.device)
    if seed is not None:
        generator.manual_seed(seed)

    result = []
    with torch.no_grad():
        for _ in range(new_chars):
            cropped = context[:, -context_length:]
            logits = model(cropped)
            last_logits = logits[:, -1, :] / temperature
            probs = nn.functional.softmax(last_logits, dim=-1)

            next_token = torch.multinomial(probs, 1, generator=generator)
            context = torch.cat((context, next_token), dim=-1)
            result.append(itos[next_token.item()])

    return "".join(result)


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Tuple[GPT, dict, dict, dict]:
    """Load a checkpoint saved by train.py. Returns (model, config, stoi, itos)."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"no checkpoint at {checkpoint_path} — run train.py first.")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = GPT(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        model_dim=config["model_dim"],
        num_blocks=config["num_blocks"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
        seed=None,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, config, checkpoint["stoi"], checkpoint["itos"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/model.pt"))
    parser.add_argument("--prompt", type=str, default="\n")
    parser.add_argument("--new-chars", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8,
                         help="Lower = more predictable, higher = more random.")
    parser.add_argument("--seed", type=int, default=None,
                         help="Set for reproducible sampling; omit for fresh output each run.")
    args = parser.parse_args()

    device = pick_device()
    model, config, stoi, itos = load_checkpoint(args.checkpoint, device)

    unknown = set(args.prompt) - set(stoi)
    if unknown:
        raise ValueError(f"prompt contains characters not seen during training: {unknown}")

    context = torch.tensor([[stoi[ch] for ch in args.prompt]], dtype=torch.long, device=device)

    output = generate(
        model, context, args.new_chars, config["context_length"], itos,
        temperature=args.temperature, seed=args.seed,
    )
    print(args.prompt + output)


if __name__ == "__main__":
    main()
