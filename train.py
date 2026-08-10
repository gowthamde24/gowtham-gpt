"""Train a character-level GPT on a text corpus.

Usage:
    python train.py
    python train.py --data path/to/corpus.txt --epochs 5000 --model-dim 256
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from data import CharVocabulary, get_batch
from model import GPT
from utils import pick_device


def train(
    model: GPT,
    data: torch.Tensor,
    epochs: int,
    context_length: int,
    batch_size: int,
    lr: float,
    log_every: int = 100,
) -> float:
    """Run the AdamW + cross-entropy training loop. Returns the final loss."""
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        x, y = get_batch(data, context_length, batch_size)
        x, y = x.to(device), y.to(device)

        logits = model(x)
        b, t, vocab_size = logits.shape
        loss = F.cross_entropy(logits.view(b * t, vocab_size), y.view(b * t))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % log_every == 0 or epoch == epochs:
            print(f"epoch {epoch:>6}/{epochs}   loss {loss.item():.4f}")

    return round(loss.item(), 4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/corpus/input.txt"),
                         help="Path to a plain-text training corpus.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/model.pt"),
                         help="Where to save the trained model + vocab.")
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = pick_device()
    print(f"device: {device}")

    text = args.data.read_text()
    vocab = CharVocabulary(text)
    data = torch.tensor(vocab.encode(text), dtype=torch.long)
    print(f"corpus: {len(text):,} chars, vocab size {vocab.vocab_size}")

    if len(data) <= args.context_length:
        raise ValueError(
            f"corpus has only {len(data)} tokens, which is <= --context-length "
            f"({args.context_length}). Use a longer corpus or a smaller context length."
        )

    model = GPT(
        vocab_size=vocab.vocab_size,
        context_length=args.context_length,
        model_dim=args.model_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout=args.dropout,
        seed=args.seed,
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"model: {num_params:,} parameters")

    final_loss = train(model, data, args.epochs, args.context_length, args.batch_size, args.lr)
    print(f"final training loss: {final_loss}")

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "stoi": vocab.stoi,
            "itos": vocab.itos,
            "config": {
                "vocab_size": vocab.vocab_size,
                "context_length": args.context_length,
                "model_dim": args.model_dim,
                "num_blocks": args.num_blocks,
                "num_heads": args.num_heads,
                "dropout": args.dropout,
            },
        },
        args.checkpoint,
    )
    print(f"checkpoint saved to {args.checkpoint}")


if __name__ == "__main__":
    main()
