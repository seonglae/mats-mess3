"""Training loop for the transformer on non-ergodic Mess3 data."""

import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from .data import NonErgodicMess3Dataset, create_dataloaders
from .model import create_model


def train(
    components: list[tuple[float, float]],
    n_layers: int = 3,
    d_model: int = 64,
    n_heads: int = 4,
    n_ctx: int = 17,
    seq_length: int = 16,
    n_sequences_per_component: int = 20000,
    batch_size: int = 4096,
    lr: float = 5e-4,
    n_epochs: int = 100,
    device: str = "mps",
    save_dir: str | None = None,
    log_every: int = 10,
) -> tuple[HookedTransformer, NonErgodicMess3Dataset, list[dict]]:
    """Train transformer on non-ergodic Mess3 mixture.

    Returns:
        model: Trained transformer.
        dataset: The training dataset.
        metrics: List of per-step metric dicts.
    """
    model = create_model(
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        n_ctx=n_ctx,
        device=device,
    )

    dataset, loader = create_dataloaders(
        components=components,
        n_sequences_per_component=n_sequences_per_component,
        seq_length=seq_length,
        batch_size=batch_size,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)

    metrics = []
    step = 0
    start_time = time.time()

    model.train()
    for epoch in range(n_epochs):
        epoch_losses = []
        for batch in loader:
            input_ids = batch["input_ids"].to(device)  # (B, seq_length+1)

            # Forward pass: predict next token
            logits = model(input_ids)  # (B, seq_length+1, vocab)

            # Loss: cross-entropy on positions 1: (predict token at position t from context 0:t)
            # Input:  BOS  t0  t1  t2 ... t_{L-1}
            # Target:  t0  t1  t2  t3 ... (shifted)
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                input_ids[:, 1:].reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            loss_val = loss.item()
            epoch_losses.append(loss_val)

            if step % log_every == 0:
                elapsed = time.time() - start_time
                metric = {
                    "step": step,
                    "epoch": epoch,
                    "loss": loss_val,
                    "elapsed_s": elapsed,
                }
                metrics.append(metric)

        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1}/{n_epochs}, avg loss: {avg_loss:.4f}, step: {step}")

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path / "model.pt")
        with open(save_path / "metrics.json", "w") as f:
            json.dump(metrics, f)
        with open(save_path / "config.json", "w") as f:
            json.dump(
                {
                    "components": components,
                    "n_layers": n_layers,
                    "d_model": d_model,
                    "n_heads": n_heads,
                    "n_ctx": n_ctx,
                    "seq_length": seq_length,
                    "n_sequences_per_component": n_sequences_per_component,
                    "batch_size": batch_size,
                    "lr": lr,
                    "n_epochs": n_epochs,
                },
                f,
                indent=2,
            )

    return model, dataset, metrics
