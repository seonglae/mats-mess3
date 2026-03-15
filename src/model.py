"""Transformer model using TransformerLens for interpretability."""

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig


def create_model(
    n_layers: int = 3,
    d_model: int = 64,
    n_heads: int = 4,
    d_mlp: int | None = None,
    n_ctx: int = 17,  # seq_length + 1 for BOS
    d_vocab: int = 4,  # {BOS, 0, 1, 2}
    device: str = "mps",
) -> HookedTransformer:
    """Create a small decoder-only transformer with TransformerLens.

    Args:
        n_layers: Number of transformer layers.
        d_model: Model/residual stream dimension.
        n_heads: Number of attention heads.
        d_mlp: MLP hidden dimension (default: 4 * d_model).
        n_ctx: Context window size.
        d_vocab: Vocabulary size.
        device: Device to use.

    Returns:
        HookedTransformer model.
    """
    if d_mlp is None:
        d_mlp = 4 * d_model

    cfg = HookedTransformerConfig(
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        d_head=d_model // n_heads,
        d_mlp=d_mlp,
        n_ctx=n_ctx,
        d_vocab=d_vocab,
        act_fn="gelu",
        normalization_type="LN",  # pre-norm
        positional_embedding_type="standard",
        device=device,
    )

    model = HookedTransformer(cfg)
    return model
