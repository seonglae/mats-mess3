"""Analysis of residual stream geometry: PCA, belief regression, visualization."""

import numpy as np
import torch
from jaxtyping import Float
from numpy import ndarray
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from transformer_lens import HookedTransformer

from .mess3 import compute_belief_states, compute_meta_beliefs


@torch.no_grad()
def extract_activations(
    model: HookedTransformer,
    input_ids: torch.Tensor,
    layer: int | None = None,
) -> dict[str, Float[ndarray, "batch seq dim"]]:
    """Extract residual stream activations at each layer.

    Args:
        model: Trained transformer.
        input_ids: Input token IDs of shape (batch, seq_length+1).
        layer: If specified, only extract at this layer. Otherwise extract all.

    Returns:
        Dict mapping hook name to activations array.
    """
    activations = {}

    def make_hook(name):
        def hook_fn(value, hook):
            activations[name] = value.detach().cpu().numpy()
        return hook_fn

    hooks = []
    if layer is not None:
        name = f"blocks.{layer}.hook_resid_post"
        hooks.append((name, make_hook(name)))
    else:
        # Embedding
        hooks.append(("hook_embed", make_hook("hook_embed")))
        hooks.append(("hook_pos_embed", make_hook("hook_pos_embed")))
        # Each layer's residual stream
        for l in range(model.cfg.n_layers):
            name = f"blocks.{l}.hook_resid_post"
            hooks.append((name, make_hook(name)))

    model.run_with_hooks(input_ids, fwd_hooks=hooks)
    return activations


def extract_all_activations(
    model: HookedTransformer,
    dataset,
    n_samples: int = 5000,
    batch_size: int = 512,
    device: str = "mps",
) -> dict:
    """Extract activations for a subset of the dataset.

    Returns:
        Dict with keys:
            'activations': dict of layer_name -> (n_samples, seq_len+1, d_model)
            'labels': (n_samples,) component labels
            'raw_tokens': (n_samples, seq_len) original tokens
            'input_ids': (n_samples, seq_len+1) with BOS
    """
    n_samples = min(n_samples, len(dataset))
    indices = np.random.choice(len(dataset), n_samples, replace=False)

    all_input_ids = []
    all_labels = []
    all_raw_tokens = []

    for idx in indices:
        item = dataset[idx]
        all_input_ids.append(item["input_ids"])
        all_labels.append(item["label"])
        all_raw_tokens.append(item["raw_tokens"])

    input_ids = torch.stack(all_input_ids)
    labels = np.array(all_labels)
    raw_tokens = torch.stack(all_raw_tokens).numpy()

    # Extract activations in batches
    all_activations = {}
    model.eval()

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = input_ids[start:end].to(device)
        batch_acts = extract_activations(model, batch)

        for name, acts in batch_acts.items():
            if name not in all_activations:
                all_activations[name] = []
            all_activations[name].append(acts)

    # Concatenate batches
    for name in all_activations:
        all_activations[name] = np.concatenate(all_activations[name], axis=0)

    return {
        "activations": all_activations,
        "labels": labels,
        "raw_tokens": raw_tokens,
        "input_ids": input_ids.numpy(),
    }


def pca_analysis(
    activations: Float[ndarray, "n_samples seq_len d_model"],
    positions: list[int] | None = None,
) -> dict:
    """Perform PCA on residual stream activations.

    Args:
        activations: Shape (n_samples, seq_len, d_model).
        positions: Which sequence positions to include. None = all non-BOS.

    Returns:
        Dict with PCA results: explained_variance_ratio, n_components_95, etc.
    """
    n_samples, seq_len, d_model = activations.shape

    if positions is None:
        # Skip BOS (position 0)
        positions = list(range(1, seq_len))

    # Flatten: (n_samples * len(positions), d_model)
    X = activations[:, positions, :].reshape(-1, d_model)

    pca = PCA()
    pca.fit(X)

    cev = np.cumsum(pca.explained_variance_ratio_)
    n_95 = int(np.searchsorted(cev, 0.95) + 1)
    n_99 = int(np.searchsorted(cev, 0.99) + 1)

    return {
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance": cev,
        "n_components_95": n_95,
        "n_components_99": n_99,
        "components": pca.components_,  # principal directions
        "mean": pca.mean_,
        "pca": pca,
    }


def pca_by_position(
    activations: Float[ndarray, "n_samples seq_len d_model"],
) -> list[dict]:
    """Run PCA separately for each context position."""
    n_samples, seq_len, d_model = activations.shape
    results = []
    for t in range(seq_len):
        X = activations[:, t, :]
        pca = PCA()
        pca.fit(X)
        cev = np.cumsum(pca.explained_variance_ratio_)
        results.append({
            "position": t,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": cev,
            "n_components_95": int(np.searchsorted(cev, 0.95) + 1),
        })
    return results


def belief_regression(
    activations: Float[ndarray, "n_samples seq_len d_model"],
    raw_tokens: Float[ndarray, "n_samples seq_len_tokens"],
    components: list[tuple[float, float]],
    labels: Float[ndarray, "n_samples"],
    positions: list[int] | None = None,
) -> dict:
    """Linear regression from activations to ground-truth belief states.

    Tests whether belief states are linearly encoded in the residual stream.

    Args:
        activations: Shape (n_samples, seq_len, d_model). seq_len includes BOS.
        raw_tokens: Shape (n_samples, seq_len_tokens). Original tokens {0,1,2}.
        components: List of (alpha, x) tuples.
        labels: Component label for each sequence.
        positions: Positions to use (in activation space, 0=BOS). Default: all non-BOS.

    Returns:
        Dict with regression results per component and overall.
    """
    n_samples, seq_len, d_model = activations.shape
    seq_len_tokens = raw_tokens.shape[1]

    if positions is None:
        positions = list(range(1, seq_len))

    results = {}

    # Per-component belief regression
    for k, (alpha, x_param) in enumerate(components):
        mask = labels == k
        if mask.sum() == 0:
            continue

        comp_tokens = raw_tokens[mask]
        beliefs = compute_belief_states(comp_tokens, alpha, x_param)
        # beliefs shape: (n_comp, seq_len_tokens, 3)

        # Activations for this component (skip BOS -> positions 1:)
        comp_acts = activations[mask]

        # Align: activation at position t (1-indexed due to BOS) corresponds to
        # belief at token position t-1
        X_list = []
        Y_list = []
        for p in positions:
            token_pos = p - 1  # account for BOS
            if 0 <= token_pos < seq_len_tokens:
                X_list.append(comp_acts[:, p, :])
                Y_list.append(beliefs[:, token_pos, :])

        if not X_list:
            continue

        X = np.concatenate(X_list, axis=0)
        Y = np.concatenate(Y_list, axis=0)

        # Fit linear regression
        reg = Ridge(alpha=1e-4)
        reg.fit(X, Y)
        Y_pred = reg.predict(X)

        rmse = np.sqrt(np.mean((Y - Y_pred) ** 2))
        r2 = r2_score(Y, Y_pred)

        results[f"component_{k}"] = {
            "alpha": alpha,
            "x": x_param,
            "rmse": rmse,
            "r2": r2,
            "n_samples": int(mask.sum()),
            "regressor": reg,
        }

    # Meta-belief regression (posterior over components)
    meta_beliefs_list = []
    meta_acts_list = []

    for k, (alpha, x_param) in enumerate(components):
        mask = labels == k
        if mask.sum() == 0:
            continue

    # Compute meta-beliefs for all sequences
    meta_beliefs = compute_meta_beliefs(raw_tokens, components)
    # meta_beliefs shape: (n_samples, seq_len_tokens, n_components)

    X_meta = []
    Y_meta = []
    for p in positions:
        token_pos = p - 1
        if 0 <= token_pos < seq_len_tokens:
            X_meta.append(activations[:, p, :])
            Y_meta.append(meta_beliefs[:, token_pos, :])

    if X_meta:
        X_meta = np.concatenate(X_meta, axis=0)
        Y_meta = np.concatenate(Y_meta, axis=0)

        reg_meta = Ridge(alpha=1e-4)
        reg_meta.fit(X_meta, Y_meta)
        Y_meta_pred = reg_meta.predict(X_meta)

        results["meta_belief"] = {
            "rmse": float(np.sqrt(np.mean((Y_meta - Y_meta_pred) ** 2))),
            "r2": float(r2_score(Y_meta, Y_meta_pred)),
            "regressor": reg_meta,
        }

    return results


def compute_cev_over_training(
    model: HookedTransformer,
    dataset,
    checkpoints: list[dict],
    n_samples: int = 2000,
    device: str = "mps",
) -> list[dict]:
    """Track CEV evolution over training checkpoints.

    This is for post-hoc analysis if checkpoints are saved.
    """
    # This would need model checkpoints - skip for now
    # The main training loop can call pca_analysis periodically instead
    raise NotImplementedError("Use periodic PCA during training instead")
