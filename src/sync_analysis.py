"""Synchronization dynamics analysis.

Additional analysis (item 4 of work test):
How quickly does the transformer identify the active component?
How does meta-belief vs within-component belief evolve across positions and layers?
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from .mess3 import compute_belief_states, compute_meta_beliefs


def position_wise_regression(
    activations: np.ndarray,
    raw_tokens: np.ndarray,
    components: list[tuple[float, float]],
    labels: np.ndarray,
) -> dict:
    """Probe for component identity and belief state at each position separately.

    Returns R^2 for meta-belief and per-component belief at each position.
    """
    n_samples, seq_len, d_model = activations.shape
    K = len(components)
    seq_len_tokens = raw_tokens.shape[1]

    # Compute ground truths
    all_beliefs = {}
    for k, (alpha, x_param) in enumerate(components):
        mask = labels == k
        if mask.sum() > 0:
            all_beliefs[k] = compute_belief_states(raw_tokens[mask], alpha, x_param)

    meta_beliefs = compute_meta_beliefs(raw_tokens, components)

    results = {"positions": [], "meta_r2": [], "component_r2": {k: [] for k in range(K)}}

    for pos in range(1, seq_len):  # skip BOS
        tp = pos - 1
        if tp >= seq_len_tokens:
            break

        results["positions"].append(pos)

        # Meta-belief at this position
        X = activations[:, pos, :]
        Y_meta = meta_beliefs[:, tp, :]

        Xt, Xe, Yt, Ye = train_test_split(X, Y_meta, test_size=0.2, random_state=42)
        reg = Ridge(alpha=1e-4)
        reg.fit(Xt, Yt)
        r2 = r2_score(Ye, reg.predict(Xe))
        results["meta_r2"].append(float(r2))

        # Per-component belief at this position
        for k, (alpha, x_param) in enumerate(components):
            mask = labels == k
            if mask.sum() == 0 or k not in all_beliefs:
                results["component_r2"][k].append(0.0)
                continue

            X_k = activations[mask, pos, :]
            Y_k = all_beliefs[k][:, tp, :]

            Xkt, Xke, Ykt, Yke = train_test_split(X_k, Y_k, test_size=0.2, random_state=42)
            reg_k = Ridge(alpha=1e-4)
            reg_k.fit(Xkt, Ykt)
            r2_k = r2_score(Yke, reg_k.predict(Xke))
            results["component_r2"][k].append(float(r2_k))

    return results


def layer_wise_probing(
    all_activations: dict[str, np.ndarray],
    raw_tokens: np.ndarray,
    components: list[tuple[float, float]],
    labels: np.ndarray,
) -> dict:
    """Probe for component identity and belief at each layer.

    This reveals the order of representation construction:
    does the model first identify the component, then refine beliefs?
    """
    K = len(components)
    seq_len_tokens = raw_tokens.shape[1]

    # Ground truths (using all non-BOS positions)
    meta_beliefs = compute_meta_beliefs(raw_tokens, components)

    all_component_beliefs = {}
    for k, (alpha, x_param) in enumerate(components):
        mask = labels == k
        if mask.sum() > 0:
            all_component_beliefs[k] = compute_belief_states(raw_tokens[mask], alpha, x_param)

    # Sort layers
    layer_names = sorted(
        [k for k in all_activations if "resid_post" in k],
        key=lambda x: int(x.split(".")[1])
    )
    if "hook_embed" in all_activations:
        layer_names = ["hook_embed"] + layer_names

    results = {"layers": [], "meta_r2": [], "avg_belief_r2": [], "component_r2": {}}

    for layer_name in layer_names:
        acts = all_activations[layer_name]
        n_samples, seq_len, d_model = acts.shape

        short = layer_name.replace("blocks.", "L").replace(".hook_resid_post", "").replace("hook_", "")
        results["layers"].append(short)

        # Meta-belief regression (all positions)
        X_meta, Y_meta = [], []
        for pos in range(1, seq_len):
            tp = pos - 1
            if tp < seq_len_tokens:
                X_meta.append(acts[:, pos, :])
                Y_meta.append(meta_beliefs[:, tp, :])

        X_meta = np.concatenate(X_meta)
        Y_meta = np.concatenate(Y_meta)

        Xmt, Xme, Ymt, Yme = train_test_split(X_meta, Y_meta, test_size=0.2, random_state=42)
        reg = Ridge(alpha=1e-4)
        reg.fit(Xmt, Ymt)
        results["meta_r2"].append(float(r2_score(Yme, reg.predict(Xme))))

        # Per-component belief regression
        r2_components = []
        for k, (alpha, x_param) in enumerate(components):
            mask = labels == k
            if mask.sum() == 0 or k not in all_component_beliefs:
                continue

            X_k, Y_k = [], []
            beliefs_k = all_component_beliefs[k]
            comp_acts = acts[mask]

            for pos in range(1, seq_len):
                tp = pos - 1
                if tp < beliefs_k.shape[1]:
                    X_k.append(comp_acts[:, pos, :])
                    Y_k.append(beliefs_k[:, tp, :])

            X_k = np.concatenate(X_k)
            Y_k = np.concatenate(Y_k)

            Xkt, Xke, Ykt, Yke = train_test_split(X_k, Y_k, test_size=0.2, random_state=42)
            reg_k = Ridge(alpha=1e-4)
            reg_k.fit(Xkt, Ykt)
            r2_k = r2_score(Yke, reg_k.predict(Xke))
            r2_components.append(float(r2_k))

            if k not in results["component_r2"]:
                results["component_r2"][k] = []
            results["component_r2"][k].append(float(r2_k))

        results["avg_belief_r2"].append(float(np.mean(r2_components)) if r2_components else 0.0)

    return results


def plot_sync_dynamics(pos_results: dict, exp_dir, components: list):
    """Plot synchronization dynamics: R2 by position."""
    K = len(components)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Meta-belief R2 by position
    ax1.plot(pos_results["positions"], pos_results["meta_r2"], "k-o", markersize=3, label="Meta-belief")
    ax1.set_xlabel("Context Position")
    ax1.set_ylabel("R$^2$")
    ax1.set_title("Meta-Synchronization: When Is the Component Identified?")
    ax1.legend()
    ax1.set_ylim(-0.1, 1.05)

    # Per-component belief R2 by position
    for k in range(K):
        if k in pos_results["component_r2"]:
            alpha, x_param = components[k]
            ax2.plot(
                pos_results["positions"],
                pos_results["component_r2"][k],
                "o-", markersize=3,
                label=f"C{k} (a={alpha}, x={x_param})"
            )

    ax2.set_xlabel("Context Position")
    ax2.set_ylabel("R$^2$")
    ax2.set_title("Within-Component Belief Tracking by Position")
    ax2.legend(fontsize=8)
    ax2.set_ylim(-0.1, 1.05)

    fig.tight_layout()
    fig.savefig(exp_dir / "figures" / "sync_dynamics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_layer_probing(layer_results: dict, exp_dir, components: list):
    """Plot layer-wise probing: where does each representation emerge?"""
    K = len(components)
    fig, ax = plt.subplots(figsize=(8, 5))

    layers = layer_results["layers"]
    x_pos = range(len(layers))

    ax.plot(x_pos, layer_results["meta_r2"], "ks-", markersize=6, label="Meta-belief (component ID)")
    ax.plot(x_pos, layer_results["avg_belief_r2"], "ro-", markersize=6, label="Avg within-component belief")

    for k in range(K):
        if k in layer_results["component_r2"]:
            alpha, x_param = components[k]
            ax.plot(
                x_pos, layer_results["component_r2"][k],
                "--", markersize=3, alpha=0.5,
                label=f"C{k} belief (a={alpha})"
            )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(layers)
    ax.set_xlabel("Layer")
    ax.set_ylabel("R$^2$")
    ax.set_title("Layer-wise Probing: Component ID vs Belief State")
    ax.legend(fontsize=7)
    ax.set_ylim(-0.1, 1.05)

    fig.tight_layout()
    fig.savefig(exp_dir / "figures" / "layer_probing.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def analyze_attention_patterns(
    model,
    components: list[tuple[float, float]],
    seq_length: int,
    n_samples: int = 2000,
    device: str = "cuda",
    exp_dir=None,
):
    """Analyze attention patterns for source discrimination.

    Do attention heads specialize for meta-synchronization (attending to
    early positions most informative for component identification)?
    """
    import torch
    from .mess3 import generate_mess3_sequences_fast

    rng = np.random.default_rng(456)
    K = len(components)
    per_comp = n_samples // K

    all_tokens = []
    all_labels = []
    for k, (alpha, x_param) in enumerate(components):
        tokens, _ = generate_mess3_sequences_fast(alpha, x_param, per_comp, seq_length, rng)
        all_tokens.append(tokens)
        all_labels.append(np.full(per_comp, k))

    tokens = np.concatenate(all_tokens)
    labels = np.concatenate(all_labels)
    bos = np.zeros((len(tokens), 1), dtype=np.int64)
    input_ids = torch.tensor(np.concatenate([bos, tokens + 1], axis=1), dtype=torch.long).to(device)

    # Extract attention patterns
    attn_patterns = {}
    def make_hook(name):
        def fn(value, hook):
            attn_patterns[name] = value.detach().cpu().numpy()
        return fn

    hooks = []
    n_layers = model.cfg.n_layers
    for l in range(n_layers):
        name = f"blocks.{l}.attn.hook_pattern"
        hooks.append((name, make_hook(name)))

    model.eval()
    with torch.no_grad():
        # Process in batches
        all_patterns = {}
        bsz = 256
        for i in range(0, len(input_ids), bsz):
            attn_patterns = {}
            batch = input_ids[i:i+bsz]
            model.run_with_hooks(batch, fwd_hooks=hooks)
            for name, pat in attn_patterns.items():
                if name not in all_patterns:
                    all_patterns[name] = []
                all_patterns[name].append(pat)

        for name in all_patterns:
            all_patterns[name] = np.concatenate(all_patterns[name])

    # Visualize attention patterns per layer (averaged)
    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4))
    if n_layers == 1:
        axes = [axes]

    for l in range(n_layers):
        name = f"blocks.{l}.attn.hook_pattern"
        if name in all_patterns:
            # Average over batch and heads: (seq, seq)
            avg_pattern = all_patterns[name].mean(axis=(0, 1))
            axes[l].imshow(avg_pattern, cmap="Blues", aspect="auto")
            axes[l].set_xlabel("Key Position")
            axes[l].set_ylabel("Query Position")
            axes[l].set_title(f"Layer {l} Avg Attention")

    fig.suptitle("Attention Patterns")
    fig.tight_layout()
    if exp_dir:
        fig.savefig(exp_dir / "figures" / "attention_patterns.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return all_patterns, labels


def run_sync_analysis(
    all_activations: dict[str, np.ndarray],
    raw_tokens: np.ndarray,
    components: list[tuple[float, float]],
    labels: np.ndarray,
    exp_dir,
    n_layers: int,
    model=None,
    device: str = "cuda",
):
    """Run full synchronization dynamics analysis."""
    print("\n--- Synchronization Dynamics Analysis ---")

    last_layer = f"blocks.{n_layers-1}.hook_resid_post"
    acts_last = all_activations[last_layer]

    # Position-wise analysis
    print("Position-wise probing...")
    pos_results = position_wise_regression(acts_last, raw_tokens, components, labels)
    plot_sync_dynamics(pos_results, exp_dir, components)

    # Layer-wise analysis
    print("Layer-wise probing...")
    layer_results = layer_wise_probing(all_activations, raw_tokens, components, labels)
    plot_layer_probing(layer_results, exp_dir, components)

    # Attention pattern analysis
    if model is not None:
        print("Attention pattern analysis...")
        seq_length = raw_tokens.shape[1]
        analyze_attention_patterns(model, components, seq_length, n_samples=2000, device=device, exp_dir=exp_dir)

    # Save results
    import json
    with open(exp_dir / "sync_analysis.json", "w") as f:
        json.dump({"position_wise": pos_results, "layer_wise": layer_results}, f, indent=2)

    # Print summary
    print(f"  Meta-belief R2 at pos 1: {pos_results['meta_r2'][0]:.3f}")
    print(f"  Meta-belief R2 at final pos: {pos_results['meta_r2'][-1]:.3f}")
    for i, layer in enumerate(layer_results["layers"]):
        print(f"  {layer}: meta R2={layer_results['meta_r2'][i]:.3f}, belief R2={layer_results['avg_belief_r2'][i]:.3f}")

    return pos_results, layer_results
