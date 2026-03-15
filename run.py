"""Main experiment: Non-ergodic Mess3 transformer training and analysis.

MATS Summer 2026 Work Test - Paul Riechers & Adam Shai Stream

Staged approach:
  Phase 1: Single Mess3 - verify gasket emerges in residual stream
  Phase 2: Non-ergodic mixture - analyze how transformer represents multiple components
"""

import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from src.mess3 import (
    compute_belief_states,
    compute_meta_beliefs,
    generate_mess3_sequences_fast,
    mess3_transition_matrices,
)
from src.model import create_model

# ============================================================
# Configuration
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def compute_entropy_rate(alpha: float, x: float) -> float:
    """Compute entropy rate H(Z|S) for Mess3 process.

    This is the theoretical lower bound on cross-entropy loss
    achievable by a model with perfect state knowledge.
    """
    T = mess3_transition_matrices(alpha, x)
    H = 0.0
    for s in range(3):
        # P(z|s) = sum_j T^(z)[s,j]
        for z in range(3):
            p = T[z, s, :].sum()
            if p > 0:
                H -= (1 / 3) * p * np.log(p)  # pi(s) = 1/3 for Mess3
    return H


def generate_batch(batch_size, seq_length, components, rng):
    """Generate a fresh batch on-the-fly."""
    K = len(components)
    per_comp = batch_size // K
    remainder = batch_size - per_comp * K

    all_tokens = []
    all_labels = []

    for k, (alpha, x_param) in enumerate(components):
        n = per_comp + (1 if k < remainder else 0)
        tokens, _ = generate_mess3_sequences_fast(alpha, x_param, n, seq_length, rng)
        all_tokens.append(tokens)
        all_labels.append(np.full(n, k))

    tokens = np.concatenate(all_tokens, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    perm = rng.permutation(len(tokens))
    tokens = tokens[perm]
    labels = labels[perm]

    bos = np.zeros((len(tokens), 1), dtype=np.int64)
    input_ids = np.concatenate([bos, tokens + 1], axis=1)

    return torch.tensor(input_ids, dtype=torch.long), labels, tokens


def train_and_analyze(
    components: list[tuple[float, float]],
    exp_name: str,
    n_layers: int = 4,
    d_model: int = 128,
    n_heads: int = 4,
    seq_length: int = 16,
    batch_size: int = 8192,
    lr: float = 5e-4,
    n_steps: int = 50000,
    log_every: int = 200,
    pca_every: int = 2000,
    analysis_samples: int = 5000,
):
    """Full training + analysis pipeline for a given set of components."""

    config = {
        "components": components,
        "n_layers": n_layers,
        "d_model": d_model,
        "n_heads": n_heads,
        "d_mlp": 4 * d_model,
        "seq_length": seq_length,
        "n_ctx": seq_length + 1,
        "d_vocab": 4,
        "batch_size": batch_size,
        "lr": lr,
        "n_steps": n_steps,
    }

    # Print entropy rates
    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"Components: {components}")
    print(f"Model: {n_layers}L, d={d_model}, heads={n_heads}")
    print(f"Training: {n_steps} steps, batch={batch_size}, lr={lr}")
    for k, (alpha, x_param) in enumerate(components):
        H = compute_entropy_rate(alpha, x_param)
        zeta = 1 - 3 * x_param
        print(f"  C{k} (a={alpha}, x={x_param}): H={H:.4f} nats, zeta={zeta:.2f}")
    print(f"Uniform CE: {np.log(3):.4f} nats")
    print(f"Device: {DEVICE}")
    print(f"{'='*60}\n")

    # Setup
    exp_dir = Path(f"experiments/{exp_name}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "figures").mkdir(exist_ok=True)
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Create model
    model = create_model(
        n_layers=n_layers, d_model=d_model, n_heads=n_heads,
        d_mlp=4 * d_model, n_ctx=seq_length + 1, d_vocab=4, device=DEVICE,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    rng = np.random.default_rng(42)

    # ---- Training ----
    metrics = []
    pca_history = []
    start = time.time()

    model.train()
    for step in range(1, n_steps + 1):
        input_ids, _, _ = generate_batch(batch_size, seq_length, components, rng)
        input_ids = input_ids.to(DEVICE)

        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            input_ids[:, 1:].reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % log_every == 0:
            elapsed = time.time() - start
            l = loss.item()
            metrics.append({"step": step, "loss": l, "elapsed_s": elapsed})
            print(f"  Step {step:>6d}/{n_steps}, loss={l:.4f}, {elapsed:.0f}s")

        if step % pca_every == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                pca_data = _quick_pca(model, components, seq_length, d_model, n_layers)
                pca_data["step"] = step
                pca_history.append(pca_data)
                print(f"    -> PCA dims@95%: {pca_data['n95']}")
            model.train()

    # Save metrics
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)
    with open(exp_dir / "pca_history.json", "w") as f:
        json.dump([{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in e.items()} for e in pca_history], f)

    # ---- Plot training loss ----
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([m["step"] for m in metrics], [m["loss"] for m in metrics])
    for k, (alpha, x_param) in enumerate(components):
        H = compute_entropy_rate(alpha, x_param)
        ax.axhline(y=H, color=f"C{k}", linestyle=":", alpha=0.7, label=f"H(C{k})={H:.3f}")
    ax.axhline(y=np.log(3), color="gray", linestyle="--", alpha=0.5, label=f"Uniform={np.log(3):.3f}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cross-Entropy Loss (nats)")
    ax.set_title(f"Training Loss - {exp_name}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(exp_dir / "figures" / "training_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Plot CEV over training ----
    if pca_history:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        for entry in pca_history:
            cev = np.array(entry["cev"])
            ax1.plot(range(1, len(cev) + 1), cev, label=f"Step {entry['step']}")
        ax1.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5)
        ax1.set_xlabel("Number of Components")
        ax1.set_ylabel("Cumulative Explained Variance")
        ax1.set_xlim(0, 30)
        ax1.set_ylim(0.5, 1.01)
        ax1.legend(fontsize=7)
        ax1.set_title("CEV Over Training")

        ax2.plot([e["step"] for e in pca_history], [e["n95"] for e in pca_history], "o-")
        ax2.set_xlabel("Training Step")
        ax2.set_ylabel("Dims for 95% CEV")
        ax2.set_title("Effective Dimensionality")
        fig.tight_layout()
        fig.savefig(exp_dir / "figures" / "cev_over_training.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ---- Full Analysis ----
    print("\n--- Analysis ---")
    model.eval()

    # Generate analysis data
    rng_analysis = np.random.default_rng(123)
    input_ids, labels, raw_tokens = generate_batch(analysis_samples, seq_length, components, rng_analysis)
    input_ids_dev = input_ids.to(DEVICE)

    # Extract activations at all layers
    activations = {}
    hooks = []
    def make_hook(name):
        def fn(value, hook):
            activations[name] = value.cpu().numpy()
        return fn

    hooks.append(("hook_embed", make_hook("hook_embed")))
    for l in range(n_layers):
        name = f"blocks.{l}.hook_resid_post"
        hooks.append((name, make_hook(name)))

    # Run in batches
    all_acts = {}
    bsz = 512
    for i in range(0, analysis_samples, bsz):
        activations = {}
        batch = input_ids_dev[i:i+bsz]
        model.run_with_hooks(batch, fwd_hooks=hooks)
        for name, act in activations.items():
            if name not in all_acts:
                all_acts[name] = []
            all_acts[name].append(act)

    for name in all_acts:
        all_acts[name] = np.concatenate(all_acts[name], axis=0)

    # PCA per layer
    print("PCA by layer:")
    cev_dict = {}
    for name, acts in all_acts.items():
        X = acts[:, 1:, :].reshape(-1, d_model)
        pca = PCA(n_components=min(30, d_model))
        pca.fit(X)
        cev = np.cumsum(pca.explained_variance_ratio_)
        n95 = int(np.searchsorted(cev, 0.95) + 1)
        short = name.replace("blocks.", "L").replace(".hook_resid_post", "").replace("hook_", "")
        cev_dict[short] = pca.explained_variance_ratio_
        print(f"  {short}: dims@95% = {n95}")

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, evr in cev_dict.items():
        ax.plot(range(1, len(evr) + 1), np.cumsum(evr), label=name)
    ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("CEV")
    ax.set_xlim(0, 30)
    ax.set_ylim(0.5, 1.01)
    ax.legend()
    ax.set_title("CEV by Layer")
    fig.tight_layout()
    fig.savefig(exp_dir / "figures" / "cev_by_layer.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # PCA projections (final layer) colored by component
    last = f"blocks.{n_layers-1}.hook_resid_post"
    acts_last = all_acts[last]
    X = acts_last[:, 1:, :].reshape(-1, d_model)
    L_rep = np.repeat(labels, seq_length)

    pca = PCA(n_components=6)
    X_pca = pca.fit_transform(X)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i in range(3):
        pc1, pc2 = 2 * i, 2 * i + 1
        axes[i].scatter(X_pca[:, pc1], X_pca[:, pc2], c=L_rep, s=0.5, alpha=0.3, cmap="tab10")
        axes[i].set_xlabel(f"PC{pc1+1} ({pca.explained_variance_ratio_[pc1]:.1%})")
        axes[i].set_ylabel(f"PC{pc2+1} ({pca.explained_variance_ratio_[pc2]:.1%})")
        axes[i].set_title(f"PCs {pc1+1}-{pc2+1}")
    fig.suptitle(f"PCA Projections (Final Layer) - {exp_name}")
    fig.tight_layout()
    fig.savefig(exp_dir / "figures" / "pca_projections.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # PCA projections by context position
    selected_pos = [1, 2, 4, 8, 12, min(16, seq_length)]
    selected_pos = sorted(set(p for p in selected_pos if p <= seq_length))
    fig, axes = plt.subplots(1, len(selected_pos), figsize=(4 * len(selected_pos), 4))
    if len(selected_pos) == 1:
        axes = [axes]
    pca_all = PCA(n_components=4)
    pca_all.fit(X)
    for i, pos in enumerate(selected_pos):
        Xp = acts_last[:, pos, :]
        Xp_pca = pca_all.transform(Xp)
        axes[i].scatter(Xp_pca[:, 0], Xp_pca[:, 1], c=labels, s=1, alpha=0.3, cmap="tab10")
        axes[i].set_title(f"Pos {pos}")
    fig.suptitle(f"Geometry by Position (Final Layer)")
    fig.tight_layout()
    fig.savefig(exp_dir / "figures" / "geometry_by_position.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Layer progression
    layer_names = ["hook_embed"] + [f"blocks.{l}.hook_resid_post" for l in range(n_layers)]
    fig, axes = plt.subplots(1, len(layer_names), figsize=(4 * len(layer_names), 4))
    for i, name in enumerate(layer_names):
        acts = all_acts[name]
        Xl = acts[:, 1:, :].reshape(-1, d_model)
        Ll = np.repeat(labels, seq_length)
        pca_l = PCA(n_components=2)
        Xl_pca = pca_l.fit_transform(Xl)
        axes[i].scatter(Xl_pca[:, 0], Xl_pca[:, 1], c=Ll, s=0.5, alpha=0.2, cmap="tab10")
        short = name.replace("blocks.", "L").replace(".hook_resid_post", "").replace("hook_", "")
        axes[i].set_title(short)
    fig.suptitle("Layer Progression")
    fig.tight_layout()
    fig.savefig(exp_dir / "figures" / "layer_progression.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Belief regression per component
    print("Belief regression:")
    K = len(components)
    reg_results = {}

    for k, (alpha, x_param) in enumerate(components):
        mask = labels == k
        comp_tokens = raw_tokens[mask]
        beliefs = compute_belief_states(comp_tokens, alpha, x_param)
        comp_acts = acts_last[mask]

        X_reg = []
        Y_reg = []
        for p in range(1, seq_length + 1):
            tp = p - 1
            if tp < beliefs.shape[1]:
                X_reg.append(comp_acts[:, p, :])
                Y_reg.append(beliefs[:, tp, :])

        X_reg = np.concatenate(X_reg, axis=0)
        Y_reg = np.concatenate(Y_reg, axis=0)

        reg = Ridge(alpha=1e-4)
        reg.fit(X_reg, Y_reg)
        Y_pred = reg.predict(X_reg)
        rmse = np.sqrt(np.mean((Y_reg - Y_pred) ** 2))
        r2 = r2_score(Y_reg, Y_pred)
        reg_results[f"component_{k}"] = {"alpha": alpha, "x": x_param, "rmse": rmse, "r2": r2}
        print(f"  C{k} (a={alpha}, x={x_param}): R2={r2:.4f}, RMSE={rmse:.4f}")

    # Meta-belief regression
    if K > 1:
        meta_beliefs = compute_meta_beliefs(raw_tokens, components)
        X_meta = []
        Y_meta = []
        for p in range(1, seq_length + 1):
            tp = p - 1
            if tp < meta_beliefs.shape[1]:
                X_meta.append(acts_last[:, p, :])
                Y_meta.append(meta_beliefs[:, tp, :])
        X_meta = np.concatenate(X_meta, axis=0)
        Y_meta = np.concatenate(Y_meta, axis=0)

        reg_m = Ridge(alpha=1e-4)
        reg_m.fit(X_meta, Y_meta)
        Y_meta_pred = reg_m.predict(X_meta)
        r2_meta = r2_score(Y_meta, Y_meta_pred)
        rmse_meta = np.sqrt(np.mean((Y_meta - Y_meta_pred) ** 2))
        reg_results["meta_belief"] = {"r2": r2_meta, "rmse": rmse_meta}
        print(f"  Meta-belief: R2={r2_meta:.4f}, RMSE={rmse_meta:.4f}")

    with open(exp_dir / "regression_results.json", "w") as f:
        json.dump(reg_results, f, indent=2, default=str)

    # Ground truth gaskets
    fig, axes = plt.subplots(2, K, figsize=(4 * K, 8))
    if K == 1:
        axes = axes.reshape(2, 1)

    v0, v1, v2 = np.array([0, 0]), np.array([1, 0]), np.array([0.5, np.sqrt(3) / 2])

    for k, (alpha, x_param) in enumerate(components):
        mask = labels == k
        comp_tokens = raw_tokens[mask]
        beliefs = compute_belief_states(comp_tokens, alpha, x_param)

        # True gasket (last position, after 16 tokens of context)
        b = beliefs[:, -1, :]
        xy = b[:, 0:1] * v0 + b[:, 1:2] * v1 + b[:, 2:3] * v2
        axes[0, k].scatter(xy[:, 0], xy[:, 1], s=0.5, alpha=0.3)
        triangle = plt.Polygon([v0, v1, v2], fill=False, edgecolor="gray")
        axes[0, k].add_patch(triangle)
        axes[0, k].set_xlim(-0.1, 1.1)
        axes[0, k].set_ylim(-0.1, 1.0)
        axes[0, k].set_aspect("equal")
        axes[0, k].set_title(f"True C{k} (a={alpha}, x={x_param})")

        # Predicted gasket (from regression)
        comp_acts_last = acts_last[mask, -1, :]
        reg = Ridge(alpha=1e-4)
        reg.fit(comp_acts_last, b)
        b_pred = np.clip(reg.predict(comp_acts_last), 0, 1)
        b_pred /= b_pred.sum(axis=1, keepdims=True)
        xy_pred = b_pred[:, 0:1] * v0 + b_pred[:, 1:2] * v1 + b_pred[:, 2:3] * v2
        r2_last = r2_score(b, b_pred)
        axes[1, k].scatter(xy_pred[:, 0], xy_pred[:, 1], s=0.5, alpha=0.3)
        triangle2 = plt.Polygon([v0, v1, v2], fill=False, edgecolor="gray")
        axes[1, k].add_patch(triangle2)
        axes[1, k].set_xlim(-0.1, 1.1)
        axes[1, k].set_ylim(-0.1, 1.0)
        axes[1, k].set_aspect("equal")
        axes[1, k].set_title(f"Predicted C{k} (R2={r2_last:.3f})")

    fig.suptitle("Belief Geometry: True vs Predicted")
    fig.tight_layout()
    fig.savefig(exp_dir / "figures" / "belief_gaskets.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Per-layer R2
    print("Per-layer belief R2:")
    layer_r2 = {}
    for l_name in [f"blocks.{l}.hook_resid_post" for l in range(n_layers)]:
        acts_l = all_acts[l_name]
        short = l_name.replace("blocks.", "L").replace(".hook_resid_post", "")
        r2s = {}
        for k, (alpha, x_param) in enumerate(components):
            mask = labels == k
            comp_tokens = raw_tokens[mask]
            beliefs = compute_belief_states(comp_tokens, alpha, x_param)
            comp_acts = acts_l[mask]
            X_r, Y_r = [], []
            for p in range(1, seq_length + 1):
                tp = p - 1
                if tp < beliefs.shape[1]:
                    X_r.append(comp_acts[:, p, :])
                    Y_r.append(beliefs[:, tp, :])
            X_r = np.concatenate(X_r)
            Y_r = np.concatenate(Y_r)
            reg = Ridge(alpha=1e-4)
            reg.fit(X_r, Y_r)
            r2s[f"C{k}"] = r2_score(Y_r, reg.predict(X_r))
        layer_r2[short] = r2s
        print(f"  {short}: {r2s}")

    with open(exp_dir / "layer_regression.json", "w") as f:
        json.dump(layer_r2, f, indent=2)

    # Dims by position
    dims_by_pos = []
    for t in range(seq_length + 1):
        Xt = acts_last[:, t, :]
        pca_t = PCA(n_components=min(30, d_model))
        pca_t.fit(Xt)
        cev_t = np.cumsum(pca_t.explained_variance_ratio_)
        dims_by_pos.append(int(np.searchsorted(cev_t, 0.95) + 1))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(dims_by_pos)), dims_by_pos, "o-")
    ax.set_xlabel("Context Position")
    ax.set_ylabel("Dims for 95% CEV")
    ax.set_title("Effective Dimensionality by Position")
    fig.tight_layout()
    fig.savefig(exp_dir / "figures" / "dims_by_position.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nDone! Results in {exp_dir}")
    return model, exp_dir


def _quick_pca(model, components, seq_length, d_model, n_layers, n_samples=2000):
    rng = np.random.default_rng(999)
    input_ids, _, _ = generate_batch(n_samples, seq_length, components, rng)
    input_ids = input_ids.to(DEVICE)

    last = f"blocks.{n_layers-1}.hook_resid_post"
    activations = {}
    def hook_fn(value, hook):
        activations["a"] = value.cpu().numpy()

    model.run_with_hooks(input_ids, fwd_hooks=[(last, hook_fn)])
    X = activations["a"][:, 1:, :].reshape(-1, d_model)

    pca = PCA(n_components=min(30, d_model))
    pca.fit(X)
    cev = np.cumsum(pca.explained_variance_ratio_)
    n95 = int(np.searchsorted(cev, 0.95) + 1)
    return {"cev": cev, "evr": pca.explained_variance_ratio_, "n95": n95}


# ============================================================
# Main
# ============================================================

def main():
    # Phase 1: Single Mess3 (verify gasket reproduction)
    print("=" * 60)
    print("PHASE 1: Single Mess3 (alpha=0.85, x=0.05)")
    print("=" * 60)
    train_and_analyze(
        components=[(0.85, 0.05)],
        exp_name="phase1_single_mess3",
        n_layers=3,
        d_model=64,
        n_heads=4,
        seq_length=16,
        batch_size=8192,
        n_steps=20000,
        pca_every=2000,
    )

    # Phase 2: Non-ergodic mixture (3 well-separated components)
    print("\n" + "=" * 60)
    print("PHASE 2: Non-ergodic Mess3 Mixture")
    print("=" * 60)
    train_and_analyze(
        components=[
            (0.95, 0.03),  # Very clear, very persistent (H=0.34, zeta=0.91)
            (0.85, 0.05),  # Standard Mess3 (H=0.61, zeta=0.85)
            (0.65, 0.15),  # Moderate (H=0.97, zeta=0.55)
        ],
        exp_name="phase2_nonergodic_3comp",
        n_layers=4,
        d_model=128,
        n_heads=4,
        seq_length=16,
        batch_size=8192,
        n_steps=50000,
        pca_every=5000,
    )


if __name__ == "__main__":
    main()
