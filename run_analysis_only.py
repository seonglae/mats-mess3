"""Run analysis only on a trained model (Phase 2).

Loads the model state from the training that already completed,
runs all analysis with the updated code (train/test split, sync dynamics, etc).
"""

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from src.bayes_optimal import bayes_optimal_loss_mixture, bayes_optimal_loss_single
from src.mess3 import compute_belief_states, compute_meta_beliefs, generate_mess3_sequences_fast
from src.model import create_model
from src.sync_analysis import run_sync_analysis

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Phase 2 config
COMPONENTS = [
    (0.95, 0.03),
    (0.85, 0.05),
    (0.65, 0.15),
]
N_LAYERS = 4
D_MODEL = 128
N_HEADS = 4
SEQ_LENGTH = 16
ANALYSIS_SAMPLES = 5000

EXP_DIR = Path("experiments/phase2_nonergodic_3comp")


def generate_batch(batch_size, seq_length, components, rng):
    K = len(components)
    per_comp = batch_size // K
    remainder = batch_size - per_comp * K
    all_tokens, all_labels = [], []
    for k, (alpha, x_param) in enumerate(components):
        n = per_comp + (1 if k < remainder else 0)
        tokens, _ = generate_mess3_sequences_fast(alpha, x_param, n, seq_length, rng)
        all_tokens.append(tokens)
        all_labels.append(np.full(n, k))
    tokens = np.concatenate(all_tokens)
    labels = np.concatenate(all_labels)
    perm = rng.permutation(len(tokens))
    tokens, labels = tokens[perm], labels[perm]
    bos = np.zeros((len(tokens), 1), dtype=np.int64)
    input_ids = np.concatenate([bos, tokens + 1], axis=1)
    return torch.tensor(input_ids, dtype=torch.long), labels, tokens


def main():
    print(f"Device: {DEVICE}")
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    (EXP_DIR / "figures").mkdir(exist_ok=True)

    # Recreate model and load weights if saved, otherwise retrain is needed
    model = create_model(
        n_layers=N_LAYERS, d_model=D_MODEL, n_heads=N_HEADS,
        d_mlp=4 * D_MODEL, n_ctx=SEQ_LENGTH + 1, d_vocab=4, device=DEVICE,
    )

    # Try to load saved model
    model_path = EXP_DIR / "model.pt"
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Loaded model from {model_path}")
    else:
        print("No saved model found. Training fresh (20K steps)...")
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=0)
        rng = np.random.default_rng(42)
        model.train()
        for step in range(1, 5001):
            input_ids, _, _ = generate_batch(8192, SEQ_LENGTH, COMPONENTS, rng)
            input_ids = input_ids.to(DEVICE)
            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                input_ids[:, 1:].reshape(-1),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 500 == 0:
                print(f"  Step {step}/5000, loss={loss.item():.4f}")
        torch.save(model.state_dict(), model_path)
        print(f"Saved model to {model_path}")

    # Bayes-optimal
    print("Computing Bayes-optimal...")
    bayes_info = bayes_optimal_loss_mixture(COMPONENTS, SEQ_LENGTH, n_samples_per_comp=5000)

    def to_ser(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: to_ser(v) for k, v in obj.items()}
        if isinstance(obj, (np.float64, np.float32)): return float(obj)
        return obj
    with open(EXP_DIR / "bayes_optimal.json", "w") as f:
        json.dump(to_ser(bayes_info), f, indent=2)
    print(f"  Oracle avg: {bayes_info['oracle_avg']:.4f}")
    print(f"  Bayesian avg: {bayes_info['bayesian_avg']:.4f}")

    # Per-position loss
    model.eval()
    with torch.no_grad():
        eval_ids, _, _ = generate_batch(4096, SEQ_LENGTH, COMPONENTS, np.random.default_rng(777))
        eval_ids = eval_ids.to(DEVICE)
        logits_eval = model(eval_ids)
        per_pos_losses = []
        for t in range(SEQ_LENGTH):
            ce = torch.nn.functional.cross_entropy(logits_eval[:, t, :], eval_ids[:, t + 1]).item()
            per_pos_losses.append(ce)
        print(f"  Per-position loss: {[f'{l:.3f}' for l in per_pos_losses]}")
    with open(EXP_DIR / "per_position_loss.json", "w") as f:
        json.dump(per_pos_losses, f)

    # Per-position loss plot
    fig, ax = plt.subplots(figsize=(8, 5))
    positions = list(range(1, SEQ_LENGTH + 1))
    ax.plot(positions, per_pos_losses, "ko-", markersize=4, label="Model")
    ax.plot(positions, bayes_info["oracle_per_pos"], "b--", label="Bayes-optimal (oracle)")
    ax.plot(positions, bayes_info["bayesian_per_pos"], "r--", label="Bayes-optimal (meta-inference)")
    ax.axhline(y=np.log(3), color="gray", linestyle=":", alpha=0.5, label="Uniform")
    ax.set_xlabel("Context Position")
    ax.set_ylabel("Cross-Entropy (nats)")
    ax.set_title("Per-Position Loss vs Bayes-Optimal")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(EXP_DIR / "figures" / "per_position_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Generate analysis data
    print("Generating analysis data...")
    rng_a = np.random.default_rng(123)
    input_ids, labels, raw_tokens = generate_batch(ANALYSIS_SAMPLES, SEQ_LENGTH, COMPONENTS, rng_a)
    input_ids_dev = input_ids.to(DEVICE)

    # Extract activations
    print("Extracting activations...")
    all_acts = {}
    hooks = []
    def make_hook(name):
        def fn(value, hook):
            all_acts_batch[name] = value.detach().cpu().numpy()
        return fn

    hook_list = [("hook_embed", make_hook("hook_embed"))]
    for l in range(N_LAYERS):
        name = f"blocks.{l}.hook_resid_post"
        hook_list.append((name, make_hook(name)))

    bsz = 512
    for i in range(0, ANALYSIS_SAMPLES, bsz):
        all_acts_batch = {}
        batch = input_ids_dev[i:i+bsz]
        model.run_with_hooks(batch, fwd_hooks=hook_list)
        for name, act in all_acts_batch.items():
            if name not in all_acts:
                all_acts[name] = []
            all_acts[name].append(act)

    for name in all_acts:
        all_acts[name] = np.concatenate(all_acts[name])

    K = len(COMPONENTS)

    # PCA per layer
    print("PCA by layer:")
    cev_dict = {}
    for name, acts in all_acts.items():
        X = acts[:, 1:, :].reshape(-1, D_MODEL)
        pca = PCA(n_components=min(30, D_MODEL))
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
    fig.savefig(EXP_DIR / "figures" / "cev_by_layer.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # PCA projections
    last = f"blocks.{N_LAYERS-1}.hook_resid_post"
    acts_last = all_acts[last]
    X_all = acts_last[:, 1:, :].reshape(-1, D_MODEL)
    L_rep = np.repeat(labels, SEQ_LENGTH)

    pca = PCA(n_components=6)
    X_pca = pca.fit_transform(X_all)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i in range(3):
        pc1, pc2 = 2*i, 2*i+1
        axes[i].scatter(X_pca[:, pc1], X_pca[:, pc2], c=L_rep, s=0.5, alpha=0.3, cmap="tab10")
        axes[i].set_xlabel(f"PC{pc1+1} ({pca.explained_variance_ratio_[pc1]:.1%})")
        axes[i].set_ylabel(f"PC{pc2+1} ({pca.explained_variance_ratio_[pc2]:.1%})")
    fig.suptitle("PCA Projections (Final Layer)")
    fig.tight_layout()
    fig.savefig(EXP_DIR / "figures" / "pca_projections.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Geometry by position
    selected_pos = [1, 2, 4, 8, 12, 16]
    pca_all = PCA(n_components=4)
    pca_all.fit(X_all)
    fig, axes = plt.subplots(1, len(selected_pos), figsize=(4*len(selected_pos), 4))
    for i, pos in enumerate(selected_pos):
        Xp = acts_last[:, pos, :]
        Xp_pca = pca_all.transform(Xp)
        axes[i].scatter(Xp_pca[:, 0], Xp_pca[:, 1], c=labels, s=1, alpha=0.3, cmap="tab10")
        axes[i].set_title(f"Pos {pos}")
    fig.suptitle("Geometry by Position")
    fig.tight_layout()
    fig.savefig(EXP_DIR / "figures" / "geometry_by_position.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Layer progression
    layer_names = ["hook_embed"] + [f"blocks.{l}.hook_resid_post" for l in range(N_LAYERS)]
    fig, axes = plt.subplots(1, len(layer_names), figsize=(4*len(layer_names), 4))
    for i, name in enumerate(layer_names):
        acts = all_acts[name]
        Xl = acts[:, 1:, :].reshape(-1, D_MODEL)
        Ll = np.repeat(labels, SEQ_LENGTH)
        pca_l = PCA(n_components=2)
        Xl_pca = pca_l.fit_transform(Xl)
        axes[i].scatter(Xl_pca[:, 0], Xl_pca[:, 1], c=Ll, s=0.5, alpha=0.2, cmap="tab10")
        short = name.replace("blocks.", "L").replace(".hook_resid_post", "").replace("hook_", "")
        axes[i].set_title(short)
    fig.suptitle("Layer Progression")
    fig.tight_layout()
    fig.savefig(EXP_DIR / "figures" / "layer_progression.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Belief regression (with train/test split)
    print("Belief regression (held-out):")
    reg_results = {}
    for k, (alpha, x_param) in enumerate(COMPONENTS):
        mask = labels == k
        comp_tokens = raw_tokens[mask]
        beliefs = compute_belief_states(comp_tokens, alpha, x_param)
        comp_acts = acts_last[mask]
        X_reg, Y_reg = [], []
        for p in range(1, SEQ_LENGTH + 1):
            tp = p - 1
            if tp < beliefs.shape[1]:
                X_reg.append(comp_acts[:, p, :])
                Y_reg.append(beliefs[:, tp, :])
        X_reg = np.concatenate(X_reg)
        Y_reg = np.concatenate(Y_reg)
        X_tr, X_te, Y_tr, Y_te = train_test_split(X_reg, Y_reg, test_size=0.2, random_state=42)
        reg = Ridge(alpha=1e-4)
        reg.fit(X_tr, Y_tr)
        Y_pred = reg.predict(X_te)
        rmse = np.sqrt(np.mean((Y_te - Y_pred) ** 2))
        r2 = r2_score(Y_te, Y_pred)
        reg_results[f"component_{k}"] = {"alpha": alpha, "x": x_param, "rmse": float(rmse), "r2": float(r2)}
        print(f"  C{k} (a={alpha}, x={x_param}): R2={r2:.4f}, RMSE={rmse:.4f}")

    # Meta-belief regression
    meta_beliefs = compute_meta_beliefs(raw_tokens, COMPONENTS)
    X_meta, Y_meta = [], []
    for p in range(1, SEQ_LENGTH + 1):
        tp = p - 1
        if tp < meta_beliefs.shape[1]:
            X_meta.append(acts_last[:, p, :])
            Y_meta.append(meta_beliefs[:, tp, :])
    X_meta = np.concatenate(X_meta)
    Y_meta = np.concatenate(Y_meta)
    Xm_tr, Xm_te, Ym_tr, Ym_te = train_test_split(X_meta, Y_meta, test_size=0.2, random_state=42)
    reg_m = Ridge(alpha=1e-4)
    reg_m.fit(Xm_tr, Ym_tr)
    r2_meta = r2_score(Ym_te, reg_m.predict(Xm_te))
    rmse_meta = np.sqrt(np.mean((Ym_te - reg_m.predict(Xm_te)) ** 2))
    reg_results["meta_belief"] = {"r2": float(r2_meta), "rmse": float(rmse_meta)}
    print(f"  Meta-belief: R2={r2_meta:.4f}, RMSE={rmse_meta:.4f}")

    with open(EXP_DIR / "regression_results.json", "w") as f:
        json.dump(reg_results, f, indent=2)

    # Gasket plots
    v0, v1, v2 = np.array([0, 0]), np.array([1, 0]), np.array([0.5, np.sqrt(3) / 2])
    fig, axes = plt.subplots(2, K, figsize=(4*K, 8))
    for k, (alpha, x_param) in enumerate(COMPONENTS):
        mask = labels == k
        comp_tokens = raw_tokens[mask]
        beliefs = compute_belief_states(comp_tokens, alpha, x_param)
        b = beliefs[:, -1, :]
        xy = b[:, 0:1] * v0 + b[:, 1:2] * v1 + b[:, 2:3] * v2
        axes[0, k].scatter(xy[:, 0], xy[:, 1], s=0.5, alpha=0.3)
        tri = plt.Polygon([v0, v1, v2], fill=False, edgecolor="gray")
        axes[0, k].add_patch(tri)
        axes[0, k].set_xlim(-0.1, 1.1); axes[0, k].set_ylim(-0.1, 1.0)
        axes[0, k].set_aspect("equal")
        axes[0, k].set_title(f"True C{k} (a={alpha}, x={x_param})")

        comp_acts_last = acts_last[mask, -1, :]
        a_tr, a_te, b_tr, b_te = train_test_split(comp_acts_last, b, test_size=0.2, random_state=42)
        reg = Ridge(alpha=1e-4)
        reg.fit(a_tr, b_tr)
        b_pred = np.clip(reg.predict(a_te), 0, 1)
        b_pred /= b_pred.sum(axis=1, keepdims=True)
        xy_pred = b_pred[:, 0:1] * v0 + b_pred[:, 1:2] * v1 + b_pred[:, 2:3] * v2
        r2_last = r2_score(b_te, b_pred)
        axes[1, k].scatter(xy_pred[:, 0], xy_pred[:, 1], s=0.5, alpha=0.3)
        tri2 = plt.Polygon([v0, v1, v2], fill=False, edgecolor="gray")
        axes[1, k].add_patch(tri2)
        axes[1, k].set_xlim(-0.1, 1.1); axes[1, k].set_ylim(-0.1, 1.0)
        axes[1, k].set_aspect("equal")
        axes[1, k].set_title(f"Predicted C{k} (R2={r2_last:.3f})")

    fig.suptitle("Belief Geometry: True vs Predicted")
    fig.tight_layout()
    fig.savefig(EXP_DIR / "figures" / "belief_gaskets.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Per-layer regression
    print("Per-layer belief R2:")
    layer_r2 = {}
    for l_name in [f"blocks.{l}.hook_resid_post" for l in range(N_LAYERS)]:
        acts_l = all_acts[l_name]
        short = l_name.replace("blocks.", "L").replace(".hook_resid_post", "")
        r2s = {}
        for k, (alpha, x_param) in enumerate(COMPONENTS):
            mask = labels == k
            comp_tokens = raw_tokens[mask]
            beliefs = compute_belief_states(comp_tokens, alpha, x_param)
            comp_acts = acts_l[mask]
            X_r, Y_r = [], []
            for p in range(1, SEQ_LENGTH + 1):
                tp = p - 1
                if tp < beliefs.shape[1]:
                    X_r.append(comp_acts[:, p, :])
                    Y_r.append(beliefs[:, tp, :])
            X_r = np.concatenate(X_r)
            Y_r = np.concatenate(Y_r)
            Xr_tr, Xr_te, Yr_tr, Yr_te = train_test_split(X_r, Y_r, test_size=0.2, random_state=42)
            reg = Ridge(alpha=1e-4)
            reg.fit(Xr_tr, Yr_tr)
            r2s[f"C{k}"] = float(r2_score(Yr_te, reg.predict(Xr_te)))
        layer_r2[short] = r2s
        print(f"  {short}: {r2s}")
    with open(EXP_DIR / "layer_regression.json", "w") as f:
        json.dump(layer_r2, f, indent=2)

    # Dims by position
    dims_by_pos = []
    for t in range(SEQ_LENGTH + 1):
        Xt = acts_last[:, t, :]
        pca_t = PCA(n_components=min(30, D_MODEL))
        pca_t.fit(Xt)
        cev_t = np.cumsum(pca_t.explained_variance_ratio_)
        dims_by_pos.append(int(np.searchsorted(cev_t, 0.95) + 1))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(dims_by_pos)), dims_by_pos, "o-")
    ax.set_xlabel("Context Position")
    ax.set_ylabel("Dims for 95% CEV")
    ax.set_title("Effective Dimensionality by Position")
    fig.tight_layout()
    fig.savefig(EXP_DIR / "figures" / "dims_by_position.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Synchronization dynamics
    run_sync_analysis(all_acts, raw_tokens, COMPONENTS, labels, EXP_DIR, N_LAYERS, model=model, device=DEVICE)

    print(f"\nDone! Results in {EXP_DIR}")


if __name__ == "__main__":
    main()
