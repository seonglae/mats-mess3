"""Main experiment: Non-ergodic Mess3 transformer training and analysis.

MATS Summer 2026 Work Test - Paul Riechers & Adam Shai Stream
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.analysis import (
    belief_regression,
    extract_all_activations,
    pca_analysis,
    pca_by_position,
)
from src.data import NonErgodicMess3Dataset, create_dataloaders
from src.model import create_model
from src.visualize import (
    plot_belief_simplex,
    plot_cev,
    plot_ground_truth_gaskets,
    plot_layer_progression,
    plot_pca_by_position,
    plot_pca_projections,
    plot_regression_results,
    plot_training_loss,
)

# ============================================================
# Configuration
# ============================================================

COMPONENTS = [
    (0.90, 0.05),  # C0: High clarity, high persistence (zeta=0.85)
    (0.60, 0.15),  # C1: Medium clarity, medium persistence (zeta=0.55)
    (0.85, 0.30),  # C2: High clarity, low persistence (zeta=0.10)
    (0.50, 0.10),  # C3: Low clarity, high persistence (zeta=0.70)
]

CONFIG = {
    # Data
    "components": COMPONENTS,
    "n_sequences_per_component": 20000,
    "seq_length": 16,
    # Model
    "n_layers": 3,
    "d_model": 64,
    "n_heads": 4,
    "d_mlp": 256,
    "n_ctx": 17,  # seq_length + BOS
    "d_vocab": 4,  # BOS + {0, 1, 2}
    # Training
    "batch_size": 4096,
    "lr": 5e-4,
    "n_epochs": 200,
    "weight_decay": 0,
    # Analysis
    "n_analysis_samples": 5000,
}

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def setup_experiment() -> Path:
    """Create timestamped experiment directory."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    exp_dir = Path(f"experiments/{timestamp}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "figures").mkdir(exist_ok=True)

    with open(exp_dir / "config.json", "w") as f:
        json.dump(CONFIG, f, indent=2)

    return exp_dir


def train_model(dataset, exp_dir: Path) -> tuple:
    """Train the transformer and return model + metrics."""
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=CONFIG["batch_size"], shuffle=True, drop_last=True
    )

    model = create_model(
        n_layers=CONFIG["n_layers"],
        d_model=CONFIG["d_model"],
        n_heads=CONFIG["n_heads"],
        d_mlp=CONFIG["d_mlp"],
        n_ctx=CONFIG["n_ctx"],
        d_vocab=CONFIG["d_vocab"],
        device=DEVICE,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])

    metrics = []
    step = 0
    start_time = time.time()

    # Periodic PCA tracking
    pca_history = []

    model.train()
    for epoch in range(CONFIG["n_epochs"]):
        epoch_losses = []
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)

            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                input_ids[:, 1:].reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        elapsed = time.time() - start_time
        metric = {"step": step, "epoch": epoch, "loss": avg_loss, "elapsed_s": elapsed}
        metrics.append(metric)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{CONFIG['n_epochs']}, loss: {avg_loss:.4f}, elapsed: {elapsed:.1f}s")

        # Track PCA periodically
        if (epoch + 1) % 50 == 0 or epoch == 0 or epoch == CONFIG["n_epochs"] - 1:
            model.eval()
            with torch.no_grad():
                pca_data = quick_pca(model, dataset, n_samples=2000)
                pca_data["epoch"] = epoch
                pca_data["step"] = step
                pca_history.append(pca_data)
            model.train()

    # Save metrics
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)

    with open(exp_dir / "pca_history.json", "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable = []
        for entry in pca_history:
            s = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in entry.items()}
            serializable.append(s)
        json.dump(serializable, f)

    return model, metrics, pca_history


def quick_pca(model, dataset, n_samples=2000):
    """Quick PCA on final layer activations."""
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    input_ids = torch.stack([dataset[i]["input_ids"] for i in indices]).to(DEVICE)

    # Get final layer activations
    last_layer = CONFIG["n_layers"] - 1
    hook_name = f"blocks.{last_layer}.hook_resid_post"

    activations = {}
    def hook_fn(value, hook):
        activations["acts"] = value.cpu().numpy()

    model.run_with_hooks(input_ids, fwd_hooks=[(hook_name, hook_fn)])

    acts = activations["acts"]  # (n_samples, seq_len, d_model)
    # Use all non-BOS positions
    X = acts[:, 1:, :].reshape(-1, CONFIG["d_model"])

    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(30, CONFIG["d_model"]))
    pca.fit(X)

    cev = np.cumsum(pca.explained_variance_ratio_)
    n95 = int(np.searchsorted(cev, 0.95) + 1)

    return {
        "cev": cev,
        "evr": pca.explained_variance_ratio_,
        "n_components_95": n95,
    }


def run_analysis(model, dataset, exp_dir: Path):
    """Full post-training analysis."""
    print("\n=== Running Analysis ===")

    # 1. Extract activations
    print("Extracting activations...")
    model.eval()
    data = extract_all_activations(
        model, dataset, n_samples=CONFIG["n_analysis_samples"],
        batch_size=512, device=DEVICE,
    )

    # 2. PCA analysis per layer
    print("PCA analysis...")
    cev_dict = {}
    for name, acts in data["activations"].items():
        if "resid_post" not in name and "embed" not in name:
            continue
        pca_result = pca_analysis(acts)
        short = name.replace("blocks.", "L").replace(".hook_resid_post", "").replace("hook_", "")
        cev_dict[short] = pca_result["explained_variance_ratio"]
        print(f"  {short}: dims for 95% = {pca_result['n_components_95']}")

    fig = plot_cev(cev_dict, title="CEV by Layer")
    fig.savefig(exp_dir / "figures" / "cev_by_layer.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. PCA projections colored by component
    last_layer_name = f"blocks.{CONFIG['n_layers']-1}.hook_resid_post"
    if last_layer_name in data["activations"]:
        acts = data["activations"][last_layer_name]

        fig = plot_pca_projections(acts, data["labels"], layer_name="Final Layer")
        fig.savefig(exp_dir / "figures" / "pca_projections_final.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig = plot_pca_by_position(acts, data["labels"], layer_name="Final Layer")
        fig.savefig(exp_dir / "figures" / "pca_by_position.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 4. Layer progression
    fig = plot_layer_progression(data["activations"], data["labels"])
    fig.savefig(exp_dir / "figures" / "layer_progression.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 5. Belief state regression
    print("Belief regression...")
    if last_layer_name in data["activations"]:
        reg_results = belief_regression(
            data["activations"][last_layer_name],
            data["raw_tokens"],
            COMPONENTS,
            data["labels"],
        )

        reg_summary = {}
        for key, val in reg_results.items():
            reg_summary[key] = {k: v for k, v in val.items() if k != "regressor"}
            if "regressor" not in val:
                continue
            print(f"  {key}: R2={val['r2']:.4f}, RMSE={val['rmse']:.4f}")

        with open(exp_dir / "regression_results.json", "w") as f:
            json.dump(reg_summary, f, indent=2, default=str)

        # Plot regression results
        fig = plot_regression_results(
            data["activations"][last_layer_name],
            data["raw_tokens"],
            COMPONENTS,
            data["labels"],
            reg_results,
        )
        fig.savefig(exp_dir / "figures" / "belief_regression.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 6. PCA by position analysis
    print("Position analysis...")
    if last_layer_name in data["activations"]:
        pos_results = pca_by_position(data["activations"][last_layer_name])
        dims_by_pos = [r["n_components_95"] for r in pos_results]

        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.plot(range(len(dims_by_pos)), dims_by_pos, "o-")
        ax.set_xlabel("Context Position")
        ax.set_ylabel("Dimensions for 95% CEV")
        ax.set_title("Effective Dimensionality by Position")
        fig.tight_layout()
        fig.savefig(exp_dir / "figures" / "dims_by_position.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 7. Per-layer belief regression
    print("Per-layer regression...")
    layer_r2s = {}
    for name, acts in data["activations"].items():
        if "resid_post" not in name:
            continue
        try:
            reg = belief_regression(acts, data["raw_tokens"], COMPONENTS, data["labels"])
            short = name.replace("blocks.", "L").replace(".hook_resid_post", "")
            r2_vals = {}
            for k, v in reg.items():
                if "r2" in v:
                    r2_vals[k] = v["r2"]
            layer_r2s[short] = r2_vals
        except Exception as e:
            print(f"  Regression failed for {name}: {e}")

    if layer_r2s:
        with open(exp_dir / "layer_regression.json", "w") as f:
            json.dump(layer_r2s, f, indent=2)

    print("\nAnalysis complete!")
    return data, reg_results if last_layer_name in data["activations"] else {}


def main():
    print(f"Device: {DEVICE}")
    print(f"Components: {COMPONENTS}")
    print(f"Config: n_layers={CONFIG['n_layers']}, d_model={CONFIG['d_model']}")

    exp_dir = setup_experiment()
    print(f"Experiment dir: {exp_dir}")

    # 1. Ground truth gaskets
    print("\n=== Ground Truth Gaskets ===")
    fig = plot_ground_truth_gaskets(COMPONENTS, n_sequences=3000, seq_length=200)
    fig.savefig(exp_dir / "figures" / "ground_truth_gaskets.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Create dataset
    print("\n=== Creating Dataset ===")
    dataset = NonErgodicMess3Dataset(
        components=COMPONENTS,
        n_sequences_per_component=CONFIG["n_sequences_per_component"],
        seq_length=CONFIG["seq_length"],
    )
    print(f"Total sequences: {len(dataset)}")
    print(f"Sequences per component: {CONFIG['n_sequences_per_component']}")

    # 3. Train
    print("\n=== Training ===")
    model, metrics, pca_history = train_model(dataset, exp_dir)

    # Plot training loss
    fig = plot_training_loss(metrics)
    fig.savefig(exp_dir / "figures" / "training_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot CEV over training
    if pca_history:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        for entry in pca_history:
            cev = np.array(entry["cev"]) if isinstance(entry["cev"], list) else entry["cev"]
            ax.plot(range(1, len(cev) + 1), cev, label=f"Epoch {entry['epoch']}")
        ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Number of Components")
        ax.set_ylabel("Cumulative Explained Variance")
        ax.set_title("CEV Over Training")
        ax.set_xlim(0, 25)
        ax.set_ylim(0.5, 1.01)
        ax.legend()
        fig.tight_layout()
        fig.savefig(exp_dir / "figures" / "cev_over_training.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Dims for 95% over training
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        epochs_list = [e["epoch"] for e in pca_history]
        dims_list = [e["n_components_95"] for e in pca_history]
        ax.plot(epochs_list, dims_list, "o-")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Dimensions for 95% CEV")
        ax.set_title("Effective Dimensionality Over Training")
        fig.tight_layout()
        fig.savefig(exp_dir / "figures" / "dims_over_training.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 4. Analysis
    data, reg_results = run_analysis(model, dataset, exp_dir)

    print(f"\n=== Done! Results in {exp_dir} ===")


if __name__ == "__main__":
    main()
