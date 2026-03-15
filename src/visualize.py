"""Visualization: residual stream geometry, belief states, PCA."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA

from .mess3 import compute_belief_states, mess3_transition_matrices


def plot_belief_simplex(
    beliefs: np.ndarray,
    labels: np.ndarray | None = None,
    title: str = "Belief States in 2-Simplex",
    ax: plt.Axes | None = None,
    alpha: float = 0.3,
    s: float = 1.0,
) -> plt.Axes:
    """Plot belief states projected onto 2D simplex coordinates.

    Maps 3D probability vectors to 2D using barycentric coordinates.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    # Barycentric to Cartesian
    # Vertices of equilateral triangle
    v0 = np.array([0, 0])
    v1 = np.array([1, 0])
    v2 = np.array([0.5, np.sqrt(3) / 2])

    xy = beliefs[:, 0:1] * v0 + beliefs[:, 1:2] * v1 + beliefs[:, 2:3] * v2

    if labels is not None:
        scatter = ax.scatter(xy[:, 0], xy[:, 1], c=labels, s=s, alpha=alpha, cmap="tab10")
        plt.colorbar(scatter, ax=ax, label="Component")
    else:
        ax.scatter(xy[:, 0], xy[:, 1], s=s, alpha=alpha, color="blue")

    # Draw simplex boundary
    triangle = plt.Polygon([v0, v1, v2], fill=False, edgecolor="gray", linewidth=1)
    ax.add_patch(triangle)

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.05)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("State 1")
    ax.set_ylabel("State 2")

    return ax


def plot_ground_truth_gaskets(
    components: list[tuple[float, float]],
    n_sequences: int = 5000,
    seq_length: int = 100,
    figsize: tuple = (12, 4),
) -> plt.Figure:
    """Plot the ground-truth Sierpinski gasket for each component."""
    K = len(components)
    fig, axes = plt.subplots(1, K, figsize=figsize)
    if K == 1:
        axes = [axes]

    for k, (alpha_param, x_param) in enumerate(components):
        rng = np.random.default_rng(42 + k)
        T = mess3_transition_matrices(alpha_param, x_param)

        # Generate long sequence and track beliefs
        eta = np.ones(3) / 3.0
        beliefs = []
        state = rng.integers(0, 3)

        for _ in range(n_sequences * seq_length):
            # Sample token
            probs = np.array([T[z, state, :].sum() for z in range(3)])
            probs /= probs.sum()
            z = rng.choice(3, p=probs)

            # Update belief
            new_eta = eta @ T[z]
            eta = new_eta / new_eta.sum()
            beliefs.append(eta.copy())

            # Sample next state given token
            trans_probs = T[z, state, :]
            trans_probs /= trans_probs.sum()
            state = rng.choice(3, p=trans_probs)

        beliefs = np.array(beliefs)

        zeta = 1 - 3 * x_param
        plot_belief_simplex(
            beliefs,
            title=f"C{k}: alpha={alpha_param}, x={x_param}\nzeta={zeta:.2f}",
            ax=axes[k],
            alpha=0.05,
            s=0.5,
        )

    fig.suptitle("Ground-Truth Belief Geometries (Sierpinski Gaskets)", y=1.02)
    fig.tight_layout()
    return fig


def plot_cev(
    explained_variance_ratios: dict[str, np.ndarray],
    title: str = "Cumulative Explained Variance",
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """Plot CEV curves for different layers/conditions."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for name, evr in explained_variance_ratios.items():
        cev = np.cumsum(evr)
        ax.plot(range(1, len(cev) + 1), cev, label=name)

    ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5, label="95%")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0, 30)
    ax.set_ylim(0.5, 1.01)
    fig.tight_layout()
    return fig


def plot_pca_projections(
    activations: np.ndarray,
    labels: np.ndarray,
    layer_name: str = "",
    positions: list[int] | None = None,
    n_components: int = 6,
    figsize: tuple = (15, 10),
) -> plt.Figure:
    """Plot activations projected onto top PCA components, colored by component label."""
    n_samples, seq_len, d_model = activations.shape

    if positions is None:
        positions = list(range(1, seq_len))

    # Flatten
    X = activations[:, positions, :].reshape(-1, d_model)
    L = np.repeat(labels, len(positions))

    pca = PCA(n_components=min(n_components, d_model))
    X_pca = pca.fit_transform(X)

    n_pairs = n_components // 2
    fig, axes = plt.subplots(1, n_pairs, figsize=figsize)
    if n_pairs == 1:
        axes = [axes]

    for i in range(n_pairs):
        pc1, pc2 = 2 * i, 2 * i + 1
        if pc2 >= X_pca.shape[1]:
            break
        scatter = axes[i].scatter(
            X_pca[:, pc1], X_pca[:, pc2],
            c=L, s=0.5, alpha=0.3, cmap="tab10"
        )
        axes[i].set_xlabel(f"PC{pc1+1} ({pca.explained_variance_ratio_[pc1]:.1%})")
        axes[i].set_ylabel(f"PC{pc2+1} ({pca.explained_variance_ratio_[pc2]:.1%})")
        axes[i].set_title(f"PCs {pc1+1}-{pc2+1}")

    plt.colorbar(scatter, ax=axes[-1], label="Component")
    fig.suptitle(f"PCA Projections - {layer_name}")
    fig.tight_layout()
    return fig


def plot_pca_by_position(
    activations: np.ndarray,
    labels: np.ndarray,
    layer_name: str = "",
    selected_positions: list[int] | None = None,
    figsize: tuple = (15, 10),
) -> plt.Figure:
    """Plot PC1-PC2 projections at different context positions."""
    n_samples, seq_len, d_model = activations.shape

    if selected_positions is None:
        selected_positions = [1, 2, 4, 8, 12, 16]
    selected_positions = [p for p in selected_positions if p < seq_len]

    n_pos = len(selected_positions)
    fig, axes = plt.subplots(1, n_pos, figsize=(4 * n_pos, 4))
    if n_pos == 1:
        axes = [axes]

    # Fit PCA on all positions combined
    X_all = activations[:, 1:, :].reshape(-1, d_model)
    pca = PCA(n_components=4)
    pca.fit(X_all)

    for i, pos in enumerate(selected_positions):
        X = activations[:, pos, :]
        X_pca = pca.transform(X)

        axes[i].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, s=1, alpha=0.3, cmap="tab10")
        axes[i].set_title(f"Position {pos}")
        axes[i].set_xlabel(f"PC1")
        axes[i].set_ylabel(f"PC2")

    fig.suptitle(f"Geometry by Context Position - {layer_name}")
    fig.tight_layout()
    return fig


def plot_layer_progression(
    all_activations: dict[str, np.ndarray],
    labels: np.ndarray,
    figsize: tuple = (15, 8),
) -> plt.Figure:
    """Show how geometry evolves across layers."""
    layer_names = sorted(
        [k for k in all_activations if "resid_post" in k],
        key=lambda x: int(x.split(".")[1])
    )

    if "hook_embed" in all_activations:
        layer_names = ["hook_embed"] + layer_names

    n_layers = len(layer_names)
    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 4))
    if n_layers == 1:
        axes = [axes]

    for i, name in enumerate(layer_names):
        acts = all_activations[name]
        n_samples, seq_len, d_model = acts.shape

        # Use positions 1: (skip BOS)
        X = acts[:, 1:, :].reshape(-1, d_model)
        L = np.repeat(labels, seq_len - 1)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        axes[i].scatter(X_pca[:, 0], X_pca[:, 1], c=L, s=0.5, alpha=0.2, cmap="tab10")
        short_name = name.replace("blocks.", "L").replace(".hook_resid_post", "")
        axes[i].set_title(short_name)
        axes[i].set_xlabel("PC1")
        axes[i].set_ylabel("PC2")

    fig.suptitle("Layer Progression: Residual Stream Geometry")
    fig.tight_layout()
    return fig


def plot_training_loss(metrics: list[dict], figsize: tuple = (8, 4)) -> plt.Figure:
    """Plot training loss curve."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    steps = [m["step"] for m in metrics]
    losses = [m["loss"] for m in metrics]
    ax.plot(steps, losses)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Training Loss")
    ax.set_yscale("log")
    fig.tight_layout()
    return fig


def plot_regression_results(
    activations: np.ndarray,
    raw_tokens: np.ndarray,
    components: list[tuple[float, float]],
    labels: np.ndarray,
    regression_results: dict,
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """Plot predicted vs true belief geometry for each component."""
    K = len(components)
    fig, axes = plt.subplots(2, K, figsize=figsize)

    for k, (alpha_param, x_param) in enumerate(components):
        mask = labels == k
        if mask.sum() == 0:
            continue

        comp_tokens = raw_tokens[mask]
        beliefs = compute_belief_states(comp_tokens, alpha_param, x_param)

        # True beliefs (last position)
        true_beliefs = beliefs[:, -1, :]
        plot_belief_simplex(true_beliefs, title=f"True C{k}", ax=axes[0, k], alpha=0.3, s=1)

        # Predicted beliefs via regression
        reg_key = f"component_{k}"
        if reg_key in regression_results:
            reg = regression_results[reg_key]["regressor"]
            comp_acts = activations[mask, -1, :]  # last position (last activation before unembedding)
            pred_beliefs = reg.predict(comp_acts)
            # Clip to simplex
            pred_beliefs = np.clip(pred_beliefs, 0, 1)
            pred_beliefs /= pred_beliefs.sum(axis=1, keepdims=True)
            r2 = regression_results[reg_key]["r2"]
            plot_belief_simplex(
                pred_beliefs,
                title=f"Predicted C{k} (R2={r2:.3f})",
                ax=axes[1, k], alpha=0.3, s=1,
            )

    fig.suptitle("Belief State Regression: True vs Predicted")
    fig.tight_layout()
    return fig
