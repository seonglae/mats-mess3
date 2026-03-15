"""Bayes-optimal loss computation.

Computes the theoretical minimum cross-entropy loss achievable by a perfect
Bayesian observer at each context position. This is essential for:
1. Knowing the lower bound on loss
2. Understanding how much room the model has to improve
3. Measuring the "synchronization gap" (how far from optimal)
"""

import numpy as np
from .mess3 import mess3_transition_matrices, generate_mess3_sequences_fast


def bayes_optimal_loss_single(
    alpha: float, x: float, seq_length: int, n_samples: int = 10000, seed: int = 42
) -> np.ndarray:
    """Compute Bayes-optimal cross-entropy at each position for a single Mess3.

    The Bayes-optimal predictor knows the process parameters and maintains
    exact belief states. Its cross-entropy at position t is:
        CE(t) = E[-log P(z_t | z_{1:t-1})]
    where the expectation is over sequences.

    Returns:
        per_position_ce: Array of shape (seq_length,) with CE at each position.
    """
    rng = np.random.default_rng(seed)
    T = mess3_transition_matrices(alpha, x)

    tokens, _ = generate_mess3_sequences_fast(alpha, x, n_samples, seq_length, rng)

    per_pos_ce = np.zeros(seq_length)

    # Start from uniform prior
    eta = np.ones((n_samples, 3)) / 3.0

    for t in range(seq_length):
        z = tokens[:, t]

        # P(z_t | z_{1:t-1}) = eta @ T^(z_t) @ 1
        # For each sample, compute the probability of the actual token
        log_probs = np.zeros(n_samples)
        new_eta = np.zeros_like(eta)

        for token_val in range(3):
            mask = z == token_val
            if mask.any():
                # eta @ T^(z) gives unnormalized next belief
                updated = eta[mask] @ T[token_val]
                # P(z) = sum of updated (before normalization)
                p_z = updated.sum(axis=1)
                log_probs[mask] = np.log(p_z + 1e-30)
                new_eta[mask] = updated / p_z[:, None]

        per_pos_ce[t] = -log_probs.mean()
        eta = new_eta

    return per_pos_ce


def bayes_optimal_loss_mixture(
    components: list[tuple[float, float]],
    seq_length: int,
    n_samples_per_comp: int = 5000,
    seed: int = 42,
) -> dict:
    """Compute Bayes-optimal CE for the non-ergodic mixture.

    Two levels of optimality:
    1. Oracle: knows which component generated the sequence
    2. Bayesian: must infer the component from observations

    Returns dict with:
        'oracle_per_pos': CE if you know the component (array of shape seq_length)
        'bayesian_per_pos': CE with meta-inference (array of shape seq_length)
        'per_component': dict of per-component CE arrays
    """
    rng = np.random.default_rng(seed)
    K = len(components)

    # Per-component optimal CE
    per_component = {}
    for k, (alpha, x_param) in enumerate(components):
        per_component[k] = bayes_optimal_loss_single(alpha, x_param, seq_length, n_samples_per_comp, seed + k)

    # Oracle: average over components (weighted equally)
    oracle_per_pos = np.mean([per_component[k] for k in range(K)], axis=0)

    # Bayesian: must infer component
    # Generate mixed data and compute Bayesian CE
    n_total = n_samples_per_comp * K
    all_tokens = []
    all_labels = []
    for k, (alpha, x_param) in enumerate(components):
        tokens, _ = generate_mess3_sequences_fast(alpha, x_param, n_samples_per_comp, seq_length, rng)
        all_tokens.append(tokens)
        all_labels.append(np.full(n_samples_per_comp, k))

    tokens = np.concatenate(all_tokens, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # For each sequence, maintain beliefs under ALL components
    # and compute the mixture prediction
    transition_matrices = [mess3_transition_matrices(a, x) for a, x in components]
    bayesian_per_pos = np.zeros(seq_length)

    # Belief over components (meta-belief)
    log_component_lik = np.zeros((n_total, K))  # cumulative log-likelihood
    # Belief within each component
    etas = [np.ones((n_total, 3)) / 3.0 for _ in range(K)]

    for t in range(seq_length):
        z = tokens[:, t]

        # For each component, compute P(z_t | z_{1:t-1}, component=k)
        p_z_given_k = np.zeros((n_total, K))
        new_etas = [np.zeros_like(eta) for eta in etas]

        for k in range(K):
            T = transition_matrices[k]
            for token_val in range(3):
                mask = z == token_val
                if mask.any():
                    updated = etas[k][mask] @ T[token_val]
                    p_z = updated.sum(axis=1)
                    p_z_given_k[mask, k] = p_z
                    new_etas[k][mask] = updated / (p_z[:, None] + 1e-30)

        # Bayesian mixture prediction:
        # P(z_t | z_{1:t-1}) = sum_k P(component=k | z_{1:t-1}) * P(z_t | z_{1:t-1}, k)
        # where P(component=k | z_{1:t-1}) propto P(z_{1:t-1} | k) * P(k)
        log_component_posterior = log_component_lik - np.log(K)  # uniform prior
        log_component_posterior -= log_component_posterior.max(axis=1, keepdims=True)
        component_posterior = np.exp(log_component_posterior)
        component_posterior /= component_posterior.sum(axis=1, keepdims=True)

        # Mixture prediction
        p_z_mixture = (component_posterior * p_z_given_k).sum(axis=1)
        bayesian_per_pos[t] = -np.log(p_z_mixture + 1e-30).mean()

        # Update cumulative log-likelihoods
        log_component_lik += np.log(p_z_given_k + 1e-30)
        etas = new_etas

    return {
        "oracle_per_pos": oracle_per_pos,
        "bayesian_per_pos": bayesian_per_pos,
        "per_component": per_component,
        "oracle_avg": float(oracle_per_pos.mean()),
        "bayesian_avg": float(bayesian_per_pos.mean()),
    }
