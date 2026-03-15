"""Mess3 process: 3-state nonunifilar HMM with Sierpinski gasket belief geometry."""

import numpy as np
from jaxtyping import Float, Int
from numpy import ndarray


def mess3_transition_matrices(
    alpha: float = 0.85, x: float = 0.05
) -> Float[ndarray, "3 3 3"]:
    """Build the 3 labeled transition matrices T^(z) for z in {0, 1, 2}.

    T^(z)_{i,j} = P(emit z, transition to state j | state i)

    Args:
        alpha: Emission clarity parameter in [0, 1].
        x: State transition noise in (0, 0.5].

    Returns:
        Array of shape (3, 3, 3) where T[z] is the transition matrix for token z.
    """
    beta = (1 - alpha) / 2
    y = 1 - 2 * x

    T = np.zeros((3, 3, 3))

    # T^(0)
    T[0] = [
        [alpha * y, beta * x, beta * x],
        [alpha * x, beta * y, beta * x],
        [alpha * x, beta * x, beta * y],
    ]

    # T^(1)
    T[1] = [
        [beta * y, alpha * x, beta * x],
        [beta * x, alpha * y, beta * x],
        [beta * x, alpha * x, beta * y],
    ]

    # T^(2)
    T[2] = [
        [beta * y, beta * x, alpha * x],
        [beta * x, beta * y, alpha * x],
        [beta * x, beta * x, alpha * y],
    ]

    return T


def generate_mess3_sequences(
    alpha: float,
    x: float,
    n_sequences: int,
    seq_length: int,
    rng: np.random.Generator | None = None,
) -> tuple[Int[ndarray, "n_seq seq_len"], Int[ndarray, "n_seq seq_len"]]:
    """Generate sequences from a Mess3 process.

    Args:
        alpha: Emission clarity.
        x: State transition noise.
        n_sequences: Number of sequences to generate.
        seq_length: Length of each sequence.
        rng: Random number generator.

    Returns:
        tokens: Array of shape (n_sequences, seq_length) with values in {0, 1, 2}.
        states: Array of shape (n_sequences, seq_length) with hidden states in {0, 1, 2}.
    """
    if rng is None:
        rng = np.random.default_rng()

    T = mess3_transition_matrices(alpha, x)
    # Net transition matrix (for sampling next state + emission jointly)
    # T_net[i] = sum over z of T[z][i] = row-stochastic transition matrix
    T_net = T.sum(axis=0)  # shape (3, 3)

    tokens = np.zeros((n_sequences, seq_length), dtype=np.int64)
    states = np.zeros((n_sequences, seq_length), dtype=np.int64)

    # Start from stationary distribution (uniform for Mess3)
    current_states = rng.integers(0, 3, size=n_sequences)

    for t in range(seq_length):
        states[:, t] = current_states

        # For each sequence, sample (token, next_state) jointly
        # P(z, j | i) = T[z][i, j]
        # Flatten to 9 outcomes per state: (z, j) for z in {0,1,2}, j in {0,1,2}
        for i in range(n_sequences):
            s = current_states[i]
            # Build joint distribution over (token, next_state)
            probs = np.array([T[z, s, :] for z in range(3)]).flatten()  # 9 values
            probs = probs / probs.sum()  # normalize for numerical safety
            idx = rng.choice(9, p=probs)
            z, j = divmod(idx, 3)
            tokens[i, t] = z
            current_states[i] = j

    return tokens, states


def generate_mess3_sequences_fast(
    alpha: float,
    x: float,
    n_sequences: int,
    seq_length: int,
    rng: np.random.Generator | None = None,
) -> tuple[Int[ndarray, "n_seq seq_len"], Int[ndarray, "n_seq seq_len"]]:
    """Vectorized sequence generation from Mess3 process.

    Same interface as generate_mess3_sequences but faster for large batches.
    """
    if rng is None:
        rng = np.random.default_rng()

    T = mess3_transition_matrices(alpha, x)

    tokens = np.zeros((n_sequences, seq_length), dtype=np.int64)
    states = np.zeros((n_sequences, seq_length), dtype=np.int64)

    # Start from uniform stationary distribution
    current_states = rng.integers(0, 3, size=n_sequences)

    # Precompute joint (token, next_state) distributions for each state
    # joint_probs[s] = flattened probability over 9 outcomes (z*3 + j)
    joint_probs = np.zeros((3, 9))
    for s in range(3):
        for z in range(3):
            for j in range(3):
                joint_probs[s, z * 3 + j] = T[z, s, j]

    # Precompute cumulative distributions for fast sampling
    joint_cdf = np.cumsum(joint_probs, axis=1)

    for t in range(seq_length):
        states[:, t] = current_states

        # Sample for all sequences at once
        u = rng.random(n_sequences)

        for s in range(3):
            mask = current_states == s
            if not mask.any():
                continue
            # Use searchsorted on cumulative distribution
            indices = np.searchsorted(joint_cdf[s], u[mask])
            indices = np.clip(indices, 0, 8)
            tokens[mask, t] = indices // 3
            current_states[mask] = indices % 3

    return tokens, states


def compute_belief_states(
    tokens: Int[ndarray, "n_seq seq_len"],
    alpha: float,
    x: float,
) -> Float[ndarray, "n_seq seq_len 3"]:
    """Compute ground-truth belief states for given token sequences.

    The belief state at position t is the posterior distribution over hidden states
    after observing tokens[0:t+1], starting from the uniform prior.

    Args:
        tokens: Token sequences, values in {0, 1, 2}.
        alpha: Mess3 alpha parameter.
        x: Mess3 x parameter.

    Returns:
        beliefs: Array of shape (n_seq, seq_len, 3) where beliefs[i, t] is
            the belief state (probability over 3 hidden states) after observing
            tokens[i, 0:t+1].
    """
    T = mess3_transition_matrices(alpha, x)
    n_seq, seq_len = tokens.shape
    beliefs = np.zeros((n_seq, seq_len, 3))

    # Start from uniform prior (stationary distribution for Mess3)
    eta = np.ones((n_seq, 3)) / 3.0

    for t in range(seq_len):
        z = tokens[:, t]  # (n_seq,)

        # Apply belief update: eta' = (eta @ T^(z)) / norm
        # Need to select the right T matrix for each sequence
        new_eta = np.zeros_like(eta)
        for token_val in range(3):
            mask = z == token_val
            if mask.any():
                new_eta[mask] = eta[mask] @ T[token_val]

        # Normalize
        norms = new_eta.sum(axis=1, keepdims=True)
        new_eta = new_eta / norms

        beliefs[:, t] = new_eta
        eta = new_eta

    return beliefs


def compute_component_log_likelihoods(
    tokens: Int[ndarray, "n_seq seq_len"],
    components: list[tuple[float, float]],
) -> Float[ndarray, "n_seq seq_len n_components"]:
    """Compute cumulative log-likelihoods under each component.

    For computing the posterior over component identity (meta-belief).

    Args:
        tokens: Token sequences.
        components: List of (alpha, x) parameter tuples.

    Returns:
        log_likelihoods: Cumulative log P(z_{1:t} | component k) for each
            sequence, position, and component.
    """
    n_seq, seq_len = tokens.shape
    K = len(components)
    log_liks = np.zeros((n_seq, seq_len, K))

    for k, (alpha, x_param) in enumerate(components):
        T = mess3_transition_matrices(alpha, x_param)
        # Start from uniform prior
        eta = np.ones((n_seq, 3)) / 3.0
        cum_log_lik = np.zeros(n_seq)

        for t in range(seq_len):
            z = tokens[:, t]
            new_eta = np.zeros_like(eta)
            for token_val in range(3):
                mask = z == token_val
                if mask.any():
                    new_eta[mask] = eta[mask] @ T[token_val]

            # P(z_t | z_{1:t-1}, component k) = sum of new_eta (before norm)
            token_prob = new_eta.sum(axis=1)
            cum_log_lik += np.log(token_prob + 1e-30)
            log_liks[:, t, k] = cum_log_lik

            # Normalize for next step
            eta = new_eta / new_eta.sum(axis=1, keepdims=True)

    return log_liks


def compute_meta_beliefs(
    tokens: Int[ndarray, "n_seq seq_len"],
    components: list[tuple[float, float]],
    prior: Float[ndarray, "n_components"] | None = None,
) -> Float[ndarray, "n_seq seq_len n_components"]:
    """Compute posterior over component identity at each position.

    P(component k | z_{1:t}) proportional to P(z_{1:t} | component k) * P(component k)

    Args:
        tokens: Token sequences.
        components: List of (alpha, x) tuples.
        prior: Prior over components (uniform if None).

    Returns:
        meta_beliefs: Posterior over components at each position.
    """
    log_liks = compute_component_log_likelihoods(tokens, components)
    K = len(components)

    if prior is None:
        log_prior = np.zeros(K)
    else:
        log_prior = np.log(prior)

    # log P(k | z_{1:t}) = log P(z_{1:t} | k) + log P(k) - log P(z_{1:t})
    log_posterior = log_liks + log_prior[None, None, :]

    # Softmax for numerical stability
    log_posterior -= log_posterior.max(axis=2, keepdims=True)
    posterior = np.exp(log_posterior)
    posterior /= posterior.sum(axis=2, keepdims=True)

    return posterior
