# Datasets Design: Non-Ergodic Mess3 Mixture

## Overview
Construct a non-ergodic training dataset where each sequence comes from one Mess3 ergodic component (one specific parameterization). The model must learn to:
1. Identify which Mess3 component generated the current sequence (in-context)
2. Track belief state within that component (Bayesian filtering)

## Mess3 Parameterizations

### Parameter Space
- **alpha** in [0, 1]: emission clarity (higher = more deterministic emissions)
- **x** in (0, 0.5]: state transition noise (lower = more persistent states)
- Derived: beta = (1 - alpha) / 2, y = 1 - 2x

### Chosen Components
We use K ergodic components with distinct parameters to ensure clearly different dynamics:

| Component | alpha | x    | Description |
|-----------|-------|------|-------------|
| C0        | 0.90  | 0.05 | High clarity, high persistence (sharp gasket) |
| C1        | 0.60  | 0.15 | Medium clarity, medium persistence (diffuse gasket) |
| C2        | 0.85  | 0.30 | High clarity, low persistence (fast mixing) |
| C3        | 0.50  | 0.10 | Low clarity, high persistence (noisy but structured) |

**Rationale**: Varying both alpha and x gives distinct spectral properties:
- zeta = 1 - 3x: C0=0.85, C1=0.55, C2=0.10, C3=0.70
- Different zeta means different belief convergence rates

### Alternative: Fewer Components
For simpler analysis, start with K=2:
- C0: alpha=0.85, x=0.05 (standard Mess3)
- C1: alpha=0.60, x=0.15 (diffuse Mess3)

## Data Generation

### Sequence Generation
```python
def generate_sequence(alpha, x, length):
    """Generate one sequence from Mess3(alpha, x)."""
    beta = (1 - alpha) / 2
    y = 1 - 2 * x

    # Build transition matrices T^(z) for z in {0, 1, 2}
    # Sample from stationary distribution pi = [1/3, 1/3, 1/3]
    # Emit tokens sequentially using T^(z) matrices
    ...
```

### Dataset Construction
```python
def build_dataset(components, n_sequences_per_component, seq_length):
    """
    Each sequence is labeled by component but label is NOT given to model.
    Model sees only token sequences.
    """
    sequences = []
    labels = []  # for analysis only, not training
    for k, (alpha, x) in enumerate(components):
        for _ in range(n_sequences_per_component):
            seq = generate_sequence(alpha, x, seq_length)
            sequences.append(seq)
            labels.append(k)
    return sequences, labels
```

### Hyperparameters
- **Sequence length**: 16 tokens (recommended 8-16 in work test)
- **Sequences per component**: ~10,000-50,000
- **Total dataset**: K * n_per_component sequences
- **Vocabulary**: {0, 1, 2} (3 tokens) + BOS token = 4 tokens
- **Shuffle**: sequences are shuffled (component identity not given)

## Ground Truth Computation

### Belief States
For each sequence and each component k, compute ground-truth belief:
```
eta_k^(z_{1:t}) = (pi * T_k^(z_1) * ... * T_k^(z_t)) / norm
```
where T_k are the transition matrices for component k.

### Meta-Belief (Process Identity)
After observing z_{1:t}, the posterior over components:
```
P(component=k | z_{1:t}) = P(z_{1:t} | component=k) * P(component=k) / P(z_{1:t})
```
where P(z_{1:t} | component=k) = pi_k * T_k^(z_1) * ... * T_k^(z_t) * 1

This is crucial for understanding what the transformer should represent.

## Notes
- Non-ergodicity: statistics differ across sequences (different components)
- Within each sequence: ergodic (single stationary process)
- The "context window" must be long enough for the model to distinguish components
- Key diagnostic: does the model learn to identify the component from early tokens?
