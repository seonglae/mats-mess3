# Literature Review: Mess3, Belief State Geometry & Factored Representations

## Core Papers

### 1. Transformers Represent Belief State Geometry in Their Residual Stream
- **Authors**: Adam Shai, Sarah Marzen, Lucas Teixeira, Alexander Oldenziel, Paul Riechers
- **Venue**: NeurIPS 2024
- **ArXiv**: 2405.15943
- **Key findings**:
  - Belief states are **linearly represented** in residual stream (R^2 ~ 0.95)
  - Layer-by-layer construction: embedding -> triangle -> fractal (Sierpinski gasket)
  - Transformers learn more than next-token: they encode full predictive information
  - Tested on Mess3 and RRXOR processes

### 2. Transformers Learn Factored Representations
- **Authors**: Adam Shai, Loren Amdahl-Culleton, Casper Christensen, Henry Bigelow, Fernando Rosas, Alexander Boyd, Eric Alt, Kyle Ray, Paul Riechers
- **ArXiv**: 2602.02385 (Feb 2026)
- **Key findings**:
  - Transformers factor world into parts in **orthogonal subspaces**
  - Factored representation: sum_n(d_n - 1) dimensions vs joint: prod(d_n) - 1
  - Inductive bias toward factoring even when lossy (hurts prediction)
  - Tested with 3 Mess3 + 2 Bloch Walk factors, vocab size 433
  - Architecture: GPT-2 style, 4 layers, d_model=120, d_MLP=480, context L=8
  - Code: https://github.com/Astera-org/factored-reps

### 3. Constrained Belief Updates Explain Geometric Structures in Transformer Representations
- **Authors**: Maciej Piotrowski, Paul Riechers, Daniel Filan, Adam Shai
- **ArXiv**: 2502.01954 (Feb 2025)
- **Key findings**:
  - Transformers implement constrained approximation to Bayesian filtering
  - r_1(z_{1:d}) = pi + sum_{s=1}^{d} (pi * T^|z_s * T^{d-s} - pi)
  - Each correction decays exponentially with distance (rate zeta = 1 - 3x)
  - This is the best achievable given parallel nature of attention

### 4. Spectral Simplicity of Apparent Complexity I
- **Authors**: Paul Riechers, James Crutchfield
- **Venue**: Chaos 2018
- **Key**: Defines Mess3 process, shows 3-state generator produces infinite causal states forming Sierpinski gasket

### 5. Neural Networks Leverage Nominally Quantum and Post-Quantum Representations
- **Authors**: Paul Riechers, Thomas Elliott, Adam Shai
- **ArXiv**: 2507.07432 (2025)
- **Key**: Bloch Walk process definition, quantum GHMM representations

---

## The Mess3 Process

### Definition
- 3-state nonunifilar HMM, alphabet {0, 1, 2}
- Parameters: alpha (emission clarity), x (state persistence)
  - beta = (1 - alpha) / 2
  - y = 1 - 2x
- Standard params: alpha=0.85, x=0.05

### Transition Matrices
```
T^(0) = | alpha*y   beta*x    beta*x  |
         | alpha*x   beta*y    beta*x  |
         | alpha*x   beta*x    beta*y  |

T^(1) = | beta*y    alpha*x   beta*x  |
         | beta*x    alpha*y   beta*x  |
         | beta*x    alpha*x   beta*y  |

T^(2) = | beta*y    beta*x    alpha*x |
         | beta*x    beta*y    alpha*x |
         | beta*x    beta*x    alpha*y |
```

### Key Properties
- **Nonunifilar**: observing token doesn't uniquely determine state transition
- **Infinite Markov order**: past of unbounded length influences future
- **Causal states form Sierpinski gasket**: fractal in 2-simplex
- Net transition matrix eigenvalue: zeta = 1 - 3x (controls belief correction decay)
- Stationary distribution: uniform pi = [1/3, 1/3, 1/3]

### Belief Update
Given belief eta and observation z:
```
eta' = (eta * T^(z)) / (eta * T^(z) * 1)
```

---

## Non-Ergodic Mixture (The Work Test Setup)

### What the work test asks
- Multiple Mess3 processes with **different parameters (alpha, x)**
- Each training sequence is from **one** ergodic component
- Model must infer: (a) which parameterization, AND (b) current hidden state
- This is a **meta-synchronization** / **in-context learning** task

### Why relevant to language models
- Non-ergodicity is "at the heart of in-context learning"
- Early tokens establish which "genre/style/process" is active
- Structurally identical to how LLMs adapt in-context
- Different parameterizations = different topics/styles/domains in natural language

---

## Geometry Predictions for Non-Ergodic Mess3

### Per-component geometry
- Each Mess3(alpha, x) produces its own Sierpinski gasket in the 2-simplex
- Different params -> different fractal structures (varying contraction rates)
- alpha controls emission clarity -> gasket density/spread
- x controls state persistence -> gasket detail/resolution

### Expected transformer representation
For K different Mess3 parameterizations:
1. **Factored hypothesis**: The model separates "which process" from "belief within process"
   - Process identity: (K-1)-dimensional subspace (simplex over K components)
   - Within-process belief: 2-dimensional subspace (2-simplex for 3-state HMM)
   - Total: K+1 dimensions
2. **Joint/product hypothesis**: K separate 2D gaskets, possibly in shared or overlapping subspaces
3. **Superimposed hypothesis**: All gaskets mapped into same 2D subspace, distinguished by process-identity encoding

### Context position effects
- Position 1: token embedding only (3 points)
- Early positions: triangle fills in, belief uncertainty high
- Later positions: fractal structure emerges as belief narrows
- Decay rate per component: zeta_k = 1 - 3*x_k

### Layer effects
- Embedding: 3 vertices of simplex
- Layer 1 attention: spreads to filled triangle
- MLP layers: progressive fractal refinement
- Final layer: full Sierpinski gasket structure

---

## Related Code Repositories
- **Official (factored reps)**: https://github.com/Astera-org/factored-reps
- **Community (belief state)**: https://github.com/sanowl/BeliefStateTransformer
- **Rigorous (markov-transformers)**: https://github.com/lena-lenkeit/markov-transformers
- **Microsoft BST**: https://github.com/microsoft/BST
- **TransformerLens**: https://github.com/TransformerLensOrg/TransformerLens
