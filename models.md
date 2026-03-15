# Model Design: Transformer for Non-Ergodic Mess3

## Architecture

### Decoder-Only Transformer (GPT-2 style)
Following the paper's recommendations for small models:

| Parameter | Value | Notes |
|-----------|-------|-------|
| n_layers | 2-3 | Paper used 4 for 5-factor; we have simpler task |
| d_model | 32-64 | Small vocab (4 tokens), simple structure |
| d_mlp | 4 * d_model | Standard ratio |
| n_heads | 2-4 | |
| context_length | 16 | Recommended 8-16 |
| vocab_size | 4 | {BOS, 0, 1, 2} |
| positional_encoding | learned | Following the paper |

### Implementation
- Use TransformerLens for interpretability hooks
- Pre-norm architecture (layer norm before attention/MLP)
- Causal attention mask
- No weight decay (following the paper)

## Training

### Objective
Standard next-token prediction with cross-entropy loss.

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| optimizer | Adam |
| learning_rate | 5e-4 |
| batch_size | 4096-25000 |
| weight_decay | 0 |
| training_steps | ~5000-50000 |

### Data Pipeline
- Generate sequences on-the-fly or pre-generate
- Each batch: random mix of all components
- BOS token at start of each sequence

## Analysis Plan

### 1. Residual Stream Geometry (Primary)
- Extract activations at each layer using TransformerLens hooks
- PCA on activations: how many dimensions used?
- Cumulative Explained Variance (CEV) over training
- Compare with predictions: factored vs joint

### 2. Belief State Regression
- Linear regression: activations -> ground-truth belief states
- Per-component: do activations encode belief for each Mess3 component?
- Meta-belief: do activations encode posterior over component identity?
- Track RMSE and R^2 over training

### 3. Geometry Visualization
- Project activations to top PCs
- Color by: component identity, belief state, context position
- Look for: Sierpinski gaskets, simplexes, separation by component

### 4. Layer-by-Layer Analysis
- How does geometry change from embedding to final layer?
- Where does component identification emerge?
- Where does within-component belief structure emerge?

### 5. Context Position Analysis
- How does geometry change with context position?
- Early positions: more uncertainty, less structure
- Later positions: sharper belief, more fractal detail
- Decay rate comparison across components

## Key Predictions (Pre-Registration)

### Mathematical Prediction
The non-ergodic mixture has a hierarchical structure:
1. **Top level**: K components -> (K-1)-simplex for process identity
2. **Bottom level**: 3 hidden states per component -> 2-simplex per component

If the transformer learns a factored representation:
- d_factored = (K-1) + 2 = K+1 dimensions
- Component identity in one subspace
- Shared belief geometry in another subspace

If the transformer learns a joint representation:
- d_joint = 3K - 1 dimensions (simplex over all 3K joint states)

### Intuitive Prediction
- The model will learn to first identify the component (early layers)
- Then refine belief within that component (later layers)
- Geometry: K clusters (one per component), each containing a gasket
- The gaskets may share orientation/subspace or be in separate subspaces

### Multiple Possible Geometries
1. **Factored**: Orthogonal subspaces for identity and belief
2. **Clustered**: K distinct gaskets in different regions of same 2D space
3. **Superimposed**: Single gasket with component identity encoded in orthogonal dimension
4. **Hierarchical**: Component identity emerges first, belief conditioned on it

## Dependencies
- transformer_lens
- torch
- numpy
- matplotlib
- scikit-learn (for PCA, linear regression)
