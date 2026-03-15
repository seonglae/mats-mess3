# Roadmap

## Deadline: 2026-03-15 23:59 AoE

## Tasks

### Phase 1: Implementation (DONE)
- [x] Mess3 process (data generation, belief computation)
- [x] Non-ergodic dataset (multiple components, shuffled)
- [x] Transformer model (TransformerLens, small)
- [x] Training loop with periodic PCA tracking
- [x] Analysis: PCA, CEV, belief regression, meta-belief
- [x] Visualization: gaskets, PCA projections, layer progression

### Phase 2: Run Experiment
- [ ] Deploy to DGX 7
- [ ] Full training run (~200 epochs)
- [ ] Extract & save all analysis results

### Phase 3: Analysis & Writeup
- [ ] Analyze residual stream geometry
- [ ] Belief regression R^2 per component per layer
- [ ] CEV dimensionality over training
- [ ] Additional analysis (choose one):
  - Option A: Attention pattern analysis (what does attention attend to?)
  - Option B: Probing for component identity vs belief state (when does each emerge?)
  - Option C: Comparison: 2 vs 4 components (scaling of representations)
- [ ] Write PDF (ACL template, concise)
- [ ] Submit via form

### Experiment Queue
| Experiment | Config | Status |
|-----------|--------|--------|
| main-4comp | 4 components, 3L/64d, 200ep | pending |
