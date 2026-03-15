# Experiment State

## Current Phase: Running on DGX

## Status
- [x] Literature review complete
- [x] Mess3 data generation implemented & tested
- [x] TransformerLens model creation tested
- [x] Analysis pipeline (PCA, belief regression, sync dynamics)
- [x] Visualization code
- [x] LaTeX writeup structure (ACL format)
- [x] Mathematical predictions derived
- [x] v2: on-the-fly data, entropy rate computation
- [x] Sync analysis code (additional analysis for item 4)
- [ ] Phase 1: Single Mess3 running (10K steps, ~55 min)
- [ ] Phase 2: Non-ergodic 3-comp running (20K steps, ~110 min)
- [ ] Fill results in paper
- [ ] Compile PDF
- [ ] Submit

## Current Run
- DGX 7 (100.70.143.80)
- Phase 1: single Mess3 (0.85, 0.05), 3L/64d, 10K steps
- Phase 2: 3-comp non-ergodic, 4L/128d, 20K steps
- v1 results: loss barely dropped (1.014), R2 was 0.76-0.90 but geometry was flat
- v2 fix: on-the-fly data, entropy rate validation, better component selection

## Key Decisions
- 3 components: (0.95,0.03), (0.85,0.05), (0.65,0.15) — well-separated entropy rates
- Additional analysis: synchronization dynamics (meta-belief vs within-belief by position and layer)
- Mathematical prediction: K+1 dimensional factored representation
