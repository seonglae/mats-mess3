# Experiment State

## Current Phase: POC

## Status
- [x] Literature review complete
- [x] Mess3 data generation implemented & tested
- [x] TransformerLens model creation tested
- [x] Activation extraction tested
- [x] Analysis pipeline (PCA, belief regression) implemented
- [x] Visualization code implemented
- [ ] Full training run on DGX
- [ ] Post-training analysis
- [ ] Additional analysis (item 4 of work test)
- [ ] PDF writeup

## Latest Run
None yet - need to deploy to DGX and run.

## Key Decisions
- 4 Mess3 components: (0.90,0.05), (0.60,0.15), (0.85,0.30), (0.50,0.10)
- Model: 3 layers, d_model=64, 4 heads, context=17 (16+BOS)
- Vocab: 4 tokens (BOS + {0,1,2})
- 20K sequences per component, 200 epochs
