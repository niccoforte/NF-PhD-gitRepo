# Output-Informed Tokenization Plan

## Goal
Build a data-driven vocabulary of recurring disordered-node substructures that are associated with improved combined UT/CT performance.

## Scope (Current Phase)
- Use existing simulated samples only (better/similar/worse than periodic).
- Skip optimization loop and surrogate-driven generation for now.
- Build and validate a first output-informed tokenization pipeline.

## Recommended Baseline
- Start with supervised patch embeddings + discrete codebook (KMeans proxy for VQ).
- Use true node displacement inputs (`dx, dy`) and UT/CT performance metrics.
- Tokenize every sample into node-level discrete token IDs.

## Pipeline
1. Load common UT/CT samples with aligned indices.
2. Build sample score using normalized performance vs periodic baseline.
3. Build node graph from periodic node coordinates.
4. Extract local patch features per node (node + neighborhood statistics).
5. Learn supervised low-dimensional patch embedding (PLS).
6. Fit codebook on embeddings (KMeans) and assign token IDs.
7. Compute diagnostics:
- token usage entropy,
- elite vs non-elite token enrichment,
- MI proxy between token histogram and performance.
8. Save artifacts for downstream interpretation and FEA intervention tests.

## Fail-Fast Criteria
- Near-uniform token usage with no enrichment in elite group.
- Very low MI proxy of tokens with score.
- Unstable token enrichment across random seeds.

If these occur, stop discrete-token direction at this scale and move to node-as-token attention baseline.

## Success Criteria
- Distinct enriched tokens in Pareto-elite set.
- Robust enrichment sign across seeds and data splits.
- FEA intervention on top-enriched motifs causes expected metric shift.

## Immediate Next Experiments
1. Run baseline tokenizer with several token counts (`16, 32, 64`).
2. Compare 1-hop vs 2-hop patch definitions.
3. Repeat with UT-only, CT-only, and combined score targets.
4. Rank candidate motifs and schedule FEA intervention batch.

## Files Added
- `code/resources/tokenization.py`: pipeline implementation.
- `code/tokenization_experiment.py`: executable setup on existing `DATA` object.
