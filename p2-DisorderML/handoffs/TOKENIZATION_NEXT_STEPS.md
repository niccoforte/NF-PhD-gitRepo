# Tokenisation continuation

The maintained notebook remains in p2-DisorderML/code/; only this brief moved to handoffs/.

## Purpose and authority

Find recurring local **disorder patterns** associated with high/low UT and C(T)
performance. This is motif discovery, not the linear per-node tokenizer inside
the predictive Transformer. Keep it separate from the accuracy experiments.
The user has an existing chat named **Tokenisation**; this file is its handoff,
not an instruction to create another chat or dispatch work automatically.

Current scope: existing FEA specimens only. No design search, insertion/removal
FEA, CNN training, damage extraction or new HPC submissions are authorised by
this brief. Those require a subsequent scoped request.

Scientific sources (Obsidian vault, read-only review on 4 October):

- `Writing/p2/p2 - Backbone.md`, optional exploratory disorder-pattern analysis.
- `Writing/Research Plans/p2 - High-Performance Disorder Pattern Identification.md`,
  especially local-patch recipes. These retain competing exploratory methods.
- `Tasks/Tasks/Tokenization.md`, the existing task note.
- `Writing/p2/p2 - Project Status.md`, core paper versus conditional motif scope.

Keep the complete 722-designable-node FCC design as the core paper. Token-based
design remains conditional on held-out stability and eventual FEA intervention.
Do not edit the vault's task completion states based on this handoff.

## Actual prototype, not the older plan

`resources/tokenization.py` contains `OutputInformedTokenizer`; its embedder
weights eight patch features by absolute feature/score correlation, then runs
**PCA**, followed by KMeans. It does **not** implement PLS or VQ-VAE. PLS is a
candidate comparison. The older plan named a nonexistent
`code/tokenization_experiment.py`; do not assume that script exists.

Current patch features are central dx/dy, magnitude and angle, neighbour mean
dx/dy, and neighbour magnitude mean/std. Here dx/dy must mean **initial geometric
disorder**, not Abaqus response U1/U2. Existing helpers use a k-nearest-neighbour
graph and broadcast each specimen's global score to its patches. This is weak
supervision, not an identified contribution of that patch to performance.

Important issues to resolve in the dedicated task before reporting results:

- Use actual reference FCC connectivity (`MLfield.reference_field_edges`) for
  strut neighbourhoods, never distance thresholds on disordered coordinates.
  Keep UT and FT maps/cuts explicit; nearest-neighbour patches are a distinct
  geometric baseline, not actual load-path connectivity.
- `prepare_xy_from_data_object` can choose FT inputs while retaining UT base
  coordinates. Verify node counts and index mapping before using it.
- `_normalized_score` currently takes a positional first row as reference.
  Select periodic specimen **ID 0** explicitly and join properties by ID.
- Split by **specimen before patch expansion**. Fit scaling, correlation/PLS/PCA
  and KMeans on training specimens only. No patch from one specimen may leak
  across train and validation, and the locked test remains untouched.
- Save the fitted embedder/codebook and ID/split/config provenance, not merely
  assigned token IDs. Validate applying the exact fitted vocabulary to new data.
- Raw angle has a wrap discontinuity. Consider sin/cos or directional offsets;
  rotation/reflection invariance is NOT automatic under directional UT/FT loading.

## Relationship to the implemented local graph experiment

`MLdual.LocalFieldGraph` is an opt-in predictive block. One shared message MLP
reads endpoint embeddings plus initial-disordered dx_ij, dy_ij and strut length,
averages incoming messages by degree, and adds them to node representations
before global attention. It uses task-specific initial graphs: UT2319/FT2259
edges. It neither clusters motifs nor learns a discrete codebook; no CNN exists.
Its first matched single-seed run did not improve the joint validation score.
That does not decide whether graph embeddings are useful for motif discovery.
The completed comparison was baseline score .863022 versus local_graph .902120
(lower is better): FT-field RMSE improved about 1%, but UT-field/curve errors
worsened, especially FT curves. These predictive results do not invalidate the
separate motif-discovery hypothesis and must not be presented as motif evidence.

Potential comparisons for the Tokenisation task, not simultaneous commitments:

1. Transparent one-hop disorder descriptors with correlation-PCA/KMeans baseline.
2. PLS embeddings, with the same patches and splits, to test supervised compression.
3. One-/two-hop graph encoder embeddings and a codebook, preserving directional
   geometry and four/eight-neighbour roles without adding edge tokens.
4. A masked reference-grid CNN only if its patch geometry and FT crack treatment
   can be made explicit. No need to force an irregular nodal graph into an image.

Response-informed embeddings could also use observed displacement histories or
local increments. Label these as **mechanism analysis**: they require a response
and cannot serve as geometry-only descriptors at design time. Do not substitute
predicted responses and call them observed fracture labels. Damage belongs to
its dedicated task.
For response-informed motifs, consider both changes of temporal displacement
increments and departure from locally affine motion. These capture abrupt/local
activity more specifically than absolute U magnitude. Keep these outputs out of
geometry-only design-time inputs, and keep irregular frame spacing/missingness,
noise and smooth nonlinear deformation as confounders. Use observed histories
only for retrospective mechanism analysis until predictive utility is tested.
Thickness/generation regime is an additional possible confounder: use verified
provenance (samples/accuracy-continuation/thickness-provenance.md), not batch-ID
labels as an inferred physical input. Do not declare a high-performing motif
causal if its enrichment only reflects a generation batch.

## First evidence to collect

- Plot representative patches, actual strut connections and disorder arrows in
  physical units. Show where similar motifs occur within the body.
- Compare 16/32/64 tokens and one-/two-hop neighbourhoods, but select settings
  on development data, not the final test set.
- Report held-out enrichment, specimen-level permutation controls and bootstrap
  uncertainty. Resample specimens, not correlated patches independently.
- Distinguish UT-only, FT-only and combined objectives. A scalar composite can
  hide trade-offs; include a Pareto-elite comparison and a strength constraint
  only after its threshold is chosen. Do not silently reuse an arbitrary old score.
- Repeat across seeds; align codebooks geometrically because integer token IDs
  can permute. Compare motif stability, not raw token-number agreement.
- Uniform usage alone is not failure. Failure is absent reproducible held-out
  association or an association explained by location/global disorder.
- Enrichment is association, not causality. Controlled motif insertion/removal
  with density/feasibility checks and fresh FEA is a later, separately authorised
  confirmation step.

Use `p2-DisorderML/code/Tokenization.ipynb` as the existing notebook surface.
Human examples belong in `samples/`, run artifacts in `data/`; no duplicate
notebooks. Ask the user to settle the first score/elite definition before a
large study. No tokenisation code was changed by this reconciliation.
