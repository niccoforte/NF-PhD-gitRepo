# Nodal feature contribution study

`A0-FeatureAblation.py` / `B4_FeatureAblation.sh` implement the authorised
field-only attribution screen on the confirmed CNN2/MSE architecture, independently
for UT and FT. `PROJECT_STATUS.md` records actual submissions, not this protocol.

| Task | Runs, each seed42 |
| --- | --- |
| UT | Full-feature control; suppress reference xy, designable, interface pair, raw present, all context |
| FT | Full-feature control; suppress xy, designable, pin pair, all tip channels, tip offsets only, tip distance only, raw present, all context |

Fifteen final fits; each retains original disorder,800 slots,11 context columns,
384/8/3 Transformer, two occupancy-masked CNN blocks, physical TRAIN-only
normalization, fixed7137/793/881 IDs, masked displacement MSE, AdamW/batch2,
450 epochs/patience52 and unchanged physical-MSE selection. No curve training,
OOF banks, damage auxiliary objective or locked-test use. Group suppression
tests conditional value, not unique physical information: coordinates/tip
features and positional encoding are correlated. The CNN reference grid still
encodes spatial arrangement even when explicit coordinate channels are removed.

`MLfeature_ablation` applies a saved per-task boolean context mask immediately
before context projection. It acts after each specimen's pin flags are refreshed,
so zeroing reference pin flags alone cannot accidentally undo the ablation.
Full context width and common initialized weights are preserved. The raw
`present` channel is separate from operational attention/loss/normalization
masks; all FT absence masks remain active in every arm. No zero inactive-task
columns receive pointless separate fits. No-context retains disorder, node
position encoding, task vector and CNN grid, so it is not geometry-free.

Default old descriptors omit `context_keep` and retain identical behavior/state
keys. Independent task configs save the same two-task mask mapping, preserving
FT-only suppression through container reload without applying it to UT or curves.
`test_feature_ablation.py` checks pin refresh, gradients, absent-node invariance,
common parameters, historical defaults and actual field checkpoint reload.

```bash
bash p2-DisorderML/HPC/studies/Features/B4_FeatureAblation.sh nodal-features-261102
# Append --submit only for authorised submission.
```

The named launcher previews by default, then uses the shared final-field shell
helper for a clean immutable source, disposable15-case GPU gate and afterok
production jobs (240h/1GPU/12CPU/90000MB). Full-data identities are checked before
tiny cases; successful preflight retains only its home log, with no archive.
Outputs use `MULTI/Dual/Transformer/<suite>-<variant>-<task>-s42` and explicit
`study=nodal_feature_ablation`; a Dual storage container does not imply sharing.

Canonical audit/report: `samples/input-feature-audit/README.md`; raw results
external Z:/p2. Review global/control/TRAIN mean, collapse, actual STATUS
onset endpoints and signed strut motion, FT tip box, examples, costs and log
histories. Fresh full-feature controls bridge the earlier CNN2 results. Confirm
meaningful removals across seeds before changing the accepted feature contract;
curve-stage feature attribution remains a distinct later study.
