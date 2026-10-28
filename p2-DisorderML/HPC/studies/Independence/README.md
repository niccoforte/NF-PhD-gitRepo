# Independence study

`A0-Independence.py` selects the existing eight-case implementation;
`B4_Independence.sh SUITE [--submit]` handles banks, gates and dependencies.
Case1(i) means four separately trained models with no shared parameters or
curve-to-field gradients. Case numbering and completed results remain in
`../../DualOutputs/README.md` and `samples/independence-review/README.md`.

The completed three-seed suite is not outstanding work. Reuse its compatible
banks rather than paying for new controls. `resources/MLcrossfit.py` contains
importable cross-fitting/group logic; it is not the submission script.

`review_sharing_study.py` refreshes the completed comparison from verified
per-run metrics/provenance and saved validation arrays. It adds actual STATUS
regions, TRAIN-mean comparisons, large four-column field/curve galleries and
stage-correct training histories/costs. Independent field histories come from
final field generators, not constant field scores recorded during curve fits.
No model is retrained; original predicted-only curve-training scope stays clear.
