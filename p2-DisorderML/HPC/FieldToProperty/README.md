# Field-to-property experiment

`A0-HPC_FieldToProperty.py` is the production entry point: full data,
450 maximum epochs, early stopping52 and CUDA by default. The former `-test`
suffix has been removed from the maintained production filename. There is no
separate, duplicated production/trial script.

It predicts UT Strength/Ductility/WoF and FT K_JIC from true displacement
histories, reusing the existing encoder and trainer. Supply `--base-model-json`
for architecture/training settings and `--split-reference` for exact saved
train/validation/test IDs. Labels join by specimen ID, transforms fit on train,
validation selects, and locked-test properties are not scored. No disorder or
physical-context channels are provided. This is an oracle diagnostic, not a
deployable disorder-to-property surrogate.

Launch through `../B1_ML-new.sh` with the HPC-relative script argument
`FieldToProperty/A0-HPC_FieldToProperty.py`. A disposable preflight adds
`--nsims 64 --epochs 1` and a `preflight` run label; full runs omit these debug
overrides and retain the established240h/1GPU/12CPU/90000MB policy.

The completed production run is `field-property-261004` (30367272), distinct
from its preflight30367256. Do not rerun it just because the maintained script
moved. Immutable source snapshots and existing archive metadata remain unchanged.

Results remain `MULTI/FieldToProperty/Transformer/<label>/`: property_model.json,
model.mdl, predictions.npz, metrics.json, property-agreement and log-loss plots.
These are not curve checkpoints. See the existing
[result report](../../samples/accuracy-continuation/field-to-property.md).
