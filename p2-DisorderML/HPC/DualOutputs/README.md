# Joint UT/FT runs

`A0-HPC-Dual-test.py` is the real-data runner. `A0-HPC-Dual-trial1.py` is a small, explicit parameter preset calling that same runner: loading, training, checkpointing and diagnostics are not duplicated. `B1_ML-new.sh` stages the companion runner when launching trial 1.

## Validated smoke run

On 7 September 2026, Apocrita job **25868425** completed with exit code 0 through the home → scratch → archive workflow. It used 64 selected pairs, three epochs, batch size 2 and validation diagnostics. Scratch was cleaned after successful archive collection. See [the human-readable report](../../samples/hpc-test-report.md).

```bash
# Submit from the home-side launch directory, with REPO_ROOT pointing to the code.
sbatch --time=00:30:00 -J dual-MULTI-smoke B1_ML-new.sh DualOutputs/A0-HPC-Dual-test.py --nsims 64 --epochs 3 --batch 2 --no-range-split
sbatch -J dual-MULTI-trial1 B1_ML-new.sh DualOutputs/A0-HPC-Dual-trial1.py
```

Use a unique job/run label. Both runners collect the same result tree under `MULTI/Dual/Transformer/<run-label>/`: best model, architecture/data metadata, source hashes, per-epoch losses, physical predictions, validity masks, four sets of diagnostics and `results/input_audit/`. The launcher archives this tree and its Slurm log, including available partial artifacts after a failure. The default evaluation split is validation, not the locked test set. No dual HPO/resume entry point exists yet.

## Trial 1: HPO-informed, not dual-HPO optimised

The following independent-run `best_params.json` files were recovered from `/data/SEMS-TaoLab/Niccolo-Forte/p2/` on 7 September 2026. These are source records, not newly tuned dual results.

| Setting | UT field | FT field | UT field → full curve | FT field → full curve | Dual trial 1 |
| --- | ---: | ---: | ---: | ---: | --- |
| Token width | 256 | 256 | 256 | 96 | field 256; curve 256 |
| Attention heads | 4 | 4 | 2 | 4 | 4 in both stages |
| Encoder layers | 4 | 5 | 2 | 2 | field 4; curve 2 |
| Feed-forward multiplier | 4 | 2 | 4 | 6 | 4 in both stages |
| Encoder dropout | 0.185 | 0.264 | 0.269 | 0.031 | field 0.20; curve 0.15 |
| Learning rate | 3.32e-4 | 7.44e-5 | 2.92e-5 | 1.43e-4 | one optimiser: 1e-4 |
| AdamW weight decay | 7.21e-9 | 2.47e-9 | 3.80e-8 | 1.96e-8 | 1e-8 |
| Batch size | 2 | 1 | 8 | 8 | 2 paired specimens |

Source files, relative to that archive root:

- `UT/Field/HPO/fUT-fHPO/Transformer/best_params.json`
- `FT/Field/HPO/fFT-fHPO/Transformer/best_params.json`
- `UT/FieldToCurve/HPO/f2cUTfull-fHPO/Transformer/best_params.json`
- `FT/FieldToCurve/HPO/f2cFTfull-fHPO/Transformer/best_params.json`

Both curve HPO runs used full 201-point targets, mean pooling and a CLS token. Trial 1 retains these choices, with masked mean pooling over body tokens; the CLS token can still influence attention. ReLU encoder activation and learned positional embeddings match all four sources. The canonical order is therefore part of the checkpoint/data contract.

Field capacity matches the four-layer UT reference and is close to the five-layer FT reference. The shared curve stage uses the larger UT width, a common four-head setting and the two-layer depth found in both curve studies. One learning rate must serve the joint network; 1e-4 is an explicit compromise. These settings are a starting point, not a claim that transferring independent HPO optima is optimal.

Trial 1 uses train-standardised MSE for each of the four targets with weights 1, at most 450 epochs, plateau factor 0.7/patience 12 and early-stop patience 75. This is closer to the independent MSE studies than the smoke runner's default physical-curve CombinedCurveLoss. Pooling, head depth, separate attention/head dropouts and normalisation options are not all identical to the legacy architectures: the dual baseline intentionally has simple LayerNorm–Dropout–Linear output heads. Record that distinction in comparisons. CLI arguments override the preset; the effective configuration and preset hash are saved.

## Why keep the synthetic test file?

`test_dual_contract.py` checks active architecture and data invariants without HPC data: exactly two encoders, one call per stage, absent-node masking, joint gradients, normalisation/checkpoint round trips, pin-selection calculations, the trial preset, and runner artifact collection. It is not a legacy compatibility shim or a second training workflow. It catches regressions that a successful three-epoch training run alone cannot identify.

```bash
python -m unittest discover -s p2-DisorderML/HPC/DualOutputs -p 'test_dual_contract.py' -v
```
