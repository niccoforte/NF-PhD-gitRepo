# ML studies

Study entry points, submission scripts and study-specific report tools belong
here, not in `resources/` or the notebook directory. Run commands from repo root.
Preview is the default; only explicit `--submit` queues jobs. Existing immutable
sources, job names and archive paths are not renamed by this source cleanup.

| Study | Run / submit | Review and status |
| --- | --- | --- |
| [Independence](Independence/README.md) | `A0-Independence.py`, `B4_Independence.sh` | Completed; Case1(i) selected |
| [Input](Input/README.md) | `A0-Input.py`, `B4_Input.sh` | `review_field_inputs.py`; Mixed selected on cost/accuracy |
| [Loss](Loss/README.md) | `A0-Loss.py`, `B4_Loss.sh` | Active Mixed continuation; no adoption verdict yet |
| [Damage](Damage/README.md) | Bank builder and completed dataset reviews | New field/damage training study agreed, not implemented |
| [Locality](Locality/README.md) | No training entry point yet | Isolated degree/GNN/CNN/attention experiments planned |

`B4_Field-source-study.sh` contains shared Input/Loss submission mechanics; their
small named wrappers supply the study. The Python presets reuse the established
`HPC/DualOutputs/A0-HPC-Dual-test.py` interface. B1 preserves their relative
`studies/<Study>/` layout in scratch and stages that common runner. There is only
one underlying training implementation. Old trial/HPO and legacy DATA/MODEL
interfaces remain supported.

Imported helpers: `resources/MLcrossfit.py` owns independent training groups and
OOF banks; `resources/MLfield_sources.py` owns field-source policies and loss
comparisons; `MLfield.py` owns field losses; `MLdual.py` owns shared architectures.
These are libraries, not job entry points. Damage preprocessing remains shared
with p1 in `resources/damage.py` and `resources/damage_data.py`.

Reports write only to canonical ignored `p2-DisorderML/samples/`; run artifacts
stay outside Git under `Z:/p2`, preserving archive-relative paths. No notebooks
or analysis code belongs in the external result store. The active checklist is
`../../handoffs/HANDOFF-ML-ACCURACY.md`; changing state belongs in PROJECT_STATUS.

Validation: `test_study_layout.py` protects entry dispatch and B1 scratch staging.
Existing framework regressions remain in `HPC/DualOutputs/test_*.py`; report
tests sit beside their study tools. A source reorganization is not new accuracy
evidence, a reason to rerun completed studies, or permission to alter active jobs.
