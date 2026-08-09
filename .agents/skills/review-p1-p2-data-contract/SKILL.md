---
name: review-p1-p2-data-contract
description: Review and protect the producer-consumer contract from p1 Abaqus generation and transfer outputs through shared processing and MLdata loading to p2 training, saved runs, transfer, and diagnostics. Use when changes touch p1 A1/A2 scripts, transfer names, manifests, CSV/NPZ schemas, sample alignment, field metadata, resources/data_processing.py, resources/MLdata.py, model save/load layouts, p2 notebooks, HPC entry points, or output tokens such as Curve, Field, and FieldToCurve.
---

# Review The P1-To-P2 Data Contract

Treat the p1-to-p2 boundary like an internal API. p1 produces named CSV/NPZ files and metadata; shared code transforms and loads them; p2 assumes their sample identity, shapes, and meanings. A change can run successfully at the producer while silently pairing the wrong input and output downstream, so review both ends together.

## Review procedure

1. Identify the changed producer, transformation, or consumer.
2. Read [references/data-contract.md](references/data-contract.md) for the affected contract only.
3. Inspect the actual producer and every listed consumer; documentation is a routing aid, not proof of behavior.
4. Update all affected ends together, or define an explicit migration for existing research artifacts.
5. Run `python tools/validate_repo.py --scope contract`, then add the narrowest code-specific or real-data check available.
6. Report separately what was verified synthetically, with local research data, in Abaqus, and on HPC.

Do not reinterpret smoothing, fracture indices, property definitions, filtering, loss definitions, or node selection as mere schema work. Those are scientific changes and need established authority.

Invoke `maintain-repo-guidance` whenever this contract or its documentation changes.
