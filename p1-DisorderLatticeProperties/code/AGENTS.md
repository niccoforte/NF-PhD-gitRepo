# p1 Notebook Agent Context

This directory contains the notebook layer for p1: FEA-derived mechanics, data processing, validation, and exploratory analysis for quasi-disordered 2D lattices. Most notebooks depend on shared helpers in `resources/`; read the helper code before changing notebook behavior.

## Workflow Role

- `p1-DisorderLatticeProperties/SIMscripts/` generates Abaqus models and writes raw `transfer/` files.
- `DataProcessing.ipynb` is the current batch processing notebook. It builds `DATA(path=1, ...)` objects, loops over lattice/disorder settings, and calls `create_inputCSV`, `create_outputCSV`, and `create_fieldNPZ`.
- `InputsOutputs.ipynb` inspects and saves ML-ready curve and field datasets using `load_data`, `UTprops`, `FTprops`, `MULTIprops`, `save_MLdata`, `save_MULTIdata`, `save_field_MLdata`, and `save_MULTIfieldData`.
- `SIMresults.ipynb` is for quick local/transfer result inspection and mechanical-property calculation from output CSVs.
- `StiffnessMatrix.ipynb` handles analytical/simulation stiffness matrices, isotropy checks, and effective-property calculations.
- `ContinuumPlots.ipynb` and `ValConvPlots.ipynb` support displacement-field visualization, material/validation plots, mesh/refinement checks, and convergence/validation figures.
- `AK-DataProcessing.ipynb` and `AK-InputsOutputs.ipynb` are legacy Akash-data workflows using `DATA(path=0, ...)`; do not mix them into the current Ti data pipeline unless the user asks.
- `FunctionApproximation.ipynb`, `Sampling.ipynb`, and `QuickPlots..ipynb` are exploratory research notebooks. Treat them as prototypes unless the user makes one active.

## Data Conventions

- Current local Ti data generally resolves through `DATA(path=1, LAT=..., nnx=..., dis=..., dN=..., mechMode=...)`.
- `DATA(path=1, ...)` points into `Z:/p1/data/Ti/{dis}/{path_add}/{dN}/{LAT}/` and uses `PATH_PER` for `Z:/p1/data/Ti/per/{LAT}/`.
- `path_add` values such as `Target-xs` are part of the data tree. Keep them explicit in notebook configuration cells.
- Active lattice names include `FCC`, `tri`, `kagome`, and `hex`; older helpers also know about `square` and `45square`.
- Detailed transfer names, processed schemas, sample alignment, field metadata, and downstream consumers are maintained in the `review-p1-p2-data-contract` skill reference. Use it whenever those boundaries change.

## Shared Helpers To Respect

- `resources/data_processing.py` owns conversion from raw `transfer/` files into processed p1 CSV/NPZ files and manifests.
- `resources/MLdata.py` owns `DATA`, ML-ready loading, property extraction, field loading, split construction, scaling, and save helpers.
- `resources/calculations.py` owns curve smoothing/parsing and UT/FT mechanical-property calculations.
- `resources/lattices.py` owns geometry, node counts, connectivity, effective properties, stiffness matrices, isotropy, and anisotropy calculations.

## How To Work Here

- Before editing a notebook, identify whether the change belongs in the notebook cell or in a shared helper. Repeated data-processing logic usually belongs in `resources/`.
- Do not run whole notebooks blindly. Many cells assume `Z:/` data, OneDrive validation workbooks, Abaqus transfer files, or specific current working directories.
- Keep active notebooks compact and current. When a cell is replaced by a helper or a cleaner workflow, remove the old code path and clear stale outputs instead of keeping a fallback copy.
- Preserve processed CSV row/index alignment. Dropping failed outputs must keep matching input, output, frequency, and manifest records aligned.
- Keep the periodic reference row (`sim_id == 0`) intact unless the task is explicitly about removing or changing the reference.
- Do not silently change smoothing thresholds, fracture-index logic, stiffness fits, effective-property formulas, or outlier rules. These are scientific assumptions, not formatting details.
- If changing field-output handling, preserve sample ids, frame values, node labels, node coordinates, components, valid masks, and main-body node filtering.
- Keep notebook configuration cells explicit for `LAT`, `nnx`, `dN`, `dis`, `path_add`, `mechMode`, and output type. Avoid hidden module-level state.
