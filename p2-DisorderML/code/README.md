# p2 notebooks

Training notebooks train or tune models; post-processing notebooks inspect saved
runs. See [AGENTS.md](AGENTS.md) for the notebook roles and boundaries.

## Scientific figure exports

`ML-FigureExports.ipynb` regenerates the mechanical trade-off scatter, selected
individual response curves and verified initial-beam displacement comparison.
It calls the maintained comparison functions in
[`resources.MLmetrics`](../../resources/MLmetrics.py). It does not load a checkpoint,
run inference, train, or export new FEA data.

1. Open it in the configured repository environment with `resources` importable.
2. Set `FIGURE_MANIFEST` to the external `figure-manifest.json` supplied with the
   approved figure library. That file identifies the source CSV, physical saved
   prediction NPZ, selected specimen INPs, cases, scales, source hashes and metrics.
3. Keep `VERIFY_SOURCES=True` to reject changed source files. No current winner-file
   hash or archived poster helper is required.
4. Choose `FORMATS` (`png`, `svg`, optionally `pdf`) and `DPI`. Set `OUTPUT_ROOT` if
   needed; otherwise exports go under the saved run's
   `results/postProcessing/figures/`. Approved library paths are refused so that
   regeneration cannot silently replace accepted originals.
5. Run the notebook. It closes each exported figure and writes a compact
   `figure-regeneration.json` with input checks, selected cases and output hashes.

The explicit display controls beside each plot reproduce the accepted figure
dimensions and semantic colours. Curve residual bars are absolute pointwise
errors on the same physical response axis. Field ribbons require a verified INP
beam map aligned to the saved canonical coordinates; they preserve the initial
crack, voids and masks. They do not infer topology, deformation or evolving damage.

The generic point-based field viewers in the post-processing notebooks retain
their own defaults. Initial-beam export is an explicit verified-geometry route,
not a replacement for those viewers. Generated figures and manifests remain
outside Git; the notebook contains no machine-specific data paths or saved output.
