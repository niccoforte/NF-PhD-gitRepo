# SimCheck task checklist

Unique checklist for chat 01a1782e-8158-7792-90e7-6fd570dd3155.
Updated 2 November 2026. [Evidence and figures](README.md).
This file owns task completion; PROJECT_STATUS and the cleanup register link here.

Researcher requests a3November cutoff for the Hsieh phase. Finish with the
strongest completed evidence and quantify remaining limitations; do not extend
kagome into an open-ended agreement/rerun campaign. Derived simulation-energy
indices and length-scaled estimates retain their actual calculation provenance.

## Hsieh validation before closing this phase

- [x] Consolidate useful original validation sources and preserve their identities.
- [x] Reconcile Review lattices.py findings and apply verified foundation repairs.
- [x] Recover native histories without fabricated zero tails or dropped final samples.
- [x] Check initiation, recorded damage, STATUS and kinetic/internal energy separately.
- [x] Check initial compliance, conversion modulus, units and reference normalisation.
- [x] Implement and assess Butterworth filtering and modal controls.
- [x] Digitise Fig.10 and retain original Fig.8 references and stated inconsistencies.
- [x] Process four new3mm triangular/kagome13%/16% cases and update existing plots.
- [x] Complete the new13% hexagonal elastic controls.
- [x] Complete/process hexagonal10mm/16%: KJ=5.349MPa sqrt(m), positive area,
  endpoint KE/IE=.464%; included provisionally in Figs8a/10. Earlier transients
  require the separate rate check below; this is not full quasistatic qualification.
- [ ] Complete/process hexagonal3mm/13% restart and10mm/13% run.
  [Queue state](Z:/p1/sims/Ti/Validation/audit/new-validation/queue-resume-20261102/state.json).
  The3mm restart and both workers subsequently exited without a known cause;
  field21/200 and solver time.01068s are saved. No solver remains active at the
  18:40London2November check. Partial sources remain intact;10mm13% never started.
  [Execution review](Z:/p1/sims/Ti/Validation/audit/new-validation/deadline-execution-review-20261102.json).
- [ ] Process the slower hexagonal10mm/16% rate check, armed after those two jobs
  through the [serial continuation](Z:/p1/sims/Ti/Validation/audit/serial-continuation-20261102-status.json),
  now stopped for review because its worker exited. This control has not started.
  Time is3.333times longer, with the same end displacement and material/mesh.
  The baseline exceeds5% KE/IE until.0239s, about30% of its time to initiation.
  Compare force, work and onset; choose the accepted result from this evidence.
- [ ] Resolve the hexagonal negative-area limitation: establish a defensible
  fracture-energy calibration or explicitly delimit unavailable source-convention
  results. Do not clip area or substitute dissipation/unloading work as ASTM J.
- [ ] Finalise the existing agreement/error table. Kagome retains the density
  trend with direct3mm errors+9.82%,+14.02%,+19.42%; report these rather than
  requiring closer agreement. No kagome rerun is justified by these residuals alone.
- [ ] Freeze one final calculation/event/filter/modulus definition and verdict.
  Four unique points remain absent from the current plot: hex3mm13% and10mm13%
  need new simulation prefixes; hex3mm16% and20% have completed histories but
  negative initial-triangle area. The third planned job checks10mm16% rate
  sensitivity and cannot fill either negative-area point.
- [ ] Make direct3mm Fig.10 validation primary; describe10mm as a sqrt(10/3)
  length-scaled reference. Keep Fig.8b sqrt(3) adjustments labelled hypotheses.

## Replace methods and results after the definition is settled

- [ ] Replace superseded production exporter, initiation, filtering, stiffness and
  modulus logic in place; do not retain selectable wrong/legacy calculation paths.
- [ ] Bind each production endpoint to its intended observable and source material
  law. Hsieh's surface-initiation threshold must not silently redefine complete
  deletion labels, UT endpoints or other materials' failure thresholds.
- [ ] Reprocess affected production datasets from reusable native simulations;
  preserve sample identities, validity and downstream p1/p2 alignment.
- [ ] Reconcile with Corrections: isotropic conversion and orthotropic FCC M are
  distinct; avoid a second plane-strain reduction of already mapped beam tensors.
- [ ] Consolidate one bending-aware periodic homogenisation workflow. Retain the
  source-matched mapped B21 controls for tri/kagome/hex; unit-load specimen
  compliance is a different quantity and is also required. Retire overlapping
  active stiffness paths only after material/geometry/BC/tensor agreement and
  consumer replacement. Existing physical-material FCC controls cannot substitute
  for mapped FCC tensors; retain them only for a referenced analytical comparison
  or minimal provenance after direct mapped controls become authoritative.
- [ ] Replace historical Excel FTval values/placeholders and obsolete audit sheets,
  notebook cells, charts and paper references with the accepted results/workflow.
- [ ] Integrate final methods, caveats, notation and figures into the actual paper.
  No manuscript replacement is complete yet.
- [ ] Verify all active consumers use the accepted workflow and no obsolete results
  remain presented as current. Retain minimal recovery evidence outside active work.
- [ ] Perform final cleanup of audit variants, auxiliary solver bulk, duplicate
  exports, temporary verification/launcher files and superseded code/results.
  [Cleanup inventory](Z:/p1/sims/Ti/Validation/audit/cleanup-review.json) owns paths
  and safety dependencies; it is not a second task checklist. Protect running jobs,
  original observations and unique digitisation. Verify replacement before deletion.

## Later work already in this chat's scope

- [ ] UT validation and any missing-stiffness processing issues.
  Corrections confirmed516 hexagonal stiffness NaNs despite finite201-sample
  curves: peak indices17-19 select only2fitpoints. Of956 common UT/FT IDs,
  502 are removed by that fit failure, leaving454 paired rows. Repair the
  verified elastic-window calculation; do not impute or rerun516 simulations.
  [Read-only evidence](C:/Users/exy053/AppData/Local/Temp/p1-corrections-coordination-20261102/hex-stiffness-readonly.json).
- [ ] FCC elastic/fracture validation coordinated with Corrections.
  Inventory existing material cards first; reuse mapped controls only when their
  exact source matches. All54 inspected StiffMatrix FCC inputs, including the
  three canonical periodic controls, use physical E=123000MPa,nu=.3, not mapped
  material. This bounded inventory does not establish full-archive absence.
  [FCC inventory](C:/Users/exy053/AppData/Local/Temp/p1-corrections-coordination-20261102/fcc-source-inventory-readonly.json).
  Propagate the same measured C and its material state through FTprops/calcFT;
  replacing a scalar alone does not settle the FCC calculation.
- [ ] Square/45square orthotropic validation and specimen-axis calibration.
- [ ] PSC calculation audit, especially fracture toughness; preserve the
  supervisor-agreed specimen sizes and convergence criteria.
- [ ] Revisit conditional lattice-foundation findings only where those later
  workflows actively use them; verify downstream p1/p2 dependencies.

Corrections owns manuscript/notation work and read-only paired-population checks;
SimCheck owns this calculation/solver migration. Agree exact shared-file edits
before either chat changes resources or stiffness notebooks. Final population
selection remains deferred; paper candidate IDs are1-1000 and ID0 is reference only.
