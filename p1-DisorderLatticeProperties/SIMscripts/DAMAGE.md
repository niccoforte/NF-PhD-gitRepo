# Read-only element-to-strut damage export

Separate opt-in producer; displacement/curve pipelines and ML architecture are
unchanged. The validated scope is the two periodic Ti FCC B21 UT/FT ODB/INP pairs
in `Z:/p1/data/Ti/disNodes/0.2/FCC/sims/`, stems `Ductile-FCC-20-per-1` and
`Fracture-FCC-20-per-1`. Periodic simulation ID is 0. No recursive discovery,
ODB upgrade, FEA or job submission is involved.

## Start here

The algorithm is implemented in maintained repository code. `samples/` holds
the generated results and visual report, not the executable implementation.

| File | Use |
| --- | --- |
| `p1-DisorderLatticeProperties/SIMscripts/A2_DamageOUTpostProcess.py` | Read one explicit ODB/INP pair and write damage arrays plus provenance. Requires Abaqus Python. |
| `resources/damage.py` | Shared element-to-strut mapping, reduction and array validation. |
| `p2-DisorderML/code/review_damage.py` | Read saved arrays and rebuild the human-readable report and figures. Uses standard Python. |

There is no damage loader in `DATA`, training target/head in `MODEL`, or automatic
HPC launch integration. This is a working export-and-review route; connecting it
to ML is a separate task. Only the checked periodic FCC profile is supported.

To rebuild the report from the two existing saved products, use the existing
environment below. It does not open an ODB. `Push-Location`/`Pop-Location` keep
the chat project's working directory unchanged:

```powershell
Push-Location 'C:\Users\exy053\Documents\00-PhD-gitRepo'
.\.venv311\Scripts\python.exe -c "import runpy; runpy.run_path('p2-DisorderML/code/review_damage.py', run_name='__main__')" --ut p2-DisorderML/samples/damage-audit/UT-periodic-damage.npz --ft p2-DisorderML/samples/damage-audit/FT-periodic-damage.npz --output-dir p2-DisorderML/samples/damage-audit
Pop-Location
```

Read `p2-DisorderML/samples/damage-audit/README.md` for the visual results. The
technical reference below records how labels and files are defined; routine use
does not require reading it.

Only if re-extraction is needed, this example regenerates the selected periodic
UT product. It explicitly overwrites that derived NPZ and its adjacent JSON;
the original INP/ODB are read only. Use `--task FT`, the `Fracture-FCC-20-per-1`
source stem and `FT-periodic-damage.npz` for the selected FT product.

```powershell
Push-Location 'C:\Users\exy053\Documents\00-PhD-gitRepo'
& 'C:\SIMULIA\Commands\abaqus.bat' python -c "import runpy; runpy.run_path('p1-DisorderLatticeProperties/SIMscripts/A2_DamageOUTpostProcess.py', run_name='__main__')" `
  --odb 'Z:/p1/data/Ti/disNodes/0.2/FCC/sims/Ductile-FCC-20-per-1.odb' `
  --inp 'Z:/p1/data/Ti/disNodes/0.2/FCC/sims/Ductile-FCC-20-per-1.inp' `
  --task UT --output 'p2-DisorderML/samples/damage-audit/UT-periodic-damage.npz' `
  --partial-rule failure-only --overwrite
Pop-Location
```

The current policy is `failure-only`: verified failed parents have D=1, fully
observed undamaged active parents have D=0, and active parents with observed
damage retain raw SDEG but have no assigned scalar D. Researcher approval is
still needed before replacing this policy with `--partial-rule max`.

<details>
<summary>Optional technical reference: algorithm, saved arrays and validation</summary>

## Full algorithm

1. Parse original inline B21 INP meshes, preserving instance-scoped node/element
   identities and sets. Reject includes, transformed instances and unsupported
   element types rather than guessing. UT selects BodySet; FT AllPartSet.
2. Identify the checked periodic 800 body joints or native 788-joint FT subset.
   Traverse actual FE connectivity from joint to joint. Require degree-two
   interior nodes, straight monotonic chains, no loops, complete unique FE
   coverage and unique unordered endpoint pairs. Measure subdivisions; never
   infer membership from consecutive labels or a fixed element count.
3. Save original endpoints, FE connectivity, parent indices and chain order.
   Audit canonical indices use periodic coordinates sorted `(y,x)`, not saved
   DATA/DUAL order. The actual endpoint sets were compared exactly against
   `reference_field_edges`. Disordered original-joint identity needs separate
   validated metadata; do not nearest-match it blindly to periodic coordinates.
4. Hash INP/ODB, open the explicit ODB with `readOnly=True` in Abaqus Python,
   then verify original IDs, connectivity, initial coordinates (2e-5 mm
   tolerance) and element selection. Matching meshes/stems are evidence, not
   an independently recovered generation manifest.
5. Read the named step and every actual recorded frame, including frame 0.
   Preserve increment/time, raw SDEG, DUCTCRT and whole-element STATUS. Requested
   DMICRT is stored as DUCTCRT here; initiation is not degradation. Identify raw
   observations by instance, element, output position, integration and section.
6. STATUS=0 establishes whole-element failure and sets derived element D=1;
   STATUS=1 means active. Never infer failure by rounding/thresholding SDEG, or
   from a single fully damaged material point. Retain the independent masks.
7. Partial rules are explicit. `failure-only` assigns D=0 only for fully observed
   undamaged active elements and leaves partial numeric D as NaN. `max` uses
   maximum reported section/IP SDEG when coverage and STATUS are known. Here
   only bottom/top section points 1 and 5 at integration point 1 are stored.
   That maximum is severity, not average cross-section degradation or proof of
   unreported interior values. It needs an explicit researcher policy choice.
8. For a parent, any verified failed member sets parent D=1 even if another
   member is unknown. Otherwise maximum-rule parent D is the maximum of complete
   member values. Incomplete observations/unselected partial reduction remain
   unknown. Never use a raw sum of element failures as D.
9. Preserve coverage, numeric-label validity and failure-known separately.
   Missing does not mean intact or failed. Carry-forward is off by default;
   optional `--carry-failure` records a separate carried mask and rejects
   reactivation. Store first failure and the previous observed all-active frame
   as an onset bracket, without inventing between-frame failure times.
10. UT loading is actual field-frame mean SET-TOP U2 / initial body H=190 mm,
    matching the legacy nominal convention. Also save relative top-body versus
    bottom-body strain. FT loading is LOAD reference-point U2 in mm. Sum loading
    RF2. Keep original nodal U2/RF2 history records on their own grids; no
    zero-fill, interpolation or ordinal joins.
11. Close the ODB, confirm unchanged original hashes, save one NPZ per task and
    reopen with `allow_pickle=False`. Validate identities/shapes/masks and exact
    re-derivation from saved raw observations. Write adjacent provenance JSON.
12. Generate actual raw-value examples, several-frame parent/element maps and
    damage/loading plots. Confirm the label policy and sample before scaling
    or model integration. Initial FT absence is not evolving failure.

The STATUS/SDEG definitions are supported by the accessible
[Abaqus damage documentation](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEMATRefMap/simamat-c-damageevolductile.htm).
The files were actually read in Abaqus 2024. The link is to the accessible 2025
documentation; the 2024 web page was unavailable. Stored descriptions, positions
and values were inspected directly in the two ODBs.

## Versioned contract

`schema_version="strut_damage_v1"`. No object arrays/pickle. Axes: F actual
frames, S native task struts, E selected elements, P reported material points,
N selected FE nodes, H raw scalar history records.

| Group | Keys/layout |
| --- | --- |
| Identity/topology | simulation_id, task, strut_id/present[S], canonical_endpoint[S,2], canonical_xy[800,2], canonical_node_present[800] |
| Original membership | endpoint_instance/node_label[S,2], endpoint_xy0[S,2,2], element_instance/label[E], element_connectivity[E,2], parent_strut_index/chain_order[E], subdivision_count/mapping_residual[S] |
| Raw points/status | point_element_index/instance/position/integration/section/section_description[P], sdeg/ductcrt[F,P], status[F,E], explicit sdeg/status validity |
| Frame geometry/loading | frame_time/increment[F], node_label[N], node_xy0[N,2], node_displacement[F,N,2] and validity, global_loading/load_displacement/reaction_force/body_strain[F] |
| Raw histories | Region/variable name tables; history_region_index/variable_index/time/value[H] |
| Derived state | element_damage/sdeg_max/sdeg_complete[F,E], element_failure_observed/known/carried/failed[F,E], strut_damage/valid/coverage/observation_coverage/failed/failure_known[F,S], failed_element_count[F,S], first_failure_frame/previous_active_frame[S] |
| Provenance/policy | partial_rule, carry_failure, metadata_json with exact paths, source/code hashes, versions, units, loading/point/failure definitions and source-match evidence |

There is no DATA/MODEL consumer. Native FT has 60 fewer initial struts than UT;
a later canonical join must mark absent edges separately from failed edges.
Future stacking must validate specimen IDs, endpoints, task masks and actual
time/loading grids. Do not broadcast labels onto nodes or use future true damage
to change inference edges. Always retain failure masks: a reported maximum near
1 can occur in an active element.

`strut_coverage` is the fraction of constituent numeric D labels assigned;
`strut_observation_coverage` is raw STATUS plus expected reported-point coverage.
They differ in failure-only mode because observed partial values remain unassigned.

## Run and validate

From the repository root in existing environments:

```powershell
abaqus python -c "import runpy; runpy.run_path('p1-DisorderLatticeProperties/SIMscripts/A2_DamageOUTpostProcess.py', run_name='__main__')" --odb ODB_PATH --inp INP_PATH --task UT --output OUTPUT.npz --partial-rule failure-only
python -c "import runpy; runpy.run_path('p2-DisorderML/code/review_damage.py', run_name='__main__')" --ut UT.npz --ft FT.npz --output-dir p2-DisorderML/samples/damage-audit
python -m unittest discover -s p1-DisorderLatticeProperties/SIMscripts -p test_damage_contract.py -v
abaqus python -c "import runpy; runpy.run_path('p1-DisorderLatticeProperties/SIMscripts/test_damage_odb.py', run_name='__main__')" --npz OUTPUT.npz
```

`--partial-rule max` selects the reported-section/parent maximum; failure
detection remains independent. Existing outputs require `--overwrite`.

Actual periodic results: 21 frames/task; UT body 2,319 struts/10,075 elements,
excluding 830 grip elements; FT 2,259/4,024. Final failed parents/elements:
UT 83/83 and FT 7/7. No expected SDEG/STATUS records were missing, no STATUS
reactivation occurred, and original hashes were unchanged. Every stored SDEG
value was either 0 or 0.9999899864196777; the field grid does not resolve a smooth
intermediate softening ramp. Active partially damaged beams can have one stored
surface value near 1 and the other 0. Do not fabricate intermediate histories or
reinterpret that state as whole-element failure.

The ignored p2 `samples/damage-audit/` contains NPZs, provenance, frame CSVs,
raw-value examples and inspected figures. Producer, review code, tests and this
runbook are tracked; generated research artifacts are not force-added to Git.
Disordered/bulk processing and model integration remain separate decisions.

The independent live-ODB test compared all 296,079 saved STATUS values and all
parent failure states at every frame directly with the ODBs, plus 140,990 raw
section SDEG values at five selected frames/task. All comparisons passed.

</details>
