"""Direct elastic C(T) calibration, beginning with an isotropic verification.

Prepare with Abaqus/CAE noGUI; extraction uses Abaqus Python. Preparation never
submits a job. The half-model compliance derivative includes the symmetry factor.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path("Z:/p1/sims/SiC/Validation/audit/elastic-calibration")
CRACK_RATIOS = (.49, .495, .5, .505, .51)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def elastic_factor(compliances, W, B, modulus, load=1., crack_ratios=CRACK_RATIOS):
    """Centered derivatives from upper-half pin compliance u/P, not full opening."""
    if len(compliances) != 5 or not all(math.isfinite(x) and x > 0 for x in (W,B,modulus,load)):
        raise ValueError("Five compliances and positive dimensions/material/load required")
    if not all(math.isfinite(a) and a > 0 for a in compliances):
        raise ValueError("Positive finite compliances required")
    ratios=list(crack_ratios)
    if (len(ratios)!=5 or not all(math.isfinite(a) and 0<a<1 for a in ratios)
            or any(b<=a for a,b in zip(ratios,ratios[1:]))
            or abs(ratios[0]+ratios[4]-2*ratios[2])>1e-10
            or abs(ratios[1]+ratios[3]-2*ratios[2])>1e-10):
        raise ValueError('Five ordered symmetric crack ratios required')
    coarse = (compliances[4]-compliances[0])/((ratios[4]-ratios[0])*W)
    fine = (compliances[3]-compliances[1])/((ratios[3]-ratios[1])*W)
    if min(coarse, fine) <= 0:
        raise ValueError("Crack extension must increase compliance")
    return {"dC_half_da_coarse_per_N": coarse,
            "dC_half_da_fine_per_N": fine,
            "difference_step_relative_change": abs(fine-coarse)/fine,
            "J_fine_N_per_mm": load**2/B*fine,
            "f_fine": (modulus*B*W*fine)**.5,
            "f_coarse": (modulus*B*W*coarse)**.5,
            "half_model_rule": "J=P^2/B*d(u_upper/P)/da; full opening is2*u_upper"}


def prepare(out=ROOT):
    """Create two exact Ortega-geometry isotropic inputs, without submission."""
    from abaqus import Mdb, mdb
    import caeModules  # Register CAE model APIs, including output requests.
    from abaqusConstants import (ANALYSIS, CARTESIAN, CPS4R, DEFORMABLE_BODY,
                                 FREE, FROM_SECTION, OFF, ON, QUAD, STANDARD,
                                 TWO_D_PLANAR, UNIFORM)
    import mesh

    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/"manifest.json").exists():
        previous = json.loads((out/"manifest.json").read_text())
        for job in previous["jobs"]:
            if digest(job["input"]) != job["input_sha256"]:
                raise ValueError("Existing prepared input differs; review instead of overwrite")
        print("Existing source-bound isotropic inputs retained; no duplicate preparation", flush=True)
        return previous
    W, B, young, poisson, load = 20., 1., 10000., .3, 1.
    Mdb()
    jobs = []
    for label, global_divisions, tip_divisions in (("medium", 40., 800.), ("fine", 80., 1600.)):
        name = "ct_iso_"+label
        directory = out/name
        directory.mkdir(exist_ok=True)
        model = mdb.Model(name=name)
        sketch = model.ConstrainedSketch(name="outline", sheetSize=5*W)
        sketch.rectangle(point1=(-.25*W, 0.), point2=(W, .6*W))
        sketch.CircleByCenterPerimeter(center=(0., .275*W), point1=(.125*W, .275*W))
        part = model.Part(name="CT", dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
        part.BaseShell(sketch=sketch)
        del model.sketches[sketch.name]
        cut = model.ConstrainedSketch(name="partitions", sheetSize=5*W)
        for x in (0.,)+tuple(ratio*W for ratio in CRACK_RATIOS):
            cut.Line(point1=(x, 0.), point2=(x, .6*W))
        cut.Line(point1=(-.25*W, .025*W), point2=(W, .025*W))
        part.PartitionFaceBySketch(faces=part.faces[:], sketch=cut)
        del model.sketches[cut.name]
        material = model.Material(name="isotropic_verification")
        material.Elastic(table=((young, poisson),))
        model.HomogeneousSolidSection(name="section", material=material.name, thickness=B)
        part.Set(name="BODY", faces=part.faces[:])
        part.SectionAssignment(region=part.sets["BODY"], sectionName="section", thicknessAssignment=FROM_SECTION)
        part.setMeshControls(regions=part.faces[:], elemShape=QUAD, technique=FREE)
        part.setElementType(regions=(part.faces[:],), elemTypes=(mesh.ElemType(elemCode=CPS4R, elemLibrary=STANDARD),))
        part.seedPart(size=W/global_divisions, deviationFactor=.1, minSizeFactor=.1)
        tolerance = W*1e-7
        local_edges = part.edges.getByBoundingBox(xMin=.45*W-tolerance, xMax=.55*W+tolerance,
                                                 yMin=-tolerance, yMax=.025*W+tolerance,
                                                 zMin=-tolerance, zMax=tolerance)
        part.seedEdgeBySize(edges=local_edges, size=W/tip_divisions, deviationFactor=.1, minSizeFactor=.1)
        part.generateMesh()
        assembly = model.rootAssembly
        assembly.DatumCsysByDefault(CARTESIAN)
        instance = assembly.Instance(name="CT-1", part=part, dependent=ON)
        pin = instance.nodes.getByBoundingBox(xMin=-tolerance, xMax=tolerance,
                                             yMin=.4*W-tolerance, yMax=.4*W+tolerance,
                                             zMin=-tolerance, zMax=tolerance)
        if len(pin) != 1:
            raise ValueError("Expected one node at Ortega's upper hole point-load location")
        assembly.Set(name="PIN", nodes=pin)
        model.DisplacementBC(name="pin_x", createStepName="Initial", region=assembly.sets["PIN"], u1=0.)
        steps, previous, prior_bc = [], "Initial", None
        for i, ratio in enumerate(CRACK_RATIOS):
            step_name = "a{:03d}".format(int(round(ratio*1000)))
            model.StaticStep(name=step_name, previous=previous, nlgeom=OFF,
                             initialInc=1., maxInc=1., timePeriod=1.)
            if prior_bc:
                model.boundaryConditions[prior_bc].deactivate(step_name)
            ligament = instance.nodes.getByBoundingBox(xMin=ratio*W-tolerance, xMax=W+tolerance,
                                                       yMin=-tolerance, yMax=tolerance,
                                                       zMin=-tolerance, zMax=tolerance)
            tip_nodes = [node for node in ligament if abs(node.coordinates[0]-ratio*W) <= tolerance]
            if len(tip_nodes) != 1 or len(ligament) < 3:
                raise ValueError("Crack front must coincide with a single mesh node")
            set_name = "LIGAMENT_"+step_name
            assembly.Set(name=set_name, nodes=ligament)
            prior_bc = "symmetry_"+step_name
            model.DisplacementBC(name=prior_bc, createStepName=step_name,
                                 region=assembly.sets[set_name], u2=0.)
            if i == 0:
                model.ConcentratedForce(name="unit_load", createStepName=step_name,
                                        region=assembly.sets["PIN"], cf2=load,
                                        distributionType=UNIFORM)
            steps.append({"step":step_name, "a_over_W":ratio, "tip_node":tip_nodes[0].label,
                          "ligament_node_count":len(ligament)})
            previous = step_name
        model.fieldOutputRequests["F-Output-1"].setValues(variables=("U", "RF", "S", "E"))
        model.historyOutputRequests["H-Output-1"].setValues(variables=("ALLSE", "ALLAE", "ALLWK"))
        job = mdb.Job(name=name, model=name, type=ANALYSIS, numCpus=8, numDomains=8)
        # writeInput writes to cwd; change only to the named task-owned directory.
        import os
        working = Path.cwd()
        try:
            os.chdir(str(directory))
            job.writeInput(consistencyChecking=ON)
        finally:
            os.chdir(str(working))
        inp = directory/(name+".inp")
        nodes = [{"label":node.label, "coordinates":list(node.coordinates)} for node in instance.nodes]
        # CAE mesh connectivity contains node indices; ODB connectivity uses labels.
        elements = [{"label":element.label, "type":"CPS4R",
                     "connectivity":[instance.nodes[i].label for i in element.connectivity]}
                    for element in instance.elements]
        import re
        element_kinds = re.findall(r"(?m)^\*Element, type=([^\r\n]+)", inp.read_text())
        if element_kinds != ["CPS4R"] or any(len(element["connectivity"]) != 4 for element in elements):
            raise ValueError("All-quad isotropic verification mesh required")
        metadata = {"job":name, "input":str(inp), "input_sha256":digest(inp),
                    "state":"prepared_not_submitted", "W_mm":W, "B_mm":B,
                    "E_MPa":young, "nu":poisson, "M_MPa":young, "P_N":load,
                    "C_state":"isotropic plane-stress verification surrogate",
                    "global_seed_mm":W/global_divisions, "tip_seed_mm":W/tip_divisions,
                    "steps":steps, "nodes":nodes, "elements":elements, "pin_node":pin[0].label,
                    "source_geometry":"Ortega Fig1: H=1.2W, L=1.25W, pin x=.25W from left, hole diameter.25W; upper pin y=.275W from symmetry plane.",
                    "load":"1 N at upper hole edge y=.4W; half model symmetry on intact ligament, pin U1=0.",
                    "limits":"Elastic continuum verification only; no plastic calibration or original beam-model edit."}
        (directory/"source.json").write_text(json.dumps(metadata,indent=2)+"\n")
        jobs.append({"job":name, "input":str(inp), "input_sha256":metadata["input_sha256"],
                     "cpus":8, "allow_solver_failure":False, "required_outputs":[".odb", ".dat"],
                     "kind":"linear_CT_isotropic_calibration", "purpose":"Verify half-model compliance derivative against ASTM before orthotropic calibration"})
        print("Prepared "+name+": {} nodes / {} CPS4R elements".format(len(nodes),len(elements)),flush=True)
    manifest = {"abaqus":"C:/SIMULIA/Commands/abaqus.bat", "queue_root":str(out/"queue"), "jobs":jobs}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest


def verified_delivery(out, manifest, job):
    """Bind a completed or individually reviewed post-analysis delivery.

    A recorded failure remains failed. Only one explicit, hash-bound native
    equilibrium review can qualify its unchanged mechanical result for use.
    """
    out = Path(out).resolve()
    manifest_path = out/'manifest.json'
    queue = Path(manifest['queue_root']).resolve()
    queue.relative_to(out)
    state_path = queue/'state.json'
    state = json.loads(state_path.read_text())
    records = [row for row in state.get('jobs',[]) if row['job']==job['job']]
    if (state.get('manifest_sha256') != digest(manifest_path)
            or len(records)!=1 or records[0].get('status') not in ('completed','failed')):
        raise ValueError('Calibration job is not a completed, manifest-bound delivery')
    status=records[0]['status']
    prepared = Path(job['input']).resolve()
    prepared.relative_to(out)
    source_path = prepared.parent/'source.json'
    source = json.loads(source_path.read_text())
    if (source.get('job') != job['job'] or source.get('input_sha256') != job['input_sha256']
            or Path(source['input']).resolve()!=prepared or digest(prepared)!=job['input_sha256']):
        raise ValueError('Prepared calibration input/source identity changed')
    run = queue/'runs'/job['job']
    delivery_path = run/'delivery.json'
    delivery = json.loads(delivery_path.read_text())
    entries = {row['file']:row for row in delivery.get('files',[])}
    if delivery.get('job')!=job['job'] or delivery.get('required_outputs_missing'):
        raise ValueError('Incomplete calibration delivery')
    for suffix in ('.inp','.odb','.dat'):
        name=job['job']+suffix
        if name not in entries or digest(run/name)!=entries[name]['sha256']:
            raise ValueError('Delivered calibration artifact changed')
    if entries[job['job']+'.inp']['sha256']!=job['input_sha256']:
        raise ValueError('Solved and prepared calibration inputs differ')
    receipt={'queue_state_sha256':digest(state_path),'delivery_sha256':digest(delivery_path),
             'source_metadata_sha256':digest(source_path),'queue_recorded_status':status,
             'calculation_usable':True}
    if status=='failed':
        review_path=queue/'reviewed-wrapper-failure.json'
        if not review_path.is_file():
            raise ValueError('Failed calibration job needs an explicit hash-bound native review')
        review=json.loads(review_path.read_text())
        if (review.get('job')!=job['job'] or review.get('state')!='accepted_control_only'
                or review.get('queue_recorded_status')!='failed' or not review.get('accepted_for_calculation')
                or review.get('queue_state_sha256')!=receipt['queue_state_sha256']
                or review.get('delivery_sha256')!=receipt['delivery_sha256']
                or review.get('source_odb_sha256')!=entries[job['job']+'.odb']['sha256']
                or review.get('input_sha256')!=job['input_sha256']
                or not review.get('native_validation',{}).get('all_expected_equilibrium_steps_verified')):
            raise ValueError('Reviewed calibration wrapper-failure identity/status changed')
        for name,sha in review['reviewed_file_sha256'].items():
            if digest(run/name)!=sha:
                raise ValueError('Reviewed calibration wrapper-failure artifact changed')
        receipt.update(wrapper_failure_review=str(review_path),wrapper_failure_review_sha256=digest(review_path))
    return source,run/(job['job']+'.odb'),entries[job['job']+'.odb']['sha256'],receipt


def reduced_tensor_input(template, C):
    """Place an already-reduced law in the verified CPS geometry, in global axes."""
    import numpy as np
    from validation_diagnostics import blocks, key

    C=np.asarray(C,dtype=float)
    if C.shape!=(3,3) or not np.isfinite(C).all() or not np.allclose(C,C.T) or np.linalg.eigvalsh(C).min()<=0:
        raise ValueError('Positive-definite symmetric reduced C required')
    if np.max(np.abs(C[:2,2]))>1e-10*np.max(np.abs(C)):
        raise ValueError('Aligned orthotropy required')
    S=np.linalg.inv(C)
    E1,E2=1/S[0,0],1/S[1,1]
    nu12=-S[0,1]/S[0,0]
    G12=1/S[2,2]
    constants=[E1,E2,math.sqrt(E1*E2),nu12,0.,0.,G12,G12,G12]
    elastic=('*Elastic, type=ENGINEERING CONSTANTS\n'
             '{:.14g}, {:.14g}, {:.14g}, {:.14g}, 0., 0., {:.14g}, {:.14g}\n{:.14g}\n').format(
                 E1,E2,constants[2],nu12,G12,G12,G12)
    generated=[]
    for block in blocks(Path(template).read_text()):
        if key(block)=='*elastic':
            generated.append(elastic)
        elif key(block)=='*solid section':
            generated.append('*Orientation, name=SOURCE_AXES\n1.,0.,0.,0.,1.,0.\n3,0.\n')
            header,rest=block.split('\n',1)
            generated.append(header+', orientation=SOURCE_AXES\n'+rest)
        else:
            generated.append(block)
    return ''.join(generated).replace('isotropic_verification','source_mapped_tensor'),constants


def prepare_orthotropic(out=ROOT):
    """Reuse the verified continuum meshes with each measured reduced tensor."""
    import numpy as np

    out=Path(out)
    isotropic=json.loads((out/'isotropic-results.json').read_text())
    if (len(isotropic)!=2 or max(abs(r['relative_difference_from_ASTM']) for r in isotropic)>.01
            or abs(isotropic[1]['f_fine']/isotropic[0]['f_fine']-1)>.01
            or max(r['difference_step_relative_change'] for r in isotropic)>.005):
        raise ValueError('Isotropic ASTM/mesh/crack-step verification is not qualified')
    if any(r['elastic_energy_relative_difference']>1e-5 or r['artificial_to_strain_energy_ratio']>.05
           for item in isotropic for r in item['steps']):
        raise ValueError('Isotropic energy qualification failed')
    directory=out/'orthotropic'
    directory.mkdir(exist_ok=True)
    manifest_path=directory/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if any(digest(job['input'])!=job['input_sha256'] for job in manifest['jobs']):
            raise ValueError('Prepared orthotropic calibration input changed')
        return manifest
    tensor_path=out.parent/'controls/control-results.json'
    tensors=[r for r in json.loads(tensor_path.read_text())['controls']
             if r['analysis']=='linear_static_periodic_rve']
    if len(tensors)!=8:
        raise ValueError('Eight source-matched tensors required')
    jobs=[]
    for tensor in tensors:
        case=tensor['case_id']
        topology,batch=case.split('-')[1],case.split('-')[3]
        C=np.array(tensor['C_aligned_MPa'])
        for mesh in ('medium','fine'):
            parent=out/('ct_iso_'+mesh)
            template=parent/('ct_iso_'+mesh+'.inp')
            source=json.loads((parent/'source.json').read_text())
            if digest(template)!=source['input_sha256']:
                raise ValueError('Verified geometry template changed')
            name='ct_{}_{}_{}'.format(topology,batch,mesh[0])
            job_dir=directory/name
            job_dir.mkdir(exist_ok=True)
            text,constants=reduced_tensor_input(template,C)
            inp=job_dir/(name+'.inp')
            inp.write_text(text,encoding='ascii')
            metadata={k:source[k] for k in ('W_mm','B_mm','P_N','steps','pin_node','global_seed_mm','tip_seed_mm')}
            metadata.update(job=name,input=str(inp),input_sha256=digest(inp),case_id=case,mesh=mesh,
                            state='prepared_not_submitted',M_MPa=tensor['M_I_pe_MPa'],
                            C_state='already_reduced_2D_law_in_CPS_surrogate',C_MPa=C.tolist(),
                            engineering_constants=constants,
                            tensor_results_sha256=digest(tensor_path),source_tensor_odb_sha256=tensor['source_odb_sha256'],
                            template=str(template),template_sha256=digest(template),
                            isotropic_gate_sha256=digest(out/'isotropic-results.json'))
            (job_dir/'source.json').write_text(json.dumps(metadata,indent=2)+'\n')
            jobs.append({'job':name,'input':str(inp),'input_sha256':metadata['input_sha256'],
                         'cpus':8,'allow_solver_failure':False,'required_outputs':['.odb','.dat'],
                         'kind':'linear_CT_orthotropic_calibration',
                         'purpose':'Direct elastic factor beyond Ortega fit; same reduced source tensor, two meshes and two crack increments.'})
    manifest={'abaqus':'C:/SIMULIA/Commands/abaqus.bat','queue_root':str(directory/'queue'),'jobs':jobs}
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def compliance_from_parameters(lam, chi, scale=10000., coupling=.2):
    """Construct an SPD engineering-shear law with the requested lambda/chi.

    The normalised off-diagonal compliance is -coupling. This is a numerical
    constitutive check, not a new lattice material or a fitted reference curve.
    """
    import numpy as np
    if not all(math.isfinite(x) for x in (lam,chi,scale,coupling)) or min(lam,scale)<=0 or not 0<=coupling<1 or chi+coupling<=0:
        raise ValueError('Stable positive constitutive parameters required')
    s11,s22=1/scale,1/(lam*scale)
    normal=math.sqrt(s11*s22)
    return np.array([[s11,-coupling*normal,0.],[-coupling*normal,s22,0.],[0.,0.,2*(chi+coupling)*normal]])


def prepare_fit_verification(out=ROOT):
    """Independently test Ortega inside its domain, retaining its fitted f."""
    import numpy as np
    from resources.calculations import calc_FaW_aniso
    from resources.lattices import calc_anisoParams, prismatic_plane_strain

    out=Path(out)
    directory=out/'ortega-verification'
    directory.mkdir(exist_ok=True)
    manifest_path=directory/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if any(digest(j['input'])!=j['input_sha256'] for j in manifest['jobs']):
            raise ValueError('Prepared fitted-equation verification changed')
        return manifest
    parameters=[('c025',1.,.25),('c05',1.,.5),('fcc',1.,.57457199),('c2',1.,2.),('c5',1.,5.),
                ('l025c05',.25,.5),('l025c2',.25,2.),('l4c05',4.,.5),('l4c2',4.,2.)]
    jobs=[]
    for tag,lam,chi in parameters:
        S=compliance_from_parameters(lam,chi)
        C=np.linalg.inv(S)
        measured=calc_anisoParams(S=S)
        if not np.allclose(measured,(lam,chi),rtol=1e-12):
            raise ValueError('Constructed material does not reproduce requested parameters')
        M=prismatic_plane_strain(C,C_state='plane_strain')['M_I_pe']
        predicted=float(calc_FaW_aniso(.5,1.,S=S))
        swapped=S[np.ix_((1,0,2),(1,0,2))]
        reciprocal=float(calc_FaW_aniso(.5,1.,S=swapped))
        for mesh in ('medium','fine'):
            template=out/('ct_iso_'+mesh)/('ct_iso_'+mesh+'.inp')
            source=json.loads((template.parent/'source.json').read_text())
            if digest(template)!=source['input_sha256']:
                raise ValueError('Verified mesh template changed')
            name='ov_{}_{}'.format(tag,mesh[0])
            path=directory/name
            path.mkdir(exist_ok=True)
            text,constants=reduced_tensor_input(template,C)
            inp=path/(name+'.inp')
            inp.write_text(text,encoding='ascii')
            metadata={k:source[k] for k in ('W_mm','B_mm','P_N','steps','pin_node','global_seed_mm','tip_seed_mm')}
            metadata.update(job=name,input=str(inp),input_sha256=digest(inp),case_id=tag,mesh=mesh,
                            state='prepared_not_submitted',M_MPa=float(M),C_state='already_reduced_2D_law_in_CPS_surrogate',
                            S_MPa_inverse=S.tolist(),C_MPa=C.tolist(),engineering_constants=constants,
                            lambda_requested=lam,chi_requested=chi,
                            Ortega_f_current_axes=predicted,Ortega_f_reciprocal_axes=reciprocal,
                            template=str(template),template_sha256=digest(template),
                            purpose='Validate the fitted Ortega equation against independent linear compliance inside its published parameter domain.')
            (path/'source.json').write_text(json.dumps(metadata,indent=2)+'\n')
            jobs.append({'job':name,'input':str(inp),'input_sha256':metadata['input_sha256'],
                         'cpus':8,'allow_solver_failure':False,'required_outputs':['.odb','.dat'],
                         'kind':'Ortega_fitted_equation_verification',
                         'purpose':'Independent in-domain fitted-equation check; no fracture reference values or coefficient tuning.'})
    manifest={'abaqus':'C:/SIMULIA/Commands/abaqus.bat','queue_root':str(directory/'queue'),'jobs':jobs,
              'reused_isotropic_results':str(out/'isotropic-results.json')}
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def source_consistent_ortega(S_xy, f_crack_axis):
    """Compare Ortega's loading-axis convention through invariant elastic J.

    The independently solved factor uses the crack-axis mode-I modulus.
    Ortega calls axis 1 the loading direction (Eq. 10), and its Eq. 12
    conversion must use that same order.  A geometry factor and modulus
    from different conventions must never be combined for unequal moduli.
    """
    import numpy as np
    from resources.calculations import calc_FaW_aniso
    from resources.lattices import calc_anisoParams, prismatic_plane_strain
    S_xy=np.asarray(S_xy,dtype=float)
    # prismatic_plane_strain validates positive definiteness, alignment and units.
    M_crack=prismatic_plane_strain(np.linalg.inv(S_xy),C_state='plane_strain')['M_I_pe']
    S_source=S_xy[np.ix_((1,0,2),(1,0,2))]
    lam_source,chi=calc_anisoParams(S=S_source)
    M_source=math.sqrt(2*lam_source**1.5/(1+chi))/S_source[0,0]
    f_source=float(calc_FaW_aniso(.5,1.,S=S_source))
    f_converted=f_source*math.sqrt(M_crack/M_source)
    relative=f_converted/f_crack_axis-1
    return {'Ortega_source_axis1':'loading y','Ortega_source_axis2':'crack x',
            'Ortega_source_lambda':float(lam_source),'Ortega_source_chi':float(chi),
            'Ortega_Eq12_M_MPa':float(M_source),'crack_axis_M_MPa':float(M_crack),
            'Ortega_f_source_convention':f_source,
            'independent_f_source_convention':float(f_crack_axis*math.sqrt(M_source/M_crack)),
            'Ortega_f_converted_to_crack_axis_M':float(f_converted),
            'source_consistent_K_relative_error':float(relative),
            'source_consistent_J_relative_error':float((1+relative)**2-1)}


def compare_fit_verification(out=ROOT):
    """Compare independent solved factors to the unchanged fitted equation."""
    out=Path(out)
    directory=out/'ortega-verification'
    results_path=directory/'orthotropic-results.json'
    data=json.loads(results_path.read_text())
    results=[]
    for case in sorted({r['case_id'] for r in data}):
        pair=[r for r in data if r['case_id']==case]
        if len(pair)!=2:raise ValueError('Both mesh results required')
        medium=next(r for r in pair if r['mesh']=='medium')
        fine=next(r for r in pair if r['mesh']=='fine')
        metadata=json.loads((directory/fine['job']/'source.json').read_text())
        f=float(fine['f_fine'])
        fit=metadata['Ortega_f_current_axes']
        inverse=metadata['Ortega_f_reciprocal_axes']
        source_comparison=source_consistent_ortega(metadata['S_MPa_inverse'],f)
        mesh_change=f/medium['f_fine']-1
        energy=max(r['elastic_energy_relative_difference'] for p in pair for r in p['steps'])
        artificial=max(r['artificial_to_strain_energy_ratio'] for p in pair for r in p['steps'])
        results.append({'case_id':case,'lambda':metadata['lambda_requested'],'chi':metadata['chi_requested'],
                        'M_MPa':metadata['M_MPa'],'f_independent_fine':f,'f_independent_medium':medium['f_fine'],
                        'a_over_W':.5,'crack_axis':'global x','opening_axis':'global y',
                        'mesh_factor_relative_change':mesh_change,
                        'crack_derivative_relative_change':fine['difference_step_relative_change'],
                        'Ortega_f_current_axes':fit,'Ortega_f_reciprocal_axes':inverse,
                        'current_axes_K_relative_error':fit/f-1,'reciprocal_axes_K_relative_error':inverse/f-1,
                        'current_axes_J_relative_error':(fit/f)**2-1,
                        'reciprocal_axes_J_relative_error':(inverse/f)**2-1,
                        'fine_source_odb_sha256':fine['odb_sha256'],
                        'max_energy_closure':energy,'max_artificial_to_strain':artificial,
                        **source_comparison,
                        'independent_FE_qualified':bool(abs(mesh_change)<.01 and fine['difference_step_relative_change']<.005
                                                       and energy<1e-5 and artificial<.05)})
    result={'scope':'Independent fitted-equation verification inside the published domain; no lattice-fracture or plastic-J validation.',
            'rows':results,'source_results_sha256':digest(results_path),
            'reused_isotropic_results':json.loads((out/'isotropic-results.json').read_text()),
            'coefficients_changed':False,'reference_values_used_to_fit':False,
            'axis_verdict':('Ortega Eq10 states E11 is along loading. Evaluate both its fitted factor and Eq12 modulus '
                            'in loading-y/crack-x source order. Compare J=f_source^2/M_source; if reporting K with '
                            'the crack-x/opening-y modulus, first convert f=f_source*sqrt(M_crack/M_source). '
                            'Swapping only the fitted factor leaves an inconsistent unequal-modulus calculation.'),
            'primary_source':{'paper':'Ortega et al., Composites Part A 63 (2014) 85-93, Eqs10,12,16 and Appendix A',
                              'corrigendum':'Composites Part A 74 (2015) 140, amended Eq16',
                              'coefficients_or_reference_data_tuned':False},
            'summary':{'all_FE_qualified':all(row['independent_FE_qualified'] for row in results),
                       'max_absolute_source_consistent_K_error':max(abs(row['source_consistent_K_relative_error']) for row in results),
                       'max_absolute_source_consistent_J_error':max(abs(row['source_consistent_J_relative_error']) for row in results)},
            'action':('Retain the fitted Ortega equation inside its domain. Equal-axial-modulus cases are unaffected by '
                      'source-axis conversion. Unequal-axial-modulus callers must use the matched source f/M pair or '
                      'the explicitly converted factor. These tests do not qualify extrapolation to square lattices '
                      'outside chi=0.1..10 or calibrate plastic J.')}
    (directory/'fit-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def figure7_markers(page):
    """Recover published FEM marker centres from Ortega Fig.7 vector objects.

    Uses the original PDF's axes, not fitted-result matching. Coordinates remain
    in PDF points and are preserved alongside their affine data coordinates.
    """
    x0, x1, y0, y250 = 87.727, 279.059, 215.138, 73.284
    groups = {'cross': {}, 'triangle': {}, 'circle': {}}
    for kind, objects in (('curve', page.curves), ('line', page.lines)):
        for index, item in enumerate(objects):
            width, height = item['x1']-item['x0'], item['bottom']-item['top']
            if not item.get('stroke') or not (100 < item['x0'] < 260 and 90 < item['top'] < 214):
                continue
            colour = item.get('stroking_color')
            marker = None
            if kind == 'line' and 1.8 < width < 2.1 and 1.8 < height < 2.1 and colour == (0.,0.,0.,1.):
                marker = 'cross'
            elif kind == 'curve' and 3.1 < width < 3.3 and 2.6 < height < 2.8 and len(item['pts']) == 4:
                marker = 'triangle'
            elif kind == 'curve' and 2.8 < width < 3.1 and 2.8 < height < 3.1 and len(item['pts']) == 6:
                marker = 'circle'
            if marker is None:
                continue
            x = (item['x0']+item['x1'])/2
            y = (item['top']+item['bottom'])/2
            if marker == 'triangle':
                # An equilateral triangle's plotting anchor is its geometric
                # centroid, not the midpoint of its bounding box.
                vertices=item['pts'][:3]
                x=sum(value[0] for value in vertices)/3
                y=sum(value[1] for value in vertices)/3
            point = {'pdf_x_pt': x, 'pdf_y_top_pt': y, 'object_kind': kind,
                     'object_index': index, 'marker': marker,
                     'a_over_W': .1 + .9*(x-x0)/(x1-x0),
                     'Kbar_published_FEM': 250*(y0-y)/(y0-y250),
                     'graphical_halfheight_Kbar': 250*height/(2*(y0-y250))}
            # The cross is two diagonal paths whose rounded PDF coordinates
            # can differ by0.0005pt. Coalesce that one graphic symbol only.
            groups[marker][(round(x,2),round(y,2))] = point
    parameters = {'cross':(.03162,10.),'triangle':(10.,.1),'circle':(1.,10.)}
    result=[]
    for marker, values in groups.items():
        points=sorted(values.values(),key=lambda row:row['a_over_W'])
        if len(points) != 71 or not .199 < points[0]['a_over_W'] < .201 or not .889 < points[-1]['a_over_W'] < .901:
            raise ValueError('Review changed source Fig7 marker inventory: '+marker+' '+str(len(points)))
        for point in points:
            point.update(lambda_Ortega=parameters[marker][0],chi_Ortega=parameters[marker][1])
            result.append(point)
    return result


def digitise_figure7(out=ROOT, paper=None):
    """Run once with a PDF-capable Python; retain source-bound vector readings."""
    import pdfplumber
    paper=Path(paper or ('C:/Users/exy053/OneDrive - Queen Mary, University of London/Documents/Research/00-Literature/'
        '08_Standards_Test_Methods_And_Data_Reduction/Test_Geometry_And_Specimens/'
        'Ortega2014_CompactTensionSpecimenForOrthotropicMaterials.pdf'))
    before=digest(paper)
    with pdfplumber.open(paper) as pdf:
        points=figure7_markers(pdf.pages[4])
    if digest(paper)!=before:
        raise ValueError('Published source changed during read-only digitisation')
    path=Path(out)/'ortega-verification/published-figure7.json'
    record={'source_pdf':str(paper),'source_sha256':before,
            'source_figure':'Fig7, printed page89 (PDF page5)',
            'source_doi':'10.1016/j.compositesa.2014.04.012','points':points,
            'method':'Recover original vector FEM marker centres; triangle centroid, circle centre and deduplicated cross diagonals. No fitted/reference ordinate adjustment.',
            'axis_calibration':{'x_data':[.1,1.],'x_pdf_pt':[87.727,279.059],
                                'Kbar_data':[0.,250.],'y_pdf_top_pt':[215.138,73.284]},
            'published_markers':len(points),'source_unchanged':True}
    path.write_text(json.dumps(record,indent=2)+'\n')
    return {'source_unchanged':True,'published_markers':len(points),'digitisation':str(path)}


def plot_fit_verification(out=ROOT):
    """Published Fig7 FEM markers, independent FE check and axis conversion."""
    import csv
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from resources.calculations import calc_FaW_aniso

    out=Path(out)
    directory=out/'ortega-verification'
    data_path=directory/'fit-comparison.json'
    data=json.loads(data_path.read_text())
    source=json.loads((directory/'published-figure7.json').read_text())
    paper=Path(source['source_pdf'])
    before=digest(paper)
    if before!=source['source_sha256'] or not source['source_unchanged']:
        raise ValueError('Published source differs from vector digitisation')
    points=source['points']
    for point in points:
        S=compliance_from_parameters(point['lambda_Ortega'],point['chi_Ortega'])
        a=point['a_over_W']
        point['Kbar_current_implementation']=float(calc_FaW_aniso(a,1.,S=S)/np.sqrt(a))
        point['relative_difference']=point['Kbar_current_implementation']/point['Kbar_published_FEM']-1
    with (directory/'published-figure7-points.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(points[0]));writer.writeheader();writer.writerows(points)
    rows=data['rows']
    with plt.rc_context({'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'}):
        fig,axes=plt.subplots(1,3,figsize=(15.8,4.8),gridspec_kw={'width_ratios':[1.2,1.1,1.]})
        ax=axes[0]
        symbols={'cross':'x','triangle':'^','circle':'o'}
        colours={'cross':'#222222','triangle':'#777777','circle':'#0000ff'}
        for marker in symbols:
            selected=[row for row in points if row['marker']==marker]
            first=selected[0]
            a=np.linspace(selected[0]['a_over_W'],selected[-1]['a_over_W'],200)
            S=compliance_from_parameters(first['lambda_Ortega'],first['chi_Ortega'])
            curve=[calc_FaW_aniso(value,1.,S=S)/np.sqrt(value) for value in a]
            ax.plot(a,curve,color=colours[marker],lw=1.2,
                    label=r'$\lambda_O=%g,\ \chi=%g$'%(first['lambda_Ortega'],first['chi_Ortega']))
            shown=selected[::5]+selected[-1:]
            ax.plot([row['a_over_W'] for row in shown],[row['Kbar_published_FEM'] for row in shown],
                    ls='',marker=symbols[marker],mfc='none',mec=colours[marker],ms=4,mew=.8)
        ax.set(xlabel=r'$a/W$',ylabel=r'Dimensionless SIF, $\overline{K}_I$',xlim=(.18,.91),ylim=(0,240),
               title='(a) Published Ortega Fig. 7')
        handles,labels=ax.get_legend_handles_labels()
        handles += [Line2D([],[],color='black',lw=1.2),Line2D([],[],color='black',marker='o',mfc='none',ls='',ms=4)]
        labels += ['Current implementation','Published FEM markers']
        ax.legend(handles,labels,frameon=False,fontsize=7.5,loc='upper left')
        ax=axes[1]
        xx=np.arange(len(rows))
        labels=[r'$%g,%g$'%(row['lambda'],row['chi']) for row in rows]
        published=[row['Ortega_f_source_convention']**2/row['Ortega_Eq12_M_MPa'] for row in rows]
        independent=[row['f_independent_fine']**2/row['crack_axis_M_MPa'] for row in rows]
        ax.plot(xx,published,'^',mfc='none',mec='black',ms=7,label='Published equation')
        ax.plot(xx,independent,'o',color='#0000ff',ms=3.5,label='Current study FE')
        ax.set(xlabel=r'Test tensor $(\lambda_{\rm crack},\chi)$',ylabel=r'$J_{\rm el}B^2W/P^2$ (MPa$^{-1}$)',
               title='(b) Independent FE verification')
        ax.set_xticks(xx,labels,rotation=60,ha='right',fontsize=7)
        ax.legend(frameon=False,fontsize=7.5)
        ax=axes[2]
        choices=[next(row for row in rows if row['case_id']==case) for case in ('l025c05','fcc','l4c05')]
        xx=np.arange(3)
        wrong=[100*(row['Ortega_f_source_convention']**2/row['crack_axis_M_MPa'])/
               (row['f_independent_fine']**2/row['crack_axis_M_MPa'])-100 for row in choices]
        corrected=[100*row['source_consistent_J_relative_error'] for row in choices]
        ax.axhline(0,color='.7',lw=.8)
        ax.plot(xx,wrong,'x',color='#a63c32',ms=7,label=r'Mixed $f_O^2/M_B$')
        ax.plot(xx,corrected,'o',color='#0000ff',ms=4,label=r'Matched $f_O^2/M_O=f_B^2/M_B$')
        ax.set_xticks(xx,[r'$0.25$','FCC-like\n'+r'$1$',r'$4$'])
        ax.set(xlabel=r'$\lambda_{\rm crack}$',ylabel=r'Difference in elastic $J$ from FE (%)',
               title='(c) Why the axis conversion matters')
        ax.legend(frameon=False,fontsize=7,loc='upper left')
        for ax in axes:
            ax.grid(alpha=.18);ax.set_axisbelow(True)
        fig.subplots_adjust(left=.06,right=.99,bottom=.26,top=.9,wspace=.37)
        for extension in ('png','pdf','svg'):
            fig.savefig(directory/('ortega-verification.'+extension),dpi=220)
        plt.close(fig)
    if digest(paper)!=before:
        raise ValueError('Published source changed during read-only digitisation')
    metadata={'source_pdf':str(paper),'source_sha256':before,'source_figure':'Fig7, printed page89 (PDF page5)',
              'source_doi':'10.1016/j.compositesa.2014.04.012',
              'published_data':'213 FEM marker centres recovered from original vector paths; not generated equation samples',
              'coordinate_calibration':{'x_data':[.1,1.],'x_pdf_pt':[87.727,279.059],
                                        'Kbar_data':[0.,250.],'y_pdf_top_pt':[215.138,73.284]},
              'graphical_uncertainty':'CSV preserves original marker half-height as conservative ordinate uncertainty; finite publication drawing precision remains.',
              'implementation':'Current corrected Eq16 returns f=K B sqrt(W)/P; published Fig7 uses Kbar=f/sqrt(a/W).',
              'independent_FE':'Current study18 solved C(T) jobs/9 designed in-domain SPD test tensors; not experimental or published raw FE data.',
              'axis_example':choices,'own_FE_source_sha256':digest(data_path),
              'published_comparison_max_absolute_relative_error':max(abs(row['relative_difference']) for row in points),
              'no_parameters_fitted':True,'no_new_jobs_submitted':True,
              'scope':'Published equation/FEM reproduction and independent elastic numerical verification only; not plastic-J or lattice-failure validation.'}
    (directory/'ortega-verification.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (directory/'ortega-verification-caption.txt').write_text(
        'Elastic C(T) verification. (a) Original FEM marker centres from Ortega et al. (2014), Fig.7, '
        'compared with the current implementation of the corrected published equation. The published ordinate '
        'is Kbar=f/sqrt(a/W); source lambda and chi are in Ortega loading-axis order. For readability every '
        'fifth marker is drawn; all213 recovered markers and their vector positions are retained in the CSV. '
        '(b) The published equation is compared with independently solved current-study FE compliance derivatives '
        'for nine designed in-domain tensors at a/W=.5; these are numerical verification cases, not invented '
        'experimental observations or published raw data. (c) Using the source factor with the crack-axis modulus '
        'mixes normalisations for unequal axes. Source-paired fO/MO and converted fB/MB preserve elastic J. '
        'Equal axial moduli, including the FCC-like case, need no numerical conversion. These elastic comparisons '
        'do not validate plastic eta, negative-area partitions or lattice-failure criteria.\n',encoding='utf-8')
    return metadata


def extract(out=ROOT):
    """Read settled delivered isotropic controls; do not submit or extrapolate."""
    from odbAccess import openOdb
    out = Path(out)
    manifest = json.loads((out/"manifest.json").read_text())
    result = []
    for job in manifest["jobs"]:
        source,odb_path,before,receipt = verified_delivery(out,manifest,job)
        odb = openOdb(str(odb_path),readOnly=True)
        try:
            pin = odb.rootAssembly.nodeSets["PIN"]
            rows = []
            for item in source["steps"]:
                step = odb.steps[item["step"]]
                frame = step.frames[-1]
                if abs(float(step.timePeriod)-1.)>1e-12 or abs(float(frame.frameValue)-1.)>1e-10:
                    raise ValueError('Calibration equilibrium step is incomplete')
                values = frame.fieldOutputs["U"].getSubset(region=pin).values
                if len(values) != 1:
                    raise ValueError("Expected one load-point displacement")
                u = float(values[0].data[1])
                reaction = sum(float(value.data[1]) for value in frame.fieldOutputs["RF"].values)
                energy_regions = [region for region in step.historyRegions.values()
                                  if "ALLSE" in region.historyOutputs]
                if len(energy_regions) != 1:
                    raise ValueError("Expected one global elastic-energy region")
                outputs = energy_regions[0].historyOutputs
                strain_energy = float(outputs["ALLSE"].data[-1][1])
                artificial = float(outputs["ALLAE"].data[-1][1])
                expected_energy = .5*source["P_N"]*u
                if (not all(math.isfinite(x) for x in (u,reaction,strain_energy,artificial))
                        or min(u,strain_energy,expected_energy)<=0
                        or abs(reaction+source['P_N'])>1e-5*source['P_N']):
                    raise ValueError('Nonfinite/nonpositive or unbalanced calibration equilibrium')
                rows.append(dict(item,u_upper_mm=u,C_half_mm_per_N=u/source["P_N"],
                                 vertical_reaction_N=reaction,
                                 force_balance_error_N=reaction+source["P_N"],
                                 ALLSE_N_mm=strain_energy,ALLAE_N_mm=artificial,
                                 half_Pu_N_mm=expected_energy,
                                 elastic_energy_relative_difference=abs(strain_energy+artificial-expected_energy)/expected_energy,
                                 artificial_to_strain_energy_ratio=artificial/strain_energy))
            metrics = elastic_factor([row["C_half_mm_per_N"] for row in rows],
                                     source["W_mm"],source["B_mm"],source["M_MPa"],source["P_N"],
                                     [row['a_over_W'] for row in rows])
            orthotropic='case_id' in source
            expected = None if orthotropic else 9.659078631008239
            result.append({"job":job["job"], "input_sha256":job["input_sha256"],
                           **receipt,
                           "odb":str(odb_path), "odb_sha256":before, "steps":rows,
                           **metrics, "ASTM_f_at_a_over_W_0p5":expected,
                           "relative_difference_from_ASTM":metrics["f_fine"]/expected-1 if expected else None,
                           "case_id":source.get('case_id'),"mesh":source.get('mesh'),
                           "acceptance_pending":"Check force/energy and difference-step/mesh convergence before use."})
        finally:
            odb.close()
        if digest(odb_path) != before:
            raise ValueError("Delivered isotropic ODB changed during extraction")
    name='orthotropic-results.json' if any(row['case_id'] for row in result) else 'isotropic-results.json'
    (out/name).write_text(json.dumps(result,indent=2)+"\n")
    return result


def qualify(out=ROOT):
    """Freeze one factor per source case after measured numerical checks."""
    out=Path(out)
    base_path=out/'orthotropic/orthotropic-results.json'
    base=json.loads(base_path.read_text())
    refinement_path=out/'increment-refinement/orthotropic-results.json'
    refinement=json.loads(refinement_path.read_text()) if refinement_path.exists() else []
    isotropic_path=out/'isotropic-results.json'
    isotropic=json.loads(isotropic_path.read_text())
    if (len(isotropic)!=2 or max(abs(r['relative_difference_from_ASTM']) for r in isotropic)>.01
            or abs(isotropic[1]['f_fine']/isotropic[0]['f_fine']-1)>.01):
        raise ValueError('Independent isotropic factor verification failed')
    results=[]
    for case in sorted({row['case_id'] for row in base}):
        pair=[row for row in base if row['case_id']==case]
        if len(pair)!=2 or {r['mesh'] for r in pair}!={'medium','fine'}:
            raise ValueError('Exactly two source-matched meshes required')
        medium=next(r for r in pair if r['mesh']=='medium')
        fine=next(r for r in pair if r['mesh']=='fine')
        extra=[r for r in refinement if r['case_id']==case]
        if len(extra)>1:raise ValueError('Conflicting refinement results')
        selected=extra[0] if extra else fine
        mesh_error=abs(selected['f_fine']/medium['f_fine']-1)
        if (mesh_error>.01 or selected['difference_step_relative_change']>.005
                or any(step['elastic_energy_relative_difference']>1e-5
                       or step['artificial_to_strain_energy_ratio']>.05
                       for item in pair+extra for step in item['steps'])):
            raise ValueError('Numerical calibration not qualified: '+case)
        results.append({'case_id':case,'factor':selected['f_fine'],
                        'method':'direct_elastic_CT_half_compliance_derivative',
                        'selected_job':selected['job'],'selected_odb_sha256':selected['odb_sha256'],
                        'medium_mesh_factor':medium['f_fine'],'fine_mesh_factor':fine['f_fine'],
                        'selected_vs_medium_factor_relative_change':mesh_error,
                        'selected_crack_derivative_relative_change':selected['difference_step_relative_change'],
                        'increment_refinement_used':bool(extra),
                        'selected_queue_status':selected['queue_recorded_status'],
                        'selected_calculation_usable':selected.get('calculation_usable',True),
                        'a_over_W':.5,'state':'numerically_qualified_elastic_factor'})
    if len(results)!=8:raise ValueError('Eight calibrated source cases required')
    result={'state':'complete','factors':results,
            'qualification_limits':{'factor_vs_ASTM_isotropic':.01,'factor_mesh_change':.01,
                                    'compliance_derivative_increment_change':.005,
                                    'elastic_energy_closure':1e-5,'artificial_to_elastic_energy':.05},
            'limit_origin':'Declared numerical engineering checks, not an ASTM qualification certificate.',
            'isotropic_verification':isotropic,'source_results_sha256':digest(base_path),
            'refinement_sha256':digest(refinement_path) if refinement else None,
            'scope':'Linear elastic continuum factors for the same reduced source tensors; no nonlinear plastic eta calibration or fracture re-simulation.'}
    (out/'calibration-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=ROOT)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare",action="store_true")
    action.add_argument("--extract",action="store_true")
    action.add_argument("--prepare-orthotropic",action="store_true")
    action.add_argument("--qualify",action="store_true")
    action.add_argument("--prepare-fit-verification",action="store_true")
    action.add_argument("--compare-fit-verification",action="store_true")
    action.add_argument("--plot-verification",action="store_true")
    action.add_argument("--digitise-published",action="store_true")
    arguments = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else sys.argv[1:]
    # CAE inserts its own launcher flags before the user arguments.
    args = parser.parse_known_args(arguments)[0] if "-cae" in arguments else parser.parse_args(arguments)
    print(json.dumps(prepare(args.out) if args.prepare else
                     prepare_orthotropic(args.out) if args.prepare_orthotropic else
                     prepare_fit_verification(args.out) if args.prepare_fit_verification else
                     compare_fit_verification(args.out) if args.compare_fit_verification else
                     plot_fit_verification(args.out) if args.plot_verification else
                     digitise_figure7(args.out) if args.digitise_published else
                     qualify(args.out) if args.qualify else extract(args.out),indent=2))
