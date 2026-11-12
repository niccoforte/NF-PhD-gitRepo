"""Source-bound Omidi2023 snub-square C(T) reproduction; preparation never submits.

The authors' published CT dimensions, measured constituent curve and original
snub-square wire pattern are used. The released STL verifies phase, notch and
pin geometry; its density-labelled thicknesses differ from paper Table2, so
the paper's .45/.56/.67mm sections remain explicit. Final paper figures are
outside this helper's output directory.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import csv
import struct

import numpy as np

ROOT = Path("Z:/p1/sims/Ti/Validation/audit/omidi")
PERIOD = (1+math.sqrt(3))*6.
SECTION_T = {.2: .45, .25: .56, .3: .67}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2)+"\n"
    if immutable and path.exists() and path.read_text(encoding="utf-8") != text:
        raise ValueError("Refuse changed prepared provenance: "+str(path))
    path.write_text(text, encoding="utf-8")


def author_cell(length=6.):
    """Literal20line snub-square motif in authors' Snub_Square.py, no Mohr model."""
    q, p = math.sqrt(3), 1+math.sqrt(3)
    first = [((0,0),(q/2,.5)),((0,0),(q/2,-.5)),((q/2,-.5),(q/2,.5)),
             ((q/2,-.5),(q,0)),((q/2,.5),(q,0))]
    second = [((q,0),(q+.5,q/2)),((q,0),(q+.5,-q/2)),((q,0),(p,0)),
              ((q+.5,q/2),(p,0)),((q+.5,-q/2),(p,0))]
    shift1, shift2 = np.array([p/2,p/2]), np.array([-p/2,p/2])
    lines = [np.asarray(edge, dtype=float) for edge in first+second]
    lines += [np.asarray(edge)+shift1 for edge in first]
    lines += [np.asarray(edge)+shift2 for edge in second]
    return [line*length for line in lines]


def clip_line(edge, lower, upper):
    a, b = np.asarray(edge, dtype=float)
    vector = b-a
    lo, hi = 0., 1.
    for axis in (0,1):
        if abs(vector[axis]) < 1e-12:
            if not lower[axis]-1e-10 <= a[axis] <= upper[axis]+1e-10:
                return None
        else:
            values = sorted(((lower[axis]-a[axis])/vector[axis],(upper[axis]-a[axis])/vector[axis]))
            lo, hi = max(lo,values[0]), min(hi,values[1])
    return np.array([a+lo*vector,a+hi*vector]) if hi-lo > 1e-9 else None


def periodic_lines(nx=15, ny=11):
    """Original motif tiled and split at every existing endpoint; deduplicated."""
    from scipy.spatial import cKDTree
    upper=np.array([nx*PERIOD,ny*PERIOD])
    lines=[]
    for i in range(-1,nx+1):
        for j in range(-1,ny+1):
            for edge in author_cell():
                # The released printed C(T) uses the same motif with a half-bar
                # horizontal shift relative to the boundary-layer generator.
                clipped=clip_line(edge+np.array([i,j])*PERIOD+np.array([3.,0.]),np.zeros(2),upper)
                if clipped is not None:lines.append(clipped)
    lines += [np.array(edge,float) for edge in [((0,0),(upper[0],0)),
              ((0,upper[1]),upper),((0,0),(0,upper[1])),((upper[0],0),upper)]]
    points=np.unique(np.round(np.asarray(lines).reshape(-1,2),8),axis=0)
    tree=cKDTree(points)
    split={}
    for a,b in lines:
        v=b-a;length=np.linalg.norm(v)
        ids=tree.query_ball_point((a+b)/2,length/2+1e-6)
        q=points[ids];t=(q-a)@v/(length*length)
        good=(t>=-1e-7)&(t<=1+1e-7)&(np.linalg.norm(q-(a+t[:,None]*v),axis=1)<1e-6)
        ts=sorted(set(np.round(np.clip(t[good],0,1),10)))
        for left,right in zip(ts[:-1],ts[1:]):
            edge=np.round([a+left*v,a+right*v],8)
            key=tuple(sorted(tuple(x) for x in edge))
            if np.linalg.norm(edge[1]-edge[0])>1e-6:split[key]=edge
    return list(split.values())


def stl_oracle(path):
    """Point membership in the exact front-face STL triangles, not a raster."""
    import matplotlib.tri as mtri
    data=Path(path).read_bytes()
    count=struct.unpack("<I",data[80:84])[0]
    dtype=np.dtype([("normal","<f4",(3,)),("v","<f4",(3,3)),("attribute","<u2")])
    records=np.frombuffer(data,dtype=dtype,offset=84,count=count)
    front=(abs(records["normal"][:,1])>.9)&np.all(abs(records["v"][:,:,1])<1e-5,axis=1)
    triangles=records["v"][front][...,[0,2]].astype(float)
    points,inverse=np.unique(triangles.reshape(-1,2),axis=0,return_inverse=True)
    cells=inverse.reshape(-1,3)
    area=np.cross(points[cells[:,1]]-points[cells[:,0]],points[cells[:,2]]-points[cells[:,0]])
    cells=cells[abs(area)>1e-10]
    finder=mtri.Triangulation(points[:,0],points[:,1],triangles=cells).get_trifinder()
    # 15x11period centre dimensions plus twice the bar half-width.
    width=float(records["v"][:,:,0].max())
    half=(width-15*PERIOD)/2
    centers=np.array([[3*PERIOD,1.5*PERIOD],[3*PERIOD,9.5*PERIOD]])
    def inside(xy):
        xy=np.atleast_2d(xy)
        solid=finder(xy[:,0]+half,xy[:,1]+half)>=0
        outside_pins=np.min(np.linalg.norm(xy[:,None,:]-centers[None,:,:],axis=2),axis=1)>=7.-1e-7
        return solid&outside_pins
    return inside,half,centers


def ct_graph(stl):
    inside,half,centers=stl_oracle(stl)
    source=periodic_lines()
    kept=[]
    for edge in source:
        a,b=edge;v=b-a
        ts=np.linspace(0,1,65)
        good=inside(a+ts[:,None]*v)
        runs=np.flatnonzero(np.diff(np.r_[False,good,False].astype(int)))
        for begin,end in zip(runs[::2],runs[1::2]):
            lo,hi=ts[begin],ts[end-1]
            if begin:
                left,right=ts[begin-1],lo
                for _ in range(32):
                    mid=(left+right)/2
                    if inside(a+mid*v)[0]:right=mid
                    else:left=mid
                lo=right
            if end<len(ts):
                left,right=hi,ts[end]
                for _ in range(32):
                    mid=(left+right)/2
                    if inside(a+mid*v)[0]:left=mid
                    else:right=mid
                hi=left
            if (hi-lo)*np.linalg.norm(v)>1e-4:
                kept.append(np.round([a+lo*v,a+hi*v],7))
    unique={tuple(sorted(tuple(p)for p in e)):e for e in kept}
    lines=list(unique.values())
    points,inverse=np.unique(np.asarray(lines).reshape(-1,2),axis=0,return_inverse=True)
    edges=inverse.reshape(-1,2)
    # The printed specimen is one connected solid; ignore only isolated wire slivers.
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    adjacency=coo_matrix((np.ones(edges.size),(edges.ravel(),edges[:,::-1].ravel())),shape=(len(points),len(points)))
    nc,labels=connected_components(adjacency,directed=False)
    counts=np.bincount(labels)
    keep=labels==int(np.argmax(counts))
    discarded=int(np.count_nonzero(~keep))
    if discarded>.01*len(points):raise ValueError("Wire graph disconnected; source reconstruction needs review")
    lines=[points[edge]for edge in edges if keep[edge].all()]
    proof={"STL_sha256":sha(stl),"STL_halfwidth_mm":half,"source_segments":len(source),
           "retained_segments":len(lines),"disconnected_nodes_discarded":discarded,
           "pin_centers_mm":centers.tolist(),"pin_rigid_region_radius_mm":7.,
           "W_mm":12*PERIOD,"a_mm":3*PERIOD,"H_mm":11*PERIOD,"overall_x_mm":15*PERIOD,
           "notch_phase":"Exact author STL solid membership on source-snubsquare centrelines",
           "pin_idealisation":"Rigid pin/ring region; translations coupled with freely rotating pin RP"}
    return lines,proof


def mesh_graph(lines, per_bar=20):
    nodes={};lookup={};elements=[];ends=[]
    def node(xy):
        key=tuple(np.round(xy,7))
        if key not in lookup:
            label=len(lookup)+1;lookup[key]=label;nodes[label]=key
        return lookup[key]
    for a,b in lines:
        count=max(1,int(math.ceil(per_bar*np.linalg.norm(b-a)/6.-1e-6)))
        labels=[node(xy)for xy in np.linspace(a,b,count+1)]
        elements.extend(zip(labels[:-1],labels[1:]))
        ends.extend((labels[0],labels[-1]))
    return nodes,elements,sorted(set(ends))


def measured_plastic(source):
    """Measured Fig4 true stress/strain, strain-binned median and monotone envelope."""
    import openpyxl
    sheet=openpyxl.load_workbook(source,data_only=True).worksheets[0]
    pairs=[]
    for row in sheet.iter_rows(min_row=2,values_only=True):
        e,s=row[12],row[13]
        if isinstance(e,(int,float))and isinstance(s,(int,float))and 0<e<.14 and s>0:
            pairs.append((e,s))
    raw=np.array(pairs)
    samples=[]
    bins=np.arange(0,.1401,.0005)
    for low,high in zip(bins[:-1],bins[1:]):
        rows=raw[(raw[:,0]>=low)&(raw[:,0]<high)]
        if len(rows):samples.append(np.median(rows,axis=0))
    curve=np.asarray(samples)
    curve[:,1]=np.maximum.accumulate(curve[:,1])
    plastic=curve[:,0]-curve[:,1]/2000.
    good=plastic>1e-6
    curve,plastic=curve[good],plastic[good]
    increasing=np.r_[True,np.diff(np.maximum.accumulate(plastic))>1e-8]
    curve,plastic=curve[increasing],plastic[increasing]
    table=[(float(curve[0,1]),0.)]+[(float(s),float(e))for s,e in zip(curve[:,1],plastic)]
    # Drop an equal-stress near-duplicate first row; hardening strains stay strictly increasing.
    table=[row for i,row in enumerate(table) if i==0 or row[1]>table[i-1][1]]
    return table,{"source":str(source),"sha256":sha(source),"physical_E_MPa":2000.,
                  "physical_nu":.3,"raw_points":len(raw),"plastic_table_points":len(table),
                  "processing":".0005 true-strain-bin median; nondecreasing stress envelope; plasticstrain=trueStrain-stress/2000",
                  "failure_total_tensile_LE11":.1}


def nset(name,labels):
    return "*Nset,nset="+name+"\n"+"".join(",".join(map(str,labels[i:i+16]))+"\n"for i in range(0,len(labels),16))


def prepare(out=ROOT):
    out=Path(out)
    source=out/'source'
    lines,geometry=ct_graph(source/'Snub Square_rho0.20.STL')
    nodes,edges,ends=mesh_graph(lines)
    plastic,material=measured_plastic(source/'Fig 4.xlsx')
    centers=np.array(geometry['pin_centers_mm'])
    pin_nodes=[]
    for center in centers:
        labels=[label for label in ends if abs(np.linalg.norm(np.asarray(nodes[label])-center)-7.)<1e-4]
        if len(labels)<4:raise ValueError('Insufficient pin-contact beam endpoints')
        pin_nodes.append(labels)
    lower,upper=len(nodes)+1,len(nodes)+2
    records=[]
    for rho,t in SECTION_T.items():
        job='omidi_ss_r'+str(int(round(rho*100)))
        directory=out/job;directory.mkdir(parents=True,exist_ok=True)
        text='*Heading\n** Source-bound Omidi2023 snub-square CT, measured parent curve\n*Node\n'
        text+=''.join('%d,%.12g,%.12g\n'%(label,*xy)for label,xy in nodes.items())
        text+=''.join('%d,%.12g,%.12g\n'%(label,*xy)for label,xy in zip((lower,upper),centers))
        text+='*Element,type=B21,elset=Body\n'+''.join('%d,%d,%d\n'%(i+1,*edge)for i,edge in enumerate(edges))
        text+='*Beam Section,elset=Body,material=Polymer,section=RECT\n15.,%.12g\n0.,0.,1.\n'%t
        text+='*Material,name=Polymer\n*Elastic\n%.12g,%.12g\n'%(2000/(1-.3**2),.3/(1-.3))
        text+='*Plastic\n'+''.join('%.12g,%.12g\n'%row for row in plastic)
        text+=nset('Load',[upper])+nset('Fixity',[lower])
        for rp,center,labels in zip((lower,upper),centers,pin_nodes):
            for label in labels:
                x,y=np.asarray(nodes[label])-center
                text+='*Equation\n3\n%d,1,1.,%d,1,-1.,%d,6,%.12g\n'%(label,rp,rp,y)
                text+='*Equation\n3\n%d,2,1.,%d,2,-1.,%d,6,%.12g\n'%(label,rp,rp,-x)
        text+='*Step,name=Loading,nlgeom=YES,inc=10000\n*Static\n.005,1.,1.e-8,.005\n'
        text+='*Boundary\nFixity,1,2\nLoad,1,1\nLoad,2,2,12.\n'
        text+='*Output,field,frequency=1\n*Element Output,elset=Body,directions=YES\nLE,S\n'
        text+='*Output,history,frequency=1\n*Node Output,nset=Load\nU2,RF2\n'
        text+='*Node Output,nset=Fixity\nU2,RF2\n*Energy Output\nALLSE,ALLPD,ALLWK\n*End Step\n'
        inp=directory/(job+'.inp')
        if inp.exists()and inp.read_text(encoding='ascii')!=text:raise ValueError('Refuse modified prepared model')
        inp.write_text(text,encoding='ascii')
        meta=dict(geometry,job=job,input=str(inp),input_sha256=sha(inp),relative_density=rho,
                  B_mm=15.,section_t_mm=t,l_mm=6.,elements=len(edges),nodes=len(nodes),
                  pin_endpoint_counts=list(map(len,pin_nodes)),lower_RP=lower,upper_RP=upper,
                  maximum_opening_mm=12.,material=material,
                  assigned_E_MPa=2000/(1-.3**2),assigned_nu=.3/(1-.3),
                  plane_state='Released author beam convention: mapped constituent E/(1-nu²),nu/(1-nu); measured plastic strains preserved',
                  source_generator_sha256=sha(source/'Snub_Square.py'),
                  source_CAD_discrepancy='Density-labelled STL thicknesses differ from paperTable2; CAD supplies centreline topology/phase, Table2 supplies assigned sections',
                  criterion='First maximum surface tensile LE11 reaches .1; record global location and pre/cross bracket',
                  state='prepared_not_submitted')
        save_json(directory/'source.json',meta,immutable=True);records.append(meta)
    save_json(out/'prepared-CT.json',records,immutable=True)
    manifest={'abaqus':'C:/SIMULIA/Commands/abaqus.bat','queue_root':str(out/'queue'),
              'jobs':[{'job':r['job'],'input':r['input'],'input_sha256':r['input_sha256'],
                       'cpus':8,'required_outputs':['.odb','.dat','.sta'],'purpose':'Published snub-square CT reproduction'}for r in records]}
    save_json(out/'ct-manifest.json',manifest,immutable=True)
    print(json.dumps({'manifest':str(out/'ct-manifest.json'),'elements':len(edges),'nodes':len(nodes),
                      'pin_nodes':list(map(len,pin_nodes)),'geometry':geometry},indent=2))
    return records


def prepare_tensor(out=ROOT):
    """Three small source-matched periodic B21 controls, not an axial Mohr sum."""
    from validation_diagnostics import periodic_patch, prepare_rve
    out=Path(out); target=out/'tensor'; target.mkdir(exist_ok=True)
    nodes,elements,_=mesh_graph(periodic_lines(5,5))
    rows=[[i+1,*edge]for i,edge in enumerate(elements)]
    coords,edges,classes,periods,origin,mismatch=periodic_patch(nodes,rows,'square',PERIOD)
    records=[]
    for density,thickness in SECTION_T.items():
        name='omidi_ss_r'+str(round(100*density));ct=out/name/(name+'.inp')
        case={'topology':'snubsquare','l_mm':'6','nominal_relative_density':str(density),
              'nnx':'1','case_id':'omidi-ss-'+str(density)}
        meta=prepare_rve(target,case,ct,coords,edges,classes,periods,[15.,thickness],
                         [2000/(1-.3**2),.3/(1-.3)])
        meta.update(source_topology='snub-square',origin_mm=origin.tolist(),
                    opposite_boundary_mismatch_mm=mismatch,
                    material_state='Author mapped constituent; use C directly without another plane-strain reduction',
                    reason='Full normal/shear stiffness including internal translation/rotation and beam bending/shear; no D-Mohr axial-affine calculation',
                    estimate_basis='Approximately400 B21 elements and three linear perturbation steps; planning allowance only',
                    estimated_wall_seconds=30,
                    published_closed_form={'E_over_Es':.280*density,'G_over_Es':.443*density**3,'nu':.433},
                    published_TableC2_lambda=1.,published_TableC2_chi={.2:7.65,.25:4.74,.3:3.16}[density])
        save_json(Path(meta['input']).parent/'source.json',meta)
        records.append(meta)
    manifest={'abaqus':'C:/SIMULIA/Commands/abaqus.bat','queue_root':str(out/'tensor-queue'),
              'jobs':[{'job':m['job'],'input':m['input'],'input_sha256':m['input_sha256'],'cpus':8,
                       'required_outputs':['.odb','.dat','.sta'],'purpose':'Source-matched snub-square periodic tensor; three independent strain steps'}for m in records]}
    save_json(out/'tensor-manifest.json',manifest,immutable=True)
    save_json(out/'prepared-tensor.json',records)
    print(json.dumps([{'job':m['job'],'nodes':m['nodes'],'elements':m['elements']}for m in records],indent=2))
    return records


def tensor_properties(C):
    """Engineering shear convention, x along crack and y along opening."""
    C=np.asarray(C,float); symmetry=float(np.max(abs(C-C.T))/np.max(abs(C)))
    if C.shape!=(3,3) or symmetry>1e-5 or np.linalg.eigvalsh((C+C.T)/2).min()<=0:
        raise ValueError('Tensor must be symmetric positive definite')
    S=np.linalg.inv(C);lam=float(S[0,0]/S[1,1])
    chi=float((2*S[0,1]+S[2,2])/(2*np.sqrt(S[0,0]*S[1,1])))
    return dict(C_MPa=C.tolist(),S_MPa_inv=S.tolist(),lambda_crack_axis=lam,chi=chi,
                M_crack_axis_MPa=float(np.sqrt(2*lam**1.5/(1+chi))/S[0,0]),
                E_x_MPa=float(1/S[0,0]),E_y_MPa=float(1/S[1,1]),
                nu_xy=float(-S[0,1]/S[0,0]),G_xy_MPa=float(1/S[2,2]),
                symmetry_relative_error=symmetry)


def gather_tensor(out=ROOT,queue=None):
    """Abaqus Python: receipt-bound read-only tensor extraction with energy check."""
    from odbAccess import openOdb
    from validation_controls import verified_control_delivery
    out=Path(out);queue=Path(queue or out/'tensor-queue')
    completed={r['job']for r in json.loads((queue/'state.json').read_text())['jobs']if r['status']=='completed'}
    results=[]
    for path in sorted((out/'tensor').glob('rve_*/source.json')):
        meta=json.loads(path.read_text())
        if meta['job']not in completed:continue
        odb_path,receipt=verified_control_delivery(meta,queue,completed)
        before=sha(odb_path);odb=openOdb(str(odb_path),readOnly=True)
        columns=[];errors=[]
        try:
            for i,label in enumerate(('Exx','Eyy','Gxy')):
                step=odb.steps[label];frame=step.frames[-1]
                rf={v.nodeLabel:float(v.data[0])for v in frame.fieldOutputs['RF'].values}
                reaction=np.array([rf[n]for n in meta['macro_nodes']])
                columns.append(reaction/(meta['volume_mm3']*meta['macrostrain']))
                energy=[reg.historyOutputs['ALLSE'].data[-1][1]for reg in step.historyRegions.values()if 'ALLSE'in reg.historyOutputs]
                if len(energy)!=1:raise ValueError('Unique RVE energy required')
                predicted=.5*reaction[i]*meta['macrostrain']
                error=float(abs(energy[0]-predicted)/max(abs(predicted),1e-30));errors.append(error)
                if error>1e-5:raise ValueError('Macro-reaction/stored-energy mismatch')
        finally:odb.close()
        if sha(odb_path)!=before:raise ValueError('ODB changed during read-only extraction')
        result=dict(meta,**tensor_properties(np.column_stack(columns)),
                    energy_relative_errors=errors,odb=str(odb_path),odb_sha256=before,
                    delivery_sha256=sha(receipt),read_only=True,state='completed_and_extracted')
        patch=np.load(path.parent/'patch.npz');lengths=np.linalg.norm(patch['coords'][patch['edges'][:,1]]-patch['coords'][patch['edges'][:,0]],axis=1)
        result['beam_volume_relative_density']=float(lengths.sum()*meta['section_B_t_mm'][1]/np.prod(meta['periods_mm']))
        save_json(path.parent/'result.json',result);results.append(result)
    save_json(out/'tensor-results.json',results)
    print(json.dumps([{k:r[k]for k in ('job','E_x_MPa','G_xy_MPa','lambda_crack_axis','chi')}for r in results],indent=2))
    return results


def checked_delivery(meta,queue,accepted):
    from validation_modal import job_directory
    if meta['job']not in accepted or sha(meta['input'])!=meta['input_sha256']:
        raise ValueError('Unreviewed job or changed prepared input')
    run=job_directory(Path(queue)/'runs',meta['job']);receipt=run/'delivery.json'
    delivery=json.loads(receipt.read_text());files={r['file']:r['sha256']for r in delivery['files']}
    if delivery['job']!=meta['job']:raise ValueError('Delivery job identity mismatch')
    for suffix in ('.inp','.odb','.dat'):
        file=run/(meta['job']+suffix)
        if file.name not in files or sha(file)!=files[file.name]:raise ValueError('Changed/incomplete delivery')
    if files[meta['job']+'.inp']!=meta['input_sha256']:raise ValueError('Solved input differs from prepared input')
    return run/(meta['job']+'.odb'),receipt


def gather(out=ROOT,queue=None,reviewed_partial=()):
    """Source-matched first local tensile-strain event; never a global peak index."""
    from odbAccess import openOdb
    out=Path(out);queue=Path(queue or out/'queue');state=json.loads((queue/'state.json').read_text())
    status={r['job']:r['status']for r in state['jobs']};reviewed_partial=set(reviewed_partial)
    if any(status.get(j)!='failed'for j in reviewed_partial):raise ValueError('Only explicitly reviewed failed prefixes may be extracted')
    accepted={j for j,s in status.items()if s=='completed'}|reviewed_partial
    records=[]
    for meta in json.loads((out/'prepared-CT.json').read_text()):
        if meta['job']not in accepted:continue
        for path,digest in ((out/'source/Snub_Square.py',meta['source_generator_sha256']),
                            (out/'source/Snub Square_rho0.20.STL',meta['STL_sha256']),
                            (Path(meta['material']['source']),meta['material']['sha256'])):
            if sha(path)!=digest:raise ValueError('Published source changed')
        odb_path,receipt=checked_delivery(meta,queue,accepted);before=sha(odb_path)
        odb=openOdb(str(odb_path),readOnly=True)
        try:
            step=odb.steps['Loading']
            regions=[region.historyOutputs for name,region in step.historyRegions.items()
                     if name.endswith('.'+str(meta['upper_RP'])) and 'RF2'in region.historyOutputs and 'U2'in region.historyOutputs]
            if len(regions)!=1:raise ValueError('Unique upper pin history required')
            u,rf=regions[0]['U2'].data,regions[0]['RF2'].data
            if len(u)!=len(rf)or any(abs(a[0]-b[0])>1e-10 for a,b in zip(u,rf)):raise ValueError('Unpaired pin histories')
            curve=np.array([[a[0],a[1],b[1]]for a,b in zip(u,rf)])
            rows=[];event=None;engineering_event=None;section_points=None
            for frame in step.frames:
                if 'LE'not in frame.fieldOutputs:continue
                field=frame.fieldOutputs['LE'];values=field.values
                if not values:continue
                if tuple(field.componentLabels)[0]!='LE11':raise ValueError('First saved beam strain component must be LE11')
                if section_points is None:
                    section_points=sorted({int(v.sectionPoint.number)for v in values if v.sectionPoint})
                    if not {1,5}.issubset(section_points):raise ValueError('Both default rectangular-beam surface points1/5 required')
                v=max(values,key=lambda value:value.data[0])
                row={'time':float(frame.frameValue),'max_LE11':float(v.data[0]),'element':int(v.elementLabel),
                     'integration_point':int(v.integrationPoint),'section_point':int(v.sectionPoint.number)if v.sectionPoint else None,
                     'opening_mm':float(np.interp(frame.frameValue,curve[:,0],curve[:,1])),
                     'force_N':float(np.interp(frame.frameValue,curve[:,0],curve[:,2]))}
                rows.append(row)
                if engineering_event is None and row['max_LE11']>=math.log1p(.1):engineering_event={'previous':rows[-2]if len(rows)>1 else None,'crossing':row}
                if row['max_LE11']>=.1:
                    event={'previous':rows[-2]if len(rows)>1 else None,'crossing':row};break
            mesh={n.label:np.array(n.coordinates[:2],float)for inst in odb.rootAssembly.instances.values()for n in inst.nodes}
            el={e.label:e.connectivity for inst in odb.rootAssembly.instances.values()for e in inst.elements}
            for bracket in (event,engineering_event):
                if bracket:
                    for endpoint in ('previous','crossing'):
                        row=bracket[endpoint]
                        if row:row['element_centroid_mm']=np.mean([mesh[n]for n in el[row['element']]],axis=0).tolist()
        finally:odb.close()
        if sha(odb_path)!=before:raise ValueError('ODB changed during read-only extraction')
        directory=Path(meta['input']).parent
        with (directory/'native-curve.csv').open('w',newline='')as f:
            writer=csv.writer(f);writer.writerow(('step_time','opening_mm','force_N'));writer.writerows(curve)
        result=dict(meta,odb=str(odb_path),odb_sha256=before,delivery_sha256=sha(receipt),read_only=True,
                    execution_state=status[meta['job']],native_samples=len(curve),event=event,
                    saved_section_points=section_points,
                    engineering_point_one_event=engineering_event,field_event_history=rows,
                    criterion_measure_limit='Publication does not explicitly distinguish engineering and logarithmic failure strain; both brackets retained',
                    state='completed_and_extracted'if event else 'recorded_prefix_without_initiation')
        save_json(directory/'result.json',result);records.append(result)
    save_json(out/'CT-results.json',records)
    print(json.dumps([{'job':r['job'],'state':r['state'],'event':r['event']}for r in records],indent=2))
    return records


def published_points(source):
    from openpyxl import load_workbook
    workbook=load_workbook(source/'Fig 10.xlsx',read_only=True,data_only=True)
    rows=list(workbook['Sheet2'].iter_rows(values_only=True));workbook.close()
    data=[]
    for density in SECTION_T:
        result={'relative_density':density}
        for label,xcol,ycol in (('published_boundary_layer',0,1),('published_experiment1',8,9),
                                ('published_experiment2',16,17),('published_CT_FE',23,24)):
            match=[r[ycol]for r in rows if len(r)>ycol and isinstance(r[xcol],(float,int))and abs(r[xcol]-density)<1e-8]
            if len(match)!=1:raise ValueError('Unique source point required: '+label+str(density))
            result[label]=float(match[0])
        data.append(result)
    return data


def positive_endpoint_force(endpoint):
    """Require the recorded tensile reaction; never manufacture its sign."""
    try:
        force=float(endpoint['force_N'])
    except (KeyError,TypeError,ValueError) as error:
        raise ValueError('Missing or invalid recorded initiation force') from error
    if not math.isfinite(force) or force<=0:
        raise ValueError('Recorded initiation force must be finite and positive')
    return force


def compare(out=ROOT):
    """Published CT/BL/experiment data versus independently reproduced lattice FE."""
    from resources.calculations import calc_FaW_aniso
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out);reference=published_points(out/'source')
    tensors=json.loads((out/'tensor-results.json').read_text());results=json.loads((out/'CT-results.json').read_text())
    comparison=[];excluded=[]
    for row in results:
        if not row['event']or not row['event']['previous']:
            excluded.append({'job':row['job'],'reason':'No recorded pre-initiation endpoint','plotted':False})
            continue
        endpoint=row['event']['previous']
        try:force=positive_endpoint_force(endpoint)
        except ValueError as error:
            raw=endpoint.get('force_N')
            if isinstance(raw,(float,int))and not math.isfinite(raw):raw=repr(raw)
            excluded.append({'job':row['job'],'reason':str(error),'recorded_force_N':raw,'plotted':False})
            continue
        rho=row['relative_density'];tensor=next(t for t in tensors if abs(float(t['case_id'].split('-')[-1])-rho)<1e-8)
        C=np.array(tensor['C_MPa']);C_loading=C[np.ix_([1,0,2],[1,0,2])]
        f=float(calc_FaW_aniso(row['a_mm'],row['W_mm'],C=C_loading))
        K=force*f/(row['B_mm']*math.sqrt(row['W_mm']*1000))
        normalized=K/(45*math.sqrt(6/1000))
        pub=next(p for p in reference if p['relative_density']==rho)
        item=dict(pub,job=row['job'],K_MPa_sqrt_m=K,normalized_K=normalized,
                  source_Ortega_factor=f,lambda_parameter=tensor['lambda_crack_axis'],chi=tensor['chi'],
                  initiation_opening_mm=endpoint['opening_mm'],initiation_force_N=force,
                  event_location_mm=endpoint['element_centroid_mm'],
                  relative_difference_vs_published_CT=normalized/pub['published_CT_FE']-1,
                  relative_difference_vs_published_BL=normalized/pub['published_boundary_layer']-1,
                  endpoint='Last recorded sample before local surface LE11=.1',
                  tensor_result_sha256=sha(Path(tensor['input']).parent/'result.json'),CT_result_sha256=sha(Path(row['input']).parent/'result.json'))
        comparison.append(item)
    save_json(out/'comparison.json',comparison)
    save_json(out/'comparison-exclusions.json',excluded)
    fig,ax=plt.subplots(figsize=(6.2,4.5),constrained_layout=True)
    x=np.array([r['relative_density']for r in reference])*100
    for label,marker,title in (('published_boundary_layer','s','Omidi–St-Pierre: boundary layer FE'),
                                ('published_CT_FE','^','Omidi–St-Pierre: C(T) FE'),
                                ('published_experiment1','v','Omidi–St-Pierre: experiment 1'),
                                ('published_experiment2','D','Omidi–St-Pierre: experiment 2')):
        ax.plot(x,[r[label]for r in reference],marker,ms=7,mfc='none',mec='black',ls='none',label=title)
    if comparison:ax.plot([100*r['relative_density']for r in comparison],[r['normalized_K']for r in comparison],
                          'o-',color='#c62828',ms=4.5,lw=1,label='Current study: discrete C(T) FE, Ortega')
    ax.set(xlabel='Relative density (%)',ylabel=r'$K_{Ic}/(\sigma_{0s}\sqrt{\ell})$',xticks=[20,25,30])
    ax.legend(frameon=False,fontsize=8);ax.spines[['top','right']].set_visible(False)
    for extension in ('png','svg'):fig.savefig(out/('omidi-lattice-validation.'+extension),dpi=240)
    plt.close(fig)
    print(json.dumps(comparison,indent=2));return comparison


def plot_tensor(out=ROOT):
    """Show independently recovered beam coefficients against published formulas."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out)
    data=sorted(json.loads((out/'tensor-results.json').read_text()),
                key=lambda r:float(r['case_id'].split('-')[-1]))
    if len(data)!=3:raise ValueError('All three verified source-matched tensors required')
    rho=np.array([float(r['case_id'].split('-')[-1])for r in data])
    Es=np.array([r['elastic_MPa_nu'][0]for r in data])
    fig,axes=plt.subplots(1,3,figsize=(10,3.4),constrained_layout=True)
    measured=[np.array([r['E_x_MPa']for r in data])/Es,
              np.array([r['G_xy_MPa']for r in data])/Es,
              np.array([r['chi']for r in data])]
    source=[.280*rho,.443*rho**3,np.array([7.65,4.74,3.16])]
    labels=[r'$E_x/E_s$',r'$G_{xy}/E_s$',r'$\chi$']
    for ax,values,reference,label in zip(axes,measured,source,labels):
        ax.plot(rho*100,reference,'^--',color='black',mfc='none',ms=7,lw=1,
                label='Omidi–St-Pierre: published')
        ax.plot(rho*100,values,'o-',color='#c62828',ms=4.5,lw=1,
                label='Current study: periodic beam FE')
        ax.set(xlabel='Relative density (%)',ylabel=label,xticks=[20,25,30])
        ax.spines[['top','right']].set_visible(False)
    axes[0].legend(frameon=False,fontsize=7.5)
    for extension in ('png','svg'):fig.savefig(out/('omidi-tensor-comparison.'+extension),dpi=240)
    plt.close(fig)
    save_json(out/'tensor-comparison.json',{
        'density':rho.tolist(),'beam_normalized_E':measured[0].tolist(),
        'beam_normalized_G':measured[1].tolist(),'beam_chi':measured[2].tolist(),
        'published_E_coefficient':source[0].tolist(),'published_G_coefficient':source[1].tolist(),
        'published_TableC2_chi':source[2].tolist(),
        'normalization':'Each coefficient uses its own assigned constituent Young modulus; beam FE uses the authors mapped constituent convention',
        'scope':'Independent periodic beam material-property verification; this plot is not fracture initiation validation',
        'difference_cause':'Not isolated. Full B21 finite-thickness shear/bending versus published leading-order coefficients; not an inferred source error.',
        'tensor_results_sha256':sha(out/'tensor-results.json')})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','prepare-tensor','gather-tensor','gather','compare','plot-tensor'))
    parser.add_argument('--out',type=Path,default=ROOT)
    parser.add_argument('--queue','--queue-root',dest='queue',type=Path)
    parser.add_argument('--reviewed-partial',nargs='*',default=[])
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.out)
    elif args.action=='prepare-tensor':prepare_tensor(args.out)
    elif args.action=='gather-tensor':gather_tensor(args.out,args.queue)
    elif args.action=='gather':gather(args.out,args.queue,args.reviewed_partial)
    elif args.action=='compare':compare(args.out)
    elif args.action=='plot-tensor':plot_tensor(args.out)
