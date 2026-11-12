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
import sys

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


def history_at_time(curve,time):
    """Keep ambiguous repeated timestamps unknown rather than choosing a force."""
    curve=np.asarray(curve,float);time=float(time)
    matches=np.flatnonzero(abs(curve[:,0]-time)<=1e-12)
    if len(matches):
        values=curve[matches,1:]
        opening=float(values[0,0])if np.allclose(values[:,0],values[0,0],rtol=1e-8,atol=1e-10)else None
        if not np.isfinite(values).all()or not np.allclose(values,values[0],rtol=1e-8,atol=1e-10):
            return {'opening_mm':opening,'force_N':None,'history_valid':False,
                    'history_issue':'Conflicting native samples at the same timestamp','native_indices':matches.tolist()}
        return {'opening_mm':opening,'force_N':float(values[0,1]),'history_valid':True,
                'history_issue':None,'native_indices':matches.tolist()}
    right=int(np.searchsorted(curve[:,0],time))
    if right==0 or right==len(curve):
        return {'opening_mm':None,'force_N':None,'history_valid':False,'history_issue':'Field outside recorded history','native_indices':[]}
    left=right-1;ends=[history_at_time(curve,float(curve[i,0]))for i in (left,right)]
    if not all(row['history_valid']for row in ends):
        return {'opening_mm':None,'force_N':None,'history_valid':False,'history_issue':'Interpolation touches ambiguous native time','native_indices':[left,right]}
    return {'opening_mm':float(np.interp(time,curve[:,0],curve[:,1])),
            'force_N':float(np.interp(time,curve[:,0],curve[:,2])),'history_valid':True,
            'history_issue':None,'native_indices':[left,right]}


def maximum_beam_tension(field,component='LE11',element_labels=None):
    """Bulk ODB arrays retain element/integration/section identity, avoiding API loops."""
    if tuple(field.componentLabels)[0]!=component:raise ValueError('First saved beam component must be '+component)
    maxima=[];section_points=set()
    for block in field.bulkDataBlocks:
        data=np.asarray(block.data)
        if not len(data):continue
        if not np.isfinite(data).all():raise ValueError('Nonfinite saved beam field data')
        labels=np.asarray(block.elementLabels)
        points=np.asarray(block.integrationPoints)
        if len(labels)!=len(data)or len(points)!=len(data):raise ValueError('Unexpected beam bulk field identity layout')
        section=int(block.sectionPoint.number);section_points.add(section)
        if element_labels is not None:
            keep=np.isin(labels,list(element_labels));data,labels,points=data[keep],labels[keep],points[keep]
            if not len(data):continue
        index=int(np.argmax(data[:,0]))
        maxima.append({'max_'+component:float(data[index,0]),'element':int(labels[index]),
                       'integration_point':int(points[index]),'section_point':section})
    if not maxima:return None,section_points
    if not {1,5}.issubset(section_points):raise ValueError('Both default rectangular-beam surface points1/5 required')
    return max(maxima,key=lambda row:row['max_'+component]),section_points


def gather(out=ROOT,queue=None,reviewed_partial=()):
    """Source-matched first local tensile-strain event; never a global peak index."""
    from odbAccess import openOdb
    out=Path(out);queue=Path(queue or out/'queue');reviewed_partial=set(reviewed_partial)
    state=json.loads((queue/'state.json').read_text())
    status={r['job']:r['status']for r in state['jobs']}
    if any(status.get(j)!='failed'for j in reviewed_partial):raise ValueError('Only explicitly reviewed failed prefixes may be extracted')
    accepted={j for j,s in status.items()if s=='completed'}|reviewed_partial
    if (out/'CT-results.json').exists():
        retained={r['job']for r in json.loads((out/'CT-results.json').read_text())}
        if retained-accepted:raise ValueError('Extraction would drop retained CT identities; explicitly review retained failed prefixes')
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
            if not np.isfinite(curve[:,0]).all()or(np.diff(curve[:,0])<0).any():raise ValueError('Native times must be finite and nondecreasing')
            duplicate_times=np.unique(curve[:-1,0][np.diff(curve[:,0])==0])
            history_issues=[dict(time=float(t),**history_at_time(curve,t))for t in duplicate_times]
            rows=[];event=None;engineering_event=None;section_points=set()
            for frame in step.frames:
                if 'LE'not in frame.fieldOutputs:continue
                maximum,points=maximum_beam_tension(frame.fieldOutputs['LE'])
                section_points.update(points)
                if maximum is None:continue
                row=dict(time=float(frame.frameValue),**maximum,**history_at_time(curve,frame.frameValue))
                rows.append(row)
                if engineering_event is None and row['max_LE11']>=math.log1p(.1):engineering_event={'previous':rows[-2]if len(rows)>1 else None,'crossing':row}
                if row['max_LE11']>=.1:
                    event={'previous':rows[-2]if len(rows)>1 else None,'crossing':row};break
            mesh={n.label:np.array(n.coordinates[:2],float)for inst in odb.rootAssembly.instances.values()for n in inst.nodes}
            el={e.label:e.connectivity for inst in odb.rootAssembly.instances.values()for e in inst.elements}
            for row in rows:
                row['element_centroid_mm']=np.mean([mesh[n]for n in el[row['element']]],axis=0).tolist()
        finally:odb.close()
        if sha(odb_path)!=before:raise ValueError('ODB changed during read-only extraction')
        directory=Path(meta['input']).parent
        with (directory/'native-curve.csv').open('w',newline='')as f:
            writer=csv.writer(f);writer.writerow(('step_time','opening_mm','force_N'));writer.writerows(curve)
        result=dict(meta,odb=str(odb_path),odb_sha256=before,delivery_sha256=sha(receipt),read_only=True,
                    execution_state=status[meta['job']],native_samples=len(curve),event=event,
                    saved_section_points=sorted(section_points),native_history_issues=history_issues,
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


def prepare_elastic(out=ROOT):
    """Frozen source CT mesh, elastic-brittle strength criterion, one unit load."""
    from validation_diagnostics import blocks,key
    out=Path(out);target=out/'elastic';target.mkdir(exist_ok=True)
    records=[]
    for source in json.loads((out/'prepared-CT.json').read_text()):
        if sha(source['input'])!=source['input_sha256']:raise ValueError('Source CT input changed')
        job=source['job'].replace('omidi_ss_','omidi_el_');directory=target/job;directory.mkdir(exist_ok=True)
        prefix=[]
        for block in blocks(Path(source['input']).read_text(encoding='ascii')):
            if key(block)=='*step':break
            if key(block)not in ('*heading','*plastic'):prefix.append(block)
        text='*Heading\n** Published snub-square elastic-brittle CT unit-load derivative\n'+''.join(prefix)
        text+='*Step,name=Elastic,perturbation\n*Static\n*Boundary\nFixity,1,2\nLoad,1,1\n*Cload\nLoad,2,1.\n'
        text+='*Output,field\n*Element Output,elset=Body\nS\n*Node Output,nset=Load\nU,RF\n*Node Output,nset=Fixity\nU,RF\n'
        text+='*Output,history\n*Energy Output\nALLSE\n*End Step\n'
        inp=directory/(job+'.inp')
        if inp.exists()and inp.read_text(encoding='ascii')!=text:raise ValueError('Refuse changed elastic input')
        inp.write_text(text,encoding='ascii')
        meta=dict(source,job=job,input=str(inp),input_sha256=sha(inp),source_inp=source['input'],
                  source_sha256=source['input_sha256'],analysis='linear_elastic_brittle_CT',
                  state='prepared_not_submitted',applied_force_N=1.,
                  criterion='First tensile section S11 reaches arbitrary parent strength; unit-strength normalization cancels that strength',
                  changes=['Retain exact source-bound CT nodes, B21 elements, sections, mapped E/nu, pin MPCs',
                           'Remove plasticity and geometrical nonlinearity for published elastic-brittle branch',
                           'Replace prescribed opening by1N upper-pin load; retain lowerxy and upperx constraints'],
                  published_reference={'source':'Omidi2023 Table3 elastic-brittle snub-square','D':.95,'density_exponent':1.5,
                                       'normalization':'KIc/(sigma_fs*sqrt(l))','reference_kind':'published fitted density law, not raw FE markers'})
        save_json(directory/'source.json',meta,immutable=True);records.append(meta)
    save_json(target/'prepared.json',records,immutable=True)
    manifest={'abaqus':'C:/SIMULIA/Commands/abaqus.bat','queue_root':str(out/'elastic-queue'),
              'jobs':[{'job':m['job'],'input':m['input'],'input_sha256':m['input_sha256'],'cpus':8,
                       'required_outputs':['.odb','.dat','.sta'],'purpose':'Published elastic-brittle snub-square discrete CT reproduction'}for m in records]}
    save_json(target/'manifest.json',manifest,immutable=True)
    print(json.dumps({'manifest':str(target/'manifest.json'),'jobs':[r['job']for r in records]},indent=2))
    return records


def prepare_ring_rotation_control(out=ROOT):
    """One explicit check: a rigid ring also constrains attached beam rotations."""
    from validation_diagnostics import blocks,key,rows
    out=Path(out);base=json.loads((out/'elastic/prepared.json').read_text())[0]
    if base['relative_density']!=.2 or sha(base['input'])!=base['input_sha256']:
        raise ValueError('Unchanged20%elastic source required')
    text=Path(base['input']).read_text(encoding='ascii');pairs={}
    for block in blocks(text):
        if key(block)!='*equation':continue
        values=rows(block)[1].split(',');node,rp=int(values[0]),int(values[3])
        if node in pairs and pairs[node]!=rp:raise ValueError('Conflicting source ring coupling')
        pairs[node]=rp
    if len(pairs)!=20:raise ValueError('Expected20source ring attachment nodes')
    additions=''.join('*Equation\n2\n%d,6,1.,%d,6,-1.\n'%(node,rp)for node,rp in sorted(pairs.items()))
    text=text.replace('*Step,name=Elastic,perturbation',additions+'*Step,name=Elastic,perturbation')
    target=out/'ring-rotation';directory=target/'elastic/omidi_el_r20_weld';directory.mkdir(parents=True,exist_ok=True)
    job='omidi_el_r20_weld';inp=directory/(job+'.inp')
    if inp.exists()and inp.read_text(encoding='ascii')!=text:raise ValueError('Refuse changed ring-rotation control')
    inp.write_text(text,encoding='ascii')
    meta=dict(base,job=job,input=str(inp),input_sha256=sha(inp),source_inp=base['input'],
              source_sha256=base['input_sha256'],ring_rotation_pairs=[[n,r]for n,r in sorted(pairs.items())],
              state='prepared_not_submitted',
              changes=['Retain entire unit-load elastic source input; add20theta_endpoint=theta_ringRP equations',
                       'Both ring RPs remain free to rotate; joined source-CAD bars follow that rigid rotation'],
              purpose='Resolve whether an unintended hinge at the declared rigid ring creates the grip stress maximum; not fitted to reference toughness')
    save_json(directory/'source.json',meta,immutable=True);save_json(target/'elastic/prepared.json',[meta],immutable=True)
    manifest={'abaqus':'C:/SIMULIA/Commands/abaqus.bat','queue_root':str(target/'queue'),
              'jobs':[{'job':job,'input':str(inp),'input_sha256':sha(inp),'cpus':8,
                       'required_outputs':['.odb','.dat','.sta'],'purpose':meta['purpose']}]}
    save_json(target/'manifest.json',manifest,immutable=True)
    print(json.dumps({'manifest':str(target/'manifest.json'),'rotation_equations':len(pairs)},indent=2))
    return meta


def prepare_boundary_layer(out=ROOT):
    """CAE only: execute inspected author geometry/K fields; write, never submit."""
    from abaqus import mdb
    from abaqusConstants import ON
    out=Path(out);target=out/'boundary-layer';target.mkdir(exist_ok=True)
    source=out/'source/Snub_Square.py';text=source.read_text()
    if sha(source)!='9561d895af76778cf818108036b6604494ed34fd5fe1f53bf11fd1137c5ddb8f':
        raise ValueError('Released author generator changed')
    substitutions=[('p = 0.001','p = 0.2'),('L = 10.00','L = 6.00'),
                   ('Es = 210000','Es = 2000'),('KI = 0.250*math.sqrt(1000)','KI = 1.0'),('cpus = 4','cpus = 8')]
    for original,replacement in substitutions:
        if text.count(original)!=1:raise ValueError('Expected unique author substitution: '+original)
        text=text.replace(original,replacement)
    ns={name:getattr(math,name)for name in ('sqrt','atan','tan','pi')}
    # No mdb.submit, os.system or process launcher occurs in this inspected source.
    exec(compile(text,str(source),'exec'),ns)
    model=mdb.models['Snub_square'];records=[]
    field_code=text[text.index('b__11 = 1/E'):text.index('print("--- %s seconds ---Before Job')]
    import os
    original_directory=Path.cwd()
    try:
        for density in SECTION_T:
            ns.update(p=density,t=density*6/2.679,E=.28*density*2000,G=.445*density**3*2000)
            model.profiles['Rect'].setValues(b=ns['t'])
            for name in list(model.boundaryConditions.keys()):
                if name.startswith('Node-'):del model.boundaryConditions[name]
            exec(compile(field_code,str(source)+'[boundaryfields]','exec'),ns)
            name='omidi_bl_r'+str(round(100*density))+'_109L';directory=target/name;directory.mkdir(exist_ok=True)
            inp=directory/(name+'.inp')
            if inp.exists():raise ValueError('Refuse to overwrite prepared author boundary-layer input')
            os.chdir(directory)
            analysis=mdb.Job(name=name,model='Snub_square',numCpus=8,numDomains=8)
            analysis.writeInput(consistencyChecking=ON)
            instance=ns['PartF'];part=ns['AssemblePart']
            meta={'job':name,'input':str(inp),'input_sha256':sha(inp),'source_inp':str(source),
                  'source_sha256':sha(source),'source_generator':str(source),'source_generator_sha256':sha(source),
                  'state':'prepared_not_submitted','relative_density':density,'l_mm':6.,'section_B_t_mm':[1.,ns['t']],
                  'assigned_E_MPa':2000/(1-.3**2),'assigned_nu':.3/(1-.3),
                  'physical_E_MPa':2000.,'physical_nu':.3,'prescribed_K_MPa_sqrt_mm':1.,
                  'source_field_E_MPa':ns['E'],'source_field_G_MPa':ns['G'],'source_field_nu':ns['nu'],
                  'source_field_lambda':ns['lam'],'source_field_chi':ns['rho'],
                  'W_mm':ns['W_x'],'H_mm':ns['W_y'],'domain_width_over_l':ns['W_x']/6,
                  'nodes':len(part.nodes),'elements':len(part.elements),'boundary_nodes':len(ns['BoundaryNodes']),
                  'source_script_substitutions':substitutions,
                  'source_invariants':['Exact author motif,40x40period tiling, notch and translated origin',
                                       'Exact author orthotropic K displacement expressions and boundary-node selection',
                                       'B21,10elements/bar, linear Elastic mapped constituent; no plastic law or damage'],
                  'scope':'109l released author domain, compared to published300l study; finite-domain limit remains to be checked',
                  'published_reference':{'source':'Table3 elastic-brittle snub-square','D':.95,'density_exponent':1.5}}
            save_json(directory/'source.json',meta,immutable=True);records.append(meta)
    finally:os.chdir(original_directory)
    save_json(target/'prepared.json',records,immutable=True)
    manifest={'abaqus':'C:/SIMULIA/Commands/abaqus.bat','queue_root':str(target/'queue'),
              'jobs':[{'job':m['job'],'input':m['input'],'input_sha256':m['input_sha256'],'cpus':8,
                       'required_outputs':['.odb','.dat','.sta'],'purpose':'Released author snub-square elastic prescribed-K boundary-layer reproduction'}for m in records]}
    save_json(target/'manifest.json',manifest,immutable=True)
    print(json.dumps({'manifest':str(target/'manifest.json'),'jobs':[{k:m[k]for k in ('job','nodes','elements','source_field_chi')}for m in records]},indent=2))
    return records


def orthotropic_K_displacement(xy,E,G,nu,KI=1.):
    """Equal-axis orthotropic mode-I field with principal complex-root branches."""
    values=np.asarray([E,G,nu,KI],float)
    if not np.isfinite(values).all()or E<=0 or G<=0:raise ValueError('Finite positive field moduli required')
    chi=E/(2*G)-nu
    if chi<=1:raise ValueError('This real-root snub-square field requires chi>1')
    beta1=math.sqrt((chi+1)/2)+math.sqrt((chi-1)/2)
    beta2=math.sqrt((chi+1)/2)-math.sqrt((chi-1)/2)
    points=np.asarray(xy,float);x,y=points[...,0],points[...,1]
    # atan2 preserves both crack-face quadrants. No manual V offset is needed.
    a1=np.arctan2(beta1*y,x);a2=np.arctan2(beta2*y,x)
    r1=np.sqrt(np.hypot(x,beta1*y));r2=np.sqrt(np.hypot(x,beta2*y))
    scale=2*KI/(E*math.sqrt(2*math.pi)*(beta1-beta2))
    u=scale*((beta1**2+nu)*beta2*r1*np.cos(a1/2)-(beta2**2+nu)*beta1*r2*np.cos(a2/2))
    v=-scale*((beta2+nu*beta1)*beta2*r1*np.sin(a1/2)-(beta1+nu*beta2)*beta1*r2*np.sin(a2/2))
    return np.stack((u,v),axis=-1)


def prepare_corrected_boundary_layer(out=ROOT):
    """Keep source mesh, correct angular branches and match retained mapped tensor."""
    from validation_diagnostics import blocks,key,rows
    out=Path(out);source_records=json.loads((out/'boundary-layer/prepared.json').read_text())
    tensors=json.loads((out/'tensor-results.json').read_text());target=out/'boundary-layer/corrected';target.mkdir(exist_ok=True)
    records=[]
    for source in source_records:
        if sha(source['input'])!=source['input_sha256']:raise ValueError('Unsubmitted source input changed')
        density=source['relative_density'];tensor=next(t for t in tensors if abs(float(t['case_id'].split('-')[-1])-density)<1e-8)
        S=np.array(tensor['S_MPa_inv']);C=np.array(tensor['C_MPa'])
        if abs(S[0,0]/S[1,1]-1)>1e-5 or np.max(abs(C[[0,1],2]))/np.max(abs(C))>1e-6:
            raise ValueError('Equal-axis orthotropic field requires verified source symmetry')
        E=2/(S[0,0]+S[1,1]);nu=-S[0,1]*E;G=1/S[2,2]
        source_blocks=blocks(Path(source['input']).read_text());nodes={};sets={};offset=None
        for block in source_blocks:
            kind=key(block);head=block.splitlines()[0]
            if kind=='*node':
                for line in rows(block):
                    values=line.split(',');nodes[int(values[0])]=np.array(list(map(float,values[1:3])))
            elif kind=='*instance':
                if offset is not None:raise ValueError('Only one source instance supported')
                offset=np.array(list(map(float,rows(block)[0].split(','))))[:2]
            elif kind=='*nset'and 'instance='in head.lower():
                name=next(v.split('=',1)[1].strip()for v in head.split(',')if v.strip().lower().startswith('nset='))
                sets[name]=[int(v)for line in rows(block)for v in line.split(',')if v.strip()]
        if offset is None:raise ValueError('Source assembly translation missing')
        corrected=[];boundary_nodes=set()
        for block in source_blocks:
            kind=key(block)
            if kind=='*boundary':
                lines=[]
                for line in rows(block):
                    values=[v.strip()for v in line.split(',')];labels=sets[values[0]]
                    if len(labels)!=1:raise ValueError('Source BC must select one actual boundary node')
                    node=labels[0];boundary_nodes.add(node)
                    u=orthotropic_K_displacement(nodes[node]+offset,E,G,nu)
                    dof=int(values[1]);lines.append('%s,%s,%s,%.14g\n'%(values[0],values[1],values[2],u[dof-1]))
                block=block.splitlines()[0]+'\n'+''.join(lines)
            elif kind=='*static':block='*Static\n1.,1.,.0001,1.\n'
            elif kind=='*beam section':
                block=block.splitlines()[0]+'\n1.,%.14g\n0.,0.,1.\n'%tensor['section_B_t_mm'][1]
            elif kind=='*elastic':
                block='*Elastic\n%.14g,%.14g\n'%tuple(tensor['elastic_MPa_nu'])
            elif kind=='*output'and 'field'in block.splitlines()[0].lower():
                block='*Output,field\n*Element Output\nS\n*Node Output\nU,RF\n'
            corrected.append(block)
        text=''.join(corrected);job=source['job'].replace('omidi_bl_','omidi_blK_')
        directory=target/job;directory.mkdir(exist_ok=True);inp=directory/(job+'.inp')
        if inp.exists()and inp.read_text(encoding='ascii')!=text:raise ValueError('Refuse changed corrected BL input')
        inp.write_text(text,encoding='ascii')
        meta=dict(source,job=job,input=str(inp),input_sha256=sha(inp),source_inp=source['input'],source_sha256=source['input_sha256'],
                  state='prepared_not_submitted',K_field_E_MPa=E,K_field_G_MPa=G,K_field_nu=nu,
                  K_field_chi=E/(2*G)-nu,K_field_lambda=1.,K_field_material_state='Retained mapped beam tensor; no second plane-strain reduction',
                  tensor_result_sha256=sha(Path(tensor['input']).parent/'result.json'),tensor_result=str(Path(tensor['input']).parent/'result.json'),
                  corrections=['Replace atan principal-quadrant angles by atan2 and remove V quadrant offsets',
                               'Use retained mapped full beam tensor S for field E/G/nu; verified equal-axis orthotropic symmetry',
                               'Use paper Table2 rounded section width and full-precision mapped material to exactly match retained RVE',
                               'One unit-amplitude linear increment; exact source mesh/notch/boundary nodes retained'],
                  source_analytic_section_B_t_mm=source['section_B_t_mm'],
                  section_B_t_mm=tensor['section_B_t_mm'],actual_beam_relative_density=tensor['beam_volume_relative_density'],
                  section_rounding_relative=tensor['section_B_t_mm'][1]/source['section_B_t_mm'][1]-1,
                  corrected_boundary_nodes=len(boundary_nodes))
        save_json(directory/'source.json',meta,immutable=True);records.append(meta)
    save_json(target/'prepared.json',records,immutable=True)
    manifest={'abaqus':'C:/SIMULIA/Commands/abaqus.bat','queue_root':str(target/'queue'),
              'jobs':[{'job':m['job'],'input':m['input'],'input_sha256':m['input_sha256'],'cpus':8,
                       'required_outputs':['.odb','.dat','.sta'],'purpose':'Author discrete BL mesh with verified physical K branches and retained mapped beam tensor'}for m in records]}
    save_json(target/'manifest.json',manifest,immutable=True)
    print(json.dumps({'manifest':str(target/'manifest.json'),'jobs':[{k:r[k]for k in ('job','corrected_boundary_nodes','K_field_chi')}for r in records]},indent=2))
    return records


def prepare_boundary_layer_checks(out=ROOT,journal_phase=False):
    """CAE only: source109l journal refinement and218l domain checks at20%."""
    from abaqus import mdb
    from abaqusConstants import ON,FINER
    import regionToolset,os
    out=Path(out);campaign='journal-phase'if journal_phase else'corrected'
    target=out/'boundary-layer'/('journal-phase/checks'if journal_phase else'checks');target.mkdir(exist_ok=True)
    source=out/'source/Snub_Square.py'
    if sha(source)!='9561d895af76778cf818108036b6604494ed34fd5fe1f53bf11fd1137c5ddb8f':raise ValueError('Author source changed')
    base=json.loads((out/'boundary-layer'/campaign/'prepared.json').read_text())[0]
    tensor=json.loads(Path(base['tensor_result']).read_text());S=np.array(tensor['S_MPa_inv'])
    E=2/(S[0,0]+S[1,1]);nu=-S[0,1]*E;G=1/S[2,2]
    original_directory=Path.cwd();records=[]
    try:
        for label,power,refine in (('near50',2,True),('domain218',3,False)):
            text=journal_phase_source(source.read_text())if journal_phase else source.read_text()
            for original,replacement in [('p = 0.001','p = 0.2'),('L = 10.00','L = 6.00'),
                                          ('Es = 210000','Es = 2000'),('KI = 0.250*math.sqrt(1000)','KI = 1.0'),
                                          ('XX_YY = 2','XX_YY = '+str(power))]:
                if text.count(original)!=1:raise ValueError('Source substitution ambiguous')
                text=text.replace(original,replacement)
            if 'Snub_square'in mdb.models:del mdb.models['Snub_square']
            ns={name:getattr(math,name)for name in ('sqrt','atan','tan','pi')}
            exec(compile(text[:text.index('BoundaryNodes =')],str(source)+'[mesh]','exec'),ns)
            model=mdb.models['Snub_square'];part=ns['AssemblePart'];assembly=ns['TriAssembly']
            model.profiles['Rect'].setValues(b=.45)
            model.materials['Elastic'].elastic.setValues(table=(tuple(tensor['elastic_MPa_nu']),))
            near_count=0
            if refine:
                tip=np.array([ns['ac'],ns['Cy']]);near=[]
                for edge in part.edges:
                    vertices=edge.getVertices()
                    if len(vertices)!=2:raise ValueError('Straight source beam edge required')
                    a,b=[np.array(part.vertices[i].pointOn[0][:2])for i in vertices]
                    direction=b-a;t=float(np.clip(np.dot(tip-a,direction)/np.dot(direction,direction),0,1))
                    if np.linalg.norm(a+t*direction-tip)<=180.+1e-6:near.append(edge)
                near_count=len(near);part.deleteMesh()
                part.seedEdgeBySize(edges=tuple(near),size=6/50.,constraint=FINER)
                part.generateMesh();assembly.regenerate()
            exec(compile(text[text.index('BoundaryNodes ='):text.index('b__11 =')],str(source)+'[boundaryselection]','exec'),ns)
            instance=ns['PartF'];seen=set()
            for node in ns['BoundaryNodes']:
                if node.label in seen:continue
                seen.add(node.label);xy=node.coordinates[:2];u=orthotropic_K_displacement(xy,E,G,nu)
                region=regionToolset.Region(nodes=instance.nodes.sequenceFromLabels(labels=(node.label,)))
                model.DisplacementBC(name='K-'+str(node.label),createStepName='DisplStep',region=region,u1=float(u[0]),u2=float(u[1]))
            model.steps['DisplStep'].setValues(initialInc=1.,maxInc=1.)
            model.fieldOutputRequests['F-Output-1'].setValuesInStep(stepName='DisplStep',numIntervals=1,variables=('S','U','RF'))
            job=('omidi_journal_r20_'if journal_phase else'omidi_blK_r20_')+label;directory=target/job;directory.mkdir(exist_ok=True);inp=directory/(job+'.inp')
            if inp.exists():raise ValueError('Refuse existing convergence input')
            os.chdir(directory);analysis=mdb.Job(name=job,model='Snub_square',numCpus=8,numDomains=8)
            analysis.writeInput(consistencyChecking=ON)
            meta=dict(base,job=job,input=str(inp),input_sha256=sha(inp),source_inp=str(source),source_sha256=sha(source),
                      state='prepared_not_submitted',analysis='author_boundary_layer_convergence',
                      W_mm=ns['W_x'],H_mm=ns['W_y'],domain_width_over_l=ns['W_x']/6.,
                      nodes=len(part.nodes),elements=len(part.elements),corrected_boundary_nodes=len(seen),
                      refinement='50elements/bar on every bar intersecting r<=30l; outer10/bar'if refine else '10elements/bar throughout',
                      refined_source_edges=near_count,convergence_control=label,
                      baseline_result=str(Path(base['input']).parent/'result.json'),
                      corrections=['Retain Figure2c phase'if journal_phase else'Retain released source notch phase',
                                   'Verified principal-root K field from retained mapped beam tensor; Table2t.45',
                                   'One unit-amplitude linear increment; no fixture or local failure mask'])
            save_json(directory/'source.json',meta,immutable=True);records.append(meta)
    finally:os.chdir(original_directory)
    save_json(target/'prepared.json',records,immutable=True)
    manifest={'abaqus':'C:/SIMULIA/Commands/abaqus.bat','queue_root':str(target/'queue'),
              'jobs':[{'job':m['job'],'input':m['input'],'input_sha256':m['input_sha256'],'cpus':8,
                       'required_outputs':['.odb','.dat','.sta'],'purpose':'Targeted external snub-square benchmark '+m['convergence_control']+' numerical check'}for m in records]}
    save_json(target/'manifest.json',manifest,immutable=True)
    print(json.dumps({'manifest':str(target/'manifest.json'),'cases':[{k:r[k]for k in ('job','nodes','elements','refined_source_edges')}for r in records]},indent=2))
    return records


def journal_phase_source(text):
    """Figure2c centres the notch on a vertical bar; released code offsets it."""
    old='Cy=W_x/2-(sqrt(3)+1)/4*L'
    if text.count(old)!=1:raise ValueError('One reviewed source notch-phase assignment required')
    return text.replace(old,'Cy=W_x/2',1)


def prepare_journal_phase(out=ROOT):
    """CAE only: Figure2c notch phase, same source topology and verified K field."""
    from abaqus import mdb
    from abaqusConstants import ON
    import regionToolset,os
    out=Path(out);target=out/'boundary-layer/journal-phase';target.mkdir(exist_ok=True)
    source=out/'source/Snub_Square.py'
    if sha(source)!='9561d895af76778cf818108036b6604494ed34fd5fe1f53bf11fd1137c5ddb8f':raise ValueError('Author source changed')
    text=journal_phase_source(source.read_text())
    for old,new in [('p = 0.001','p = 0.2'),('L = 10.00','L = 6.00'),
                    ('Es = 210000','Es = 2000'),('KI = 0.250*math.sqrt(1000)','KI = 1.0')]:
        if text.count(old)!=1:raise ValueError('Source substitution ambiguous')
        text=text.replace(old,new)
    ns={name:getattr(math,name)for name in ('sqrt','atan','tan','pi')}
    exec(compile(text[:text.index('BoundaryNodes =')],str(source)+'[Figure2c phase]','exec'),ns)
    exec(compile(text[text.index('BoundaryNodes ='):text.index('b__11 =')],str(source)+'[boundaries]','exec'),ns)
    model=mdb.models['Snub_square'];part=ns['AssemblePart'];instance=ns['PartF']
    tensors=json.loads((out/'tensor-results.json').read_text());records=[];original_directory=Path.cwd()
    try:
        for density,width in SECTION_T.items():
            tensor=next(t for t in tensors if abs(float(t['case_id'].split('-')[-1])-density)<1e-8)
            S=np.array(tensor['S_MPa_inv']);E=2/(S[0,0]+S[1,1]);nu=-S[0,1]*E;G=1/S[2,2]
            model.profiles['Rect'].setValues(a=15.,b=width)
            model.materials['Elastic'].elastic.setValues(table=(tuple(tensor['elastic_MPa_nu']),))
            for name in list(model.boundaryConditions.keys()):del model.boundaryConditions[name]
            seen=set()
            for node in ns['BoundaryNodes']:
                if node.label in seen:continue
                seen.add(node.label);u=orthotropic_K_displacement(node.coordinates[:2],E,G,nu)
                region=regionToolset.Region(nodes=instance.nodes.sequenceFromLabels(labels=(node.label,)))
                model.DisplacementBC(name='K-'+str(node.label),createStepName='DisplStep',region=region,u1=float(u[0]),u2=float(u[1]))
            model.steps['DisplStep'].setValues(initialInc=1.,maxInc=1.)
            model.fieldOutputRequests['F-Output-1'].setValuesInStep(stepName='DisplStep',numIntervals=1,variables=('S','U','RF'))
            job='omidi_journal_r'+str(round(100*density));directory=target/job;directory.mkdir(exist_ok=True);inp=directory/(job+'.inp')
            if inp.exists():raise ValueError('Refuse existing journal-phase input')
            os.chdir(directory);analysis=mdb.Job(name=job,model='Snub_square',numCpus=8,numDomains=8);analysis.writeInput(consistencyChecking=ON)
            tensor_path=Path(tensor['input']).parent/'result.json'
            meta=dict(job=job,input=str(inp),input_sha256=sha(inp),source_inp=str(source),source_sha256=sha(source),
                      source_generator=str(source),source_generator_sha256=sha(source),state='prepared_not_submitted',
                      analysis='published_figure2c_phase_boundary_layer',relative_density=density,l_mm=6.,section_B_t_mm=[15.,width],
                      assigned_E_MPa=tensor['elastic_MPa_nu'][0],assigned_nu=tensor['elastic_MPa_nu'][1],
                      prescribed_K_MPa_sqrt_mm=1.,W_mm=ns['W_x'],H_mm=ns['W_y'],domain_width_over_l=ns['W_x']/6.,
                      nodes=len(part.nodes),elements=len(part.elements),corrected_boundary_nodes=len(seen),
                      refinement='10 B21 elements/bar',tensor_result=str(tensor_path),tensor_result_sha256=sha(tensor_path),
                      K_field_E_MPa=E,K_field_G_MPa=G,K_field_nu=nu,K_field_chi=tensor['chi'],K_field_lambda=1.,
                      notch_phase_evidence='Published Fig2c and vertical first-failure bar in section3.1; notch centre Cy=W/2',
                      released_phase='Cy=W/2-(1+sqrt(3))*l/4 retained in separate diagnostic, not overwritten',
                      corrections=['Figure2c vertical-bar-centred notch, fixed before new results',
                                   'Principal-root physical mode-I field from retained mapped full beam tensor',
                                   'Table2 section widths and one linear increment; source motif and notch width retained'],
                      scope='109l source-derived domain; mesh/domain qualification separate from literature agreement')
            save_json(directory/'source.json',meta,immutable=True);records.append(meta)
    finally:os.chdir(original_directory)
    save_json(target/'prepared.json',records,immutable=True)
    manifest=dict(abaqus='C:/SIMULIA/Commands/abaqus.bat',queue_root=str(target/'queue'),
                  jobs=[dict(job=r['job'],input=r['input'],input_sha256=r['input_sha256'],cpus=8,
                             required_outputs=['.odb','.dat','.sta'],purpose='Figure2c phase matched elastic snub-square benchmark')for r in records])
    save_json(target/'manifest.json',manifest,immutable=True)
    print(json.dumps({'manifest':str(target/'manifest.json'),'elements':len(part.elements)}));return records


def gather_boundary_layer(out=ROOT,queue=None,checks=False,campaign=None):
    """Read-only completed physical-K jobs, checking global work and force balance."""
    from odbAccess import openOdb
    from validation_controls import verified_control_delivery
    if campaign=='journal-checks':campaign='journal-phase/checks';checks=True
    out=Path(out);target=out/'boundary-layer'/(campaign or ('checks'if checks else 'corrected'));queue=Path(queue or target/'queue')
    completed={r['job']for r in json.loads((queue/'state.json').read_text())['jobs']if r['status']=='completed'}
    records=[]
    for meta in json.loads((target/'prepared.json').read_text()):
        if meta['job']not in completed:continue
        odb_path,receipt=verified_control_delivery(meta,queue,completed);before=sha(odb_path)
        odb=openOdb(str(odb_path),readOnly=True)
        try:
            step=odb.steps['DisplStep'];frame=step.frames[-1]
            if abs(frame.frameValue-1)>1e-8:raise ValueError('Full unit-K amplitude not reached')
            maximum,points=maximum_beam_tension(frame.fieldOutputs['S'],'S11')
            if not maximum or maximum['max_S11']<=0:raise ValueError('Finite positive tensile stress required')
            instance=next(iter(odb.rootAssembly.instances.values()))
            node_max=max(n.label for n in instance.nodes);u=np.zeros((node_max+1,2));rf=np.zeros_like(u)
            for name,array in (('U',u),('RF',rf)):
                for block in frame.fieldOutputs[name].bulkDataBlocks:
                    values=np.asarray(block.data);labels=np.asarray(block.nodeLabels,dtype=int)
                    if not np.isfinite(values).all():raise ValueError('Nonfinite BL nodal '+name)
                    array[labels]=values[:,:2]
            net=rf.sum(axis=0);scale=np.linalg.norm(rf,axis=1).sum()
            if not math.isfinite(scale)or scale<=0 or np.linalg.norm(net)/scale>1e-5:raise ValueError('Boundary-layer global force imbalance')
            energies=[reg.historyOutputs['ALLSE'].data[-1][1]for reg in step.historyRegions.values()if 'ALLSE'in reg.historyOutputs]
            predicted=float(.5*np.sum(u*rf))
            if len(energies)!=1 or not math.isfinite(energies[0])or predicted<=0:raise ValueError('Finite positive BL work/energy required')
            error=abs(energies[0]-predicted)/predicted
            if error>1e-4:raise ValueError('BL boundary reaction work/stored-energy mismatch')
            mesh={n.label:np.array(n.coordinates[:2])for n in instance.nodes}
            element=next(e for e in instance.elements if e.label==maximum['element'])
            endpoints=np.array([mesh[n]for n in element.connectivity]);centroid=endpoints.mean(axis=0)
            # ODB instance coordinates must already carry the published translation.
            xmin=min(point[0]for point in mesh.values());xmax=max(point[0]for point in mesh.values())
            if not xmin<0<xmax:raise ValueError('ODB mesh is not in translated crack-tip coordinates')
            maximum['element_centroid_mm']=centroid.tolist()
            maximum['element_vector_mm']=(endpoints[1]-endpoints[0]).tolist()
        finally:odb.close()
        if sha(odb_path)!=before:raise ValueError('BL ODB changed during read-only extraction')
        normalized=meta['prescribed_K_MPa_sqrt_mm']/(maximum['max_S11']*math.sqrt(meta['l_mm']))
        reference=.95*meta['relative_density']**1.5
        result=dict(meta,state='completed_and_extracted',read_only=True,odb=str(odb_path),odb_sha256=before,
                    delivery_sha256=sha(receipt),maximum_surface_stress=maximum,saved_section_points=sorted(points),
                    global_force_resultant_N=net.tolist(),force_balance_relative=float(np.linalg.norm(net)/scale),
                    boundary_work_N_mm=predicted,ALLSE_N_mm=float(energies[0]),energy_relative_error=float(error),
                    normalized_K=normalized,published_fitted_K=reference,relative_difference=normalized/reference-1,
                    reference_kind='Published Table3 elastic-brittle fit .95*rho^1.5, not raw FE markers')
        if checks:
            baseline=json.loads(Path(meta['baseline_result']).read_text())
            result.update(baseline_result_sha256=sha(meta['baseline_result']),
                          relative_change_vs_baseline=normalized/baseline['normalized_K']-1)
        save_json(Path(meta['input']).parent/'result.json',result);records.append(result)
    save_json(target/'results.json',records)
    print(json.dumps([{k:r[k]for k in ('job','normalized_K','published_fitted_K','relative_difference','maximum_surface_stress','energy_relative_error')}for r in records],indent=2))
    return records


def plot_boundary_layer(out=ROOT,campaign='corrected'):
    """External discrete-lattice validation, preserving its source-fit qualification."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    target=Path(out)/'boundary-layer'/campaign;records=json.loads((target/'results.json').read_text())
    records.sort(key=lambda r:r['relative_density'])
    fig,ax=plt.subplots(figsize=(6.2,4.5),constrained_layout=True)
    rho=np.linspace(.18,.32,100)
    ax.plot(rho*100,.95*rho**1.5,'--',color='black',lw=1.2,label='Omidi–St-Pierre: published elastic fit')
    if records:ax.plot([100*r['relative_density']for r in records],[r['normalized_K']for r in records],
                        'o-',color='#c62828',ms=5,lw=1,label='Current study: discrete lattice FE')
    ax.set(xlabel='Relative density (%)',ylabel=r'$K_{Ic}/(\sigma_{fs}\sqrt{\ell})$',xticks=[20,25,30])
    ax.legend(frameon=False,fontsize=8.5);ax.spines[['top','right']].set_visible(False)
    for extension in ('png','svg'):fig.savefig(target/('omidi-boundary-layer-validation.'+extension),dpi=240)
    plt.close(fig)


def elastic_energy_check(force,reaction,opening,energies):
    """Fail closed on unavailable channels before equilibrium/work comparisons."""
    if not all(math.isfinite(value)for value in (reaction,opening,force))or force<=0:
        raise ValueError('Finite reaction/opening and positive finite applied force required')
    if abs(reaction+force)>1e-5*force:raise ValueError('Unit-load global reaction imbalance')
    if len(energies)!=1 or not math.isfinite(energies[0])or opening<=0:
        raise ValueError('Unique finite energy and positive finite compliance required')
    error=abs(energies[0]-.5*force*opening)/(.5*force*opening)
    if error>1e-5:raise ValueError('Unit-load work/energy mismatch')
    return float(error)


def gather_elastic(out=ROOT,queue=None):
    """Read-only delivered unit-load CT stresses/compliance; no source model edits."""
    from odbAccess import openOdb
    from validation_controls import verified_control_delivery
    out=Path(out);queue=Path(queue or out/'elastic-queue')
    completed={r['job']for r in json.loads((queue/'state.json').read_text())['jobs']if r['status']=='completed'}
    records=[]
    for meta in json.loads((out/'elastic/prepared.json').read_text()):
        if meta['job']not in completed:continue
        odb_path,receipt=verified_control_delivery(meta,queue,completed);before=sha(odb_path)
        odb=openOdb(str(odb_path),readOnly=True)
        try:
            step=odb.steps['Elastic'];frame=step.frames[-1]
            maximum,points=maximum_beam_tension(frame.fieldOutputs['S'],'S11')
            U={v.nodeLabel:float(v.data[1])for v in frame.fieldOutputs['U'].values}
            reaction=sum(float(v.data[1])for v in frame.fieldOutputs['RF'].values)
            opening=U[meta['upper_RP']];force=meta['applied_force_N']
            energies=[reg.historyOutputs['ALLSE'].data[-1][1]for reg in step.historyRegions.values()if 'ALLSE'in reg.historyOutputs]
            error=elastic_energy_check(force,reaction,opening,energies)
            if not maximum or maximum['max_S11']<=0:raise ValueError('Positive maximum surface tensile stress required')
            mesh={n.label:np.array(n.coordinates[:2])for inst in odb.rootAssembly.instances.values()for n in inst.nodes}
            elements={e.label:e.connectivity for inst in odb.rootAssembly.instances.values()for e in inst.elements}
            centers={label:np.mean([mesh[n]for n in conn],axis=0)for label,conn in elements.items()}
            tip=np.array([6*PERIOD,5.5*PERIOD])
            tip_region={label for label,point in centers.items()if np.linalg.norm(point-tip)<=2*meta['l_mm']}
            tip_bar={label for label,point in centers.items()if abs(point[0]-tip[0])<1e-4 and abs(point[1]-tip[1])<meta['l_mm']/2+1e-4}
            region_maximum,_=maximum_beam_tension(frame.fieldOutputs['S'],'S11',tip_region)
            bar_maximum,_=maximum_beam_tension(frame.fieldOutputs['S'],'S11',tip_bar)
            if not region_maximum or not bar_maximum:raise ValueError('Expected source crack-tip bar/region missing')
            for peak in (maximum,region_maximum,bar_maximum):
                if peak:peak['element_centroid_mm']=centers[peak['element']].tolist()
        finally:odb.close()
        if sha(odb_path)!=before:raise ValueError('Elastic ODB changed during read-only extraction')
        result=dict(meta,maximum_surface_stress=maximum,saved_section_points=sorted(points),
                    crack_tip_region_maximum=region_maximum,first_vertical_tip_bar_maximum=bar_maximum,
                    crack_tip_mm=tip.tolist(),crack_tip_region_radius_mm=2*meta['l_mm'],
                    compliance_mm_N=opening/force,global_reaction_RF2_N=reaction,
                    energy_relative_error=float(error),ALLSE_N_mm=float(energies[0]),
                    odb=str(odb_path),odb_sha256=before,delivery_sha256=sha(receipt),read_only=True,
                    state='completed_and_extracted')
        save_json(Path(meta['input']).parent/'result.json',result);records.append(result)
    save_json(out/'elastic/results.json',records)
    print(json.dumps([{'job':r['job'],'max_stress':r['maximum_surface_stress'],'energy_error':r['energy_relative_error']}for r in records],indent=2))
    return records


def compare_elastic(out=ROOT):
    """Actual discrete CT elastic-brittle FE against published Table3 density law."""
    from resources.calculations import calc_FaW_aniso
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out);tensors=json.loads((out/'tensor-results.json').read_text());records=[]
    for row in json.loads((out/'elastic/results.json').read_text()):
        rho=row['relative_density'];tensor=next(t for t in tensors if abs(float(t['case_id'].split('-')[-1])-rho)<1e-8)
        C=np.array(tensor['C_MPa']);C=C[np.ix_([1,0,2],[1,0,2])]
        f=float(calc_FaW_aniso(row['a_mm'],row['W_mm'],C=C))
        stress=row['maximum_surface_stress']['max_S11']/row['applied_force_N']
        normalized=f/(stress*row['B_mm']*math.sqrt(row['W_mm']*row['l_mm']))
        reference=.95*rho**1.5
        tip_bar=normalized*row['maximum_surface_stress']['max_S11']/row['first_vertical_tip_bar_maximum']['max_S11']
        tip_region=normalized*row['maximum_surface_stress']['max_S11']/row['crack_tip_region_maximum']['max_S11']
        result=dict(relative_density=rho,job=row['job'],normalized_K=normalized,published_fitted_K=reference,
                    global_criterion='Global surface maximum; current model is pin-controlled, not crack-tip fracture',
                    first_vertical_tip_bar_normalized_K=tip_bar,tip_region_normalized_K=tip_region,
                    first_vertical_tip_bar_relative_difference=tip_bar/reference-1,
                    tip_region_relative_difference=tip_region/reference-1,
                    first_vertical_tip_bar_maximum=row['first_vertical_tip_bar_maximum'],
                    crack_tip_region_maximum=row['crack_tip_region_maximum'],
                    relative_difference=normalized/reference-1,Ortega_factor=f,
                    lambda_parameter=tensor['lambda_crack_axis'],chi=tensor['chi'],
                    M_MPa=tensor['M_crack_axis_MPa'],S11_MPa_per_N=stress,
                    maximum_surface_stress=row['maximum_surface_stress'],
                    reference_kind='Published Table3 elastic-brittle fit .95*rho^1.5, not raw data markers',
                    scope='Discrete finite CT unit-load strength criterion plus Ortega, compared with published large prescribed-K boundary-layer elastic-brittle fit',
                    CT_result_sha256=sha(Path(row['input']).parent/'result.json'),
                    tensor_result_sha256=sha(Path(tensor['input']).parent/'result.json'))
        records.append(result)
    records.sort(key=lambda r:r['relative_density']);save_json(out/'elastic/comparison.json',records)
    fig,ax=plt.subplots(figsize=(6.2,4.5),constrained_layout=True)
    rho=np.linspace(.18,.32,100)
    ax.plot(rho*100,.95*rho**1.5,'--',color='black',lw=1.2,label='Omidi–St-Pierre: published elastic fit')
    if records:ax.plot([100*r['relative_density']for r in records],[r['normalized_K']for r in records],
                        'o-',color='#c62828',ms=5,lw=1,label='Current study: discrete C(T) FE, Ortega')
    ax.set(xlabel='Relative density (%)',ylabel=r'$K_{Ic}/(\sigma_{fs}\sqrt{\ell})$',xticks=[20,25,30])
    ax.legend(frameon=False,fontsize=8.5);ax.spines[['top','right']].set_visible(False)
    for extension in ('png','svg'):fig.savefig(out/'elastic'/('omidi-elastic-validation.'+extension),dpi=240)
    plt.close(fig);print(json.dumps(records,indent=2));return records


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    action_argument=parser.add_argument('action',choices=('prepare','prepare-tensor','gather-tensor','gather','compare','plot-tensor',
                                        'prepare-elastic','gather-elastic','compare-elastic','prepare-ring-rotation','prepare-boundary-layer',
                                        'prepare-corrected-boundary-layer','gather-boundary-layer','plot-boundary-layer',
                                        'prepare-boundary-layer-checks','gather-boundary-layer-checks','prepare-journal-phase','prepare-journal-checks'))
    parser.add_argument('--out',type=Path,default=ROOT)
    parser.add_argument('--queue','--queue-root',dest='queue',type=Path)
    parser.add_argument('--reviewed-partial',nargs='*',default=[])
    parser.add_argument('--campaign',choices=('corrected','checks','journal-phase','journal-checks'),default='corrected')
    argv=sys.argv[1:]
    if '-noGUI'in argv:
        starts=[i for i,value in enumerate(argv)if value in action_argument.choices]
        if len(starts)!=1:raise ValueError('One explicit Omidi action required after CAE arguments')
        argv=argv[starts[0]:]
    if '--'in argv:argv=argv[argv.index('--')+1:]
    if argv and Path(argv[0]).name=='validation_omidi.py':argv=argv[1:]
    args=parser.parse_args(argv)
    if args.action=='prepare':prepare(args.out)
    elif args.action=='prepare-tensor':prepare_tensor(args.out)
    elif args.action=='gather-tensor':gather_tensor(args.out,args.queue)
    elif args.action=='gather':gather(args.out,args.queue,args.reviewed_partial)
    elif args.action=='compare':compare(args.out)
    elif args.action=='plot-tensor':plot_tensor(args.out)
    elif args.action=='prepare-elastic':prepare_elastic(args.out)
    elif args.action=='gather-elastic':gather_elastic(args.out,args.queue)
    elif args.action=='compare-elastic':compare_elastic(args.out)
    elif args.action=='prepare-ring-rotation':prepare_ring_rotation_control(args.out)
    elif args.action=='prepare-boundary-layer':prepare_boundary_layer(args.out)
    elif args.action=='prepare-corrected-boundary-layer':prepare_corrected_boundary_layer(args.out)
    elif args.action=='gather-boundary-layer':gather_boundary_layer(args.out,args.queue,campaign=args.campaign)
    elif args.action=='plot-boundary-layer':plot_boundary_layer(args.out,args.campaign)
    elif args.action=='prepare-boundary-layer-checks':prepare_boundary_layer_checks(args.out)
    elif args.action=='gather-boundary-layer-checks':gather_boundary_layer(args.out,args.queue,True)
    elif args.action=='prepare-journal-phase':prepare_journal_phase(args.out)
    elif args.action=='prepare-journal-checks':prepare_boundary_layer_checks(args.out,True)
