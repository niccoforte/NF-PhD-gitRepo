"""Independent raw-ODB comparison of one explicit mapped FCC damage product."""
import argparse
import json
from pathlib import Path

import numpy as np
from resources.damage import require, validate_damage_arrays


def verify(path):
    from odbAccess import openOdb
    with np.load(path,allow_pickle=False) as saved:
        a={key:saved[key] for key in saved.files}
    validate_damage_arrays(a)
    metadata=json.loads(str(a['metadata_json']))
    require(Path(metadata['odb_path']).stem == metadata['source_stem'], 'Source stem mismatch')
    require(int(a['simulation_id']) == metadata['simulation_id'], 'Specimen ID mismatch')
    element_labels=a['element_label']
    parents=a['parent_strut_index']
    instance=metadata['mapping']['instance']
    point_keys=[(str(a['point_instance'][p]),int(element_labels[e]),int(a['point_integration'][p]),int(a['point_section'][p]))
                for p,e in enumerate(a['point_element_index'])]
    odb=openOdb(metadata['odb_path'],readOnly=True)
    statuses_checked,sdeg_checked=0,0
    try:
        frames=odb.steps[metadata['step']].frames
        require(len(frames)==len(a['frame_time']),'Frame count changed')
        for j,frame in enumerate(frames):
            require(frame.frameValue==a['frame_time'][j],'Recorded time mismatch')
            raw={v.elementLabel:float(v.data) for v in frame.fieldOutputs['STATUS'].values
                 if v.instance.name==instance}
            direct=np.array([raw.get(int(e),np.nan) for e in element_labels])
            require(np.array_equal(direct,a['status'][j],equal_nan=True),'Raw STATUS differs from saved values')
            statuses_checked+=len(direct)
            for parent in range(len(a['strut_id'])):
                members=direct[parents==parent]
                expected=bool((members==0).any())
                require(expected==bool(a['strut_failed'][j,parent]),'Independent parent failure mismatch')
                known=expected or bool(np.isfinite(members).all())
                require(known==bool(a['strut_failure_known'][j,parent]),'Independent parent known-mask mismatch')
            if j in {0,5,7,9,len(frames)-1}:
                raw_points={(v.instance.name,v.elementLabel,v.integrationPoint,v.sectionPoint.number):float(v.data)
                            for v in frame.fieldOutputs['SDEG'].values if v.instance.name==instance}
                require(np.array_equal(np.array([raw_points.get(k,np.nan) for k in point_keys]),a['sdeg'][j],equal_nan=True),
                        'Raw section/IP SDEG differs from saved values')
                sdeg_checked+=len(point_keys)
    finally:
        odb.close()
    summary=dict(task=str(a['task']),source_stem=metadata['source_stem'],
                 simulation_id=metadata['simulation_id'],raw_status_values_verified=statuses_checked,
                 raw_sdeg_values_spotchecked=sdeg_checked,all_frame_parent_failures_verified=True,
                 all_frame_parent_known_masks_verified=True,
                 raw_missing_values_in_checked_products=int((~a['status_valid']).sum()+(~a['sdeg_valid']).sum()),read_only=True)
    Path(path).with_suffix('.verification.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--npz',required=True)
    verify(parser.parse_args().npz)
