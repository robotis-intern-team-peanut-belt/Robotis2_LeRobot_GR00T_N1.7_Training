#!/usr/bin/env python3
"""Build the reviewed boundary-safe Task 700013 essential v3 derivative."""
from __future__ import annotations

import json, os, shutil, tempfile
from collections import defaultdict
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SRC = Path('/NHNHOME/WORKSPACE/26motir001_D/intern_team/training/artifacts/datasets/Task_700013_OMX_Insert_F2_Intern_MCAP_left_gripper_fix_v1_lerobot_v30')
DST = Path('/NHNHOME/WORKSPACE/26motir001_D/intern_team/training/artifacts/datasets/Task_700013_OMX_Insert_F2_Intern_MCAP_essential_lerobot_v30')
FPS = 15.0
RAW = {0:(3,),1:(3,),2:(1,3),3:(1,3),4:(3,),5:(3,),6:(3,),7:(1,3),8:(3,),9:(1,3),10:(1,3),11:(1,3),12:(3,),13:(3,),14:(3,),15:(1,3),16:(3,),17:(3,),18:(1,3),19:(1,3),20:(3,),21:(3,),22:(3,),23:(1,3),24:(1,3),25:(3,),26:(1,3),27:(1,3),28:(3,),29:(0,1,3),30:(1,2,3),31:(3,4),32:(1,3),33:(1,3),34:(1,3),35:(3,),36:(3,),37:(3,),38:(3,),40:(0,3),41:(3,),42:(0,),43:(3,),44:(0,),45:(3,),46:(3,),47:(3,),48:(3,),49:(3,),50:(3,),51:(3,),52:(3,),54:(3,),55:(3,),56:(3,),57:(3,),58:(3,),59:(3,),61:(3,),62:(3,),63:(3,),64:(3,),65:(3,),66:(3,),67:(3,),69:(3,),70:(3,),71:(3,),72:(0,3),73:(3,),74:(3,),75:(3,),76:(3,),77:(3,),78:(3,),79:(3,),80:(1,3),81:(3,),82:(3,),83:(1,3),84:(3,),85:(3,),86:(3,),87:(3,),88:(3,),89:(3,),90:(3,),91:(3,),92:(3,),93:(3,),94:(3,),95:(3,),96:(3,),97:(3,),98:(1,3),99:(0,1,3)}
EX = {k:set(v) for k,v in RAW.items()}

def jwrite(path, value):
    tmp=path.with_name('.'+path.name+'.tmp'); tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n'); os.replace(tmp,path)
def pwrite(path, table):
    tmp=path.with_name('.'+path.name+'.tmp'); pq.write_table(table,tmp,compression='snappy'); os.replace(tmp,path)
def stats(values):
    a=np.asarray(values)
    convert=lambda value: np.atleast_1d(value).astype(float).tolist()
    return {'mean':convert(np.mean(a,0)),'std':convert(np.maximum(np.std(a,0),1e-3)),'min':convert(np.min(a,0)),'max':convert(np.max(a,0)),'count':[len(a)]}
def ann(frames, names, task):
    blocks=[]; start=0; current=int(frames[0]['subtask_index'])
    for i,row in enumerate(frames[1:],1):
        sub=int(row['subtask_index'])
        if sub != current:
            blocks.append({'frame_duration':[start,i],'sub_task_idx':current,'sub_task_instruction':names[current]}); start,current=i,sub
    blocks.append({'frame_duration':[start,len(frames)],'sub_task_idx':current,'sub_task_instruction':names[current]})
    return {'data_folder':'','meta_data':{'task_duration':len(frames),'valid_duration':[0,len(frames)]},'sub_task_annotation':blocks,'task_name':task}

def main():
    if DST.exists(): raise SystemExit(f'output exists: {DST}')
    info=json.loads((SRC/'meta/info.json').read_text())
    if info.get('codebase_version')!='v3.0' or info.get('total_episodes')!=100: raise SystemExit('unexpected source')
    ts=[pq.read_table(p) for p in sorted((SRC/'data').rglob('*.parquet'))]; schema=ts[0].schema
    by=defaultdict(list)
    for r in pa.concat_tables(ts).to_pylist(): by[int(r['episode_index'])].append(dict(r))
    et=pq.read_table(next((SRC/'meta/episodes').rglob('*.parquet'))); erows={int(r['episode_index']):r for r in et.to_pylist()}
    subnames={int(r['subtask_index']):r['subtask'] for r in pq.read_table(SRC/'meta/subtasks.parquet').to_pylist()}
    stage=Path(tempfile.mkdtemp(prefix='.'+DST.name+'.tmp-',dir=DST.parent))
    try:
        shutil.copytree(SRC/'meta',stage/'meta'); (stage/'data/chunk-000').mkdir(parents=True); (stage/'annotations/chunk-000').mkdir(parents=True)
        for f in (SRC/'videos').rglob('*.mp4'):
            out=stage/f.relative_to(SRC); out.parent.mkdir(parents=True,exist_ok=True); os.link(f,out)
        output=[]; episodes=[]; mapping=[]; fmap={}; retained=defaultdict(int); removed=defaultdict(int); gi=0
        for source_ep in range(100):
            frames=sorted(by[source_ep],key=lambda r:r['frame_index']); keep=[r['subtask_index'] not in EX.get(source_ep,set()) for r in frames]
            for r,k in zip(frames,keep): (retained if k else removed)[int(r['subtask_index'])]+=1
            pos=0
            while pos<len(frames):
                if not keep[pos]: pos+=1; continue
                end=pos+1
                while end<len(frames) and keep[end]: end+=1
                segment=[dict(r) for r in frames[pos:end]]; newep=len(episodes)
                for nf,r in enumerate(segment):
                    fmap[(source_ep,int(r['frame_index']))]=(newep,nf); r['episode_index']=newep; r['frame_index']=nf; r['index']=gi; r['timestamp']=nf/FPS; output.append(r); gi+=1
                e=dict(erows[source_ep]); e['episode_index']=newep; e['length']=len(segment); e['data/chunk_index']=e['data/file_index']=0; e['dataset_from_index']=gi-len(segment); e['dataset_to_index']=gi; e['meta/episodes/chunk_index']=e['meta/episodes/file_index']=0; e['full_episode_index']=source_ep
                for cam in ('observation.images.rgb.cam_head','observation.images.rgb.cam_left_wrist','observation.images.rgb.cam_right_wrist'):
                    a=f'videos/{cam}/from_timestamp'; b=f'videos/{cam}/to_timestamp'; base=float(erows[source_ep][a]); e[a]=base+pos/FPS; e[b]=base+end/FPS
                for key in ('timestamp','frame_index','index','episode_index','task_index','subtask_index','observation.state','action'):
                    for stat,val in stats([r[key] for r in segment]).items(): e[f'stats/{key}/{stat}']=val
                episodes.append(e); mapping.append({'essential_episode_index':newep,'source_episode_index':source_ep,'source_frame_range':[pos,end],'length':len(segment)})
                source_ann=json.loads(next((SRC/'annotations').rglob(f'episode_{source_ep:06d}.json')).read_text())
                jwrite(stage/'annotations/chunk-000'/f'episode_{newep:06d}.json',ann(segment,subnames,source_ann['task_name'])); pos=end
        pwrite(stage/'data/chunk-000/file-000.parquet',pa.Table.from_pylist(output,schema=schema))
        shutil.rmtree(stage/'meta/episodes'); (stage/'meta/episodes/chunk-000').mkdir(parents=True); pwrite(stage/'meta/episodes/chunk-000/file-000.parquet',pa.Table.from_pylist(episodes,schema=et.schema))
        reuse=pq.read_table(SRC/'meta/frame_reuse.parquet'); rr=[]
        for r in reuse.to_pylist():
            found=fmap.get((int(r['episode_index']),int(r['target_frame_index'])))
            if found is not None: r=dict(r); r['episode_index'],r['target_frame_index']=found; rr.append(r)
        pwrite(stage/'meta/frame_reuse.parquet',pa.Table.from_pylist(rr,schema=reuse.schema))
        for ip in (stage/'meta/info.json',stage/'info.json'):
            if ip.exists(): d=json.loads(ip.read_text()); d['repo_id']='local/'+DST.name; d['total_episodes']=len(episodes); d['total_frames']=len(output); jwrite(ip,d)
        sp=stage/'meta/stats.json'; sd=json.loads(sp.read_text())
        for key in ('timestamp','frame_index','index','episode_index','task_index','subtask_index','observation.state','action'): sd[key]=stats([r[key] for r in output])
        jwrite(sp,sd)
        receipt={'schema_version':1,'kind':'task700013_reviewed_essential_v3','source_root':str(SRC),'output_root':str(DST),'source_total_episodes':100,'source_total_frames':sum(len(v) for v in by.values()),'essential_total_episodes':len(episodes),'essential_total_frames':len(output),'excluded_episode_subtasks':{str(k):sorted(v) for k,v in EX.items()},'subtask_index_map':{'0':'left servo','1':'left electronic','2':'left cable','3':'left OMX-F3','4':'left OMX-Base'},'retained_frames_by_subtask':dict(retained),'excluded_frames_by_subtask':dict(removed),'source_segment_map':mapping,'video_policy':'hardlinked source MP4s with precise kept timestamp intervals','horizon_guard':'each output episode is one contiguous kept run; no horizon crosses an excluded span','review_note':'EP_100 was blank and source IDs are 0..99.'}
        jwrite(stage/'ESSENTIAL_SELECTION_RECEIPT.json',receipt); os.replace(stage,DST); print(json.dumps(receipt,indent=2,sort_keys=True))
    except BaseException:
        shutil.rmtree(stage,ignore_errors=True); raise
if __name__=='__main__': main()
