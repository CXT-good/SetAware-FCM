from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from cell_level.baselines.flowsom import FlowSOM
from dataio.flow_dataset import _read_csv_with_fallback
from dataio.marker_mapping import MARKER_TO_INDEX

def paths(xlsx, sheet):
    df=pd.read_excel(xlsx, sheet_name=sheet); return [str(Path(p).resolve()) for p in df.csv_path.astype(str) if str(p).endswith('.csv') and Path(p).is_file()]
def load_map(p, key):
    root=json.load(open(p,encoding='utf8'))
    d=root.get(key) or root.get('datasets', {}).get(key)
    if d is None:
        for value in root.values():
            if isinstance(value, dict) and key in value:
                d=value[key]; break
    if d is None: raise KeyError(f'dataset mapping not found: {key}')
    return {str(x):i for i,x in enumerate(d['class_list'])}, d['class_list']
def features(p,label):
    h=_read_csv_with_fallback(p,nrows=0).columns.tolist(); return [c for c in h if c in MARKER_TO_INDEX and c not in {label,'Time','event_ID'}]
def sample_file(p, cols, label, mapping, n, rng):
    header=set(_read_csv_with_fallback(p,nrows=0).columns)
    missing=[c for c in cols if c not in header]
    if missing: raise ValueError(f'missing required markers {missing}: {p}')
    df=_read_csv_with_fallback(p,usecols=lambda c:c in set(cols+[label]),dtype={label:str})
    x=df.reindex(columns=cols).apply(pd.to_numeric,errors='coerce').fillna(0).to_numpy('float32')
    y=df[label].astype(str).str.strip().map(mapping).fillna(0).astype('int64').to_numpy()
    take=min(n,len(x)); idx=rng.choice(len(x),take,replace=False) if take<len(x) else np.arange(len(x)); return x[idx],y[idx]
def parse_args(argv=None):
 p=argparse.ArgumentParser()
 p.add_argument('--split_xlsx',required=True)
 p.add_argument('--class_mapping_path',required=True)
 p.add_argument('--dataset_key',required=True)
 p.add_argument('--output_dir',required=True)
 p.add_argument('--sample_per_tube',type=int,default=10000)
 p.add_argument('--xdim',type=int,default=10)
 p.add_argument('--ydim',type=int,default=10)
 p.add_argument('--meta_clusters',type=int,default=30)
 p.add_argument('--epochs',type=int,default=10)
 p.add_argument('--seed',type=int,default=42)
 p.add_argument('--label_column',default='label')
 p.add_argument('--marker_columns',nargs='+',default=None)
 return p.parse_args(argv)
def main():
 a=parse_args()
 out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=False); ps=paths(a.split_xlsx,'train'); mapping,classes=load_map(a.class_mapping_path,a.dataset_key); cols=list(a.marker_columns) if a.marker_columns else features(ps[0],a.label_column); rng=np.random.default_rng(a.seed)
 xs=[];ys=[]
 for i,path in enumerate(ps):
  x,y=sample_file(path,cols,a.label_column,mapping,a.sample_per_tube,rng); xs.append(x);ys.append(y); print(f'{i+1}/{len(ps)} {Path(path).name}: {len(x)}',flush=True)
 x=np.concatenate(xs);y=np.concatenate(ys); scaler=StandardScaler().fit(x); z=scaler.transform(x).astype('float32'); model=FlowSOM(a.xdim,a.ydim,random_state=a.seed).fit(z,a.epochs).fit_meta(a.meta_clusters,a.seed).fit_label_mapping(z,y,len(classes))
 np.savez_compressed(out/'flowsom_model.npz',**model.state_dict(),mean=scaler.mean_,scale=scaler.scale_)
 json.dump({'dataset_key':a.dataset_key,'features':cols,'class_list':classes,'seed':a.seed,'sample_per_tube':a.sample_per_tube,'train_tubes':ps},open(out/'run_meta.json','w',encoding='utf8'),ensure_ascii=False,indent=2)
 pd.DataFrame(model.meta_class_counts,columns=classes).assign(meta_cluster=np.arange(len(model.meta_class_counts)),predicted_class=model.meta_to_class).to_csv(out/'meta_cluster_label_mapping.csv',index=False)
 print('COMPLETE',flush=True)
if __name__=='__main__': main()
