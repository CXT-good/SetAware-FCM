from __future__ import annotations
import argparse,json,os
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import confusion_matrix
from cell_level.baselines.flowsom import FlowSOM
from dataio.flow_dataset import _read_csv_with_fallback
def parse_args(argv=None):
 p=argparse.ArgumentParser()
 p.add_argument('--model_dir',required=True)
 p.add_argument('--split_xlsx',required=True)
 p.add_argument('--output_dir',required=True)
 p.add_argument('--label_column',default='label')
 p.add_argument('--chunksize',type=int,default=200000)
 p.add_argument('--skip_cell_dumps',action='store_true')
 return p.parse_args(argv)
def main():
 a=parse_args()
 d=np.load(Path(a.model_dir)/'flowsom_model.npz',allow_pickle=True); state={k:d[k] for k in ('xdim','ydim','sigma','learning_rate','codebook','node_to_meta','meta_to_class','meta_class_counts')}; state.update({k:state[k].item() for k in ('xdim','ydim','sigma','learning_rate')}); model=FlowSOM.from_state_dict(state); meta=json.load(open(Path(a.model_dir)/'run_meta.json',encoding='utf8')); cols=meta['features']; classes=meta['class_list']; mp={str(x):i for i,x in enumerate(classes)}; mean,scale=d['mean'],d['scale']; out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True); pred_dir=out/'per_tube_predictions'; dfp=pd.read_excel(a.split_xlsx,sheet_name='test'); cm=np.zeros((len(classes),len(classes)),dtype=np.int64)
 if not a.skip_cell_dumps: pred_dir.mkdir(exist_ok=True)
 for path in [Path(x).resolve() for x in dfp.csv_path.astype(str) if str(x).endswith('.csv') and Path(x).is_file()]:
  rows=[] if not a.skip_cell_dumps else None
  for q in pd.read_csv(path,usecols=lambda c:c in set(cols+[a.label_column]),chunksize=a.chunksize):
   x=q.reindex(columns=cols).apply(pd.to_numeric,errors='coerce').fillna(0).to_numpy('float32'); y=q[a.label_column].astype(str).str.strip().map(mp).fillna(0).astype(int).to_numpy(); pred=model.predict((x-mean)/np.where(scale==0,1,scale)); cm+=confusion_matrix(y,pred,labels=np.arange(len(classes)))
   if rows is not None: rows.append(pd.DataFrame({'true_label':y,'pred_label':pred}))
  if rows is not None: pd.concat(rows).to_csv(pred_dir/(path.stem+'_predict.csv'),index=False)
  print(path.name,flush=True)
 tp=np.diag(cm).astype(float); rs=cm.sum(1);cs=cm.sum(0); prec=np.divide(tp,cs,out=np.zeros_like(tp),where=cs>0);rec=np.divide(tp,rs,out=np.zeros_like(tp),where=rs>0);f1=np.divide(2*prec*rec,prec+rec,out=np.zeros_like(tp),where=prec+rec>0); total=cm.sum(); pd.DataFrame({'label':classes,'accuracy':[(tp[i]+total-rs[i]-cs[i]+tp[i])/total for i in range(len(classes))],'precision':prec,'recall':rec,'f1':f1,'support':rs.astype(int)}).to_csv(out/'metrics_per_class.csv',index=False);np.save(out/'agg_confusion.npy',cm);print('COMPLETE',flush=True)
if __name__=='__main__': main()
