from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from common.patient_paper import patient_table, write_json
from patient_level.baselines.gmm_fisher_svm.model import save_outputs
from patient_level.baselines.gmm_fisher_svm.model_per_tube import (
    fit_feature_pipeline,
    patient_ids_labels,
)


def cv_score(train_rows, ids, y, k, C, classes, args):
    counts=pd.Series(y).value_counts()
    n_splits=int(min(args.inner_cv, counts.min())) if len(counts)>1 else 0
    if n_splits<2:
        return np.nan
    skf=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=args.seed)
    scores=[]
    rows_by_pid={pid: train_rows[train_rows['_patient_id'].astype(str)==str(pid)] for pid in ids}
    for fold,(tr,va) in enumerate(skf.split(np.zeros(len(y)),y),start=1):
        tr_ids=[ids[i] for i in tr]; va_ids=[ids[i] for i in va]
        tr_rows=pd.concat([rows_by_pid[i] for i in tr_ids], ignore_index=True)
        va_rows=pd.concat([rows_by_pid[i] for i in va_ids], ignore_index=True)
        Xtr,Xva,_=fit_feature_pipeline(tr_rows,tr_ids,va_rows,va_ids,k,args,seed_offset=fold*10000)
        enc=LabelEncoder().fit(y[tr])
        clf=SVC(kernel='linear',C=float(C),class_weight='balanced',probability=True,random_state=args.seed)
        clf.fit(Xtr,enc.transform(y[tr]))
        pred=enc.inverse_transform(clf.predict(Xva))
        score=balanced_accuracy_score(y[va],pred) if len(classes)==2 else f1_score(y[va],pred,average='macro',zero_division=0)
        scores.append(score)
        print(f"[CV] K={k} C={C} fold={fold}/{n_splits} score={score:.4f}", flush=True)
    return float(np.mean(scores))


def main():
    ap=argparse.ArgumentParser(description='Per-tube GMM-Fisher Vector-SVM patient-level baseline')
    ap.add_argument('--split_xlsx',required=True)
    ap.add_argument('--output_dir',required=True)
    ap.add_argument('--k_values',default='8,16,32')
    ap.add_argument('--c_values',default='0.1,1,10')
    ap.add_argument('--inner_cv',type=int,default=3)
    ap.add_argument('--header_scan_limit',type=int,default=2000)
    ap.add_argument('--max_events_per_file',type=int,default=30000)
    ap.add_argument('--max_events_per_patient_tube',type=int,default=50000)
    ap.add_argument('--max_gmm_events',type=int,default=200000)
    ap.add_argument('--clip_quantile',type=float,default=0.01)
    ap.add_argument('--bootstrap',type=int,default=500)
    ap.add_argument('--seed',type=int,default=2024)
    args=ap.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    train_rows=patient_table(args.split_xlsx,'train')
    test_rows=patient_table(args.split_xlsx,'test')
    train_ids,y_train=patient_ids_labels(train_rows)
    test_ids,y_test=patient_ids_labels(test_rows)
    classes=np.array(sorted(pd.unique(y_train)))
    print(f"[INFO] train patients={len(train_ids)} test patients={len(test_ids)} classes={classes.tolist()} tubes={sorted(train_rows['_tube_id'].astype(str).unique().tolist())}", flush=True)
    k_values=[int(x) for x in args.k_values.split(',') if x]
    c_values=[float(x) for x in args.c_values.split(',') if x]
    best=None; cv_rows=[]
    for k in k_values:
        for C in c_values:
            try:
                score=cv_score(train_rows,train_ids,y_train,k,C,classes,args)
            except Exception as exc:
                print(f"[WARN] CV failed K={k} C={C}: {exc}", flush=True)
                score=np.nan
            cv_rows.append({'K':k,'C':C,'cv_score':score})
            if np.isfinite(score) and (best is None or score>best['cv_score']):
                best={'K':k,'C':C,'cv_score':score}
    if best is None:
        best={'K':16 if 16 in k_values else k_values[0], 'C':1.0 if 1.0 in c_values else c_values[0], 'cv_score':np.nan}
    pd.DataFrame(cv_rows).to_csv(out/'cv_results.csv',index=False)
    print(f"[INFO] best={best}", flush=True)
    Xtr,Xte,pipes=fit_feature_pipeline(train_rows,train_ids,test_rows,test_ids,best['K'],args,seed_offset=77777)
    enc=LabelEncoder().fit(y_train)
    clf=SVC(kernel='linear',C=float(best['C']),class_weight='balanced',probability=True,random_state=args.seed)
    clf.fit(Xtr,enc.transform(y_train))
    pred=enc.inverse_transform(clf.predict(Xte)); proba=clf.predict_proba(Xte)
    metrics=save_outputs(out,test_ids,y_test,pred,proba,enc.classes_,args.bootstrap,args.seed)
    serial_pipes=[]
    for p in pipes:
        serial_pipes.append({'tube':p['tube'],'markers':p['markers'],'dim':p['dim']})
    joblib.dump({'pipes':pipes,'svm':clf,'encoder':enc,'best':best,'args':vars(args)}, out/'gmm_fisher_svm_per_tube.joblib')
    write_json(out/'metadata.json', {'method':'gmm_fisher_svm_per_tube','paper':'Wang et al. 2025 adapted per tube','args':vars(args),'best':best,'classes':enc.classes_.tolist(),'tubes':serial_pipes,'metrics':metrics})
    print('[DONE]', metrics, flush=True)

if __name__=='__main__':
    main()
