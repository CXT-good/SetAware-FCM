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
from patient_level.baselines.gmm_fisher_svm.model import (
    fisher_matrix,
    fit_gmm,
    fit_preprocess,
    make_patient_cells,
    marker_columns_from_paths,
    save_outputs,
)


def cv_score(train_ids, y, train_cells, args, k: int, C: float, classes):
    counts=pd.Series(y).value_counts()
    n_splits=int(min(args.inner_cv, counts.min())) if len(counts)>1 else 0
    if n_splits < 2:
        return np.nan
    skf=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    scores=[]
    for fold,(tr,va) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        fold_train=[train_cells[i] for i in tr]; fold_val=[train_cells[i] for i in va]
        lo,hi,scaler=fit_preprocess(fold_train,args.clip_quantile,args.max_gmm_events,args.max_gmm_events_per_patient,args.seed+fold)
        gmm=fit_gmm(fold_train,lo,hi,scaler,k,args.max_gmm_events,args.max_gmm_events_per_patient,args.seed+fold)
        Xtr=fisher_matrix(fold_train,lo,hi,scaler,gmm); Xva=fisher_matrix(fold_val,lo,hi,scaler,gmm)
        enc=LabelEncoder().fit(y[tr])
        clf=SVC(kernel="linear", C=float(C), class_weight="balanced", probability=True, random_state=args.seed)
        clf.fit(Xtr, enc.transform(y[tr]))
        pred=enc.inverse_transform(clf.predict(Xva))
        if len(classes)==2:
            score=balanced_accuracy_score(y[va], pred)
        else:
            score=f1_score(y[va], pred, average="macro", zero_division=0)
        scores.append(score)
        print(f"[CV] K={k} C={C} fold={fold}/{n_splits} score={score:.4f}", flush=True)
    return float(np.mean(scores))


def main():
    ap=argparse.ArgumentParser(description="GMM-Fisher Vector-SVM patient-level baseline after Wang et al. 2025")
    ap.add_argument("--split_xlsx", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--marker_mode", default="dataset_marker")
    ap.add_argument("--k_values", default="8,16,32")
    ap.add_argument("--c_values", default="0.1,1,10")
    ap.add_argument("--inner_cv", type=int, default=3)
    ap.add_argument("--header_scan_limit", type=int, default=2000)
    ap.add_argument("--max_cells_per_patient", type=int, default=100000)
    ap.add_argument("--max_gmm_events", type=int, default=200000)
    ap.add_argument("--max_gmm_events_per_patient", type=int, default=3000)
    ap.add_argument("--clip_quantile", type=float, default=0.01)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--seed", type=int, default=2024)
    args=ap.parse_args()

    out=Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    train_rows=patient_table(args.split_xlsx,"train")
    test_rows=patient_table(args.split_xlsx,"test")
    all_paths=train_rows["_csv_path"].astype(str).tolist()+test_rows["_csv_path"].astype(str).tolist()
    markers=marker_columns_from_paths(all_paths,args.header_scan_limit)
    print(f"[INFO] aligned markers ({len(markers)}): {markers}", flush=True)
    write_json(out/"selected_markers.json", {"marker_mode":args.marker_mode,"markers":markers,"header_scan_limit":args.header_scan_limit})

    train_ids,y_train,train_cells=make_patient_cells(train_rows,markers,args.max_cells_per_patient,args.seed)
    test_ids,y_test,test_cells=make_patient_cells(test_rows,markers,args.max_cells_per_patient,args.seed+10000)
    classes=np.array(sorted(pd.unique(y_train)))
    print(f"[INFO] train patients={len(train_ids)} test patients={len(test_ids)} classes={classes.tolist()}", flush=True)

    k_values=[int(x) for x in str(args.k_values).split(',') if x]
    c_values=[float(x) for x in str(args.c_values).split(',') if x]
    cv_rows=[]; best=None
    for k in k_values:
        for C in c_values:
            try:
                s=cv_score(train_ids,y_train,train_cells,args,k,C,classes)
            except Exception as exc:
                print(f"[WARN] CV failed K={k} C={C}: {exc}", flush=True)
                s=np.nan
            cv_rows.append({"K":k,"C":C,"cv_score":s})
            if np.isfinite(s) and (best is None or s>best["cv_score"]):
                best={"K":k,"C":C,"cv_score":s}
    if best is None:
        best={"K":16 if 16 in k_values else k_values[0], "C":1.0 if 1.0 in c_values else c_values[0], "cv_score":np.nan}
    pd.DataFrame(cv_rows).to_csv(out/"cv_results.csv", index=False)
    print(f"[INFO] best params: {best}", flush=True)

    lo,hi,scaler=fit_preprocess(train_cells,args.clip_quantile,args.max_gmm_events,args.max_gmm_events_per_patient,args.seed)
    gmm=fit_gmm(train_cells,lo,hi,scaler,best["K"],args.max_gmm_events,args.max_gmm_events_per_patient,args.seed)
    Xtr=fisher_matrix(train_cells,lo,hi,scaler,gmm); Xte=fisher_matrix(test_cells,lo,hi,scaler,gmm)
    enc=LabelEncoder().fit(y_train)
    clf=SVC(kernel="linear", C=float(best["C"]), class_weight="balanced", probability=True, random_state=args.seed)
    clf.fit(Xtr, enc.transform(y_train))
    pred=enc.inverse_transform(clf.predict(Xte))
    proba=clf.predict_proba(Xte)
    classes_out=enc.classes_
    metrics=save_outputs(out,test_ids,y_test,pred,proba,classes_out,args.bootstrap,args.seed)
    joblib.dump({"scaler":scaler,"clip_low":lo,"clip_high":hi,"gmm":gmm,"svm":clf,"encoder":enc,"markers":markers,"args":vars(args),"best":best}, out/"gmm_fisher_svm.joblib")
    write_json(out/"metadata.json", {"method":"gmm_fisher_svm", "paper":"Wang et al. 2025", "args":vars(args), "best":best, "markers":markers, "classes":classes_out.tolist(), "metrics":metrics})
    print("[DONE]", metrics, flush=True)


if __name__ == "__main__":
    main()
