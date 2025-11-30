#!/usr/bin/env python3
"""
train_rf.py — Train a RandomForest with Platt calibration (via CalibratedClassifierCV),
select an F1-optimal threshold on the validation split, evaluate on the test split,
and save model artifacts + predictions + plots.
"""
import os, json, argparse
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, precision_recall_curve,
                             roc_curve, confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.utils import check_random_state
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator as FPGen
from rdkit.Chem import DataStructs
import matplotlib.pyplot as plt
from joblib import dump

def morgan_fp(smiles, nbits=2048, radius=2):
    gen = FPGen.GetMorganGenerator(radius=radius, fpSize=nbits)
    X = np.zeros((len(smiles), nbits), dtype=np.uint8)
    ok = np.zeros((len(smiles),), dtype=bool)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(str(s)) if pd.notna(s) else None
        if m is None:
            continue
        bv = gen.GetFingerprint(m)
        arr = np.zeros((nbits,), dtype=int)
        DataStructs.ConvertToNumpyArray(bv, arr)
        X[i, :] = arr.astype(np.uint8)
        ok[i] = True
    return X, ok

def ece_score(y_true, p_pred, n_bins=10):
    # remove NaNs
    mask = np.isfinite(p_pred)
    y = np.array(y_true)[mask].astype(int)
    p = np.array(p_pred)[mask].astype(float)
    if len(y) == 0:
        return np.nan
    bins = np.linspace(0.0, 1.0, n_bins+1)
    idx = np.digitize(p, bins) - 1
    ece = 0.0
    total = len(y)
    for b in range(n_bins):
        sel = idx == b
        if not np.any(sel):
            continue
        conf = p[sel].mean()
        acc = y[sel].mean()
        ece += (sel.sum()/total) * abs(acc - conf)
    return float(ece)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset.csv")
    ap.add_argument("--nbits", type=int, default=2048)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--model_out", default="models/rf_calibrated.joblib")
    ap.add_argument("--meta_out", default="models/model_meta.json")
    args = ap.parse_args()

    rng = check_random_state(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)

    df = pd.read_csv(args.data)
    for col in ["smiles", "is_potent_any", "split"]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing in {args.data}")
    # Build fingerprints for all rows (keep parse_ok flags)
    X, ok = morgan_fp(df["smiles"].astype(str).tolist(), nbits=args.nbits, radius=args.radius)
    y = df["is_potent_any"].astype(int).to_numpy()
    split = df["split"].astype(str).to_numpy()

    def sel(s):
        m = (split == s) & ok
        return X[m], y[m], m

    Xtr, ytr, mtr = sel("train")
    Xva, yva, mva = sel("valid")
    Xte, yte, mte = sel("test")

    # Base RF
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=1,
        class_weight=None, random_state=args.seed, n_jobs=-1
    )
    # Platt calibration via 5-fold CV on the training set
    clf = CalibratedClassifierCV(estimator=rf, method="sigmoid", cv=5)
    clf.fit(Xtr, ytr)

    # Predict calibrated probabilities
    p_val = clf.predict_proba(Xva)[:,1] if len(yva) else np.array([])
    p_test = clf.predict_proba(Xte)[:,1] if len(yte) else np.array([])

    # Select threshold on validation to maximize F1
    thr_grid = np.linspace(0.01, 0.99, 99)
    best_f1, best_thr = -1.0, 0.5
    if len(yva):
        for t in thr_grid:
            yhat = (p_val >= t).astype(int)
            f1 = f1_score(yva, yhat)
            if f1 > best_f1:
                best_f1, best_thr = f1, float(t)

    # Evaluate on test at best_thr
    if len(yte):
        roc = roc_auc_score(yte, p_test)
        ap = average_precision_score(yte, p_test)
        brier = brier_score_loss(yte, p_test)
        ece = ece_score(yte, p_test, n_bins=10)
        yhat = (p_test >= best_thr).astype(int)
        prec = precision_score(yte, yhat)
        rec = recall_score(yte, yhat)
        f1t = f1_score(yte, yhat)
        cm = confusion_matrix(yte, yhat, labels=[0,1]).tolist()
    else:
        roc = ap = brier = ece = prec = rec = f1t = np.nan
        cm = [[0,0],[0,0]]

    # Save artifacts
    dump(clf, args.model_out)
    meta = {
        "featurizer": {"nbits": args.nbits, "radius": args.radius},
        "decision": {"threshold": best_thr},
        "data": {"path": args.data},
        "metrics_test": {
            "roc_auc": float(roc), "ap": float(ap),
            "brier": float(brier), "ece": float(ece),
            "precision": float(prec), "recall": float(rec), "f1": float(f1t),
            "confusion_matrix": cm
        }
    }
    with open(args.meta_out, "w") as f:
        json.dump(meta, f, indent=2)

    # Save predictions
    if len(yva):
        pd.DataFrame({"p_rf": p_val, "y_true": yva}).to_csv(os.path.join(args.outdir, "pred_valid.csv"), index=False)
    if len(yte):
        pd.DataFrame({"p_rf": p_test, "y_true": yte}).to_csv(os.path.join(args.outdir, "pred_test.csv"), index=False)

    # Save metrics JSON
    with open(os.path.join(args.outdir, "metrics_test.json"), "w") as f:
        json.dump(meta["metrics_test"], f, indent=2)

    # Plots
    if len(yte):
        fpr, tpr, _ = roc_curve(yte, p_test)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.plot([0,1],[0,1],'--')
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC (test)")
        plt.savefig(os.path.join(args.outdir, "fig_roc.png"), dpi=200, bbox_inches="tight")
        plt.close()

        precs, recs, _ = precision_recall_curve(yte, p_test)
        plt.figure()
        plt.plot(recs, precs)
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR (test)")
        plt.savefig(os.path.join(args.outdir, "fig_pr.png"), dpi=200, bbox_inches="tight")
        plt.close()

        # Calibration curve (10 bins)
        bins = np.linspace(0,1,11)
        idx = np.digitize(p_test, bins) - 1
        conf = []; acc = []
        for b in range(10):
            sel = idx==b
            if not np.any(sel): 
                conf.append(np.nan); acc.append(np.nan); continue
            conf.append(p_test[sel].mean()); acc.append(yte[sel].mean())
        plt.figure()
        plt.plot([0,1],[0,1],'--')
        plt.plot(conf, acc, marker='o')
        plt.xlabel("Mean predicted probability"); plt.ylabel("Observed positive rate")
        plt.title("Calibration (test)")
        plt.savefig(os.path.join(args.outdir, "fig_calibration.png"), dpi=200, bbox_inches="tight")
        plt.close()

        # Confusion matrix
        from matplotlib import pyplot as plt
        cm_arr = np.array(cm)
        plt.figure()
        plt.imshow(cm_arr, interpolation='nearest')
        for i in range(2):
            for j in range(2):
                plt.text(j, i, cm_arr[i,j], ha="center", va="center")
        plt.xticks([0,1], ["Pred 0","Pred 1"]); plt.yticks([0,1], ["True 0","True 1"])
        plt.title(f"Confusion (thr={best_thr:.4f})"); plt.colorbar()
        plt.savefig(os.path.join(args.outdir, "fig_confusion.png"), dpi=200, bbox_inches="tight")
        plt.close()

    print(f"[train_rf] Saved model -> {args.model_out}")
    print(f"[train_rf] Threshold (valid, F1-opt) -> {best_thr:.4f}")
    print(f"[train_rf] Test metrics ->", meta["metrics_test"])

if __name__ == "__main__":
    main()
