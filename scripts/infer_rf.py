#!/usr/bin/env python3
"""
infer_rf.py — Minimal, reproducible inference for a calibrated Random Forest (RF)
on Morgan fingerprints, producing per-compound probabilities.

Usage
-----
python scripts/infer_rf.py \
  --in_csv data/dataset.csv \
  --out_csv results/pred_rf.csv \
  --model models/rf_calibrated.joblib \
  --smiles_col smiles \
  --nbits 2048 --radius 2 \
  --threshold 0.6906

Notes
-----
- Expects a column with SMILES (default: "smiles").
- Generates Morgan (ECFP-like) fingerprints and feeds them to a calibrated RF.
- Writes an output CSV preserving original rows and adds:
    * p_rf: predicted probability for the positive class
    * y_pred_rf (optional): binary decision if --threshold is provided
    * parse_ok: whether the SMILES was parsed successfully
- Rows with parse failures are retained with parse_ok=False and p_rf NaN.
- If a column `is_potent_any` exists, it will be preserved (useful for evaluation).
"""

import argparse
import sys
import os
import json
import numpy as np
import pandas as pd

from joblib import load
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

# Try newer generator API if available; otherwise fall back
try:
    from rdkit.Chem import rdFingerprintGenerator as FPGen
    HAS_FPGEN = True
except Exception:
    HAS_FPGEN = False


def morgan_fp_array(smiles_list, nbits=2048, radius=2):
    """Convert a list of SMILES to a 2D numpy array of Morgan fingerprints.
    Returns (X, keep_mask) where keep_mask[i] is True if smiles_list[i] parsed OK.
    """
    X = np.zeros((len(smiles_list), nbits), dtype=np.uint8)
    keep = np.zeros((len(smiles_list),), dtype=bool)
    if HAS_FPGEN:
        gen = FPGen.GetMorganGenerator(radius=radius, fpSize=nbits)
    for i, smi in enumerate(smiles_list):
        m = Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
        if m is None:
            continue
        if HAS_FPGEN:
            bv = gen.GetFingerprint(m)
        else:
            bv = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nbits)
        arr = np.zeros((nbits,), dtype=int)
        DataStructs.ConvertToNumpyArray(bv, arr)
        X[i, :] = arr.astype(np.uint8)
        keep[i] = True
    return X, keep


def main():
    ap = argparse.ArgumentParser(description="Inference with calibrated RF on Morgan fingerprints.")
    ap.add_argument("--in_csv", required=True, help="Input CSV with a SMILES column.")
    ap.add_argument("--out_csv", required=True, help="Output CSV with probabilities (and optional decisions).")
    ap.add_argument("--model", default="models/rf_calibrated.joblib", help="Path to calibrated RF (.joblib).")
    ap.add_argument("--smiles_col", default="smiles", help="Name of the SMILES column in input CSV.")
    ap.add_argument("--nbits", type=int, default=2048, help="Fingerprint bit size (default: 2048).")
    ap.add_argument("--radius", type=int, default=2, help="Morgan radius (default: 2).")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Optional decision threshold; if provided, outputs y_pred_rf.")
    ap.add_argument("--meta_json", default="models/model_meta.json",
                    help="Optional JSON with featurizer {nbits,radius} and default threshold.")
    args = ap.parse_args()

    # Load data
    df = pd.read_csv(args.in_csv)
    if args.smiles_col not in df.columns:
        sys.exit(f"[ERROR] SMILES column '{args.smiles_col}' not found in {args.in_csv}. Columns={list(df.columns)}")

    # If meta_json exists, let it override nbits/radius and threshold (unless CLI provided threshold)
    if os.path.exists(args.meta_json):
        try:
            meta = json.load(open(args.meta_json, "r"))
            nb = meta.get("featurizer", {}).get("nbits")
            rd = meta.get("featurizer", {}).get("radius")
            if nb: args.nbits = int(nb)
            if rd: args.radius = int(rd)
            if args.threshold is None:
                t_star = meta.get("decision", {}).get("threshold")
                if t_star is not None:
                    args.threshold = float(t_star)
        except Exception as e:
            print(f"[WARN] Failed to parse meta_json '{args.meta_json}': {e}", file=sys.stderr)

    # Compute fingerprints (keeping original row order, and a parse_ok mask)
    smiles = df[args.smiles_col].astype(str).tolist()
    X, keep_mask = morgan_fp_array(smiles, nbits=args.nbits, radius=args.radius)

    # Load model
    clf = load(args.model)

    # Predict probabilities for parseable rows
    p = np.full((len(df),), np.nan, dtype=float)
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X[keep_mask])
        # Convention: positive class is [:,1]
        if proba.shape[1] == 2:
            p[keep_mask] = proba[:, 1]
        else:
            # In rare cases with single-column proba, use column 0
            p[keep_mask] = proba[:, 0]
    else:
        # Fallback to decision_function then map to [0,1] via sigmoid
        from scipy.special import expit
        scores = clf.decision_function(X[keep_mask])
        p[keep_mask] = expit(scores)

    out = df.copy()
    out["parse_ok"] = keep_mask
    out["p_rf"] = p

    # Optional hard decision
    if args.threshold is not None:
        y_pred = np.where(np.isfinite(p) & (p >= args.threshold), 1, 0).astype("Int64")
        y_pred[~np.isfinite(p)] = pd.NA
        out["y_pred_rf"] = y_pred

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    # Small summary to stderr
    n_total = len(out)
    n_ok = int(out["parse_ok"].sum())
    n_fail = n_total - n_ok
    msg = f"[infer_rf] Done. n_total={n_total}, parsed={n_ok}, failed={n_fail}, wrote='{args.out_csv}'"
    if args.threshold is not None:
        msg += f", threshold={args.threshold}"
    print(msg, file=sys.stderr)


if __name__ == "__main__":
    main()
