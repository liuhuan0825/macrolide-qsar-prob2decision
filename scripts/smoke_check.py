import pandas as pd, numpy as np
from pathlib import Path


LABEL_CANDS = ["is_potent_any", "label", "y"]
SMILES_CANDS = ["SMILES", "smiles"]

def pick_col(df, cands):
    for c in cands:
        if c in df.columns: return c
    raise KeyError(f"Columns not found: {cands}")

df = pd.read_csv("data/dataset.csv")
smiles_col = pick_col(df, SMILES_CANDS)
label_col  = pick_col(df, LABEL_CANDS)

df = df[[smiles_col, label_col]].dropna()
df = df.sample(n=min(128, len(df)), random_state=0)
from rdkit import Chem
from rdkit.Chem import AllChem

def featurize(s):
    m = Chem.MolFromSmiles(s)
    if m is None: return None
    fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
    arr = np.zeros((1,), dtype=int)
    # 转成 python 列表
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.tolist()

rows = []
for s, y in zip(df[smiles_col].tolist(), df[label_col].tolist()):
    x = featurize(s)
    if x is not None:
        rows.append((x, int(y)))

X = [r[0] for r in rows]
y = [r[1] for r in rows]

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

clf = RandomForestClassifier(n_estimators=50, random_state=0, n_jobs=2)
clf.fit(X, y)
proba = clf.predict_proba(X)[:, 1]
auc = roc_auc_score(y, proba)

print(f"SMOKE OK: n={len(y)}, RF ROC-AUC={auc:.3f}")
