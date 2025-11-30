
import pandas as pd, numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

LABELS = ["is_potent_any","label","y"]
SMILES = ["SMILES","smiles"]

df = pd.read_csv("data/dataset.csv")
label = next(c for c in LABELS if c in df)
smiles = next(c for c in SMILES if c in df)
df = df[[smiles,label]].dropna()
df = df.sample(n=min(128, len(df)), random_state=0)

X, y = [], []
for s, yy in zip(df[smiles], df[label]):
    m = Chem.MolFromSmiles(s)
    if m is None: continue
    fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
    arr = np.zeros((2048,), dtype=int)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    X.append(arr); y.append(int(yy))

clf = RandomForestClassifier(n_estimators=50, random_state=0, n_jobs=2)
clf.fit(X, y)
proba = clf.predict_proba(X)[:,1]
auc = roc_auc_score(y, proba)
print(f"SMOKE OK: n={len(y)}, RF ROC-AUC={auc:.3f}")
