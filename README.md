# Prob2Decision: Hybrid QSAR for Veterinary Macrolides (RF + D-MPNN)

Decision-focused QSAR pipeline that turns **probabilities into decisions**:
- **Calibrated Random Forest (RF; Morgan fingerprints)**
- **Validation-set thresholding (F1-optimal)**
- **Test-set evaluation** (ROC/PR, Brier/ECE, confusion matrix)

This bundle is **reviewer-ready**: it contains a standardized dataset, minimal training & inference scripts, environment, and a one-click runner.

## 🚀 Quickstart
```bash
conda env create -f env/environment.yml
conda activate gbsdl

# Train + calibrate RF, select threshold on valid, evaluate on test, and plot
bash run_all.sh
```

## Data
- `data/dataset.csv` with columns: `smiles,is_potent_any,split` where `split∈{train,valid,test}`.
- SMILES canonicalized via RDKit; labels are binary (1=potent under any tested condition).

## Outputs
- `models/rf_calibrated.joblib` — calibrated RF model
- `models/model_meta.json` — featurizer config and selected threshold
- `results/pred_valid.csv`, `results/pred_test.csv` — probabilities per row
- `results/metrics_test.json` — AUC/AP, Brier, ECE, F1/precision/recall
- `results/fig_roc.png`, `results/fig_pr.png`, `results/fig_calibration.png`, `results/fig_confusion.png`

## License & Citation
- Code: MIT (see `LICENSE`)
- Cite using `CITATION.cff` (GitHub provides citation formats automatically)

## Notes
- For D-MPNN/Chemprop, you can extend this repo later; the evaluation protocol stays the same.
