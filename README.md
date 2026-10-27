# Milestone 03 - Baseline Models, Evaluation and Initial Findings

Out-of-school prediction for children aged 5-16 in Pakistan (ASER 2025). This folder is self-contained: the Milestone 2
input files are in `data/`, the brief and rubric in `docs/`, and every Milestone 3 deliverable is produced from them.

## Deliverables

| Rubric item | Where |
|---|---|
| Executable notebook | `Milestone03_BaselineModels.ipynb` (executed in place, outputs visible) |
| Modular scripts | `src/` - see module map below; `src/run_all.py` reproduces everything |
| Manuscript section | `report/Milestone03_Report.docx` (export to PDF from Word / Pages) |
| Tables and figures | `outputs/tables/t01..t21*.csv`, `outputs/figures/fig1..fig9*.png`, `outputs/results.json` |

## Inputs (`data/`, from Milestone 2 - read only)

| File | Content |
|---|---|
| `aser2025_training.csv` | 161,803 children x 17 columns: id, district, weight, target and the 13 features |
| `feature_manifest.json` | Column roles (target, features, id, CV group, weight) - loaded by `src/config.py` |
| `data_dictionary.csv` | Plain-English definition of every derived variable |
| `encoding_log.txt` | What Milestone 2 one-hot encoded and dropped, and why |
| `subgroup_counts.csv` | Sample sizes and prevalence per subgroup (used for edge cases and the M4 fairness roadmap) |
| `attrition.csv` | Raw 183,854 rows -> 161,803 after cleaning |
| `ITA_ASER2025_Coding_Manual.xlsx` | Original ASER survey codebook |

`docs/` holds the Milestone 3 brief (`Research Question + Instructions.docx`).

## Reproduce

```bash
cd milestone3
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/run_all.py                 # ~5 min: tuning (incl. random forest), metrics, cost analysis, importance, figures
.venv/bin/python src/make_report.py             # rebuild report/Milestone03_Report.docx from outputs/results.json
.venv/bin/jupyter nbconvert --to notebook --execute --inplace Milestone03_BaselineModels.ipynb
.venv/bin/python tools/build_notebook.py        # only if you edit the notebook cell definitions
```

Random state 42 throughout; the grouped folds, grid searches and permutation importances are deterministic.

## Module map (`src/`)

| Module | Role |
|---|---|
| `config.py` | Paths; column roles loaded from `data/feature_manifest.json` (target, 13 features, id, district group, weight); demographic subset for H5; cost ratios; thresholds grid |
| `data.py` | `load_training()` with integrity assertions (161,803 rows, no NaN, unique ids, non-overlapping province dummies); `split_xy`, `describe_split`, `province_of` |
| `cv.py` | `make_holdout()` - one fold of `StratifiedGroupKFold(5)` by district as test (~20%); `make_inner_cv()`; `tune()` = `GridSearchCV` on grouped folds refit on ROC-AUC; `cv_table`, `fold_scores` |
| *(threshold selection)* | In `run_all.py`: out-of-fold training probabilities via `cross_val_predict` on the same grouped folds; the cost-minimising threshold is chosen there and applied once to the hold-out. The hold-out is never used to pick a threshold. |
| `models.py` | `DummyClassifier(most_frequent)`; `StandardScaler -> LogisticRegression` pipeline + grid (C x class_weight); `DecisionTreeClassifier` + grid (max_depth x min_samples_leaf x class_weight); `RandomForestClassifier` (200 trees) + grid (max_depth x min_samples_leaf x class_weight) |
| `evaluate.py` | Accuracy, precision, recall, specificity, F1, NPV, ROC-AUC, PR-AUC (optionally survey-weighted); formatted confusion matrices; threshold sweep; expected cost per 1,000 for FN:FP ratios 1/3/5/10; per-1,000 operational translation; subgroup metrics; district-cluster bootstrap of an AUC difference |
| `importance.py` | statsmodels odds ratios with 95% CI; sklearn standardised coefficients; tree Gini importance; permutation importance on hold-out; H4 gender x area predicted cells |
| `plots.py` | ROC/PR, confusion heatmaps, threshold/cost curves, importance panels, odds-ratio forest, H5 fold plot, H4 bars, subgroup recall, tree top levels |
| `run_all.py` | Orchestrates the above and writes `outputs/` |
| `make_report.py` | Builds the Word report from `outputs/results.json` and the figures |

## Where each rubric requirement is met

**A. Baseline implementation and architecture selection**
- Naive benchmark: `models.build_naive` -> notebook section 3, Table 3 / `t07`.
- Three ML models: logistic regression, decision tree and random forest (`models.py`), tuned in notebook section 3 and `run_all.py`.
- Justification: notebook section 3 table; report section 2.

**B. Experimental setup and cross-validation**
- Partitioning: district-grouped, target-stratified hold-out (`cv.make_holdout`, asserts zero district overlap); `t01_split_description.csv`.
- Validation framework: inner `StratifiedGroupKFold(5)` on the training districts; per-fold variance in `t05_cv_best_summary.csv`.
- Hyperparameter exploration: full grids in `t02_cv_grid_logreg.csv`, `t03_cv_grid_tree.csv`, `t03b_cv_grid_rf.csv`, `t04_cv_grid_logreg_demo.csv`.
- H5 required comparison (3 vs 13 features, same folds, paired t-test plus a district-cluster bootstrap of the hold-out gain): `t06_h5_auc_by_fold.csv`, `results.json -> h5.holdout_bootstrap`, `fig6`.

**C. Evaluation metrics and asymmetric error cost**
- Metrics table (unweighted and ASER-weighted): `t07_holdout_metrics_threshold_0.5.csv`, `fig1`.
- Confusion matrices: `t11_confusion_*.csv`, `fig2_confusion_matrices_default.png`, `fig2_confusion_matrices_recommended.png`.
- Asymmetric cost: threshold sweeps on training OOF and hold-out `t08`; thresholds chosen on training OOF with hold-out metrics and the hold-out's own optimum for comparison `t09`; default vs recommended operating point with per-1,000 translation `t10`; `fig3`.
- Edge cases: misclassified profile `t12`, outcome share by province `t13`, subgroup metrics `t14`, `fig8`.

**D. Feature importance and insights**
- Odds ratios with CIs `t15`, `fig5`; standardised coefficients `t16`; Gini `t17` (tree) and `t17b` (forest); permutation `t18`/`t19`/`t19b`; `fig4`, `fig9`, `fig10`.
- Hypotheses H1-H5 evidence table `t21_hypothesis_evidence.csv`; H4 gender x area cells `t20`, `fig7`.
- Policy takeaways and Milestone 4 roadmap: notebook sections 9-10; report sections 7-8.

## Headline results (hold-out districts, n = 32,516)

| Model | ROC-AUC (unweighted / ASER-weighted) | PR-AUC | Recall @0.5 | Threshold chosen on training OOF (FN:FP = 5) | Hold-out recall / precision there |
|---|---|---|---|---|---|
| Majority class | 0.500 / 0.500 | 0.125 | 0.00 | - | - |
| Logistic regression (balanced, C = 0.001) | 0.674 / 0.713 | 0.214 | 0.63 | 0.54 | 0.57 / 0.20 |
| Decision tree (depth 10, leaf >= 200) | 0.711 / 0.749 | 0.236 | 0.04 | 0.16 | 0.62 / 0.24 |
| Random forest (200 trees, depth 12, leaf >= 20) | 0.725 / 0.755 | 0.254 | 0.04 | 0.13 | 0.71 / 0.22 |

Hold-out expected cost per 1,000 children at those thresholds (FN:FP = 5): tree 487, forest 496, logistic 552. The tree is the
recommended operating point; the forest ranks best (highest AUC) but its training-chosen threshold sits slightly below its hold-out
optimum, and the 2% cost gap is within fold-to-fold noise.

H5: full model +0.15 AUC over the demographic model in every CV fold (paired t p = 0.010); hold-out gain +0.13 with a
district-cluster bootstrap 95% interval of about [+0.08, +0.17]. All five Milestone 2 null
hypotheses are rejected; housing quality and digital access are the strongest household drivers, disability and
Balochistan residence the strongest risk factors, and the gender gap is concentrated in rural areas.

## Notes and caveats

- All drivers are conditional associations from one cross-sectional survey, not causal effects; the report says so before
  the policy takeaways.
- The operating threshold is selected on training out-of-fold predictions, never on the hold-out; `t09` also reports the
  hold-out's own optimum so the size of the avoided flattery is visible (about 1% of cost for logistic regression, none for the tree).

- Islamabad Capital Territory is a single district and fell entirely into the training side of the grouped split,
  so it has no hold-out metrics; with 52 positives overall it is flagged as too small for subgroup conclusions.
- `run_all.py` takes about 5 minutes end to end; the random forest grid (18 settings x 5 folds x 200 trees) is most of it.
- `encoding_log.txt` mentions imputing inside folds; the training file has no missing values, so no imputer is used
  (and `data.load_training` asserts this).
- The report is a standalone Milestone 3 section intended to be pasted into the team manuscript after the Milestone 2
  draft sections.
