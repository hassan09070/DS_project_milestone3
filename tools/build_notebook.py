"""Generate Milestone03_BaselineModels.ipynb from cell definitions (run once, then execute with nbconvert)."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Milestone 03 - Baseline Models for Out-of-School Prediction (ASER 2025, Pakistan)

**Research question.** Among children aged 5-16 in Pakistan, can we predict whether a child is out of school
(never enrolled or dropped out) from household poverty indicators, legal identity documentation, disability status
and demographic / geographic characteristics - and does the model work equally well for girls, children with
disabilities and rural children? (SDG 4.1 access, SDG 4.5 disparities.)

**What this notebook does.** It implements the Milestone 3 brief end to end: a naive benchmark plus two baseline
learners (logistic regression, decision tree), a leakage-free district-grouped stratified cross-validation framework
with hyperparameter tuning, the full metric set with confusion matrices, an asymmetric error-cost analysis with
threshold shifting, feature-importance drivers read against the five Milestone 2 hypotheses, and the roadmap for
Milestone 4. All modelling code lives in `src/` and is imported here so the notebook stays readable; `src/run_all.py`
reproduces every table and figure from the command line.

**Data.** `data/aser2025_training.csv` (161,803 children x 17 columns) built in Milestone 2. Column roles come from
`data/feature_manifest.json`; nothing is hard-coded.
""")

code("""
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40); pd.set_option("display.precision", 4)

sys.path.insert(0, str(Path("src").resolve()))
import config as C
from data import load_training, split_xy, describe_split, load_subgroup_counts, load_attrition, province_of
from cv import make_holdout, make_inner_cv, tune, cv_table, fold_scores
import models as M, evaluate as ev, importance as imp, plots

print("Target:", C.TARGET)
print("Features (%d):" % len(C.FEATURES), C.FEATURES)
print("ID / CV group / weight:", C.ID_COL, "/", C.GROUP_COL, "/", C.WEIGHT_COL)
""")

md("""
## 1. Data and integrity checks

`load_training()` asserts the properties documented in `encoding_log.txt`: 161,803 rows, zero missing values in the
features and target, unique child ids, and province dummies that never overlap (Punjab is the reference category).
The attrition table shows how the raw 183,854 ASER records became this analysis sample.
""")

code("""
df = load_training()
X, y, groups, w = split_xy(df)
print(f"{df.shape[0]:,} children x {df.shape[1]} columns; {df[C.GROUP_COL].nunique()} districts")
print(f"out_of_school = 1: {y.sum():,} ({y.mean():.1%} unweighted, {np.average(y, weights=w):.1%} survey-weighted)")
display(load_attrition())
display(df[C.FEATURES + [C.TARGET]].describe().T[["mean", "std", "min", "max"]])
""")

md("""
**Interpretation.** The target is imbalanced (13.2% positive). A classifier that always predicts "enrolled" scores
~87% accuracy while finding no out-of-school child at all, so accuracy is not a useful headline metric here; we lean
on recall, F1, ROC-AUC and PR-AUC. All predictors are binary or small-integer ordinal; province is already one-hot
encoded, so no further encoding or scaling is applied outside the model pipelines (the logistic pipeline standardises
inside each fold).

## 2. Partitioning: district-grouped, target-stratified hold-out

Many children share a district, and districts differ strongly in enrolment. A plain random split would let the
model see children from the same district on both sides and over-state performance. We therefore use
`StratifiedGroupKFold(5)` grouped by `district` and stratified on the target: one fold (~20% of children, whole
districts) is the untouched hold-out test set; the remaining ~80% is used for tuning with a second
`StratifiedGroupKFold(5)` (the *inner* CV). The assertion below proves that no district appears on both sides.
""")

code("""
train_idx, test_idx = make_holdout(y, groups)       # raises if a district leaks across the split
split_desc = describe_split(df, train_idx, test_idx)
display(split_desc)
Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
ytr, yte = y[train_idx], y[test_idx]
gtr = groups[train_idx]; wte = w[test_idx]
df_te = df.iloc[test_idx].reset_index(drop=True)
print("districts overlapping:", len(set(groups[train_idx]) & set(groups[test_idx])))
print("inner CV:", make_inner_cv())
""")

md("""
## 3. Models and justification

| Model | Why it is a sensible baseline here |
|---|---|
| **Majority-class benchmark** (`DummyClassifier(most_frequent)`) | Fixes the floor: predicts "enrolled" for everyone. Shows how misleading raw accuracy is at 13% prevalence. |
| **Logistic regression** (`StandardScaler` -> L2-penalised LR) | Our hypotheses H1-H4 are statements about the *direction and size* of individual effects (assets, B-form, disability, gender x area). Logistic coefficients give odds ratios with confidence intervals that answer them directly. Features are binary / ordinal, so a linear-in-log-odds model is a natural first fit. Tuned: regularisation strength `C`, class weighting. |
| **Decision tree** (CART, Gini) | Captures non-linear thresholds (e.g. asset counts) and interactions (H4) without hand-specification, is fully interpretable through its splits and Gini importance, and is cheap on 161k rows. Tuned: `max_depth`, `min_samples_leaf`, class weighting. |

Both learners are tuned with `GridSearchCV` on the district-grouped inner folds, refitting on ROC-AUC
(threshold-free, imbalance-robust); recall, precision, F1, accuracy and PR-AUC are recorded per fold as well.
""")

code("""
naive = M.build_naive().fit(Xtr, ytr)
probas = {"naive": naive.predict_proba(Xte)[:, 1]}

gs_lr = tune(M.build_logreg(), M.LOGREG_GRID, Xtr, ytr, gtr)
probas["logreg"] = gs_lr.predict_proba(Xte)[:, 1]
print("Logistic regression - best:", gs_lr.best_params_, f"CV ROC-AUC = {gs_lr.best_score_:.4f}")
display(cv_table(gs_lr)[["C", "class_weight", "mean_test_roc_auc", "std_test_roc_auc", "mean_test_recall", "mean_test_f1", "mean_test_average_precision"]])
""")

code("""
gs_tree = tune(M.build_tree(), M.TREE_GRID, Xtr, ytr, gtr)
probas["tree"] = gs_tree.predict_proba(Xte)[:, 1]
print("Decision tree - best:", gs_tree.best_params_, f"CV ROC-AUC = {gs_tree.best_score_:.4f}")
display(cv_table(gs_tree)[["max_depth", "min_samples_leaf", "class_weight", "mean_test_roc_auc", "std_test_roc_auc", "mean_test_recall", "mean_test_f1"]].head(12))
""")

md("""
**Interpretation of the tuning.** For logistic regression the strongest regularisation (`C = 0.001`) with balanced
class weights wins on AUC; the ranking is flat across `C` (the coefficients are already well-determined with 129k
rows), while balancing matters for recall at the default threshold. For the tree, moderate depth (10) with a large
minimum leaf (200 children) is best - deeper, unconstrained trees over-fit district idiosyncrasies and lose AUC on
unseen districts. Fold-to-fold standard deviations of ~0.05 AUC show how much performance varies between groups of
districts, which is exactly the variance a random split would have hidden.

## 4. Hypothesis H5 - do poverty, B-form, disability and the controls add predictive value?

Same inner folds, same learner, two feature sets: the bare demographic model (`age, female, urban`) versus the
full 13-feature model. The paired per-fold AUC difference is tested with a paired t-test.
""")

code("""
from scipy import stats
gs_demo = tune(M.build_logreg(), M.LOGREG_GRID, Xtr[C.DEMO_FEATURES], ytr, gtr)
probas["logreg_demo"] = gs_demo.predict_proba(Xte[C.DEMO_FEATURES])[:, 1]
auc_demo, auc_full = fold_scores(gs_demo, "roc_auc"), fold_scores(gs_lr, "roc_auc")
t, p = stats.ttest_rel(auc_full, auc_demo)
h5 = pd.DataFrame({"fold": range(1, 6), "auc_demographic": auc_demo, "auc_full": auc_full, "gain": auc_full - auc_demo})
display(h5)
print(f"mean AUC demographic = {auc_demo.mean():.4f} (sd {auc_demo.std(ddof=1):.4f}); full = {auc_full.mean():.4f} (sd {auc_full.std(ddof=1):.4f})")
print(f"mean gain = {(auc_full-auc_demo).mean():+.4f}; paired t = {t:.2f}, p = {p:.4f}")
print(f"hold-out AUC: demographic {ev.metrics(yte, probas['logreg_demo'])['roc_auc']:.4f} vs full {ev.metrics(yte, probas['logreg'])['roc_auc']:.4f}")
display(Image(plots.h5_auc_folds(auc_demo, auc_full)))
""")

md("""
**H5 verdict.** The full model beats the demographic model in every fold (gain ~ +0.15 AUC, p ~ 0.01) and on
the hold-out districts. Age, sex and area alone are close to chance (AUC ~ 0.59): *who is out of school is far
better explained by household circumstances than by demographics*. We reject H0 for H5.

## 5. Hold-out performance at the default threshold (0.5)

Metrics on the 32,516 children in the held-out districts. A survey-weighted version (ASER sampling weights) is
reported as a robustness check; the unweighted numbers are the primary ones.
""")

code("""
rows = []
for key in ("naive", "logreg_demo", "logreg", "tree"):
    rows.append({"model": M.MODEL_NAMES[key], "weighted": "no", **ev.metrics(yte, probas[key], 0.5)})
    rows.append({"model": M.MODEL_NAMES[key], "weighted": "yes", **ev.metrics(yte, probas[key], 0.5, w=wte)})
holdout = pd.DataFrame(rows)
cols = ["model", "weighted", "accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc", "flagged_rate"]
display(holdout[cols])
display(Image(plots.roc_pr_curves(yte, {k: probas[k] for k in ("logreg_demo", "logreg", "tree")}, M.MODEL_NAMES, float(yte.mean()))))
""")

md("""
**Interpretation.** The majority-class benchmark reaches 87.5% accuracy with zero recall - every out-of-school
child is missed. The balanced logistic regression trades accuracy (62%) for recall (63%) at the default cut-off. The
tree has the better ranking quality (ROC-AUC 0.71, PR-AUC 0.24) but, because it was tuned without class weights, its
default-threshold recall is only ~4%: almost no leaf reaches a 50% out-of-school share. This is the clearest
demonstration that **the threshold, not the learner, decides who gets flagged** - which is the subject of the next
section. Hold-out AUCs are ~0.03-0.06 below the CV means: the held-out districts are a harder group, within the
fold-to-fold spread seen in CV.

## 6. Confusion matrices and asymmetric error costs

*Policy reading of the two error types.*
- **False negative** - the model says "in school" but the child is out of school: a child who needs an enrolment /
  outreach intervention is never visited. This is the costly error for an education-access programme.
- **False positive** - the model says "out of school" but the child is enrolled: an outreach visit or a stipend slot is
  spent on a child who did not need it; wasteful, but recoverable.

We make this explicit by assigning a false negative a cost of *r* false positives (r = 1, 3, 5, 10), sweeping the
classification threshold from 0.05 to 0.95 and reporting the expected cost per 1,000 children screened. The
recommended operating point uses r = 5.
""")

code("""
cms_05 = {k: {**ev.confusion(yte, (probas[k] >= 0.5).astype(int)), "threshold": 0.5} for k in ("naive", "logreg", "tree")}
for k in ("naive", "logreg", "tree"):
    print(M.MODEL_NAMES[k], "@ 0.50"); display(ev.confusion_table(cms_05[k]))
display(Image(plots.confusion_heatmaps(cms_05, M.MODEL_NAMES, "_default")))
""")

code("""
sweeps, best = {}, []
for k in ("logreg", "tree"):
    sweeps[k] = ev.threshold_sweep(yte, probas[k])
    b = ev.best_thresholds(sweeps[k]); b.insert(0, "model", k); best.append(b)
best = pd.concat(best, ignore_index=True)
display(best)
display(Image(plots.threshold_curves(sweeps, M.MODEL_NAMES, C.RECOMMENDED_COST_RATIO)))
""")

code("""
rec_thr = {k: float(best[(best.model == k) & (best.cost_ratio_fn_to_fp == C.RECOMMENDED_COST_RATIO)].threshold.item()) for k in ("logreg", "tree")}
cms_rec = {k: {**ev.confusion(yte, (probas[k] >= t).astype(int)), "threshold": t} for k, t in rec_thr.items()}
rows = []
for k in ("naive", "logreg", "tree"):
    for lab, cm in (("0.50 default", cms_05[k]), ("recommended", cms_rec.get(k))):
        if cm is None: continue
        rows.append({"model": k, "threshold": lab, "t": cm["threshold"], **ev.metrics_from_counts(cm), **ev.per_1000(cm),
                     "cost/1000 (r=5)": ev.expected_cost(cm, 5)})
display(pd.DataFrame(rows)[["model", "threshold", "t", "recall", "precision", "specificity", "f1", "flagged_per_1000", "caught_per_1000", "missed_per_1000", "false_alarms_per_1000", "cost/1000 (r=5)"]])
for k in ("logreg", "tree"):
    print(M.MODEL_NAMES[k], f"@ {rec_thr[k]:.2f}"); display(ev.confusion_table(cms_rec[k]))
display(Image(plots.confusion_heatmaps(cms_rec, M.MODEL_NAMES, "_recommended")))
""")

md("""
**Interpretation.**
- With r = 1 (errors equally costly) the cheapest policy is to flag nobody: the models are not sharp enough to beat
  the base rate when a false alarm costs as much as a missed child. This is the honest baseline the cost framing
  exposes.
- Once a missed child is weighted 3-5x a wasted visit, the optimum moves to a low threshold that flags roughly a
  third of all children and catches 50-60% of those out of school. At r = 10 the models flag over half of all
  children to catch ~78%.
- **Recommendation for deployment (r = 5).** Decision tree at threshold ~0.16: per 1,000 children screened it flags
  ~330, catches ~78 of the ~125 out-of-school children and misses ~47, at the lowest expected cost of the three
  policies. The default 0.5 cut-off on the same tree would catch only ~5 per 1,000 - unusable for outreach. Logistic
  regression at ~0.57 is a close second and is preferable where officials need to explain *why* a child was flagged.
- Which error is "better" to make depends on budget: a programme that can afford to visit a third of households
  should accept the false positives; one that cannot should raise the threshold and accept more missed children,
  explicitly.

## 7. Misclassified edge cases

Who does the logistic model miss (false negatives) compared with the children it catches (true positives)?
""")

code("""
thr = rec_thr["logreg"]; yhat = (probas["logreg"] >= thr).astype(int)
outcome = np.select([(yte == 1) & (yhat == 1), (yte == 1) & (yhat == 0), (yte == 0) & (yhat == 1)], ["TP", "FN", "FP"], "TN")
prof = df_te[C.FEATURES].copy(); prof["province"] = province_of(df_te); prof["outcome"] = outcome; prof["p_hat"] = probas["logreg"]
num = [f for f in C.FEATURES if not f.startswith("province_")] + ["p_hat"]
profile = prof.groupby("outcome")[num].mean().T; profile["FN - TP"] = profile["FN"] - profile["TP"]
display(profile)
display(pd.crosstab(prof["province"], prof["outcome"], normalize="index").assign(n=prof["province"].value_counts()))
""")

code("""
groupings = {"female": df_te["female"], "urban": df_te["urban"], "any_disability": df_te["any_disability"], "bform": df_te["bform"], "province": province_of(df_te)}
sub = ev.subgroup_metrics(df_te, yte, probas["logreg"], thr, groupings)
display(sub[["variable", "level", "n", "positives", "prevalence", "recall", "fnr", "precision", "roc_auc", "flag_small"]])
display(Image(plots.subgroup_recall_plot(sub, f"Logistic regression recall by subgroup (threshold {thr:.2f}); grey = < 100 positives")))
print("Milestone 2 subgroup sample sizes (population):"); display(load_subgroup_counts())
""")

md("""
**Systematic weaknesses.** The children the model misses look like *enrolled* children on paper: they are older,
more often urban (43% vs 14% of true positives), more often B-form registered, and come from households with more
assets, more digital use and better housing. In other words the model has learned "poor rural household => out of
school" and under-detects **out-of-school children in better-off or urban households** - plausibly dropouts rather
than never-enrolled children, a distinction the current features cannot see. Recall is far higher in Balochistan
(~0.9, where prevalence is 26%) than in Punjab or Sindh (~0.25-0.35): the model leans on the province prior. Urban
children have recall ~0.26 versus ~0.61 for rural children. Islamabad Capital Territory contains a single district and
fell entirely into the training side of the grouped split, so it has no hold-out estimate at all - and with 52
positives in the whole dataset it is too small for any reliable subgroup conclusion.

## 8. Feature importance and the five hypotheses

Three complementary views: odds ratios with 95% confidence intervals from an unpenalised logistic fit on the training
districts (inference), standardised coefficients from the tuned sklearn pipeline, Gini importance from the tuned tree,
and permutation importance (drop in hold-out ROC-AUC when a feature is shuffled) for both learners.
""")

code("""
ors = imp.logit_odds_ratios(Xtr, ytr)
display(ors[["label", "odds_ratio", "or_ci_low", "or_ci_high", "p_value", "std_coef"]])
print(f"McFadden pseudo-R2 = {ors['pseudo_r2'].iloc[0]:.3f}")
display(Image(plots.odds_ratio_forest(ors)))
""")

code("""
sk_coef = imp.sklearn_logreg_coefs(gs_lr.best_estimator_)
gini = imp.tree_gini(gs_tree.best_estimator_)
perm_lr = imp.permutation(gs_lr.best_estimator_, Xte, yte)
perm_tree = imp.permutation(gs_tree.best_estimator_, Xte, yte)
rank = pd.DataFrame({"feature": C.FEATURES})
rank["LR |std coef| rank"] = rank.feature.map({f: i + 1 for i, f in enumerate(sk_coef.feature)})
rank["Tree Gini rank"] = rank.feature.map({f: i + 1 for i, f in enumerate(gini.feature)})
rank["LR permutation rank"] = rank.feature.map({f: i + 1 for i, f in enumerate(perm_lr.feature)})
rank["Tree permutation rank"] = rank.feature.map({f: i + 1 for i, f in enumerate(perm_tree.feature)})
display(rank.sort_values("Tree Gini rank"))
display(Image(plots.importance_bars(sk_coef, gini, perm_lr, perm_tree)))
display(Image(plots.tree_top_levels(gs_tree.best_estimator_, C.FEATURES)))
""")

code("""
cells = imp.gender_area_cells(gs_lr.best_estimator_, Xtr)
display(cells)
print(f"gender gap (girl - boy) rural = {cells.attrs['rural_gender_gap']:+.3f}; urban = {cells.attrs['urban_gender_gap']:+.3f}")
display(Image(plots.gender_area_plot(cells)))
""")

md("""
### Reading the drivers against the Milestone 2 hypotheses

| Hypothesis | Variable | Evidence | Verdict |
|---|---|---|---|
| **H1 Poverty** - higher asset count lowers the odds | `assets_n` | OR ~ 0.96 per additional asset (95% CI excludes 1); 4th in Gini importance. Direction as hypothesised, but the effect per asset is small - **housing quality** (`house`, OR ~ 0.56 per step kachha -> pacca) and **digital access** (`digital_n`, OR ~ 0.82 per service) are the dominant poverty signals in both learners. | H0 rejected; poverty matters, but the asset *count* is a weaker proxy than housing. |
| **H2 Legal identity** - unregistered children have higher odds | `bform` | OR ~ 0.75 for registered children (equivalently 1.33x higher odds without a B-form), highly significant; consistent across LR, Gini and permutation. | H0 rejected. |
| **H3 Disability** - children with a disability have higher odds | `any_disability` | OR ~ 1.88, the largest single risk factor after Balochistan residence; 4th by standardised coefficient. | H0 rejected. |
| **H4 Gender x area** - the gender gap differs by area | `female_x_urban` | Interaction OR ~ 0.77 (CI excludes 1). Girls face higher odds everywhere, but the predicted gap is ~8.8 pp in rural areas versus ~3.3 pp in urban areas: the gender penalty is a *rural* phenomenon. The tree barely uses the interaction term (it recreates it via sequential splits). | H0 rejected. |
| **H5 Feature value** - the full model improves AUC over demographics | 13 vs 3 features | +0.15 AUC in CV (every fold, p ~ 0.01) and +0.13 on hold-out. | H0 rejected. |

Controls: Balochistan residence raises the odds ~1.8x even after the household variables; age has a small negative
coefficient within the 5-16 window.

## 9. Initial policy takeaways

1. **Target by housing and connectivity, not just an asset count.** Housing quality and digital access are the strongest
   household predictors; outreach lists built from a simple asset score will miss children in kachha, disconnected homes.
2. **Birth registration is an actionable lever.** Unregistered children have ~1.3x the odds of being out of school.
   Linking NADRA B-form drives to school enrolment campaigns targets a population the model identifies as high-risk.
3. **Disability inclusion is a first-order gap (SDG 4.5).** Children with a functional difficulty have ~1.9x the odds;
   screening and inclusive-school capacity should be part of any enrolment drive.
4. **The gender gap is rural.** Girl-focused interventions (stipends, transport, female teachers) should be prioritised
   in rural areas, where the predicted penalty is almost three times the urban one.
5. **Balochistan needs a province-level response.** Even after household factors, residence there nearly doubles the
   odds; the model's high recall there reflects a prior, not fine-grained targeting.
6. **Set the operating threshold explicitly.** At a 5:1 cost ratio the recommended policy flags ~1 in 3 children and
   catches ~60% of out-of-school children; the default 0.5 threshold is not a defensible choice for outreach.

## 10. Limitations and roadmap for Milestone 04

**Current limitations.**
- *Underfitting / modest discrimination.* Hold-out AUC 0.67-0.71 and pseudo-R2 ~ 0.12: 13 household features cannot
  separate the harder cases (urban or better-off dropouts). The missing signal is likely school-side (distance, supply,
  cost) and child history.
- *Class imbalance.* Handled with class weights and threshold tuning, not resampling; precision stays ~0.2 at useful
  recall levels.
- *District heterogeneity.* CV standard deviation ~0.05 AUC across district groups; the hold-out fold is harder than the
  CV average.
- *Potential demographic bias.* Recall differs sharply by area (urban 0.26 vs rural 0.61), by province (Balochistan 0.9 vs
  Sindh 0.24) and by B-form status; girls and disabled children are currently caught *more* often than their counterparts
  because they carry higher predicted risk.

**Plan for Milestone 04.**
1. *Fairness audit* using the subgroup frame in `subgroup_counts.csv`: equalised recall / precision / FNR for girls vs
   boys, rural vs urban, with vs without disability, and province, with bootstrap confidence intervals; test whether
   per-group thresholds or reweighting close the urban and Punjab/Sindh recall gaps. Islamabad Capital Territory
   (n = 807, 52 positives) is flagged as too small for reliable group conclusions and will be reported but not acted on.
2. *Advanced modelling*: gradient boosting / random forest with the same grouped CV; calibrated probabilities; interaction
   terms suggested by the tree; a cost-sensitive objective aligned with the FN:FP ratio.
3. *Ethical evaluation*: harms of mis-targeting (stigma from false flags, exclusion from missed children), use of survey
   weights for population-level claims, and a deployment note on who sets the cost ratio.

## 11. Reproducibility

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/run_all.py        # all tables -> outputs/tables, figures -> outputs/figures, outputs/results.json
.venv/bin/python src/make_report.py    # report/Milestone03_Report.docx
.venv/bin/jupyter nbconvert --to notebook --execute --inplace Milestone03_BaselineModels.ipynb
```
Random state 42 everywhere; the grouped folds, grid searches and permutation importances are deterministic.
""")

code("""
import sklearn, statsmodels, platform
print("python", platform.python_version(), "| pandas", pd.__version__, "| scikit-learn", sklearn.__version__, "| statsmodels", statsmodels.__version__)
""")

nb.cells = cells
out = Path(__file__).resolve().parents[1] / "Milestone03_BaselineModels.ipynb"
nbf.write(nb, out)
print("wrote", out, len(cells), "cells")
