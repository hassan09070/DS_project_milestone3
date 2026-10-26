"""Build report/Milestone03_Report.docx from outputs/results.json and the saved figures.

    .venv/bin/python src/make_report.py

The report is written as a standalone Milestone 3 section (model architectures, experimental setup, metrics,
asymmetric error cost, feature importance, policy findings, M4 roadmap) ready to be pasted into the team manuscript.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

R = json.load(open(C.OUT_DIR / "results.json"))
T = C.TABLES_DIR
NAMES = {"naive": "Majority-class baseline", "logreg": "Logistic regression", "tree": "Decision tree",
         "logreg_demo": "Logistic regression (age, female, urban)"}


def fmt(x, nd=3):
    if isinstance(x, (int,)) and not isinstance(x, bool):
        return f"{x:,}"
    if isinstance(x, float):
        if pd.isna(x):
            return "-"
        return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.{nd}f}"
    return str(x)


def add_table(doc, df: pd.DataFrame, nd=3, font=8.5, caption: str | None = None):
    if caption:
        p = doc.add_paragraph(); r = p.add_run(caption); r.bold = True; r.font.size = Pt(9.5)
    t = doc.add_table(rows=1, cols=len(df.columns))
    t.style = "Light Grid Accent 1"
    for i, c in enumerate(df.columns):
        cell = t.rows[0].cells[i]; cell.text = str(c)
        for p in cell.paragraphs:
            for r in p.runs: r.bold = True; r.font.size = Pt(font)
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = fmt(v, nd)
            for p in cells[i].paragraphs:
                for r in p.runs: r.font.size = Pt(font)
    doc.add_paragraph()


def add_fig(doc, key_or_path, caption, width=6.3):
    path = R["figures"].get(key_or_path, key_or_path)
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph(); r = p.add_run(caption); r.italic = True; r.font.size = Pt(9)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER


def bullets(doc, items):
    for it in items:
        doc.add_paragraph(it, style="List Bullet")


def main():
    h05 = R["holdout_05"]; cvb = {r["model"]: r for r in R["cv_best"]}; h5 = R["h5"]
    tc = {(r["model"], r["threshold_label"]): r for r in R["threshold_comparison"]}
    ors = {r["feature"]: r for r in R["odds_ratios"]}
    gini = {r["feature"]: r for r in R["gini"]}
    sub = pd.DataFrame(R["subgroups_logreg"]).set_index(["variable", "level"])
    rec = R["recommended_threshold"]; ratio = R["recommended_cost_ratio"]
    split = {r["partition"]: r for r in R["split"]}
    ds = R["dataset"]
    tree_rec = tc[("tree", "recommended")]; lr_rec = tc[("logreg", "recommended")]
    bs = h5["holdout_bootstrap"]; w05 = R["holdout_05_weighted"]
    ct = {(r["model"], r["cost_ratio_fn_to_fp"]): r for r in R["cost_thresholds"]}
    tree_05 = tc[("tree", "0.50 (default)")]; lr_05 = tc[("logreg", "0.50 (default)")]

    doc = Document()
    st = doc.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(10.5)

    doc.add_heading("Milestone 03 - Baseline Models, Evaluation and Initial Findings", 0)
    doc.add_paragraph("Predicting out-of-school status among children aged 5-16 in Pakistan (ASER 2025) - "
                      "manuscript Section IV draft: Methods (modelling), Results (baseline) and preliminary Discussion.").italic = True

    # ---------------------------------------------------------------- 1
    doc.add_heading("1. Modelling objective and data", 1)
    doc.add_paragraph(
        f"The target is out_of_school (1 = never enrolled or dropped out, 0 = currently enrolled). The analysis file built in "
        f"Milestone 2 holds {ds['rows']:,} children from {ds['districts']} districts, of whom {ds['positives']:,} "
        f"({ds['prevalence']:.1%}) are out of school ({ds['weighted_prevalence']:.1%} when ASER sampling weights are applied). "
        f"The 13 predictors are age, sex, area, the sex x area interaction, B-form registration, any functional disability, a "
        f"household asset count (0-8), a digital-use count (0-5), housing structure (1-3) and four province dummies with Punjab "
        f"as reference. Columns were used exactly as prepared in Milestone 2 (feature_manifest.json); the id, district and weight "
        f"columns are never used as predictors. The data contain no missing values, so no imputation step was needed.")

    # ---------------------------------------------------------------- 2
    doc.add_heading("2. Baseline architectures and justification", 1)
    bullets(doc, [
        "Naive benchmark - majority-class classifier (predicts 'enrolled' for every child). It fixes the performance floor and "
        f"shows why accuracy is misleading at {ds['prevalence']:.1%} prevalence: it scores {h05['naive']['accuracy']:.1%} accuracy with zero recall.",
        "Logistic regression (standardised inputs, L2 penalty). Hypotheses H1-H4 concern the direction and size of individual "
        "effects, which logistic coefficients answer directly as odds ratios with confidence intervals. All predictors are binary "
        "or short ordinal scales, so a model linear in the log-odds is an appropriate first fit. Tuned: regularisation strength C "
        "and class weighting.",
        "Decision tree (CART, Gini). Captures non-linear thresholds in the asset and digital counts and interactions such as the "
        "gender x area effect (H4) without hand specification, remains interpretable through its splits and Gini importances, and "
        "is cheap on 161k rows. Tuned: maximum depth, minimum leaf size and class weighting.",
    ])

    # ---------------------------------------------------------------- 3
    doc.add_heading("3. Experimental setup and cross-validation", 1)
    doc.add_paragraph(
        "Children are clustered in districts and enrolment rates differ sharply between districts, so a random split would let "
        "the model see children from the same district on both sides and over-state performance. We therefore partition with "
        "StratifiedGroupKFold grouped by district and stratified on the target. One fold of a 5-fold grouped split is held out "
        f"as the test set ({split['test']['rows']:,} children, {split['test']['districts']} districts, "
        f"{split['test']['share_of_rows']:.1%} of rows, prevalence {split['test']['prevalence']:.1%}); the remaining "
        f"{split['train']['rows']:,} children in {split['train']['districts']} districts are used for tuning with a second 5-fold "
        "grouped stratified split (inner CV). No district appears on both sides of any split (asserted in code). Hyperparameters "
        "were selected with GridSearchCV on the inner folds, refitting on ROC-AUC; recall, precision, F1, PR-AUC and accuracy were "
        "recorded per fold to show variance. Standardisation for the logistic model is fitted inside each fold.")
    add_table(doc, pd.read_csv(T / "t01_split_description.csv"), caption="Table 1. Grouped stratified hold-out split")
    cvdf = pd.DataFrame(R["cv_best"])[["model", "best_params", "cv_roc_auc_mean", "cv_roc_auc_sd", "cv_average_precision_mean",
                                        "cv_recall_mean", "cv_f1_mean", "cv_accuracy_mean"]]
    cvdf["model"] = cvdf["model"].map(NAMES)
    cvdf.columns = ["Model", "Best parameters", "CV ROC-AUC", "SD", "CV PR-AUC", "CV recall", "CV F1", "CV accuracy"]
    add_table(doc, cvdf, caption="Table 2. Inner cross-validation results of the tuned configurations (mean over 5 district-grouped folds; recall, F1 and accuracy are at the default 0.5 threshold, ROC-AUC and PR-AUC are threshold-free)")
    lrp, trp = R["best_params"]["logreg"], R["best_params"]["tree"]
    doc.add_paragraph(
        f"Hyperparameter exploration. The logistic grid covered C in {{0.001, 0.01, 0.1, 1, 10}} x class_weight in {{none, balanced}}; "
        f"the best point was C = {lrp['C']} with {lrp['class_weight']} weights (CV ROC-AUC {cvb['logreg']['cv_roc_auc_mean']:.3f} "
        f"+/- {cvb['logreg']['cv_roc_auc_sd']:.3f}). AUC was flat across C, while balancing matters for recall at the default "
        f"threshold. The tree grid covered max_depth in {{3, 5, 7, 10, 15, none}} x min_samples_leaf in {{1, 50, 200, 1000}} x "
        f"class_weight; the best point was depth {trp['max_depth']}, minimum leaf {trp['min_samples_leaf']}, no class weights "
        f"(CV ROC-AUC {cvb['tree']['cv_roc_auc_mean']:.3f} +/- {cvb['tree']['cv_roc_auc_sd']:.3f}; fitted tree depth "
        f"{R['tree_shape']['depth']}, {R['tree_shape']['leaves']} leaves). Unconstrained trees over-fit district idiosyncrasies and "
        f"lose AUC on unseen districts. Because refitting is on ROC-AUC, a threshold-free criterion, the selected tree carries no class "
        f"weighting and at the default 0.5 cut-off flags almost nobody (recall {h05['tree']['recall']:.1%} on the hold-out, Table 3): "
        f"almost no leaf reaches a 50% out-of-school share. That is not a defect of the tree; it is why the operating threshold is set "
        f"separately, and without looking at the hold-out, in section 5. Full grids: outputs/tables/t02-t04.")

    doc.add_heading("3.1 Hypothesis H5 - value of the household features", 2)
    doc.add_paragraph(
        f"On identical inner folds, a logistic model using only age, sex and area reached ROC-AUC {h5['auc_demo_mean']:.3f} "
        f"(+/- {h5['auc_demo_sd']:.3f}) against {h5['auc_full_mean']:.3f} (+/- {h5['auc_full_sd']:.3f}) for the 13-feature model: a "
        f"gain of {h5['gain_mean']:+.3f} AUC in every fold (paired t = {h5['paired_t']:.2f}, p = {h5['p_value']:.3f}). On the hold-out "
        f"districts the AUCs were {h5['holdout_auc_demo']:.3f} and {h5['holdout_auc_full']:.3f}. Five folds are a thin basis for a t-test, so "
        f"the hold-out gain was also bootstrapped by resampling the {bs['n_clusters']} held-out districts with replacement "
        f"({bs['n_boot']:,} draws): gain {bs['diff_point']:+.3f} AUC, 95% interval [{bs['ci_low']:+.3f}, {bs['ci_high']:+.3f}], positive in "
        f"{bs['share_positive']:.0%} of draws. H0 for H5 is rejected: who is out of school is explained far better by household circumstances "
        f"than by demographics alone.")
    add_fig(doc, "h5", "Figure 1. Per-fold ROC-AUC of the demographic and full logistic models (district-grouped CV).", 4.8)

    # ---------------------------------------------------------------- 4
    doc.add_heading("4. Evaluation on the held-out districts", 1)
    ho = pd.read_csv(T / "t07_holdout_metrics_threshold_0.5.csv")
    ho = ho[["model_name", "weighted", "accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc", "flagged_rate"]]
    ho.columns = ["Model", "Weighted", "Accuracy", "Precision", "Recall", "Specificity", "F1", "ROC-AUC", "PR-AUC", "Flagged"]
    add_table(doc, ho, caption="Table 3. Hold-out metrics at the default 0.5 threshold (unweighted and ASER-weighted)")
    doc.add_paragraph(
        f"The majority-class benchmark reaches {h05['naive']['accuracy']:.1%} accuracy and finds no out-of-school child. The balanced "
        f"logistic regression trades accuracy ({h05['logreg']['accuracy']:.1%}) for recall ({h05['logreg']['recall']:.1%}) at the default "
        f"cut-off (ROC-AUC {h05['logreg']['roc_auc']:.3f}, PR-AUC {h05['logreg']['pr_auc']:.3f}). The decision tree ranks children better "
        f"(ROC-AUC {h05['tree']['roc_auc']:.3f}, PR-AUC {h05['tree']['pr_auc']:.3f}) but, tuned without class weights, almost no leaf "
        f"exceeds a 50% out-of-school share, so its default-threshold recall is only {h05['tree']['recall']:.1%}. The threshold, not the "
        f"learner, decides who is flagged. Applying the ASER sampling weights changes the picture in different directions for different "
        f"models: ROC-AUC rises from {h05['logreg']['roc_auc']:.3f} to {w05['logreg']['roc_auc']:.3f} for the full logistic model and from "
        f"{h05['tree']['roc_auc']:.3f} to {w05['tree']['roc_auc']:.3f} for the tree, but falls from {h05['logreg_demo']['roc_auc']:.3f} to "
        f"{w05['logreg_demo']['roc_auc']:.3f} for the demographic-only model. The weights up-weight households that the sample "
        f"under-represents - predominantly poorer, rural ones - where the household features carry most of their signal and where age, "
        f"sex and area alone separate least; population-level claims should therefore quote the weighted figures. Hold-out AUCs sit "
        f"0.03-0.06 below the CV means: the held-out districts are a harder group, within the fold-to-fold spread observed in CV.")
    add_fig(doc, "roc_pr", "Figure 2. ROC and precision-recall curves on the held-out districts.")
    add_fig(doc, "cm_05", "Figure 3. Confusion matrices at the default 0.5 threshold (counts; row % of each true class).")

    # ---------------------------------------------------------------- 5
    doc.add_heading("5. Asymmetric error-cost analysis", 1)
    doc.add_paragraph(
        "A false negative (model says 'in school', child is out of school) means a child who needs an enrolment or outreach "
        "intervention is never visited - the costly error for an education-access programme. A false positive (model says 'out of "
        "school', child is enrolled) spends an outreach visit or a stipend slot on a child who did not need it - wasteful but "
        "recoverable. We quantify this by pricing a false negative at r false positives (r = 1, 3, 5, 10), sweeping the threshold "
        "from 0.05 to 0.95 and computing the expected cost per 1,000 children screened. The operating threshold is a model setting, so "
        "it is chosen without touching the hold-out: each tuned pipeline is re-fitted on the five district-grouped training folds, "
        "out-of-fold probabilities are collected for every training child, the cost curve is swept on those, and the cost-minimising "
        "threshold is then applied once to the hold-out districts. Table 4 reports both the training out-of-fold metrics that drove the "
        "choice and the hold-out metrics at that threshold; its last two columns show what the hold-out's own optimum would have been, "
        "to make visible how much selecting the threshold on the test set would have flattered the result.")
    bt = pd.read_csv(T / "t09_cost_minimising_thresholds.csv")
    bt["model"] = bt["model"].map(NAMES)
    bt = bt[["model", "cost_ratio_fn_to_fp", "threshold", "oof_recall", "oof_precision", "oof_cost_per_1000",
             "holdout_recall", "holdout_precision", "holdout_flagged_rate", "holdout_cost_per_1000", "holdout_oracle_threshold", "holdout_oracle_cost_per_1000"]]
    bt.columns = ["Model", "FN:FP", "Threshold (chosen on train OOF)", "OOF recall", "OOF precision", "OOF cost/1,000",
                  "Hold-out recall", "Hold-out precision", "Hold-out flagged", "Hold-out cost/1,000", "Hold-out oracle thr.", "Oracle cost/1,000"]
    add_table(doc, bt, caption="Table 4. Cost-minimising threshold per cost ratio: chosen on training out-of-fold predictions, then evaluated once on the hold-out", font=7.5)
    add_fig(doc, "threshold", f"Figure 4. Recall/precision and expected cost versus threshold (cost ratio {ratio}:1). Solid curves are training "
            "out-of-fold predictions (used for selection); dashed curves are the hold-out (reported only); dash-dot lines mark the chosen thresholds.")
    cmp_ = pd.read_csv(T / "t10_confusion_default_vs_recommended.csv")
    cmp_ = cmp_[["model", "threshold_label", "threshold", "tp", "fn", "fp", "tn", "recall", "precision", "specificity",
                 "flagged_per_1000", "caught_per_1000", "missed_per_1000", f"cost_per_1000_ratio{ratio}"]]
    cmp_["model"] = cmp_["model"].map(NAMES)
    cmp_.columns = ["Model", "Threshold", "t", "TP", "FN", "FP", "TN", "Recall", "Precision", "Specificity", "Flagged/1,000",
                    "Caught/1,000", "Missed/1,000", f"Cost/1,000 (r={ratio})"]
    add_table(doc, cmp_, caption="Table 5. Default versus recommended operating point", font=8)
    add_fig(doc, "cm_rec", f"Figure 5. Confusion matrices at the recommended thresholds (cost ratio {ratio}:1).")
    bullets(doc, [
        "With r = 1 the cheapest policy is to flag nobody: neither model is sharp enough to beat the base rate when a false alarm "
        "costs as much as a missed child. The cost framing makes this explicit rather than hiding it behind accuracy.",
        f"At r = {ratio} the optimum moves to a low threshold. Recommended deployment: the decision tree at threshold {rec['tree']:.2f} - per "
        f"1,000 children screened it flags {tree_rec['flagged_per_1000']:.0f}, catches {tree_rec['caught_per_1000']:.0f} of the "
        f"{tree_rec['caught_per_1000'] + tree_rec['missed_per_1000']:.0f} out-of-school children and misses {tree_rec['missed_per_1000']:.0f} "
        f"(recall {tree_rec['recall']:.2f}, precision {tree_rec['precision']:.2f}), the lowest hold-out expected cost of all policies. The default "
        f"0.5 threshold on the same tree would catch only {tree_05['caught_per_1000']:.0f} per 1,000.",
        f"Logistic regression at threshold {rec['logreg']:.2f} is a close second (recall {lr_rec['recall']:.2f}, precision "
        f"{lr_rec['precision']:.2f}, {lr_rec['flagged_per_1000']:.0f} flagged per 1,000) and is preferable where officials must explain why "
        "a child was flagged.",
        f"At r = 10 both models flag over half of all children to catch {ct[('logreg',10)]['holdout_recall']:.0%}-{ct[('tree',10)]['holdout_recall']:.0%}; "
        "whether that is acceptable is a budget decision for the programme, which the cost ratio makes explicit.",
        f"Honesty check. Had the threshold been picked on the hold-out itself, the logistic optimum would have been "
        f"{ct[('logreg',ratio)]['holdout_oracle_threshold']:.2f} with cost {ct[('logreg',ratio)]['holdout_oracle_cost_per_1000']:.0f} per 1,000 against "
        f"{ct[('logreg',ratio)]['holdout_cost_per_1000']:.0f} at the training-selected {rec['logreg']:.2f}; for the tree the two coincide at "
        f"{rec['tree']:.2f}. The flattery from selecting on the test set is small here (about "
        f"{(ct[('logreg',ratio)]['holdout_cost_per_1000']/ct[('logreg',ratio)]['holdout_oracle_cost_per_1000']-1):.1%} of cost for the logistic model) but it is "
        "not zero, which is why all numbers at the recommended threshold - Tables 5-7, Figures 5-6 - use the training-selected value.",
    ])

    doc.add_heading("5.1 Misclassified edge cases", 2)
    prof = pd.read_csv(T / "t12_misclassified_profile_logreg.csv")
    prof = prof[["feature", "TP", "FN", "FP", "TN", "gap_FN_minus_TP"]]
    prof.columns = ["Feature (mean)", "True positives", "False negatives", "False positives", "True negatives", "FN - TP"]
    add_table(doc, prof, caption=f"Table 6. Mean feature values by outcome, logistic regression at threshold {rec['logreg']:.2f}")
    subt = pd.read_csv(T / "t14_subgroup_metrics_recommended_threshold.csv")
    subt = subt[subt.model == "logreg"][["variable", "level", "n", "positives", "prevalence", "recall", "fnr", "precision", "roc_auc"]]
    subt.columns = ["Variable", "Level", "n (test)", "Positives", "Prevalence", "Recall", "FNR", "Precision", "ROC-AUC"]
    add_table(doc, subt, caption="Table 7. Logistic-regression performance by subgroup on the held-out districts")
    add_fig(doc, "subgroup", "Figure 6. Recall by subgroup (logistic regression, recommended threshold).", 5.6)
    doc.add_paragraph(
        f"The children the model misses look like enrolled children on paper: they are older, far more often urban "
        f"({[r for r in R['edge_profile'] if r['feature']=='urban'][0]['FN']:.0%} of false negatives versus "
        f"{[r for r in R['edge_profile'] if r['feature']=='urban'][0]['TP']:.0%} of true positives), more often B-form registered, "
        "and live in households with more assets, more digital use and better housing. The model has learned 'poor rural household "
        "implies out of school' and under-detects out-of-school children in better-off or urban households - plausibly dropouts rather "
        f"than never-enrolled children, a distinction the current features cannot see. Recall is {sub.loc[('province','Balochistan'),'recall']:.2f} "
        f"in Balochistan but {sub.loc[('province','Sindh'),'recall']:.2f} in Sindh and {sub.loc[('province','Punjab'),'recall']:.2f} in Punjab; "
        f"{sub.loc[('urban',1),'recall']:.2f} for urban versus {sub.loc[('urban',0),'recall']:.2f} for rural children. Islamabad Capital "
        "Territory is a single district that fell entirely on the training side of the grouped split, so it has no hold-out estimate; with "
        "52 positives in the whole dataset it is too small for reliable subgroup conclusions.")

    # ---------------------------------------------------------------- 6
    doc.add_heading("6. Feature importance and the Milestone 2 hypotheses", 1)
    o = pd.read_csv(T / "t15_logit_odds_ratios.csv")[["label", "odds_ratio", "or_ci_low", "or_ci_high", "p_value", "std_coef"]]
    o.columns = ["Feature", "Odds ratio", "CI low", "CI high", "p-value", "Std. coef"]
    add_table(doc, o, caption=f"Table 8. Logistic regression odds ratios (raw units, training districts; McFadden pseudo-R2 = {R['pseudo_r2']:.3f})")
    g = pd.read_csv(T / "t17_tree_gini_importance.csv").merge(pd.read_csv(T / "t18_permutation_logreg.csv")[["feature", "auc_drop_mean"]], on="feature") \
        .merge(pd.read_csv(T / "t19_permutation_tree.csv")[["feature", "auc_drop_mean"]], on="feature", suffixes=("_lr", "_tree"))
    g = g[["label", "gini_importance", "auc_drop_mean_lr", "auc_drop_mean_tree"]]
    g.columns = ["Feature", "Tree Gini importance", "Permutation AUC drop (LR)", "Permutation AUC drop (tree)"]
    add_table(doc, g, caption="Table 9. Tree Gini importance and hold-out permutation importance")
    add_fig(doc, "odds", "Figure 7. Odds ratios with 95% confidence intervals.", 5.6)
    add_fig(doc, "importance", "Figure 8. Standardised coefficients, Gini importance and permutation importance.")
    add_fig(doc, "h4", "Figure 9. Predicted out-of-school probability by gender and area (H4).", 4.6)

    hyp = pd.DataFrame(R["hypotheses"])[["hypothesis", "feature", "H1_expectation", "odds_ratio", "or_ci_low", "or_ci_high", "p_value", "gini_rank", "H0_rejected_at_5pct"]]
    hyp.columns = ["Hypothesis", "Variable", "H1 expects", "Odds ratio", "CI low", "CI high", "p-value", "Gini rank", "H0 rejected"]
    hyp["Gini rank"] = hyp["Gini rank"].map(lambda v: "-" if pd.isna(v) else str(int(v)))
    add_table(doc, hyp, caption="Table 10. Evidence for the five hypotheses")
    gaps = R["h4_gaps"]
    bullets(doc, [
        f"H1 Poverty (assets_n): OR {ors['assets_n']['odds_ratio']:.3f} per additional asset, CI excluding 1; Gini rank {[r for r in R['hypotheses'] if r['feature']=='assets_n'][0]['gini_rank']:.0f}. "
        f"The direction is as hypothesised, but the per-asset effect is small - housing structure (OR {ors['house']['odds_ratio']:.2f} per step "
        f"kachha to pacca) and digital access (OR {ors['digital_n']['odds_ratio']:.2f} per service) are the dominant poverty signals in both learners. "
        "H0 rejected; the asset count is a weaker poverty proxy than housing.",
        f"H2 Legal identity (bform): OR {ors['bform']['odds_ratio']:.3f} for registered children, i.e. {1/ors['bform']['odds_ratio']:.2f}x higher odds "
        "without a B-form, significant and consistent across coefficient, Gini and permutation views. H0 rejected.",
        f"H3 Disability (any_disability): OR {ors['any_disability']['odds_ratio']:.2f}, the largest single risk factor after Balochistan residence. H0 rejected.",
        f"H4 Gender x area (female_x_urban): interaction OR {ors['female_x_urban']['odds_ratio']:.3f}, CI excluding 1. Girls face higher odds "
        f"everywhere, but the predicted girl-boy gap is {gaps['rural_gender_gap']*100:.1f} percentage points in rural areas against "
        f"{gaps['urban_gender_gap']*100:.1f} in urban areas: the gender penalty is a rural phenomenon. The tree barely uses the interaction "
        "term because it recreates it through sequential splits. H0 rejected.",
        f"H5 Feature value: +{h5['gain_mean']:.3f} AUC in cross-validation (every fold, p = {h5['p_value']:.3f}) and "
        f"+{h5['holdout_auc_full'] - h5['holdout_auc_demo']:.3f} on hold-out. H0 rejected.",
        f"Controls: Balochistan residence raises the odds {ors['province_Balochistan']['odds_ratio']:.2f}x even after the household variables; "
        "age has a small negative coefficient within the 5-16 window; the ICT dummy is not significant (807 children). Note that Balochistan "
        "has the largest positive standardised coefficient in the logistic model yet close to zero permutation importance on the hold-out "
        "(Table 9), and Sindh's permutation importance is slightly negative. The province dummies overlap heavily with housing quality and "
        "digital access (kachha, disconnected households are concentrated in Balochistan), so shuffling a province flag costs little once the "
        "household variables remain; the coefficient measures the province's conditional association, the permutation drop its unique "
        "contribution to ranking children in new districts. The two views disagree because the information is shared, not because one is wrong.",
    ])

    # ---------------------------------------------------------------- 7
    doc.add_heading("7. Initial policy takeaways", 1)
    doc.add_paragraph(
        "A caveat before the recommendations. ASER is a single cross-sectional household survey, and every driver above is an association "
        "conditional on the other twelve features, not an estimated causal effect. The B-form result is the clearest example: children "
        "without a registration are more likely to be out of school, but households that are more engaged with public services may obtain "
        "both the document and a school place for reasons unrelated to the document itself, in which case issuing B-forms alone would not "
        "move enrolment. The same reasoning applies to housing, digital access and disability. The takeaways below therefore identify where "
        "and for whom out-of-school risk concentrates - which is what a targeting model can legitimately inform - and name candidate levers "
        "whose causal effect would need to be established (for instance through a piloted or staggered roll-out) before being relied on as policy.")
    bullets(doc, [
        "Target by housing and connectivity, not just an asset count. Housing quality and digital access are the strongest household "
        "predictors of out-of-school status; outreach lists built from a simple asset score will miss children in kachha, disconnected homes.",
        "Birth registration is a candidate lever and a targeting marker. Unregistered children have about 1.3x the odds of being out of "
        "school; at minimum, NADRA B-form records identify a population the model flags as high-risk, and a joint registration-plus-enrolment "
        "drive is a testable intervention.",
        "Disability inclusion is a first-order gap (SDG 4.5). Children with a functional difficulty have about 1.9x the odds; screening and "
        "inclusive-school capacity belong in any enrolment drive.",
        "The gender gap is rural. Girl-focused interventions should be prioritised in rural areas, where the predicted penalty is almost "
        "three times the urban one.",
        "Balochistan needs a province-level response: residence there nearly doubles the odds after household factors are accounted for.",
        f"Set the operating threshold explicitly and choose it on training data. At a {ratio}:1 cost ratio the recommended policy flags about "
        f"{tree_rec['flagged_rate']:.0%} of children and catches {tree_rec['recall']:.0%} of out-of-school children on unseen districts; the default "
        "0.5 threshold is not a defensible choice for outreach.",
    ])

    # ---------------------------------------------------------------- 8
    doc.add_heading("8. Limitations and roadmap for Milestone 04", 1)
    doc.add_paragraph("Current limitations.")
    bullets(doc, [
        f"Modest discrimination / underfitting: hold-out ROC-AUC {h05['logreg']['roc_auc']:.2f}-{h05['tree']['roc_auc']:.2f} and pseudo-R2 "
        f"{R['pseudo_r2']:.2f}. Thirteen household features cannot separate the harder cases (urban or better-off dropouts); the missing signal "
        "is likely school-side (distance, supply, cost) and child history.",
        "Class imbalance is handled with class weights and threshold tuning, not resampling; precision stays near 0.2 at useful recall levels.",
        "District heterogeneity: CV standard deviation of about 0.05 AUC between district groups; the hold-out fold is harder than the CV average.",
        "Potential demographic bias: recall differs sharply by area, province and B-form status. Girls and disabled children are currently "
        "caught more often than their counterparts because they carry higher predicted risk, which is itself a fairness question.",
        "Associational evidence only: the drivers are conditional associations in one cross-sectional survey; none of the policy levers has an "
        "estimated causal effect (section 7).",
        "The H5 paired t-test rests on five folds; the district-cluster bootstrap of the hold-out gain (section 3.1) is the sturdier statement, "
        "and both agree.",
    ])
    doc.add_paragraph("Plan for Milestone 04.")
    bullets(doc, [
        "Fairness audit using the subgroup frame in subgroup_counts.csv: recall, precision and false-negative rate for girls vs boys, rural vs "
        "urban, with vs without disability, and by province, with bootstrap confidence intervals; test whether per-group thresholds or "
        "reweighting close the urban and Punjab/Sindh recall gaps. Islamabad Capital Territory (n = 807, 52 positives) is too small for "
        "reliable group conclusions and will be reported but not acted on.",
        "Advanced modelling: gradient boosting and random forest under the same district-grouped CV; probability calibration; interaction terms "
        "suggested by the tree; a cost-sensitive objective aligned with the FN:FP ratio.",
        "Ethical evaluation: harms of mis-targeting (stigma from false flags, exclusion from missed children), use of survey weights for "
        "population-level claims, and a deployment note on who sets the cost ratio.",
    ])

    doc.add_heading("9. Reproducibility", 1)
    doc.add_paragraph("Code: milestone3/src (config, data, cv, models, evaluate, importance, plots, run_all, make_report) and the executed "
                      "notebook Milestone03_BaselineModels.ipynb. `python src/run_all.py` regenerates every table (outputs/tables) and figure "
                      "(outputs/figures) from aser2025_training.csv with random state 42; `python src/make_report.py` rebuilds this document.")

    out = C.REPORT_DIR / "Milestone03_Report.docx"
    doc.save(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
