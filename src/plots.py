"""Publication-style figures saved to outputs/figures."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.tree import plot_tree

from config import FIGURES_DIR, LABELS

sns.set_theme(style="whitegrid", context="talk", font_scale=0.8)
PALETTE = {"naive": "#7f7f7f", "logreg": "#1f77b4", "tree": "#d62728", "logreg_demo": "#9467bd"}


def _save(fig, name: str) -> str:
    path = FIGURES_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def roc_pr_curves(y, probas: dict, names: dict, prevalence: float) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for key, p in probas.items():
        fpr, tpr, _ = roc_curve(y, p)
        from sklearn.metrics import auc, average_precision_score
        axes[0].plot(fpr, tpr, label=f"{names[key]} (AUC = {auc(fpr, tpr):.3f})", color=PALETTE.get(key))
        prec, rec, _ = precision_recall_curve(y, p)
        axes[1].plot(rec, prec, label=f"{names[key]} (AP = {average_precision_score(y, p):.3f})", color=PALETTE.get(key))
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    axes[0].set(xlabel="False-positive rate (1 - specificity)", ylabel="True-positive rate (recall)",
                title="ROC curves - hold-out districts")
    axes[1].axhline(prevalence, color="k", ls="--", lw=1, label=f"Chance (prevalence = {prevalence:.3f})")
    axes[1].set(xlabel="Recall (share of out-of-school children caught)", ylabel="Precision (share of flags that are correct)",
                title="Precision-recall curves - hold-out districts")
    for ax in axes:
        ax.legend(loc="lower right" if ax is axes[0] else "upper right", fontsize=9)
    return _save(fig, "fig1_roc_pr_curves")


def confusion_heatmaps(cms: dict, names: dict, suffix: str = "") -> str:
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.6))
    axes = np.atleast_1d(axes)
    for ax, (key, c) in zip(axes, cms.items()):
        mat = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
        row = mat.sum(axis=1, keepdims=True)
        pct = mat / np.where(row == 0, 1, row)
        annot = np.array([[f"{int(mat[i, j]):,}\n({pct[i, j]:.1%})" for j in range(2)] for i in range(2)])
        sns.heatmap(pct, annot=annot, fmt="", cmap="Blues", cbar=False, vmin=0, vmax=1, ax=ax,
                    xticklabels=["Pred. enrolled", "Pred. out of school"],
                    yticklabels=["Actual enrolled", "Actual out of school"])
        ax.set_title(f"{names[key]}\nthreshold = {c.get('threshold', 0.5):.2f}", fontsize=11)
    fig.suptitle("Confusion matrices on hold-out districts (counts; row % = share of each true class)", y=1.02, fontsize=12)
    return _save(fig, f"fig2_confusion_matrices{suffix}")


def threshold_curves(sweeps: dict, names: dict, ratio: int) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for key, sw in sweeps.items():
        axes[0].plot(sw.threshold, sw.recall, color=PALETTE.get(key), label=f"{names[key]} recall")
        axes[0].plot(sw.threshold, sw.precision, color=PALETTE.get(key), ls=":", label=f"{names[key]} precision")
        axes[1].plot(sw.threshold, sw[f"cost_per_1000_ratio{ratio}"], color=PALETTE.get(key), label=names[key])
    axes[0].axvline(0.5, color="k", lw=1, ls="--")
    axes[0].set(xlabel="Classification threshold on P(out of school)", ylabel="Metric value",
                title="Recall and precision versus threshold")
    axes[1].axvline(0.5, color="k", lw=1, ls="--")
    axes[1].set(xlabel="Classification threshold on P(out of school)", ylabel=f"Expected cost per 1,000 children\n(1 FN = {ratio} FP)",
                title=f"Expected misclassification cost (FN:FP = {ratio}:1)")
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=9)
    return _save(fig, "fig3_threshold_cost_curves")


def importance_bars(lr: pd.DataFrame, tree: pd.DataFrame, perm_lr: pd.DataFrame, perm_tree: pd.DataFrame) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    d = lr.sort_values("std_coef")
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in d.std_coef]
    axes[0, 0].barh(d.label, d.std_coef, color=colors)
    axes[0, 0].axvline(0, color="k", lw=1)
    axes[0, 0].set(title="Logistic regression: standardised coefficient\n(red raises odds of being out of school, blue lowers)", xlabel="Log-odds change per 1 SD")
    d = tree.sort_values("gini_importance")
    axes[0, 1].barh(d.label, d.gini_importance, color="#d62728")
    axes[0, 1].set(title="Decision tree: Gini importance", xlabel="Share of impurity reduction")
    for ax, pm, ttl, col in ((axes[1, 0], perm_lr, "Logistic regression: permutation importance", "#1f77b4"),
                             (axes[1, 1], perm_tree, "Decision tree: permutation importance", "#d62728")):
        d = pm.sort_values("auc_drop_mean")
        ax.barh(d.label, d.auc_drop_mean, xerr=d.auc_drop_std, color=col, ecolor="k", capsize=3)
        ax.set(title=ttl, xlabel="Drop in hold-out ROC-AUC when feature is shuffled")
    return _save(fig, "fig4_feature_importance")


def odds_ratio_forest(ors: pd.DataFrame) -> str:
    d = ors.sort_values("odds_ratio")
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.errorbar(d.odds_ratio, d.label, xerr=[d.odds_ratio - d.or_ci_low, d.or_ci_high - d.odds_ratio],
                fmt="o", color="#1f77b4", ecolor="k", capsize=3)
    ax.axvline(1, color="k", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set(xlabel="Odds ratio for being out of school (log scale, 95% CI)", title="Logistic regression odds ratios (raw units, training districts)")
    return _save(fig, "fig5_odds_ratios")


def h5_auc_folds(folds_demo: np.ndarray, folds_full: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    k = np.arange(1, len(folds_demo) + 1)
    ax.plot(k, folds_demo, "o-", color=PALETTE["logreg_demo"], label="Demographic model (age, female, urban)")
    ax.plot(k, folds_full, "o-", color=PALETTE["logreg"], label="Full model (13 features)")
    for i in range(len(k)):
        ax.plot([k[i], k[i]], [folds_demo[i], folds_full[i]], color="grey", lw=1)
    ax.set(xlabel="Grouped CV fold", ylabel="ROC-AUC", title="H5: AUC gain from poverty, B-form, disability and controls", xticks=k)
    ax.legend(fontsize=9)
    return _save(fig, "fig6_h5_auc_by_fold")


def gender_area_plot(cells: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(7, 4.8))
    piv = cells.pivot(index="urban", columns="female", values="p_out_of_school")
    piv.index = ["Rural", "Urban"]; piv.columns = ["Boys", "Girls"]
    piv.plot(kind="bar", ax=ax, color=["#1f77b4", "#d62728"], rot=0)
    ax.set(ylabel="Predicted P(out of school)", xlabel="Area", title="H4: predicted out-of-school probability by gender and area\n(other features at median)")
    for cont in ax.containers:
        ax.bar_label(cont, fmt="%.3f", fontsize=9)
    return _save(fig, "fig7_gender_area_cells")


def subgroup_recall_plot(sub: pd.DataFrame, title: str) -> str:
    d = sub.copy()
    d["group"] = d["variable"] + " = " + d["level"].astype(str)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#bbbbbb" if s else "#1f77b4" for s in d.flag_small]
    ax.barh(d.group, d.recall, color=colors)
    for i, (r, n, p) in enumerate(zip(d.recall, d.n, d.positives)):
        ax.text(r + 0.01, i, f"{r:.2f}  (n={n:,}, pos={p:,})", va="center", fontsize=8)
    ax.set(xlim=(0, 1.15), xlabel="Recall (share of out-of-school children caught)", title=title)
    return _save(fig, "fig8_subgroup_recall")


def tree_top_levels(pipe, features) -> str:
    clf = pipe.named_steps["clf"]
    fig, ax = plt.subplots(figsize=(18, 7))
    plot_tree(clf, max_depth=3, feature_names=[LABELS.get(f, f) for f in features],
              class_names=["enrolled", "out of school"], filled=True, impurity=False, proportion=True,
              rounded=True, fontsize=8, ax=ax)
    ax.set_title(f"Tuned decision tree - first three levels (full depth = {clf.get_depth()}, leaves = {clf.get_n_leaves()})")
    return _save(fig, "fig9_tree_top_levels")
