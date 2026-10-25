"""Classification metrics, confusion matrices, threshold sweeps and asymmetric cost."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, roc_auc_score)

from config import COST_RATIOS, THRESHOLD_GRID


def confusion(y, yhat, w=None) -> dict:
    cm = confusion_matrix(y, yhat, labels=[0, 1], sample_weight=w)
    tn, fp, fn, tp = cm.ravel()
    return {"tn": float(tn), "fp": float(fp), "fn": float(fn), "tp": float(tp)}


def metrics_from_counts(c: dict) -> dict:
    tn, fp, fn, tp = c["tn"], c["fp"], c["fn"], c["tp"]
    n = tn + fp + fn + tp
    safe = lambda a, b: float(a / b) if b > 0 else float("nan")
    return {
        "accuracy": safe(tp + tn, n),
        "precision": safe(tp, tp + fp),
        "recall": safe(tp, tp + fn),            # sensitivity
        "specificity": safe(tn, tn + fp),
        "f1": safe(2 * tp, 2 * tp + fp + fn),
        "npv": safe(tn, tn + fn),
        "balanced_accuracy": 0.5 * (safe(tp, tp + fn) + safe(tn, tn + fp)),
        "flagged_rate": safe(tp + fp, n),
    }


def metrics(y, proba, threshold: float = 0.5, w=None) -> dict:
    """Full metric set at one threshold, optionally sampling-weighted."""
    yhat = (proba >= threshold).astype(int)
    c = confusion(y, yhat, w)
    out = {"threshold": threshold, **c, **metrics_from_counts(c)}
    out["roc_auc"] = float(roc_auc_score(y, proba, sample_weight=w))
    out["pr_auc"] = float(average_precision_score(y, proba, sample_weight=w))
    return out


def confusion_table(c: dict) -> pd.DataFrame:
    """Formatted 2x2 with counts and row percentages."""
    tn, fp, fn, tp = c["tn"], c["fp"], c["fn"], c["tp"]
    fmt = lambda a, row: f"{int(round(a)):,} ({a / row:.1%})" if row else "0"
    return pd.DataFrame(
        {
            "Predicted: enrolled (0)": [fmt(tn, tn + fp), fmt(fn, fn + tp)],
            "Predicted: out of school (1)": [fmt(fp, tn + fp), fmt(tp, fn + tp)],
        },
        index=["Actual: enrolled (0)", "Actual: out of school (1)"],
    )


def expected_cost(c: dict, cost_fn: float, cost_fp: float = 1.0) -> float:
    """Total misclassification cost per 1,000 children, FN weighted by cost_fn."""
    n = c["tn"] + c["fp"] + c["fn"] + c["tp"]
    return 1000.0 * (cost_fn * c["fn"] + cost_fp * c["fp"]) / n


def threshold_sweep(y, proba, grid=THRESHOLD_GRID, ratios=COST_RATIOS) -> pd.DataFrame:
    rows = []
    for t in grid:
        yhat = (proba >= t).astype(int)
        c = confusion(y, yhat)
        m = metrics_from_counts(c)
        row = {"threshold": t, **c, **m}
        for r in ratios:
            row[f"cost_per_1000_ratio{r}"] = expected_cost(c, cost_fn=r)
        rows.append(row)
    return pd.DataFrame(rows)


def best_thresholds(sweep: pd.DataFrame, ratios=COST_RATIOS) -> pd.DataFrame:
    """Cost-minimising threshold for every FN:FP cost ratio."""
    rows = []
    for r in ratios:
        col = f"cost_per_1000_ratio{r}"
        best = sweep.loc[sweep[col].idxmin()]
        rows.append({
            "cost_ratio_fn_to_fp": r,
            "threshold": best["threshold"],
            "recall": best["recall"],
            "precision": best["precision"],
            "specificity": best["specificity"],
            "f1": best["f1"],
            "flagged_rate": best["flagged_rate"],
            "cost_per_1000": best[col],
        })
    return pd.DataFrame(rows)


def per_1000(c: dict) -> dict:
    """Operational translation: per 1,000 children screened."""
    n = c["tn"] + c["fp"] + c["fn"] + c["tp"]
    k = 1000.0 / n
    return {
        "flagged_per_1000": (c["tp"] + c["fp"]) * k,
        "caught_per_1000": c["tp"] * k,
        "missed_per_1000": c["fn"] * k,
        "false_alarms_per_1000": c["fp"] * k,
        "out_of_school_per_1000": (c["tp"] + c["fn"]) * k,
    }


def subgroup_metrics(df_test: pd.DataFrame, y, proba, threshold: float, groupings: dict) -> pd.DataFrame:
    """Recall/precision/FNR by subgroup, used for edge-case diagnostics and the M4 fairness roadmap.

    groupings: {"variable": pd.Series of labels aligned with df_test}
    """
    rows = []
    yhat = (proba >= threshold).astype(int)
    for var, labels in groupings.items():
        for lvl in pd.unique(labels):
            mask = (labels == lvl).to_numpy()
            if mask.sum() == 0:
                continue
            c = confusion(y[mask], yhat[mask])
            m = metrics_from_counts(c)
            pos = int(c["tp"] + c["fn"])
            rows.append({
                "variable": var, "level": lvl, "n": int(mask.sum()), "positives": pos,
                "prevalence": pos / mask.sum(),
                "recall": m["recall"], "fnr": 1 - m["recall"] if not np.isnan(m["recall"]) else np.nan,
                "precision": m["precision"], "specificity": m["specificity"], "f1": m["f1"],
                "roc_auc": float(roc_auc_score(y[mask], proba[mask])) if 0 < y[mask].sum() < mask.sum() else np.nan,
                "flag_small": pos < 100,
            })
    return pd.DataFrame(rows)
