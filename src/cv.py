"""Leakage-free partitioning: district-grouped, target-stratified splits.

Children from one district must never appear on both sides of a split, so the
model cannot memorise district-level enrolment rates. StratifiedGroupKFold keeps
whole districts together while balancing the 13.2% positive rate across folds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, precision_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from config import N_INNER_SPLITS, N_OUTER_SPLITS, RANDOM_STATE


def make_holdout(y: np.ndarray, groups: np.ndarray, test_fold: int = 0):
    """Take one fold of a 5-fold grouped stratified split as the hold-out test set (~20%)."""
    sgkf = StratifiedGroupKFold(n_splits=N_OUTER_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    folds = list(sgkf.split(np.zeros(len(y)), y, groups))
    train_idx, test_idx = folds[test_fold]
    assert not set(groups[train_idx]) & set(groups[test_idx]), "district leakage between train and test"
    return train_idx, test_idx


def make_inner_cv() -> StratifiedGroupKFold:
    return StratifiedGroupKFold(n_splits=N_INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE)


SCORING = {
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "recall": "recall",
    "precision": make_scorer(precision_score, zero_division=0),   # no warning when a fold predicts no positives
    "f1": make_scorer(f1_score, zero_division=0),
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
}


def tune(estimator, param_grid, X, y, groups, n_jobs: int = -1) -> GridSearchCV:
    """Grid search with district-grouped stratified folds; refit on ROC-AUC."""
    gs = GridSearchCV(
        estimator,
        param_grid,
        scoring=SCORING,
        refit="roc_auc",
        cv=make_inner_cv(),
        n_jobs=n_jobs,
        return_train_score=False,
    )
    gs.fit(X, y, groups=groups)
    return gs


def cv_table(gs: GridSearchCV, prefix_strip: str = "clf__") -> pd.DataFrame:
    """Tidy table of every grid point with mean/std of each CV metric."""
    res = pd.DataFrame(gs.cv_results_)
    params = [c for c in res.columns if c.startswith("param_")]
    keep = params + [f"mean_test_{m}" for m in SCORING] + [f"std_test_{m}" for m in SCORING] + ["rank_test_roc_auc"]
    out = res[keep].copy()
    out.columns = [c.replace("param_", "").replace(prefix_strip, "") for c in out.columns]
    return out.sort_values("rank_test_roc_auc").reset_index(drop=True)


def fold_scores(gs: GridSearchCV, metric: str = "roc_auc") -> np.ndarray:
    """Per-fold scores of the best grid point for one metric."""
    res = gs.cv_results_
    best = gs.best_index_
    return np.array([res[f"split{k}_test_{metric}"][best] for k in range(N_INNER_SPLITS)])
