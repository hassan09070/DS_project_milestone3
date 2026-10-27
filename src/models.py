"""Baseline architectures and their hyperparameter grids."""
from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from config import RANDOM_STATE


def build_naive() -> DummyClassifier:
    """Majority-class benchmark: predicts 'enrolled' for every child."""
    return DummyClassifier(strategy="most_frequent")


def build_logreg() -> Pipeline:
    """Standardised logistic regression (scaler is fitted inside each CV fold -> no leakage)."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(solver="lbfgs", max_iter=5000, random_state=RANDOM_STATE)),
    ])


LOGREG_GRID = {
    "clf__C": [0.001, 0.01, 0.1, 1.0, 10.0],
    "clf__class_weight": [None, "balanced"],
}


def build_tree() -> Pipeline:
    return Pipeline([
        ("clf", DecisionTreeClassifier(criterion="gini", random_state=RANDOM_STATE)),
    ])


TREE_GRID = {
    "clf__max_depth": [3, 5, 7, 10, 15, None],
    "clf__min_samples_leaf": [1, 50, 200, 1000],
    "clf__class_weight": [None, "balanced"],
}

def build_rf() -> Pipeline:
    """Random forest: many de-correlated trees averaged; each tree sees a bootstrap sample and sqrt(13) features per split."""
    return Pipeline([
        ("clf", RandomForestClassifier(n_estimators=200, max_features="sqrt", n_jobs=1, random_state=RANDOM_STATE)),
    ])


RF_GRID = {
    "clf__max_depth": [8, 12, None],
    "clf__min_samples_leaf": [20, 100, 300],
    "clf__class_weight": [None, "balanced_subsample"],
}

MODELS = ("logreg", "tree", "rf")          # the tuned learners, in reporting order

MODEL_NAMES = {
    "naive": "Majority-class baseline",
    "logreg": "Logistic regression",
    "tree": "Decision tree",
    "rf": "Random forest",
    "logreg_demo": "Logistic regression (age, female, urban)",
}
