"""Baseline architectures and their hyperparameter grids."""
from __future__ import annotations

from sklearn.dummy import DummyClassifier
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

MODEL_NAMES = {
    "naive": "Majority-class baseline",
    "logreg": "Logistic regression",
    "tree": "Decision tree",
    "logreg_demo": "Logistic regression (age, female, urban)",
}
