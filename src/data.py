"""Loading and integrity checks for the Milestone 2 training file."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (ATTRITION_CSV, FEATURES, GROUP_COL, ID_COL, SUBGROUP_CSV,
                    TARGET, TRAINING_CSV, WEIGHT_COL, PROVINCE_COLS)


def load_training() -> pd.DataFrame:
    """Read aser2025_training.csv and assert the properties documented in encoding_log.txt."""
    df = pd.read_csv(TRAINING_CSV, low_memory=False)
    expected = [ID_COL, GROUP_COL, WEIGHT_COL, TARGET, *FEATURES]
    missing = set(expected) - set(df.columns)
    assert not missing, f"training file lacks columns: {missing}"
    assert df.shape[0] == 161_803, f"unexpected row count {df.shape[0]}"
    assert df[FEATURES + [TARGET]].isna().sum().sum() == 0, "NaNs found - instructions say none exist"
    assert not df[ID_COL].duplicated().any(), "duplicate child ids"
    assert set(df[TARGET].unique()) == {0, 1}, "target must be binary"
    # one-hot sanity: at most one province dummy switched on per row
    assert (df[PROVINCE_COLS].sum(axis=1) <= 1).all(), "province dummies overlap"
    return df


def split_xy(df: pd.DataFrame, features: list[str] | None = None):
    """Return X, y, groups, weights as separate objects."""
    feats = features or FEATURES
    X = df[feats].astype(float)
    y = df[TARGET].astype(int).to_numpy()
    groups = df[GROUP_COL].to_numpy()
    w = df[WEIGHT_COL].astype(float).to_numpy()
    return X, y, groups, w


def load_subgroup_counts() -> pd.DataFrame:
    return pd.read_csv(SUBGROUP_CSV)


def load_attrition() -> pd.DataFrame:
    return pd.read_csv(ATTRITION_CSV)


def province_of(df: pd.DataFrame) -> pd.Series:
    """Recover the province label from the one-hot columns (reference = Punjab)."""
    out = pd.Series("Punjab", index=df.index, dtype=object)
    for c in PROVINCE_COLS:
        out[df[c] == 1] = c.replace("province_", "")
    return out


def describe_split(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> pd.DataFrame:
    """Rows, positives, prevalence and district counts for the two partitions."""
    rows = []
    for name, idx in (("train", train_idx), ("test", test_idx)):
        part = df.iloc[idx]
        rows.append({
            "partition": name,
            "rows": len(part),
            "share_of_rows": round(len(part) / len(df), 4),
            "positives": int(part[TARGET].sum()),
            "prevalence": round(part[TARGET].mean(), 4),
            "districts": part[GROUP_COL].nunique(),
        })
    return pd.DataFrame(rows)
