"""Feature-importance drivers: odds ratios, standardised coefficients, Gini and permutation importance."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.inspection import permutation_importance

from config import FEATURES, LABELS, RANDOM_STATE


def logit_odds_ratios(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Unpenalised logistic fit (statsmodels) on raw units -> odds ratios with 95% CI and p-values.

    Reported alongside the tuned sklearn model: the two agree in sign/ordering, statsmodels adds inference.
    """
    Xc = sm.add_constant(X, has_constant="add")
    res = sm.Logit(y, Xc).fit(disp=0, maxiter=200)
    ci = res.conf_int()
    out = pd.DataFrame({
        "feature": Xc.columns,
        "coef": res.params.values,
        "std_err": res.bse.values,
        "z": res.tvalues.values,
        "p_value": res.pvalues.values,
        "odds_ratio": np.exp(res.params.values),
        "or_ci_low": np.exp(ci[0].values),
        "or_ci_high": np.exp(ci[1].values),
    })
    out = out[out["feature"] != "const"].copy()
    # standardised coefficient = coef * SD(x): effect of a one-SD move, comparable across features
    sds = X.std(ddof=0)
    out["std_coef"] = out["coef"].values * sds.reindex(out["feature"]).values
    out["abs_std_coef"] = out["std_coef"].abs()
    out["label"] = out["feature"].map(LABELS).fillna(out["feature"])
    out["pseudo_r2"] = res.prsquared
    return out.sort_values("abs_std_coef", ascending=False).reset_index(drop=True)


def sklearn_logreg_coefs(pipe, features=FEATURES) -> pd.DataFrame:
    """Standardised coefficients from the tuned sklearn pipeline (scaler + LR)."""
    clf = pipe.named_steps["clf"]
    scale = pipe.named_steps["scale"]
    std_coef = clf.coef_.ravel()                      # already on standardised inputs
    raw_coef = std_coef / scale.scale_                # back to raw units
    return pd.DataFrame({
        "feature": features,
        "std_coef": std_coef,
        "raw_coef": raw_coef,
        "odds_ratio_raw_unit": np.exp(raw_coef),
        "abs_std_coef": np.abs(std_coef),
        "label": [LABELS.get(f, f) for f in features],
    }).sort_values("abs_std_coef", ascending=False).reset_index(drop=True)


def tree_gini(pipe, features=FEATURES) -> pd.DataFrame:
    clf = pipe.named_steps["clf"]
    return pd.DataFrame({
        "feature": features,
        "gini_importance": clf.feature_importances_,
        "label": [LABELS.get(f, f) for f in features],
    }).sort_values("gini_importance", ascending=False).reset_index(drop=True)


def permutation(pipe, X_test, y_test, features=FEATURES, n_repeats: int = 10) -> pd.DataFrame:
    r = permutation_importance(pipe, X_test, y_test, scoring="roc_auc", n_repeats=n_repeats,
                               random_state=RANDOM_STATE, n_jobs=-1)
    return pd.DataFrame({
        "feature": features,
        "auc_drop_mean": r.importances_mean,
        "auc_drop_std": r.importances_std,
        "label": [LABELS.get(f, f) for f in features],
    }).sort_values("auc_drop_mean", ascending=False).reset_index(drop=True)


def gender_area_cells(pipe, X_template: pd.DataFrame) -> pd.DataFrame:
    """Predicted out-of-school probability for the 4 gender x area cells (H4), other features at their medians."""
    med = X_template.median()
    rows = []
    for female in (0, 1):
        for urban in (0, 1):
            x = med.copy()
            x["female"], x["urban"], x["female_x_urban"] = female, urban, female * urban
            p = pipe.predict_proba(pd.DataFrame([x])[X_template.columns])[0, 1]
            rows.append({"female": female, "urban": urban,
                         "cell": f"{'Girl' if female else 'Boy'}, {'urban' if urban else 'rural'}",
                         "p_out_of_school": float(p)})
    out = pd.DataFrame(rows)
    # gender gap within each area
    rural_gap = out.loc[(out.female == 1) & (out.urban == 0), "p_out_of_school"].item() - out.loc[(out.female == 0) & (out.urban == 0), "p_out_of_school"].item()
    urban_gap = out.loc[(out.female == 1) & (out.urban == 1), "p_out_of_school"].item() - out.loc[(out.female == 0) & (out.urban == 1), "p_out_of_school"].item()
    out.attrs["rural_gender_gap"] = rural_gap
    out.attrs["urban_gender_gap"] = urban_gap
    return out
