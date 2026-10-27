"""End-to-end Milestone 3 pipeline.

    .venv/bin/python src/run_all.py

Writes outputs/tables/*.csv, outputs/figures/*.png and outputs/results.json.
Every number quoted in the report and README comes from results.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C                                            # noqa: E402
from cv import cv_table, fold_scores, make_holdout, make_inner_cv, tune       # noqa: E402
from data import describe_split, load_subgroup_counts, load_training, province_of, split_xy  # noqa: E402
import evaluate as ev                                          # noqa: E402
import importance as imp                                       # noqa: E402
import models as M                                             # noqa: E402
import plots                                                   # noqa: E402

T = C.TABLES_DIR


def save(df: pd.DataFrame, name: str) -> None:
    df.to_csv(T / f"{name}.csv", index=False)


def main() -> dict:
    t0 = time.time()
    R: dict = {"random_state": C.RANDOM_STATE, "features": C.FEATURES, "target": C.TARGET}

    # ------------------------------------------------------------------ 1. data + split
    df = load_training()
    X, y, groups, w = split_xy(df)
    train_idx, test_idx = make_holdout(y, groups)
    split_desc = describe_split(df, train_idx, test_idx)
    save(split_desc, "t01_split_description")
    R["split"] = split_desc.to_dict(orient="records")
    R["dataset"] = {"rows": int(len(df)), "positives": int(y.sum()), "prevalence": float(y.mean()),
                    "weighted_prevalence": float(np.average(y, weights=w)), "districts": int(df[C.GROUP_COL].nunique())}
    assert abs(split_desc.loc[1, "prevalence"] - y.mean()) < 0.01, "test prevalence drifted from population"

    Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
    ytr, yte = y[train_idx], y[test_idx]
    gtr = groups[train_idx]
    wte = w[test_idx]
    df_te = df.iloc[test_idx].reset_index(drop=True)
    print(f"[split] train {len(ytr):,} rows / test {len(yte):,} rows; test prevalence {yte.mean():.4f}")

    # ------------------------------------------------------------------ 2. models
    probas, fitted, cv_summary = {}, {}, []

    naive = M.build_naive().fit(Xtr, ytr)
    probas["naive"] = naive.predict_proba(Xte)[:, 1]          # constant = 0 -> AUC 0.5
    fitted["naive"] = naive

    print("[tune] logistic regression ...")
    gs_lr = tune(M.build_logreg(), M.LOGREG_GRID, Xtr, ytr, gtr)
    save(cv_table(gs_lr), "t02_cv_grid_logreg")
    fitted["logreg"] = gs_lr.best_estimator_
    probas["logreg"] = gs_lr.predict_proba(Xte)[:, 1]
    print(f"        best {gs_lr.best_params_}  CV AUC {gs_lr.best_score_:.4f}")

    print("[tune] decision tree ...")
    gs_tree = tune(M.build_tree(), M.TREE_GRID, Xtr, ytr, gtr)
    save(cv_table(gs_tree), "t03_cv_grid_tree")
    fitted["tree"] = gs_tree.best_estimator_
    probas["tree"] = gs_tree.predict_proba(Xte)[:, 1]
    print(f"        best {gs_tree.best_params_}  CV AUC {gs_tree.best_score_:.4f}")

    print("[tune] random forest ...")
    gs_rf = tune(M.build_rf(), M.RF_GRID, Xtr, ytr, gtr)
    save(cv_table(gs_rf), "t03b_cv_grid_rf")
    fitted["rf"] = gs_rf.best_estimator_
    probas["rf"] = gs_rf.predict_proba(Xte)[:, 1]
    print(f"        best {gs_rf.best_params_}  CV AUC {gs_rf.best_score_:.4f}")

    # H5: demographic-only logistic model on identical folds
    print("[tune] logistic regression, demographic features only (H5) ...")
    Xtr_demo, Xte_demo = Xtr[C.DEMO_FEATURES], Xte[C.DEMO_FEATURES]
    gs_demo = tune(M.build_logreg(), M.LOGREG_GRID, Xtr_demo, ytr, gtr)
    save(cv_table(gs_demo), "t04_cv_grid_logreg_demo")
    fitted["logreg_demo"] = gs_demo.best_estimator_
    probas["logreg_demo"] = gs_demo.predict_proba(Xte_demo)[:, 1]

    for key, gs in (("logreg", gs_lr), ("tree", gs_tree), ("rf", gs_rf), ("logreg_demo", gs_demo)):
        row = {"model": key, "best_params": json.dumps({k.replace("clf__", ""): v for k, v in gs.best_params_.items()})}
        for m in ("roc_auc", "average_precision", "recall", "precision", "f1", "accuracy", "balanced_accuracy"):
            f = fold_scores(gs, m)
            row[f"cv_{m}_mean"], row[f"cv_{m}_sd"] = float(f.mean()), float(f.std(ddof=1))
        cv_summary.append(row)
    cv_summary = pd.DataFrame(cv_summary)
    save(cv_summary, "t05_cv_best_summary")
    R["cv_best"] = cv_summary.to_dict(orient="records")

    # H5 paired comparison
    auc_demo, auc_full = fold_scores(gs_demo, "roc_auc"), fold_scores(gs_lr, "roc_auc")
    tstat, pval = stats.ttest_rel(auc_full, auc_demo)
    h5 = pd.DataFrame({"fold": np.arange(1, len(auc_demo) + 1), "auc_demographic": auc_demo, "auc_full": auc_full,
                       "auc_gain": auc_full - auc_demo})
    save(h5, "t06_h5_auc_by_fold")
    R["h5"] = {"auc_demo_mean": float(auc_demo.mean()), "auc_demo_sd": float(auc_demo.std(ddof=1)),
               "auc_full_mean": float(auc_full.mean()), "auc_full_sd": float(auc_full.std(ddof=1)),
               "gain_mean": float((auc_full - auc_demo).mean()), "gain_sd": float((auc_full - auc_demo).std(ddof=1)),
               "paired_t": float(tstat), "p_value": float(pval), "folds": int(len(auc_demo)),
               "holdout_auc_demo": float(ev.metrics(yte, probas["logreg_demo"])["roc_auc"]),
               "holdout_auc_full": float(ev.metrics(yte, probas["logreg"])["roc_auc"]),
               "best_params_demo": {k.replace("clf__", ""): v for k, v in gs_demo.best_params_.items()}}
    # 5 folds are a thin basis for a t-test: add a district-cluster bootstrap of the hold-out AUC gain
    R["h5"]["holdout_bootstrap"] = ev.cluster_bootstrap_auc_diff(yte, probas["logreg_demo"], probas["logreg"], groups[test_idx])
    print(f"        H5 hold-out AUC gain {R['h5']['holdout_bootstrap']['diff_point']:+.4f} "
          f"[{R['h5']['holdout_bootstrap']['ci_low']:+.4f}, {R['h5']['holdout_bootstrap']['ci_high']:+.4f}] (district bootstrap)")
    R["best_params"] = {"logreg": {k.replace("clf__", ""): v for k, v in gs_lr.best_params_.items()},
                        "tree": {k.replace("clf__", ""): v for k, v in gs_tree.best_params_.items()},
                        "rf": {k.replace("clf__", ""): v for k, v in gs_rf.best_params_.items()}}
    tree_clf = fitted["tree"].named_steps["clf"]
    R["tree_shape"] = {"depth": int(tree_clf.get_depth()), "leaves": int(tree_clf.get_n_leaves())}

    # ------------------------------------------------------------------ 3. hold-out metrics @ 0.5
    rows = []
    for key in ("naive", "logreg_demo", *M.MODELS):
        m = ev.metrics(yte, probas[key], 0.5)
        rows.append({"model": key, "model_name": M.MODEL_NAMES[key], "weighted": False, **m})
        mw = ev.metrics(yte, probas[key], 0.5, w=wte)
        rows.append({"model": key, "model_name": M.MODEL_NAMES[key], "weighted": True, **mw})
    holdout = pd.DataFrame(rows)
    save(holdout, "t07_holdout_metrics_threshold_0.5")
    R["holdout_05"] = holdout[~holdout.weighted].set_index("model").to_dict(orient="index")
    R["holdout_05_weighted"] = holdout[holdout.weighted].set_index("model").to_dict(orient="index")
    assert all(R["holdout_05"][k]["roc_auc"] > 0.5 for k in M.MODELS)
    assert abs(R["holdout_05"]["naive"]["accuracy"] - (1 - yte.mean())) < 1e-9

    # ------------------------------------------------------------------ 4. threshold selection (training OOF) + asymmetric cost
    # The operating threshold is a model setting, so it is chosen WITHOUT touching the hold-out: each tuned pipeline is
    # re-fitted on the district-grouped inner folds, out-of-fold probabilities are collected for every training child,
    # the cost curve is swept on those, and only the chosen threshold is then applied once to the hold-out.
    print("[threshold] out-of-fold training predictions for threshold selection ...")
    oof = {}
    for key in M.MODELS:
        oof[key] = cross_val_predict(clone(fitted[key]), Xtr, ytr, groups=gtr, cv=make_inner_cv(), method="predict_proba", n_jobs=-1)[:, 1]
    sweeps_oof, sweeps_ho, best_t, cms_05, cms_rec = {}, {}, [], {}, {}
    for key in M.MODELS:
        sw = ev.threshold_sweep(ytr, oof[key]); sw.insert(0, "model", key); sw.insert(1, "data", "train_oof"); sweeps_oof[key] = sw
        sh = ev.threshold_sweep(yte, probas[key]); sh.insert(0, "model", key); sh.insert(1, "data", "holdout"); sweeps_ho[key] = sh
        bt = ev.best_thresholds(sw); bt.insert(0, "model", key)
        # hold-out performance at the OOF-selected threshold (reported, never used for selection)
        for i, r in bt.iterrows():
            c = ev.confusion(yte, (probas[key] >= r["threshold"]).astype(int)); m = ev.metrics_from_counts(c)
            bt.loc[i, "holdout_recall"], bt.loc[i, "holdout_precision"], bt.loc[i, "holdout_specificity"] = m["recall"], m["precision"], m["specificity"]
            bt.loc[i, "holdout_flagged_rate"] = m["flagged_rate"]
            bt.loc[i, "holdout_cost_per_1000"] = ev.expected_cost(c, r["cost_ratio_fn_to_fp"])
            # what the hold-out optimum WOULD have been, to show how much selection-on-test flatters
            col = f"cost_per_1000_ratio{int(r['cost_ratio_fn_to_fp'])}"
            bt.loc[i, "holdout_oracle_threshold"] = sh.loc[sh[col].idxmin(), "threshold"]
            bt.loc[i, "holdout_oracle_cost_per_1000"] = sh[col].min()
        best_t.append(bt)
    save(pd.concat(list(sweeps_oof.values()) + list(sweeps_ho.values())), "t08_threshold_sweep")
    best_t = pd.concat(best_t, ignore_index=True)
    best_t = best_t.rename(columns={"recall": "oof_recall", "precision": "oof_precision", "specificity": "oof_specificity",
                                    "f1": "oof_f1", "flagged_rate": "oof_flagged_rate", "cost_per_1000": "oof_cost_per_1000"})
    save(best_t, "t09_cost_minimising_thresholds")
    R["cost_thresholds"] = best_t.to_dict(orient="records")

    rec_thr = {}
    for key in M.MODELS:
        thr = float(best_t[(best_t.model == key) & (best_t.cost_ratio_fn_to_fp == C.RECOMMENDED_COST_RATIO)].threshold.item())
        rec_thr[key] = thr
        cms_05[key] = {**ev.confusion(yte, (probas[key] >= 0.5).astype(int)), "threshold": 0.5}
        cms_rec[key] = {**ev.confusion(yte, (probas[key] >= thr).astype(int)), "threshold": thr}
    cms_05["naive"] = {**ev.confusion(yte, (probas["naive"] >= 0.5).astype(int)), "threshold": 0.5}
    R["recommended_threshold"] = rec_thr
    R["recommended_cost_ratio"] = C.RECOMMENDED_COST_RATIO
    R["threshold_selection"] = "cost-minimising threshold chosen on district-grouped out-of-fold training predictions; hold-out touched once"
    # recommended model = lowest hold-out expected cost at the recommended ratio, among thresholds chosen on training OOF
    at_ratio = best_t[best_t.cost_ratio_fn_to_fp == C.RECOMMENDED_COST_RATIO].set_index("model")
    R["recommended_model"] = str(at_ratio["holdout_cost_per_1000"].idxmin())
    R["model_rank_by_holdout_cost"] = at_ratio["holdout_cost_per_1000"].sort_values().round(2).to_dict()

    comp_rows = []
    for key in ("naive", *M.MODELS):
        for label, cm in (("0.50 (default)", cms_05[key]), ("recommended", cms_rec.get(key))):
            if cm is None:
                continue
            mm = ev.metrics_from_counts(cm)
            comp_rows.append({"model": key, "threshold_label": label, "threshold": cm["threshold"], **{k: cm[k] for k in ("tn", "fp", "fn", "tp")},
                              **mm, **ev.per_1000(cm),
                              **{f"cost_per_1000_ratio{r}": ev.expected_cost(cm, r) for r in C.COST_RATIOS}})
    comp = pd.DataFrame(comp_rows)
    save(comp, "t10_confusion_default_vs_recommended")
    R["threshold_comparison"] = comp.to_dict(orient="records")
    for key, cm in {**{f"{k}_05": v for k, v in cms_05.items()}, **{f"{k}_rec": v for k, v in cms_rec.items()}}.items():
        ev.confusion_table(cm).to_csv(T / f"t11_confusion_{key}.csv")

    # ------------------------------------------------------------------ 5. edge cases: misclassified profiles + subgroups
    thr_lr = rec_thr["logreg"]
    yhat = (probas["logreg"] >= thr_lr).astype(int)
    outcome = np.where((yte == 1) & (yhat == 1), "TP", np.where((yte == 1) & (yhat == 0), "FN",
                       np.where((yte == 0) & (yhat == 1), "FP", "TN")))
    prof = df_te[C.FEATURES].copy()
    prof["province"] = province_of(df_te)
    prof["outcome"] = outcome
    prof["p_hat"] = probas["logreg"]
    num_cols = [f for f in C.FEATURES if not f.startswith("province_")] + ["p_hat"]
    profile = prof.groupby("outcome")[num_cols].mean().T
    profile["gap_FN_minus_TP"] = profile["FN"] - profile["TP"]
    profile.insert(0, "feature", profile.index)
    profile.to_csv(T / "t12_misclassified_profile_logreg.csv", index=False)
    R["edge_profile"] = profile.round(4).to_dict(orient="records")
    R["edge_counts"] = pd.Series(outcome).value_counts().to_dict()
    prov_out = pd.crosstab(prof["province"], prof["outcome"], normalize="index").round(4)
    prov_out["n"] = prof["province"].value_counts()
    prov_out.reset_index().to_csv(T / "t13_outcome_share_by_province.csv", index=False)
    R["province_outcomes"] = prov_out.reset_index().to_dict(orient="records")

    groupings = {
        "female": df_te["female"], "urban": df_te["urban"], "any_disability": df_te["any_disability"],
        "bform": df_te["bform"], "province": province_of(df_te),
    }
    sub_rows = []
    for key in M.MODELS:
        s = ev.subgroup_metrics(df_te, yte, probas[key], rec_thr[key], groupings)
        s.insert(0, "model", key)
        sub_rows.append(s)
    sub = pd.concat(sub_rows, ignore_index=True)
    save(sub, "t14_subgroup_metrics_recommended_threshold")
    R["subgroups_logreg"] = sub[sub.model == "logreg"].drop(columns="model").to_dict(orient="records")
    R["subgroups_tree"] = sub[sub.model == "tree"].drop(columns="model").to_dict(orient="records")
    R["subgroups_rf"] = sub[sub.model == "rf"].drop(columns="model").to_dict(orient="records")
    R["subgroup_counts_file"] = load_subgroup_counts().to_dict(orient="records")

    # ------------------------------------------------------------------ 6. feature importance
    print("[importance] odds ratios, gini, permutation ...")
    ors = imp.logit_odds_ratios(Xtr, ytr)
    save(ors, "t15_logit_odds_ratios")
    sk_coef = imp.sklearn_logreg_coefs(fitted["logreg"])
    save(sk_coef, "t16_sklearn_logreg_std_coefs")
    gini = imp.tree_gini(fitted["tree"])
    save(gini, "t17_tree_gini_importance")
    perm_lr = imp.permutation(fitted["logreg"], Xte, yte)
    perm_tree = imp.permutation(fitted["tree"], Xte, yte)
    gini_rf = imp.tree_gini(fitted["rf"])
    save(gini_rf, "t17b_rf_gini_importance")
    perm_rf = imp.permutation(fitted["rf"], Xte, yte)
    save(perm_lr, "t18_permutation_logreg"); save(perm_tree, "t19_permutation_tree"); save(perm_rf, "t19b_permutation_rf")
    cells = imp.gender_area_cells(fitted["logreg"], Xtr)
    save(cells, "t20_h4_gender_area_cells")
    R["odds_ratios"] = ors.round(5).to_dict(orient="records")
    R["sk_coefs"] = sk_coef.round(5).to_dict(orient="records")
    R["gini"] = gini.round(5).to_dict(orient="records")
    R["perm_logreg"] = perm_lr.round(5).to_dict(orient="records")
    R["perm_tree"] = perm_tree.round(5).to_dict(orient="records")
    R["gini_rf"] = gini_rf.round(5).to_dict(orient="records")
    R["perm_rf"] = perm_rf.round(5).to_dict(orient="records")
    R["h4_cells"] = cells.to_dict(orient="records")
    R["h4_gaps"] = {"rural_gender_gap": cells.attrs["rural_gender_gap"], "urban_gender_gap": cells.attrs["urban_gender_gap"]}
    R["pseudo_r2"] = float(ors["pseudo_r2"].iloc[0])

    # hypothesis evidence table
    o = ors.set_index("feature"); g = gini.set_index("feature"); pl = perm_lr.set_index("feature"); grf = gini_rf.set_index("feature")
    hyp_rows = []
    for h, f in C.HYPOTHESIS_FEATURES.items():
        expect = {"assets_n": "OR < 1", "bform": "OR < 1", "any_disability": "OR > 1", "female_x_urban": "OR != 1"}[f]
        orv = o.loc[f, "odds_ratio"]
        supported = {"assets_n": orv < 1, "bform": orv < 1, "any_disability": orv > 1, "female_x_urban": True}[f] and o.loc[f, "p_value"] < 0.05
        hyp_rows.append({"hypothesis": h, "feature": f, "H1_expectation": expect, "odds_ratio": orv,
                         "or_ci_low": o.loc[f, "or_ci_low"], "or_ci_high": o.loc[f, "or_ci_high"], "p_value": o.loc[f, "p_value"],
                         "std_coef_rank": int(ors.index[ors.feature == f][0] + 1), "gini_rank": int(gini.index[gini.feature == f][0] + 1),
                         "rf_gini_rank": int(gini_rf.index[gini_rf.feature == f][0] + 1),
                         "perm_auc_drop_logreg": pl.loc[f, "auc_drop_mean"], "H0_rejected_at_5pct": bool(supported)})
    hyp_rows.append({"hypothesis": "H5 Feature value", "feature": "13 vs 3 features", "H1_expectation": "AUC gain > 0",
                     "odds_ratio": np.nan, "or_ci_low": np.nan, "or_ci_high": np.nan, "p_value": R["h5"]["p_value"],
                     "std_coef_rank": np.nan, "gini_rank": np.nan, "rf_gini_rank": np.nan, "perm_auc_drop_logreg": R["h5"]["gain_mean"],
                     "H0_rejected_at_5pct": bool(R["h5"]["p_value"] < 0.05 and R["h5"]["gain_mean"] > 0)})
    hyp = pd.DataFrame(hyp_rows)
    save(hyp, "t21_hypothesis_evidence")
    R["hypotheses"] = hyp.to_dict(orient="records")

    # ------------------------------------------------------------------ 7. figures
    print("[plots] ...")
    names = M.MODEL_NAMES
    figs = {
        "roc_pr": plots.roc_pr_curves(yte, {k: probas[k] for k in ("logreg_demo", *M.MODELS)}, names, float(yte.mean())),
        "cm_05": plots.confusion_heatmaps({k: cms_05[k] for k in ("naive", *M.MODELS)}, names, "_default"),
        "cm_rec": plots.confusion_heatmaps(cms_rec, names, "_recommended"),
        "threshold": plots.threshold_curves(sweeps_oof, sweeps_ho, rec_thr, names, C.RECOMMENDED_COST_RATIO),
        "importance": plots.importance_bars(sk_coef, gini, perm_lr, perm_tree),
        "importance_rf": plots.rf_importance_bars(gini_rf, perm_rf),
        "odds": plots.odds_ratio_forest(ors),
        "h5": plots.h5_auc_folds(auc_demo, auc_full),
        "h4": plots.gender_area_plot(cells),
        "subgroup": plots.subgroup_recall_plot(sub[sub.model == "logreg"], f"Logistic regression recall by subgroup (threshold {thr_lr:.2f}); grey = < 100 positives"),
        "tree": plots.tree_top_levels(fitted["tree"], C.FEATURES),
    }
    R["figures"] = figs
    R["runtime_sec"] = round(time.time() - t0, 1)

    with open(C.OUT_DIR / "results.json", "w") as fh:
        json.dump(R, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"[done] {R['runtime_sec']} s -> {C.OUT_DIR / 'results.json'}")
    return R


if __name__ == "__main__":
    main()
