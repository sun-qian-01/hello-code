"""
Second low-rank latent residual run with larger hashed interaction space and
several residual shrinkage experts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lowrank_latent_residual import load_info, make_tokens
from strong_model import qerror_from_logs


ROOT = Path(__file__).resolve().parent
SEED = 2090


def main():
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    y = np.log1p(train.Cardinality.astype(float).values)
    base_oof = pd.read_csv(ROOT / "oof_rewrite_angle_affine_oof.csv")
    base_sub = pd.read_csv(ROOT / "submission_rewrite_angle_affine.csv")
    hier_oof = pd.read_csv(ROOT / "hierarchical_template_oof.csv")
    grid_oof = pd.read_csv(ROOT / "grid_surface_oof.csv")
    base = base_oof.PredLog.values.astype(float)
    baset = np.log1p(base_sub.Cardinality.astype(float).values)
    info = load_info()
    tokens_train = make_tokens(train, info)
    tokens_test = make_tokens(test, info)
    hasher = FeatureHasher(n_features=2**18, input_type="dict", alternate_sign=False)
    H = hasher.transform(tokens_train)
    Ht = hasher.transform(tokens_test)
    feats = []
    featst = []
    for n in [64, 128, 256, 384]:
        svd = TruncatedSVD(n_components=n, random_state=SEED + n, n_iter=10)
        Z = svd.fit_transform(H)
        Zt = svd.transform(Ht)
        print("svd", n, svd.explained_variance_ratio_.sum(), flush=True)
        feats.append(Z.astype(np.float32))
        featst.append(Zt.astype(np.float32))
    # Include multiple existing predictions as coordinates in latent space.
    sub_hier = pd.read_csv(ROOT / "submission_hierarchical_template.csv")
    sub_grid = pd.read_csv(ROOT / "submission_grid_surface.csv")
    X = np.column_stack([
        *feats,
        base,
        hier_oof.PredLog.values,
        grid_oof.PredLog.values,
        base - hier_oof.PredLog.values,
        base - grid_oof.PredLog.values,
    ])
    Xt = np.column_stack([
        *featst,
        baset,
        np.log1p(sub_hier.Cardinality.astype(float).values),
        np.log1p(sub_grid.Cardinality.astype(float).values),
        baset - np.log1p(sub_hier.Cardinality.astype(float).values),
        baset - np.log1p(sub_grid.Cardinality.astype(float).values),
    ])
    resid = y - base
    folds = KFold(n_splits=5, shuffle=True, random_state=SEED)
    experts = [base, hier_oof.PredLog.values.astype(float), grid_oof.PredLog.values.astype(float)]
    expertst = [baset, np.log1p(sub_hier.Cardinality.astype(float).values), np.log1p(sub_grid.Cardinality.astype(float).values)]
    names = ["base", "hier", "grid"]
    # Linear robust experts
    for name, model, shrink in [
        ("ridge30", make_pipeline(StandardScaler(), Ridge(alpha=30.0)), 0.35),
        ("ridge100", make_pipeline(StandardScaler(), Ridge(alpha=100.0)), 0.45),
        ("huber", make_pipeline(StandardScaler(), HuberRegressor(alpha=0.03, epsilon=1.25, max_iter=250)), 0.25),
    ]:
        oof = np.zeros(len(train)); tps = []
        for tri, vai in folds.split(X):
            model.fit(X[tri], np.clip(resid[tri], -2.5, 2.5))
            oof[vai] = base[vai] + shrink * model.predict(X[vai])
            tps.append(baset + shrink * model.predict(Xt))
        experts.append(oof); expertst.append(np.column_stack(tps).mean(1)); names.append(name)
    # LightGBM residual experts
    configs = [
        ("lgb_a", 0.20, 127, 9, 0.025),
        ("lgb_b", 0.30, 191, 10, 0.02),
        ("lgb_c", 0.15, 63, 7, 0.035),
    ]
    for nm, shrink, leaves, depth, lr in configs:
        oof = np.zeros(len(train)); tps = []
        params = {
            "objective": "regression_l1", "metric": "mae", "learning_rate": lr,
            "num_leaves": leaves, "max_depth": depth, "min_data_in_leaf": 35,
            "feature_fraction": 0.75, "bagging_fraction": 0.8, "bagging_freq": 1,
            "lambda_l1": 0.2, "lambda_l2": 6.0, "verbose": -1, "num_threads": -1,
        }
        for fold, (tri, vai) in enumerate(folds.split(X), 1):
            dtr = lgb.Dataset(X[tri], label=np.clip(resid[tri], -2.5, 2.5))
            dva = lgb.Dataset(X[vai], label=np.clip(resid[vai], -2.5, 2.5))
            m = lgb.train({**params, "seed": SEED + fold}, dtr, num_boost_round=2200, valid_sets=[dva], callbacks=[lgb.early_stopping(120, verbose=False), lgb.log_evaluation(0)])
            oof[vai] = base[vai] + shrink * m.predict(X[vai], num_iteration=m.best_iteration)
            tps.append(baset + shrink * m.predict(Xt, num_iteration=m.best_iteration))
        experts.append(oof); expertst.append(np.column_stack(tps).mean(1)); names.append(nm)
    E = np.column_stack(experts); ET = np.column_stack(expertst)
    for i, nm in enumerate(names):
        q = qerror_from_logs(E[:, i], y)
        print(nm, q.mean(), np.median(q), np.percentile(q, 95), flush=True)
    def trans(z, M):
        w = np.exp(z[:M.shape[1]]); w = w / w.sum()
        p = M @ w
        a = 0.93 + 0.14 / (1 + np.exp(-z[M.shape[1]]))
        b = 0.2 * np.tanh(z[M.shape[1] + 1])
        return a * p + b
    def obj(z): return float(np.mean(qerror_from_logs(trans(z, E), y)))
    res = differential_evolution(obj, [(-7, 7)] * (E.shape[1] + 2), seed=SEED, maxiter=240, tol=1e-9, polish=True)
    pred = trans(res.x, E); predt = trans(res.x, ET); q = qerror_from_logs(pred, y)
    w = np.exp(res.x[:E.shape[1]]); w = w / w.sum()
    print("blend", q.mean(), dict(zip(names, w)), np.percentile(q, [50, 95, 99, 100]), flush=True)
    pd.DataFrame({"Id": train.Id, "Cardinality": train.Cardinality, "PredLog": pred, "QError": q}).to_csv(ROOT / "lowrank_latent_v2_oof.csv", index=False)
    pd.DataFrame({"Id": test.Id, "Cardinality": np.rint(np.maximum(np.expm1(predt), 1)).astype(np.int64)}).to_csv(ROOT / "submission_lowrank_latent_v2.csv", index=False)


if __name__ == "__main__":
    main()
