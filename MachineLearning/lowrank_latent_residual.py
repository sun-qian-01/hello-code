"""
Matrixized low-rank latent residual model.

The direct latent join simulator is too slow without torch/jax. This script
keeps the latent idea but makes it matrix-friendly:

  * build sparse-ish dense features for table combos, predicate shape, column
    thresholds and physical estimates;
  * learn low-rank latent components with TruncatedSVD over hashed interaction
    features;
  * fit residuals of the best model using Ridge/ExtraTrees/LightGBM on those
    latent components.

This is a practical approximation to join-key latent factors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from strong_model import qerror_from_logs


ROOT = Path(__file__).resolve().parent
SEED = 2080


def parse_preds(v):
    if pd.isna(v) or str(v).strip() == "":
        return []
    parts = [p.strip() for p in str(v).split(",")]
    out = []
    for i in range(0, len(parts) - 2, 3):
        try:
            val = float(parts[i + 2])
        except ValueError:
            val = 0.0
        out.append((parts[i], parts[i + 1], val))
    return out


def norm(v):
    return "" if pd.isna(v) else str(v)


def load_info():
    s = pd.read_csv(ROOT / "column_min_max_vals.csv")
    return {r["name"]: (float(r["min"]), float(r["max"]), float(r["cardinality"]), float(r["num_unique_values"])) for _, r in s.iterrows()}


def make_tokens(df: pd.DataFrame, info: Dict[str, Tuple[float, float, float, float]]) -> List[Dict[str, float]]:
    rows = []
    for r in df.to_dict("records"):
        table = norm(r["Tables"])
        join = norm(r["Join Conditions"])
        preds = parse_preds(r["Predicates"])
        d: Dict[str, float] = {}
        d[f"T={table}"] = 1.0
        d[f"J={join}"] = 1.0
        d[f"NP={len(preds)}"] = 1.0
        shape = "|".join(f"{c}{o}" for c, o, _ in preds)
        cols = "|".join(c for c, _, _ in preds)
        ops = "|".join(o for _, o, _ in preds)
        d[f"S={table}|{join}|{shape}"] = 1.0
        d[f"C={table}|{cols}"] = 1.0
        d[f"O={table}|{ops}"] = 1.0
        # Value-bin and pairwise interaction tokens. These approximate latent
        # join-key correlations without explicitly optimizing a simulator.
        value_bins = []
        for c, o, v in preds:
            mn, mx, _, nu = info.get(c, (0.0, 1.0, 1.0, 1.0))
            x = np.clip((v - mn) / max(mx - mn, 1.0), 0.0, 1.0)
            b = int(np.floor(x * 20))
            lb = int(np.floor(np.log1p(max(v, 0.0)) / np.log1p(max(mx, 1.0)) * 20))
            d[f"P={c}{o}"] = 1.0
            d[f"PB={c}{o}:{b}"] = 1.0
            d[f"PL={c}{o}:{lb}"] = 1.0
            d[f"PV={c}{o}"] = float(x)
            value_bins.append(f"{c}{o}:{b}")
        for i in range(len(value_bins)):
            for j in range(i + 1, len(value_bins)):
                d[f"PAIR={value_bins[i]}&{value_bins[j]}"] = 1.0
        rows.append(d)
    return rows


def build_latent_features(train, test):
    info = load_info()
    tokens_train = make_tokens(train, info)
    tokens_test = make_tokens(test, info)
    hasher = FeatureHasher(n_features=2**16, input_type="dict", alternate_sign=False)
    H_train = hasher.transform(tokens_train)
    H_test = hasher.transform(tokens_test)
    feats_train = []
    feats_test = []
    for n_comp in [32, 64, 128, 256]:
        svd = TruncatedSVD(n_components=n_comp, random_state=SEED + n_comp, n_iter=7)
        Z_train = svd.fit_transform(H_train)
        Z_test = svd.transform(H_test)
        feats_train.append(Z_train.astype(np.float32))
        feats_test.append(Z_test.astype(np.float32))
        print(f"SVD {n_comp}: explained={svd.explained_variance_ratio_.sum():.4f}", flush=True)
    return np.hstack(feats_train), np.hstack(feats_test)


def main():
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    y = np.log1p(train.Cardinality.astype(float).values)
    # Use best current OOF as base.
    base_oof = pd.read_csv(ROOT / "oof_rewrite_angle_affine_oof.csv")
    base_sub = pd.read_csv(ROOT / "submission_rewrite_angle_affine.csv")
    base = base_oof.PredLog.values.astype(float)
    baset = np.log1p(base_sub.Cardinality.astype(float).values)
    residual = y - base
    Z, Zt = build_latent_features(train, test)
    # Add base prediction as feature.
    X = np.column_stack([Z, base, np.abs(base - pd.read_csv(ROOT / "hierarchical_template_oof.csv").PredLog.values)])
    Xt = np.column_stack([Zt, baset, np.abs(baset - np.log1p(pd.read_csv(ROOT / "submission_hierarchical_template.csv").Cardinality.astype(float).values))])
    folds = KFold(n_splits=5, shuffle=True, random_state=SEED)
    experts = [base]
    expertst = [baset]
    names = ["base"]
    # Ridge residual expert
    oof = np.zeros(len(train)); tps=[]
    for tri, vai in folds.split(X):
        model = make_pipeline(StandardScaler(), Ridge(alpha=50.0))
        model.fit(X[tri], residual[tri])
        oof[vai] = base[vai] + 0.4 * model.predict(X[vai])
        tps.append(baset + 0.4 * model.predict(Xt))
    experts.append(oof); expertst.append(np.column_stack(tps).mean(1)); names.append("ridge_latent")
    # LGB residual expert
    oof = np.zeros(len(train)); tps=[]
    params = {
        "objective": "regression_l1", "metric": "mae", "learning_rate": 0.025,
        "num_leaves": 127, "max_depth": 9, "min_data_in_leaf": 40,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
        "lambda_l1": 0.2, "lambda_l2": 5.0, "verbose": -1, "num_threads": -1,
    }
    for fold, (tri, vai) in enumerate(folds.split(X), 1):
        dtr = lgb.Dataset(X[tri], label=np.clip(residual[tri], -2.0, 2.0))
        dva = lgb.Dataset(X[vai], label=np.clip(residual[vai], -2.0, 2.0))
        m = lgb.train({**params, "seed": SEED + fold}, dtr, num_boost_round=1800, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        oof[vai] = base[vai] + 0.25 * m.predict(X[vai], num_iteration=m.best_iteration)
        tps.append(baset + 0.25 * m.predict(Xt, num_iteration=m.best_iteration))
    experts.append(oof); expertst.append(np.column_stack(tps).mean(1)); names.append("lgb_latent")
    for i,n in enumerate(names):
        q=qerror_from_logs(experts[i],y); print(n,q.mean(),np.median(q),np.percentile(q,95),flush=True)
    E=np.column_stack(experts); ET=np.column_stack(expertst)
    def trans(z,M):
        w=np.exp(z[:M.shape[1]]); w=w/w.sum(); p=M@w
        a=0.93+0.14/(1+np.exp(-z[M.shape[1]])); b=0.2*np.tanh(z[M.shape[1]+1])
        return a*p+b
    def obj(z): return float(np.mean(qerror_from_logs(trans(z,E),y)))
    res=differential_evolution(obj,[(-7,7)]*(E.shape[1]+2),seed=SEED,maxiter=200,tol=1e-9,polish=True)
    pred=trans(res.x,E); predt=trans(res.x,ET); q=qerror_from_logs(pred,y)
    print("blend",q.mean(),np.percentile(q,[50,95,99,100]),flush=True)
    pd.DataFrame({"Id":train.Id,"Cardinality":train.Cardinality,"PredLog":pred,"QError":q}).to_csv(ROOT/"lowrank_latent_oof.csv",index=False)
    pd.DataFrame({"Id":test.Id,"Cardinality":np.rint(np.maximum(np.expm1(predt),1)).astype(np.int64)}).to_csv(ROOT/"submission_lowrank_latent.csv",index=False)
    print("saved submission_lowrank_latent.csv", flush=True)


if __name__ == "__main__":
    main()
