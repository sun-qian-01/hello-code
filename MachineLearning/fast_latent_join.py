"""
Fast latent join-key inversion.

The full L-BFGS latent model is too slow in this environment. This version uses
a much smaller parameterization:

  * K latent movie groups.
  * each table has a group distribution and each predicate column/op has a
    group correction vector;
  * query prediction is still star-join sum_k T_k * prod child_k;
  * parameters are optimized with scipy on a compact feature matrix.

It is fast enough for 5-fold OOF and still tests the latent join hypothesis.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from sklearn.model_selection import KFold

from strong_model import TABLES, qerror_from_logs


ROOT = Path(__file__).resolve().parent
SEED = 2070
CHILDREN = ["mc", "ci", "mi", "mi_idx", "mk"]


def load_col_info() -> Dict[str, Tuple[float, float, float, float]]:
    stats = pd.read_csv(ROOT / "column_min_max_vals.csv")
    return {
        row["name"]: (
            float(row["min"]),
            float(row["max"]),
            float(row["cardinality"]),
            float(row["num_unique_values"]),
        )
        for _, row in stats.iterrows()
    }


def parse_tables(v: object) -> List[str]:
    if pd.isna(v) or str(v).strip() == "":
        return []
    return [p.strip().split()[-1] for p in str(v).split(",") if p.strip()]


def parse_preds(v: object) -> List[Tuple[str, str, float]]:
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


def base_sel(col: str, op: str, val: float, info: Dict[str, Tuple[float, float, float, float]]) -> float:
    mn, mx, _, nu = info.get(col, (0.0, 1.0, 1.0, 1.0))
    x = (val - mn) / max(mx - mn, 1.0)
    if op == "=":
        s = 1.0 / max(nu, 1.0)
    elif op == "<":
        s = x
    elif op == ">":
        s = 1.0 - x
    else:
        s = 1.0
    return float(np.clip(s, 1e-9, 1.0))


def encode(df: pd.DataFrame, info: Dict[str, Tuple[float, float, float, float]], key_map: Dict[str, int] | None = None):
    if key_map is None:
        key_map = {}
    table_present = np.zeros((len(df), len(TABLES)), dtype=np.float64)
    table_base_logs = np.zeros((len(df), len(TABLES)), dtype=np.float64)
    table_key_counts: List[List[List[int]]] = [[[] for _ in TABLES] for _ in range(len(df))]
    for i, row in enumerate(df.to_dict("records")):
        tables = parse_tables(row["Tables"])
        for ti, table in enumerate(TABLES):
            table_present[i, ti] = float(table in tables)
        for col, op, val in parse_preds(row["Predicates"]):
            table = col.split(".")[0]
            if table not in TABLES:
                continue
            key = f"{col}{op}"
            if key not in key_map:
                key_map[key] = len(key_map)
            ti = TABLES.index(table)
            table_base_logs[i, ti] += math.log(base_sel(col, op, val, info))
            table_key_counts[i][ti].append(key_map[key])
    return table_present, table_base_logs, table_key_counts, key_map


class FastLatent:
    def __init__(self, info: Dict[str, Tuple[float, float, float, float]], n_keys: int, K: int, l2: float, seed: int):
        self.info = info
        self.n_keys = n_keys
        self.K = K
        self.l2 = l2
        self.rng = np.random.default_rng(seed)
        self.rows = {t: info[f"{t}.id"][2] for t in TABLES}
        self.title_rows = self.rows["t"]
        self.child_unique = {
            "mc": info["mc.movie_id"][3],
            "ci": info["ci.movie_id"][3],
            "mi": info["mi.movie_id"][3],
            "mi_idx": info["mi_idx.movie_id"][3],
            "mk": info["mk.movie_id"][3],
        }
        self.child_fan = {c: self.rows[c] / self.child_unique[c] for c in CHILDREN}
        self.child_cov = {c: self.child_unique[c] / self.title_rows for c in CHILDREN}

    @property
    def n_params(self) -> int:
        # cluster logits + child scale + key correction + alpha beta
        return self.K + len(CHILDREN) * self.K + self.n_keys * self.K + 2

    def init(self) -> np.ndarray:
        p = np.zeros(self.n_params)
        p[: self.K] = self.rng.normal(0, 0.05, self.K)
        start = self.K + len(CHILDREN) * self.K
        p[start : start + self.n_keys * self.K] = self.rng.normal(0, 0.03, self.n_keys * self.K)
        p[-2] = 1.0
        p[-1] = 0.0
        return p

    def unpack(self, p: np.ndarray):
        pos = 0
        logits = p[pos : pos + self.K]
        pos += self.K
        w = np.exp(logits - logits.max())
        title_mass = self.title_rows * w / w.sum()
        child_scale = {}
        for c in CHILDREN:
            child_scale[c] = np.exp(np.clip(p[pos : pos + self.K], -2.0, 2.0))
            pos += self.K
        key_corr = np.clip(p[pos : pos + self.n_keys * self.K].reshape(self.n_keys, self.K), -3.0, 3.0)
        return title_mass, child_scale, key_corr, p[-2], p[-1]

    def predict(self, present, base_logs, key_counts, p: np.ndarray, idx: np.ndarray | None = None) -> np.ndarray:
        if idx is None:
            idx = np.arange(len(present))
        title_mass, child_scale, key_corr, alpha, beta = self.unpack(p)
        out = np.zeros(len(idx))
        for oi, i in enumerate(idx):
            log_sel = np.repeat(base_logs[i, :, None], self.K, axis=1)
            for ti in range(len(TABLES)):
                for kid in key_counts[i][ti]:
                    log_sel[ti] += key_corr[kid]
            has = present[i] > 0.5
            if has[0]:  # title present
                vals = title_mass * np.exp(np.clip(log_sel[0], -35, 5))
                for child in CHILDREN:
                    ti = TABLES.index(child)
                    if has[ti]:
                        vals *= self.child_cov[child] * self.child_fan[child] * child_scale[child] * np.exp(np.clip(log_sel[ti], -35, 5))
                card = vals.sum()
            else:
                cards = []
                for child in CHILDREN:
                    ti = TABLES.index(child)
                    if has[ti]:
                        vals = title_mass * self.child_cov[child] * self.child_fan[child] * child_scale[child] * np.exp(np.clip(log_sel[ti], -35, 5))
                        cards.append(vals.sum())
                if cards:
                    card = np.prod(cards) / (self.title_rows ** max(len(cards) - 1, 0))
                else:
                    card = self.title_rows
            out[oi] = alpha * math.log1p(max(card, 1e-9)) + beta
        return out

    def fit(self, present, base_logs, key_counts, y, train_idx, maxiter=180):
        p0 = self.init()
        # Add a small sample objective for speed if needed.
        train_idx = np.asarray(train_idx)
        def obj(p):
            pred = self.predict(present, base_logs, key_counts, p, train_idx)
            err = pred - y[train_idx]
            abs_err = np.abs(err)
            loss = np.where(abs_err < 1.0, 0.5 * err * err, abs_err - 0.5).mean()
            return float(loss + self.l2 * np.mean(p[:-2] * p[:-2]))
        best = None
        for r in range(2):
            start = p0.copy()
            if r:
                start[:-2] += self.rng.normal(0, 0.04, len(start) - 2)
            res = minimize(obj, start, method="Powell", options={"maxiter": maxiter, "ftol": 1e-5, "xtol": 1e-5, "disp": False})
            if best is None or res.fun < best.fun:
                best = res
        return best.x


def run_config(train, test, K, l2):
    info = load_col_info()
    present, logs, keys, key_map = encode(train, info)
    pt, lt, kt, _ = encode(test, info, key_map)
    y = np.log1p(train.Cardinality.astype(float).values)
    oof = np.zeros(len(train))
    test_preds = []
    for fold, (tri, vai) in enumerate(KFold(n_splits=5, shuffle=True, random_state=SEED).split(train), 1):
        model = FastLatent(info, len(key_map), K, l2, SEED + fold)
        params = model.fit(present, logs, keys, y, tri, maxiter=120)
        oof[vai] = model.predict(present, logs, keys, params, vai)
        test_preds.append(model.predict(pt, lt, kt, params))
        q = qerror_from_logs(oof[vai], y[vai])
        print(f"  fold {fold} q={q.mean():.5f} med={np.median(q):.5f}", flush=True)
    return oof, np.column_stack(test_preds).mean(1)


def main():
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    y = np.log1p(train.Cardinality.astype(float).values)
    experts = []
    expertst = []
    names = []
    for K, l2 in [(3, 1e-3), (5, 1e-3)]:
        print(f"fast latent K={K}", flush=True)
        oof, predt = run_config(train, test, K, l2)
        q = qerror_from_logs(oof, y)
        print(f"fast_latent_K{K} mean={q.mean():.5f} p95={np.percentile(q,95):.5f}", flush=True)
        pd.DataFrame({"Id": train.Id, "Cardinality": train.Cardinality, "PredLog": oof, "QError": q}).to_csv(ROOT / f"fast_latent_K{K}_oof.csv", index=False)
        pd.DataFrame({"Id": test.Id, "Cardinality": np.rint(np.maximum(np.expm1(predt), 1)).astype(np.int64)}).to_csv(ROOT / f"submission_fast_latent_K{K}.csv", index=False)
        experts.append(oof); expertst.append(predt); names.append(f"latent{K}")
    for name, of, sf, col in [
        ("rewrite", "oof_rewrite_angle_affine_oof.csv", "submission_rewrite_angle_affine.csv", "PredLog"),
        ("hier", "hierarchical_template_oof.csv", "submission_hierarchical_template.csv", "PredLog"),
        ("grid", "grid_surface_oof.csv", "submission_grid_surface.csv", "PredLog"),
    ]:
        if (ROOT / of).exists():
            o = pd.read_csv(ROOT / of); s = pd.read_csv(ROOT / sf)
            experts.append(o[col].values.astype(float)); expertst.append(np.log1p(s.Cardinality.astype(float).values)); names.append(name)
    E = np.column_stack(experts); ET = np.column_stack(expertst)
    for i,n in enumerate(names):
        q=qerror_from_logs(E[:,i],y); print(n,q.mean(),np.median(q),np.percentile(q,95),flush=True)
    def trans(z,M):
        w=np.exp(z[:M.shape[1]]); w=w/w.sum(); p=M@w
        a=0.93+0.14/(1+np.exp(-z[M.shape[1]])); b=0.2*np.tanh(z[M.shape[1]+1])
        return a*p+b
    def obj(z): return float(np.mean(qerror_from_logs(trans(z,E),y)))
    res=differential_evolution(obj,[(-7,7)]*(E.shape[1]+2),seed=SEED,maxiter=220,tol=1e-9,polish=True)
    pred=trans(res.x,E); predt=trans(res.x,ET); q=qerror_from_logs(pred,y)
    print("blend",q.mean(),np.percentile(q,[50,95,99,100]),flush=True)
    pd.DataFrame({"Id": train.Id, "Cardinality": train.Cardinality, "PredLog": pred, "QError": q}).to_csv(ROOT/"fast_latent_blend_oof.csv",index=False)
    pd.DataFrame({"Id": test.Id, "Cardinality": np.rint(np.maximum(np.expm1(predt),1)).astype(np.int64)}).to_csv(ROOT/"submission_fast_latent_blend.csv",index=False)


if __name__ == "__main__":
    main()
