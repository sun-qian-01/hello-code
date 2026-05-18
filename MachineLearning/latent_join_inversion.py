"""
Latent join-key inversion model.

This is a deliberately heavy, nonstandard attempt to infer hidden movie_id
distribution structure from query cardinalities alone.

Assumption:
    The title/movie domain is a mixture of K latent movie groups. A query first
    selects title rows per group, then each child table contributes a
    group-specific matching fanout. Star join cardinality is approximated by:

        sum_k title_mass_k(query) * prod_child child_fanout_{child,k}(query)

    Single child-table queries are predicted by summing child fanouts over
    groups.

The group response of each predicate column/operator is represented by simple
piecewise-linear basis functions of the normalized predicate threshold. We
optimize all parameters with scipy on log-cardinality loss.

This is not meant to be pretty; it is meant to test whether hidden join-key
latent structure can provide a genuinely different signal from GBDT ensembles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from sklearn.model_selection import KFold, train_test_split

from strong_model import TABLES, qerror_from_logs


ROOT = Path(__file__).resolve().parent
SEED = 2060
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


def parse_tables(value: object) -> List[str]:
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [part.strip().split()[-1] for part in str(value).split(",") if part.strip()]


def parse_predicates(value: object) -> List[Tuple[str, str, float]]:
    if pd.isna(value) or str(value).strip() == "":
        return []
    parts = [p.strip() for p in str(value).split(",")]
    out: List[Tuple[str, str, float]] = []
    for i in range(0, len(parts) - 2, 3):
        try:
            val = float(parts[i + 2])
        except ValueError:
            val = 0.0
        out.append((parts[i], parts[i + 1], val))
    return out


def basis(norm_value: float) -> np.ndarray:
    """Compact basis for threshold value in [0, 1]."""
    x = float(np.clip(norm_value, 0.0, 1.0))
    return np.array([1.0, x, x * x, math.sqrt(max(x, 0.0)), math.log1p(9.0 * x) / math.log(10.0)], dtype=np.float64)


def baseline_selectivity(col: str, op: str, val: float, col_info: Dict[str, Tuple[float, float, float, float]]) -> Tuple[float, float]:
    cmin, cmax, _, nunique = col_info.get(col, (0.0, 1.0, 1.0, 1.0))
    norm = (val - cmin) / max(cmax - cmin, 1.0)
    if op == "=":
        sel = 1.0 / max(nunique, 1.0)
    elif op == "<":
        sel = norm
    elif op == ">":
        sel = 1.0 - norm
    else:
        sel = 1.0
    return float(np.clip(sel, 1e-9, 1.0)), float(norm)


@dataclass
class EncodedData:
    table_mask: np.ndarray
    pred_table: List[List[str]]
    pred_key: List[List[str]]
    pred_base_log: List[List[float]]
    pred_basis: List[List[np.ndarray]]
    y_log: np.ndarray | None


def encode_df(
    df: pd.DataFrame,
    col_info: Dict[str, Tuple[float, float, float, float]],
    key_to_id: Dict[str, int] | None = None,
) -> Tuple[EncodedData, Dict[str, int]]:
    if key_to_id is None:
        key_to_id = {}
    table_mask = np.zeros((len(df), len(TABLES)), dtype=np.float64)
    pred_table: List[List[str]] = []
    pred_key: List[List[str]] = []
    pred_base_log: List[List[float]] = []
    pred_basis: List[List[np.ndarray]] = []
    for i, row in enumerate(df.to_dict("records")):
        tables = parse_tables(row["Tables"])
        for j, table in enumerate(TABLES):
            table_mask[i, j] = float(table in tables)
        pts: List[str] = []
        pks: List[str] = []
        pbl: List[float] = []
        pbs: List[np.ndarray] = []
        for col, op, val in parse_predicates(row["Predicates"]):
            table = col.split(".")[0]
            key = f"{col}{op}"
            if key not in key_to_id:
                key_to_id[key] = len(key_to_id)
            sel, norm = baseline_selectivity(col, op, val, col_info)
            pts.append(table)
            pks.append(key)
            pbl.append(math.log(max(sel, 1e-12)))
            pbs.append(basis(norm))
        pred_table.append(pts)
        pred_key.append(pks)
        pred_base_log.append(pbl)
        pred_basis.append(pbs)
    y_log = np.log1p(df["Cardinality"].astype(float).values) if "Cardinality" in df.columns else None
    return EncodedData(table_mask, pred_table, pred_key, pred_base_log, pred_basis, y_log), key_to_id


class LatentJoinModel:
    def __init__(
        self,
        col_info: Dict[str, Tuple[float, float, float, float]],
        key_to_id: Dict[str, int],
        n_clusters: int = 6,
        l2: float = 1e-3,
        seed: int = SEED,
    ) -> None:
        self.col_info = col_info
        self.key_to_id = key_to_id
        self.n_keys = len(key_to_id)
        self.K = n_clusters
        self.B = len(basis(0.5))
        self.l2 = l2
        self.rng = np.random.default_rng(seed)
        self.table_rows = {t: col_info[f"{t}.id"][2] for t in TABLES}
        self.title_rows = self.table_rows["t"]
        self.child_unique = {
            "mc": col_info["mc.movie_id"][3],
            "ci": col_info["ci.movie_id"][3],
            "mi": col_info["mi.movie_id"][3],
            "mi_idx": col_info["mi_idx.movie_id"][3],
            "mk": col_info["mk.movie_id"][3],
        }
        self.child_fanout = {t: self.table_rows[t] / max(self.child_unique[t], 1.0) for t in CHILDREN}
        self.child_coverage = {t: self.child_unique[t] / self.title_rows for t in CHILDREN}

    @property
    def n_params(self) -> int:
        # cluster logits + child log fanout scale + key/cluster/basis correction
        return self.K + len(CHILDREN) * self.K + self.n_keys * self.K * self.B + 2

    def initial_params(self) -> np.ndarray:
        p = np.zeros(self.n_params, dtype=np.float64)
        pos = self.K + len(CHILDREN) * self.K
        p[pos : pos + self.n_keys * self.K * self.B] = self.rng.normal(0.0, 0.02, self.n_keys * self.K * self.B)
        # global alpha/beta on log prediction
        p[-2] = 1.0
        p[-1] = 0.0
        return p

    def unpack(self, params: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray, float, float]:
        pos = 0
        logits = params[pos : pos + self.K]
        pos += self.K
        w = np.exp(logits - np.max(logits))
        title_mass = self.title_rows * w / w.sum()
        child_scale: Dict[str, np.ndarray] = {}
        for child in CHILDREN:
            raw = params[pos : pos + self.K]
            pos += self.K
            child_scale[child] = np.exp(np.clip(raw, -3.0, 3.0))
        corr = params[pos : pos + self.n_keys * self.K * self.B].reshape(self.n_keys, self.K, self.B)
        alpha = params[-2]
        beta = params[-1]
        return title_mass, child_scale, corr, alpha, beta

    def predict_log_raw(self, data: EncodedData, params: np.ndarray) -> np.ndarray:
        title_mass, child_scale, corr, alpha, beta = self.unpack(params)
        out = np.zeros(len(data.table_mask), dtype=np.float64)
        table_index = {t: i for i, t in enumerate(TABLES)}
        key_ids = self.key_to_id
        for i in range(len(out)):
            present = {t: data.table_mask[i, table_index[t]] > 0.5 for t in TABLES}
            table_log_sel = {t: np.zeros(self.K, dtype=np.float64) for t in TABLES}
            for table, key, base_log, bs in zip(data.pred_table[i], data.pred_key[i], data.pred_base_log[i], data.pred_basis[i]):
                kid = key_ids.get(key)
                if kid is None:
                    continue
                adj = corr[kid] @ bs
                adj = np.clip(adj, -4.0, 4.0)
                table_log_sel[table] += base_log + adj
            if present["t"]:
                group_vals = title_mass * np.exp(np.clip(table_log_sel["t"], -35.0, 5.0))
                for child in CHILDREN:
                    if present[child]:
                        fan = self.child_coverage[child] * self.child_fanout[child] * child_scale[child]
                        group_vals *= fan * np.exp(np.clip(table_log_sel[child], -35.0, 5.0))
                card = group_vals.sum()
            else:
                # Single child-table query: sum child rows over clusters.
                child_cards = []
                for child in CHILDREN:
                    if present[child]:
                        fan = self.child_coverage[child] * self.child_fanout[child] * child_scale[child]
                        vals = title_mass * fan * np.exp(np.clip(table_log_sel[child], -35.0, 5.0))
                        child_cards.append(vals.sum())
                if child_cards:
                    card = np.prod(child_cards) / (self.title_rows ** max(len(child_cards) - 1, 0))
                else:
                    card = self.title_rows * np.exp(np.clip(table_log_sel["t"], -35.0, 5.0)).sum() / self.K
            out[i] = alpha * math.log1p(max(card, 1e-9)) + beta
        return out

    def fit(self, data: EncodedData, train_idx: np.ndarray, valid_idx: np.ndarray | None = None, maxiter: int = 350) -> np.ndarray:
        assert data.y_log is not None
        y = data.y_log
        p0 = self.initial_params()
        # Warm-start alpha/beta around a simple physical estimate.
        def objective(params: np.ndarray) -> float:
            pred = self.predict_log_raw(_subset_data(data, train_idx), params)
            err = pred - y[train_idx]
            # Huber-ish log loss plus small regularization on correction params.
            delta = 1.0
            abs_err = np.abs(err)
            loss = np.where(abs_err <= delta, 0.5 * err * err, delta * (abs_err - 0.5 * delta)).mean()
            reg = self.l2 * np.mean(params[:-2] * params[:-2])
            return float(loss + reg)

        best = None
        for restart in range(3):
            start = p0.copy()
            if restart:
                start[:-2] += self.rng.normal(0.0, 0.05, len(start) - 2)
            res = minimize(
                objective,
                start,
                method="L-BFGS-B",
                options={"maxiter": maxiter, "maxls": 30, "ftol": 1e-8},
            )
            if best is None or res.fun < best.fun:
                best = res
        assert best is not None
        return best.x


def _subset_data(data: EncodedData, idx: np.ndarray) -> EncodedData:
    y = data.y_log[idx] if data.y_log is not None else None
    return EncodedData(
        data.table_mask[idx],
        [data.pred_table[i] for i in idx],
        [data.pred_key[i] for i in idx],
        [data.pred_base_log[i] for i in idx],
        [data.pred_basis[i] for i in idx],
        y,
    )


def run_oof(train: pd.DataFrame, test: pd.DataFrame, n_clusters: int, l2: float) -> Tuple[np.ndarray, np.ndarray]:
    col_info = load_col_info()
    train_data, key_to_id = encode_df(train, col_info)
    test_data, _ = encode_df(test, col_info, key_to_id)
    oof = np.zeros(len(train), dtype=np.float64)
    test_preds: List[np.ndarray] = []
    folds = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (fit_idx, valid_idx) in enumerate(folds.split(train), start=1):
        model = LatentJoinModel(col_info, key_to_id, n_clusters=n_clusters, l2=l2, seed=SEED + fold)
        params = model.fit(train_data, fit_idx, valid_idx, maxiter=260)
        oof[valid_idx] = model.predict_log_raw(_subset_data(train_data, valid_idx), params)
        test_preds.append(model.predict_log_raw(test_data, params))
        q = qerror_from_logs(oof[valid_idx], train_data.y_log[valid_idx])
        print(f"  fold {fold}: q={q.mean():.5f}, med={np.median(q):.5f}", flush=True)
    return oof, np.column_stack(test_preds).mean(axis=1)


def optimize_blend(experts: List[np.ndarray], experts_test: List[np.ndarray], y_log: np.ndarray, names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    E = np.column_stack(experts)
    ET = np.column_stack(experts_test)
    for i, name in enumerate(names):
        q = qerror_from_logs(E[:, i], y_log)
        print(f"{name}: mean={q.mean():.5f}, med={np.median(q):.5f}, p95={np.percentile(q,95):.5f}", flush=True)

    def transform(z: np.ndarray, M: np.ndarray) -> np.ndarray:
        w = np.exp(z[: M.shape[1]])
        w = w / w.sum()
        p = M @ w
        alpha = 0.93 + 0.14 / (1.0 + np.exp(-z[M.shape[1]]))
        beta = 0.2 * np.tanh(z[M.shape[1] + 1])
        return alpha * p + beta

    def obj(z: np.ndarray) -> float:
        return float(np.mean(qerror_from_logs(transform(z, E), y_log)))

    res = differential_evolution(
        obj,
        [(-7, 7)] * (E.shape[1] + 2),
        seed=SEED,
        maxiter=240,
        tol=1e-9,
        polish=True,
        workers=1,
        updating="immediate",
    )
    pred = transform(res.x, E)
    pred_test = transform(res.x, ET)
    weights = np.exp(res.x[: E.shape[1]])
    weights = weights / weights.sum()
    print("blend", np.mean(qerror_from_logs(pred, y_log)), dict(zip(names, weights)), flush=True)
    print("pct", np.percentile(qerror_from_logs(pred, y_log), [50, 90, 95, 99, 99.5, 100]), flush=True)
    return pred, pred_test


def main() -> None:
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    y_log = np.log1p(train["Cardinality"].astype(float).values)
    experts: List[np.ndarray] = []
    experts_test: List[np.ndarray] = []
    names: List[str] = []
    for K, l2 in [(4, 1e-3), (6, 1e-3), (8, 2e-3)]:
        print(f"latent model K={K} l2={l2}", flush=True)
        oof, pred_test = run_oof(train, test, K, l2)
        q = qerror_from_logs(oof, y_log)
        print(f"latent_K{K}: mean={q.mean():.5f}, p95={np.percentile(q,95):.5f}", flush=True)
        pd.DataFrame({"Id": train["Id"], "Cardinality": train["Cardinality"], "PredLog": oof, "QError": q}).to_csv(ROOT / f"latent_join_K{K}_oof.csv", index=False)
        pd.DataFrame({"Id": test["Id"], "Cardinality": np.rint(np.maximum(np.expm1(pred_test), 1.0)).astype(np.int64)}).to_csv(ROOT / f"submission_latent_join_K{K}.csv", index=False)
        experts.append(oof)
        experts_test.append(pred_test)
        names.append(f"latentK{K}")

    for name, oof_file, sub_file, col in [
        ("rewrite", "oof_rewrite_angle_affine_oof.csv", "submission_rewrite_angle_affine.csv", "PredLog"),
        ("hier", "hierarchical_template_oof.csv", "submission_hierarchical_template.csv", "PredLog"),
        ("grid", "grid_surface_oof.csv", "submission_grid_surface.csv", "PredLog"),
    ]:
        if (ROOT / oof_file).exists() and (ROOT / sub_file).exists():
            o = pd.read_csv(ROOT / oof_file)
            s = pd.read_csv(ROOT / sub_file)
            experts.append(o[col].values.astype(float))
            experts_test.append(np.log1p(s["Cardinality"].astype(float).values))
            names.append(name)

    pred, pred_test = optimize_blend(experts, experts_test, y_log, names)
    q = qerror_from_logs(pred, y_log)
    pd.DataFrame({"Id": train["Id"], "Cardinality": train["Cardinality"], "PredLog": pred, "QError": q}).to_csv(ROOT / "latent_join_blend_oof.csv", index=False)
    sub = pd.DataFrame({"Id": test["Id"], "Cardinality": np.rint(np.maximum(np.expm1(pred_test), 1.0)).astype(np.int64)})
    sub.to_csv(ROOT / "submission_latent_join_blend.csv", index=False)
    print("submission stats", np.percentile(sub["Cardinality"], [0, 1, 5, 50, 95, 99, 100]), sub["Cardinality"].mean(), flush=True)


if __name__ == "__main__":
    main()
