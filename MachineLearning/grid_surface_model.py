"""
Per-template grid surface estimator.

For frequent predicate-column templates, learn the cardinality surface directly
from labels on a quantile grid. This approximates the hidden joint distribution
without assuming uniform selectivity/fanout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.optimize import differential_evolution
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from strong_model import qerror_from_logs


ROOT = Path(__file__).resolve().parent
N_FOLDS = 5
SEED = 2050


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


def parse_predicates(value: object) -> List[Tuple[str, str, float]]:
    if pd.isna(value) or str(value).strip() == "":
        return []
    parts = [p.strip() for p in str(value).split(",")]
    out = []
    for i in range(0, len(parts) - 2, 3):
        try:
            val = float(parts[i + 2])
        except ValueError:
            val = 0.0
        out.append((parts[i], parts[i + 1], val))
    return out


def norm(v: object) -> str:
    return "" if pd.isna(v) else str(v)


def key_and_vector(row: dict, col_info: Dict[str, Tuple[float, float, float, float]]) -> Tuple[str, np.ndarray]:
    preds = parse_predicates(row["Predicates"])
    table = norm(row["Tables"])
    join = norm(row["Join Conditions"])
    # Keep op in key; different monotonic directions define different surfaces.
    key = f"{table}||{join}||" + ("|".join(f"{c}{o}" for c, o, _ in preds) or "none")
    vals = []
    for c, o, v in preds:
        mn, mx, _, nu = col_info.get(c, (0.0, 1.0, 1.0, 1.0))
        nv = (v - mn) / max(mx - mn, 1.0)
        vals.extend([np.clip(nv, 0.0, 1.0), np.log1p(max(v, 0.0)) / np.log1p(max(mx, 1.0))])
    if not vals:
        vals = [0.0]
    return key, np.asarray(vals, dtype=np.float32)


def fit_surface_predict(X_fit: np.ndarray, y_fit: np.ndarray, X_pred: np.ndarray, method: str) -> np.ndarray:
    if len(X_fit) < 5:
        return np.full(len(X_pred), np.median(y_fit), dtype=np.float32)
    # Remove duplicate points by median target.
    df = pd.DataFrame(X_fit)
    df["y"] = y_fit
    grouped = df.groupby(list(range(X_fit.shape[1]))).y.median().reset_index()
    Xg = grouped.iloc[:, :-1].values.astype(np.float32)
    yg = grouped["y"].values.astype(np.float32)
    if method == "knn":
        scaler = StandardScaler()
        Xs = scaler.fit_transform(Xg)
        Ps = scaler.transform(X_pred)
        k = min(max(8, int(np.sqrt(len(Xg)))), len(Xg))
        model = KNeighborsRegressor(n_neighbors=k, weights="distance", p=2)
        model.fit(Xs, yg)
        pred = model.predict(Ps)
    elif method == "linear_nd" and Xg.shape[1] <= 6 and len(Xg) >= Xg.shape[1] + 2:
        try:
            lin = LinearNDInterpolator(Xg, yg, fill_value=np.nan)
            near = NearestNDInterpolator(Xg, yg)
            pred = lin(X_pred)
            missing = ~np.isfinite(pred)
            if missing.any():
                pred[missing] = near(X_pred[missing])
        except Exception:
            pred = np.full(len(X_pred), np.median(yg))
    else:
        pred = np.full(len(X_pred), np.median(yg))
    lo = np.percentile(yg, 1) - 0.5
    hi = np.percentile(yg, 99) + 0.5
    return np.clip(pred, lo, hi).astype(np.float32)


def surface_expert(
    keys: np.ndarray,
    keys_test: np.ndarray,
    vecs: List[np.ndarray],
    vecs_test: List[np.ndarray],
    y_log: np.ndarray,
    base_log: np.ndarray,
    base_test_log: np.ndarray,
    *,
    min_group: int,
    method: str,
    target: str,
    shrink: float,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    pred = base_log.copy()
    pred_test = base_test_log.copy()
    covered = np.zeros(len(keys), dtype=bool)
    covered_test = np.zeros(len(keys_test), dtype=bool)
    fold_id = np.zeros(len(keys), dtype=np.int16)
    for fold, (_, valid_idx) in enumerate(KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(keys), start=1):
        fold_id[valid_idx] = fold
    groups = {
        key: values.values
        for key, values in pd.Series(np.arange(len(keys))).groupby(keys)
        if len(values) >= min_group
    }
    groups_test = {key: values.values for key, values in pd.Series(np.arange(len(keys_test))).groupby(keys_test)}
    for key, all_idx in groups.items():
        dim = len(vecs[all_idx[0]])
        if any(len(vecs[i]) != dim for i in all_idx):
            continue
        X_all = np.vstack([vecs[i] for i in all_idx])
        y_all = y_log[all_idx] if target == "direct" else y_log[all_idx] - base_log[all_idx]
        for fold in range(1, N_FOLDS + 1):
            valid_mask = fold_id[all_idx] == fold
            if not valid_mask.any():
                continue
            fit_mask = ~valid_mask
            if fit_mask.sum() < max(10, min_group // 2):
                continue
            loc_pred = fit_surface_predict(X_all[fit_mask], y_all[fit_mask], X_all[valid_mask], method)
            valid_global = all_idx[valid_mask]
            pred[valid_global] = loc_pred if target == "direct" else base_log[valid_global] + shrink * loc_pred
            covered[valid_global] = True
        test_idx = groups_test.get(key)
        if test_idx is None or len(test_idx) == 0:
            continue
        if any(len(vecs_test[i]) != dim for i in test_idx):
            continue
        loc_test = fit_surface_predict(X_all, y_all, np.vstack([vecs_test[i] for i in test_idx]), method)
        pred_test[test_idx] = loc_test if target == "direct" else base_test_log[test_idx] + shrink * loc_test
        covered_test[test_idx] = True
    return pred, pred_test, float(covered.mean()), float(covered_test.mean())


def main() -> None:
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    col_info = load_col_info()
    y_log = np.log1p(train["Cardinality"].astype(float).values)
    fallback_oof = pd.read_csv(ROOT / "hierarchical_template_oof.csv")
    fallback_sub = pd.read_csv(ROOT / "submission_hierarchical_template.csv")
    base_log = fallback_oof["PredLog"].values.astype(float)
    base_test_log = np.log1p(fallback_sub["Cardinality"].astype(float).values)

    train_pairs = [key_and_vector(row, col_info) for row in train.to_dict("records")]
    test_pairs = [key_and_vector(row, col_info) for row in test.to_dict("records")]
    keys = np.array([p[0] for p in train_pairs], dtype=object)
    keys_test = np.array([p[0] for p in test_pairs], dtype=object)
    vecs = [p[1] for p in train_pairs]
    vecs_test = [p[1] for p in test_pairs]

    experts = [base_log]
    experts_test = [base_test_log]
    names = ["hier_fallback"]
    for method in ["knn", "linear_nd"]:
        for target, shrink in [("direct", 0.0), ("resid", 0.45)]:
            for min_group in [30, 80, 150, 300]:
                print(f"surface {method=} {target=} min={min_group}", flush=True)
                pred, pred_test, cov, cov_test = surface_expert(
                    keys,
                    keys_test,
                    vecs,
                    vecs_test,
                    y_log,
                    base_log,
                    base_test_log,
                    min_group=min_group,
                    method=method,
                    target=target,
                    shrink=shrink,
                )
                q = qerror_from_logs(pred, y_log)
                name = f"{method}_{target}_m{min_group}"
                print(f"  {name}: mean={q.mean():.5f} med={np.median(q):.5f} p95={np.percentile(q,95):.5f} cov={cov:.3f} test={cov_test:.3f}", flush=True)
                experts.append(pred)
                experts_test.append(pred_test)
                names.append(name)
    E = np.column_stack(experts)
    ET = np.column_stack(experts_test)
    scores = np.array([np.mean(qerror_from_logs(E[:, i], y_log)) for i in range(E.shape[1])])
    keep = np.argsort(scores)[: min(12, len(scores))]
    print("best:", flush=True)
    for i in keep:
        print(i, names[i], scores[i], flush=True)

    def transform(z: np.ndarray, M: np.ndarray) -> np.ndarray:
        w = np.exp(z[: M.shape[1]])
        w = w / w.sum()
        p = M @ w
        alpha = 0.93 + 0.14 / (1 + np.exp(-z[M.shape[1]]))
        beta = 0.2 * np.tanh(z[M.shape[1] + 1])
        return alpha * p + beta

    def obj(z: np.ndarray) -> float:
        return float(np.mean(qerror_from_logs(transform(z, E[:, keep]), y_log)))

    result = differential_evolution(
        obj,
        [(-7, 7)] * (len(keep) + 2),
        seed=SEED,
        maxiter=240,
        tol=1e-9,
        polish=True,
        workers=1,
        updating="immediate",
    )
    pred = transform(result.x, E[:, keep])
    pred_test = transform(result.x, ET[:, keep])
    q = qerror_from_logs(pred, y_log)
    print("blend", q.mean(), np.percentile(q, [50, 90, 95, 99, 99.5, 100]), flush=True)
    w = np.exp(result.x[: len(keep)])
    w = w / w.sum()
    for idx, ww in sorted(zip(keep, w), key=lambda x: -x[1]):
        print(names[idx], ww, scores[idx], flush=True)
    pd.DataFrame({"Id": train["Id"], "Cardinality": train["Cardinality"], "PredLog": pred, "QError": q}).to_csv(ROOT / "grid_surface_oof.csv", index=False)
    sub = pd.DataFrame({"Id": test["Id"], "Cardinality": np.rint(np.maximum(np.expm1(pred_test), 1.0)).astype(np.int64)})
    sub.to_csv(ROOT / "submission_grid_surface.csv", index=False)
    print("submission stats", np.percentile(sub["Cardinality"], [0, 1, 5, 50, 95, 99, 100]), sub["Cardinality"].mean(), flush=True)


if __name__ == "__main__":
    main()
