"""
Monotone template model.

This tries a more structural assumption than generic ensembles: for a fixed
query template, cardinality should be monotone with respect to predicate
thresholds. For example:
  col < x increases as x increases
  col > x decreases as x increases
  equality ids are treated as unordered/numeric signals without constraints

The script trains per-template HistGradientBoostingRegressor models with
monotonic constraints and blends them with the current best fallback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

from strong_model import qerror_from_logs


ROOT = Path(__file__).resolve().parent
N_FOLDS = 5
SEED = 2031


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
    out: List[Tuple[str, str, float]] = []
    for i in range(0, len(parts) - 2, 3):
        try:
            val = float(parts[i + 2])
        except ValueError:
            val = 0.0
        out.append((parts[i], parts[i + 1], val))
    return out


def norm(value: object) -> str:
    return "" if pd.isna(value) else str(value)


def key_for(row: dict, mode: str) -> str:
    preds = parse_predicates(row["Predicates"])
    table = norm(row["Tables"])
    join = norm(row["Join Conditions"])
    if mode == "exact":
        pred = "|".join(f"{col}{op}" for col, op, _ in preds) or "none"
        return f"{table}||{join}||{pred}"
    if mode == "cols":
        pred = "|".join(col for col, _, _ in preds) or "none"
        return f"{table}||{pred}"
    if mode == "tableop":
        pred = "|".join(f"{col.split('.')[0]}{op}" for col, op, _ in preds) or "none"
        return f"{table}||{pred}"
    raise ValueError(mode)


def build_template_arrays(
    df: pd.DataFrame,
    col_info: Dict[str, Tuple[float, float, float, float]],
    mode: str,
) -> Tuple[np.ndarray, List[np.ndarray], List[List[int]]]:
    keys: List[str] = []
    vectors: List[np.ndarray] = []
    constraints: List[List[int]] = []
    for row in df.to_dict("records"):
        preds = parse_predicates(row["Predicates"])
        keys.append(key_for(row, mode))
        x: List[float] = []
        cst: List[int] = []
        sels: List[float] = []
        for col, op, val in preds:
            cmin, cmax, _, nunique = col_info.get(col, (0.0, 1.0, 1.0, 1.0))
            crange = max(cmax - cmin, 1.0)
            norm_val = (val - cmin) / crange
            clipped = float(np.clip(norm_val, 0.0, 1.0))
            if op == "<":
                mono = 1
                sel = max(clipped, 1e-7)
            elif op == ">":
                mono = -1
                sel = max(1.0 - clipped, 1e-7)
            else:
                mono = 0
                sel = 1.0 / max(nunique, 1.0)
            # Main threshold feature gets monotonic constraint. Extra transforms
            # are unconstrained because duplicate constraints can conflict.
            x.extend([clipped, np.log1p(max(val, 0.0)) / np.log1p(max(cmax, 1.0)), np.log(max(sel, 1e-12))])
            cst.extend([mono, 0, 0])
            sels.append(sel)
        x.extend(
            [
                float(sum(np.log(np.maximum(sels, 1e-12)))) if sels else 0.0,
                float(np.mean(sels)) if sels else 1.0,
            ]
        )
        cst.extend([0, 0])
        vectors.append(np.asarray(x, dtype=np.float32))
        constraints.append(cst)
    return np.asarray(keys, dtype=object), vectors, constraints


def fit_hgb(X_fit: np.ndarray, y_fit: np.ndarray, cst: List[int], seed: int) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=220,
        max_leaf_nodes=15,
        min_samples_leaf=max(8, min(30, len(y_fit) // 20)),
        l2_regularization=0.1,
        monotonic_cst=cst,
        random_state=seed,
    )
    model.fit(X_fit, y_fit)
    return model


def fit_rf(X_fit: np.ndarray, y_fit: np.ndarray, seed: int) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=180,
        max_depth=9,
        min_samples_leaf=max(3, min(12, len(y_fit) // 80)),
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_fit, y_fit)
    return model


def make_expert(
    train_keys: np.ndarray,
    test_keys: np.ndarray,
    train_vecs: List[np.ndarray],
    test_vecs: List[np.ndarray],
    constraints: List[List[int]],
    y_log: np.ndarray,
    base_log: np.ndarray,
    base_test_log: np.ndarray,
    *,
    min_group: int,
    target: str,
    model_kind: str,
    shrink: float,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    pred = base_log.copy()
    pred_test = base_test_log.copy()
    covered = np.zeros(len(train_keys), dtype=bool)
    covered_test = np.zeros(len(test_keys), dtype=bool)
    fold_id = np.zeros(len(train_keys), dtype=np.int16)
    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (_, valid_idx) in enumerate(folds.split(train_keys), start=1):
        fold_id[valid_idx] = fold

    group_to_idx = {
        key: values.values
        for key, values in pd.Series(np.arange(len(train_keys))).groupby(train_keys)
        if len(values) >= min_group
    }
    test_group_to_idx = {key: values.values for key, values in pd.Series(np.arange(len(test_keys))).groupby(test_keys)}

    for key, all_idx in group_to_idx.items():
        dim = len(train_vecs[all_idx[0]])
        if any(len(train_vecs[i]) != dim for i in all_idx):
            continue
        X_all = np.vstack([train_vecs[i] for i in all_idx])
        cst = constraints[all_idx[0]]
        for fold in range(1, N_FOLDS + 1):
            val_idx = all_idx[fold_id[all_idx] == fold]
            fit_idx_local = np.where(fold_id[all_idx] != fold)[0]
            if len(val_idx) == 0 or len(fit_idx_local) < max(20, min_group // 2):
                continue
            fit_global = all_idx[fit_idx_local]
            y_fit = y_log[fit_global] if target == "direct" else y_log[fit_global] - base_log[fit_global]
            if model_kind == "hgb":
                model = fit_hgb(X_all[fit_idx_local], y_fit, cst, SEED + fold)
            else:
                model = fit_rf(X_all[fit_idx_local], y_fit, SEED + fold)
            # val_idx are global; map back to local positions
            val_local = np.array([np.where(all_idx == i)[0][0] for i in val_idx], dtype=int)
            local_pred = model.predict(X_all[val_local])
            lo = np.percentile(y_fit, 1) - 0.5
            hi = np.percentile(y_fit, 99) + 0.5
            local_pred = np.clip(local_pred, lo, hi)
            pred[val_idx] = local_pred if target == "direct" else base_log[val_idx] + shrink * local_pred
            covered[val_idx] = True

        test_idx = test_group_to_idx.get(key)
        if test_idx is None or len(test_idx) == 0:
            continue
        if any(len(test_vecs[i]) != dim for i in test_idx):
            continue
        y_fit = y_log[all_idx] if target == "direct" else y_log[all_idx] - base_log[all_idx]
        if model_kind == "hgb":
            model = fit_hgb(X_all, y_fit, cst, SEED + 999)
        else:
            model = fit_rf(X_all, y_fit, SEED + 999)
        local_test = model.predict(np.vstack([test_vecs[i] for i in test_idx]))
        lo = np.percentile(y_fit, 1) - 0.5
        hi = np.percentile(y_fit, 99) + 0.5
        local_test = np.clip(local_test, lo, hi)
        pred_test[test_idx] = local_test if target == "direct" else base_test_log[test_idx] + shrink * local_test
        covered_test[test_idx] = True

    return pred, pred_test, float(covered.mean()), float(covered_test.mean())


def optimize(experts: List[np.ndarray], experts_test: List[np.ndarray], y_log: np.ndarray, names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    E = np.column_stack(experts)
    ET = np.column_stack(experts_test)
    scores = np.array([np.mean(qerror_from_logs(E[:, i], y_log)) for i in range(E.shape[1])])
    keep = np.argsort(scores)[: min(12, len(scores))]
    print("individual:", flush=True)
    for i in keep:
        q = qerror_from_logs(E[:, i], y_log)
        print(f"  {i:2d} {names[i]:32s} mean={q.mean():.5f} med={np.median(q):.5f} p95={np.percentile(q,95):.5f}", flush=True)

    EE = E[:, keep]
    EET = ET[:, keep]

    def transform(z: np.ndarray, M: np.ndarray) -> np.ndarray:
        w = np.exp(z[: M.shape[1]])
        w = w / w.sum()
        p = M @ w
        alpha = 0.93 + 0.14 / (1.0 + np.exp(-z[M.shape[1]]))
        beta = 0.2 * np.tanh(z[M.shape[1] + 1])
        return alpha * p + beta

    def obj(z: np.ndarray) -> float:
        return float(np.mean(qerror_from_logs(transform(z, EE), y_log)))

    result = differential_evolution(
        obj,
        [(-7, 7)] * (EE.shape[1] + 2),
        seed=SEED,
        maxiter=240,
        tol=1e-9,
        polish=True,
        workers=1,
        updating="immediate",
    )
    pred = transform(result.x, EE)
    pred_test = transform(result.x, EET)
    w = np.exp(result.x[: EE.shape[1]])
    w = w / w.sum()
    print(f"blend={np.mean(qerror_from_logs(pred, y_log)):.5f}", flush=True)
    for idx, weight in sorted(zip(keep, w), key=lambda x: -x[1]):
        print(f"  weight {names[idx]:32s} {weight:.5f}", flush=True)
    print("pct:", np.percentile(qerror_from_logs(pred, y_log), [50, 90, 95, 99, 99.5, 100]), flush=True)
    return pred, pred_test


def main() -> None:
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    col_info = load_col_info()
    y_log = np.log1p(train["Cardinality"].astype(float).values)

    # Start from the best available fallback.
    fallback_oof = pd.read_csv(ROOT / "hierarchical_template_oof.csv") if (ROOT / "hierarchical_template_oof.csv").exists() else pd.read_csv(ROOT / "group_expert_oof.csv")
    fallback_sub = pd.read_csv(ROOT / "submission_hierarchical_template.csv") if (ROOT / "submission_hierarchical_template.csv").exists() else pd.read_csv(ROOT / "submission_group_expert.csv")
    base_col = "PredLog" if "PredLog" in fallback_oof.columns else "pred_log"
    base_log = fallback_oof[base_col].values.astype(float)
    base_test_log = np.log1p(fallback_sub["Cardinality"].astype(float).values)

    experts = [base_log]
    experts_test = [base_test_log]
    names = ["fallback_hier"]

    for mode in ["exact", "cols", "tableop"]:
        keys_train, vecs_train, cst_train = build_template_arrays(train, col_info, mode)
        keys_test, vecs_test, _ = build_template_arrays(test, col_info, mode)
        settings = [
            ("hgb", "direct", 120, 0.0),
            ("hgb", "resid", 120, 0.55),
            ("rf", "direct", 200, 0.0),
        ]
        if mode != "exact":
            settings = [("hgb", "resid", 100, 0.35), ("rf", "resid", 180, 0.25)]
        for model_kind, target, min_group, shrink in settings:
            print(f"train {mode=} {model_kind=} {target=} min={min_group}", flush=True)
            pred, pred_test, cov, cov_test = make_expert(
                keys_train,
                keys_test,
                vecs_train,
                vecs_test,
                cst_train,
                y_log,
                base_log,
                base_test_log,
                min_group=min_group,
                target=target,
                model_kind=model_kind,
                shrink=shrink,
            )
            name = f"{mode}_{model_kind}_{target}_m{min_group}"
            q = qerror_from_logs(pred, y_log)
            print(f"  {name}: mean={q.mean():.5f} med={np.median(q):.5f} p95={np.percentile(q,95):.5f} cov={cov:.3f} test={cov_test:.3f}", flush=True)
            experts.append(pred)
            experts_test.append(pred_test)
            names.append(name)

    pred, pred_test = optimize(experts, experts_test, y_log, names)
    q = qerror_from_logs(pred, y_log)
    pd.DataFrame(
        {"Id": train["Id"], "Cardinality": train["Cardinality"], "PredLog": pred, "QError": q}
    ).to_csv(ROOT / "monotone_template_oof.csv", index=False)
    sub = pd.DataFrame(
        {"Id": test["Id"], "Cardinality": np.rint(np.maximum(np.expm1(pred_test), 1.0)).astype(np.int64)}
    )
    sub.to_csv(ROOT / "submission_monotone_template.csv", index=False)
    print("submission stats:", np.percentile(sub["Cardinality"], [0, 1, 5, 50, 95, 99, 100]), sub["Cardinality"].mean(), flush=True)


if __name__ == "__main__":
    main()
