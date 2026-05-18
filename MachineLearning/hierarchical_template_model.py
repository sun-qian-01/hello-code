"""
Hierarchical template-first cardinality estimator.

This is a deliberately different angle from the global LightGBM pipeline:
queries are split by predicate template, and each template is treated as a
small low-dimensional regression problem. The model backs off through a
hierarchy when an exact template has too few examples:

    exact column+operator template -> column template -> table/operator template
    -> table combo -> global fallback

Each local expert is validated out-of-fold and only blended if it helps OOF
Mean Q-Error. This file writes several candidate submissions so online testing
can choose the variant that transfers best.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from strong_model import qerror_from_logs


ROOT = Path(__file__).resolve().parent
SEED = 2028
N_FOLDS = 5


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


def make_keys(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    keys = {name: [] for name in ["exact", "cols", "tableop", "ops", "table", "table_npred"]}
    for row in df.to_dict("records"):
        preds = parse_predicates(row["Predicates"])
        table = norm(row["Tables"])
        join = norm(row["Join Conditions"])
        exact = "|".join(f"{col}{op}" for col, op, _ in preds) or "none"
        cols = "|".join(col for col, _, _ in preds) or "none"
        tableop = "|".join(f"{col.split('.')[0]}{op}" for col, op, _ in preds) or "none"
        ops = "|".join(op for _, op, _ in preds) or "none"
        keys["exact"].append(f"{table}||{join}||{exact}")
        keys["cols"].append(f"{table}||{cols}")
        keys["tableop"].append(f"{table}||{tableop}")
        keys["ops"].append(f"{table}||{ops}")
        keys["table"].append(table)
        keys["table_npred"].append(f"{table}||{len(preds)}")
    return {k: np.array(v, dtype=object) for k, v in keys.items()}


def build_template_features(
    df: pd.DataFrame,
    col_info: Dict[str, Tuple[float, float, float, float]],
    max_preds: int = 6,
) -> np.ndarray:
    rows: List[List[float]] = []
    for row in df.to_dict("records"):
        preds = parse_predicates(row["Predicates"])
        feats: List[float] = []
        sels: List[float] = []
        for col, op, val in preds[:max_preds]:
            cmin, cmax, card, nunique = col_info.get(col, (0.0, 1.0, 1.0, 1.0))
            crange = max(cmax - cmin, 1.0)
            norm_val = (val - cmin) / crange
            clipped = float(np.clip(norm_val, 0.0, 1.0))
            if op == "=":
                sel = 1.0 / max(nunique, 1.0)
                op_code = 0.0
            elif op == "<":
                sel = max(clipped, 1e-7)
                op_code = -1.0
            elif op == ">":
                sel = max(1.0 - clipped, 1e-7)
                op_code = 1.0
            else:
                sel = 1.0
                op_code = 2.0
            sels.append(sel)
            feats.extend(
                [
                    norm_val,
                    clipped,
                    math.log1p(max(val, 0.0)) / math.log1p(max(cmax, 1.0)),
                    math.log(max(sel, 1e-12)),
                    op_code,
                    math.log1p(max(nunique, 1.0)),
                    math.log1p(max(card, 1.0)),
                ]
            )
        while len(feats) < max_preds * 7:
            feats.extend([0.0, -1.0, 0.0, 0.0, 9.0, 0.0, 0.0])
        feats.extend(
            [
                float(len(preds)),
                float(sum(op == "=" for _, op, _ in preds)),
                float(sum(op == "<" for _, op, _ in preds)),
                float(sum(op == ">" for _, op, _ in preds)),
                float(sum(math.log(max(sel, 1e-12)) for sel in sels)) if sels else 0.0,
                float(min(sels)) if sels else 1.0,
                float(max(sels)) if sels else 1.0,
                float(np.mean(sels)) if sels else 1.0,
            ]
        )
        rows.append(feats)
    return np.asarray(rows, dtype=np.float32)


def fit_predict_local_model(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_pred: np.ndarray,
    model_kind: str,
    seed: int,
) -> np.ndarray:
    n, d = X_fit.shape
    if model_kind == "ridge_poly":
        degree = 2 if d <= 24 and n >= 30 else 1
        model = make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=degree, include_bias=False),
            Ridge(alpha=3.0),
        )
    elif model_kind == "huber":
        model = make_pipeline(StandardScaler(), HuberRegressor(alpha=0.01, epsilon=1.35, max_iter=300))
    elif model_kind == "rf":
        model = RandomForestRegressor(
            n_estimators=120,
            max_depth=10,
            min_samples_leaf=max(2, min(10, n // 80)),
            random_state=seed,
            n_jobs=-1,
        )
    elif model_kind == "et":
        model = ExtraTreesRegressor(
            n_estimators=160,
            max_depth=12,
            min_samples_leaf=max(2, min(8, n // 100)),
            random_state=seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(model_kind)
    model.fit(X_fit, y_fit)
    pred = model.predict(X_pred).astype(np.float32)
    # Local robust/linear models can occasionally extrapolate wildly in sparse
    # templates. Clipping to the local target range keeps them usable as experts
    # without letting one template dominate mean Q-Error.
    lo = float(np.percentile(y_fit, 1) - 0.75)
    hi = float(np.percentile(y_fit, 99) + 0.75)
    return np.clip(pred, lo, hi).astype(np.float32)


def local_template_expert(
    keys_train: np.ndarray,
    keys_test: np.ndarray,
    X: np.ndarray,
    X_test: np.ndarray,
    y_log: np.ndarray,
    base_log: np.ndarray,
    base_test_log: np.ndarray,
    *,
    model_kind: str,
    target_kind: str,
    min_group: int,
    shrink: float,
    blend_base: float,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    pred = base_log.copy()
    pred_test = base_test_log.copy()
    covered = np.zeros(len(X), dtype=bool)
    covered_test = np.zeros(len(X_test), dtype=bool)

    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_id = np.zeros(len(X), dtype=np.int16)
    for fold, (_, valid_idx) in enumerate(folds.split(X), start=1):
        fold_id[valid_idx] = fold

    group_to_indices = {
        key: values.values
        for key, values in pd.Series(np.arange(len(keys_train))).groupby(keys_train)
        if len(values) >= min_group
    }
    test_group_to_indices = {
        key: values.values for key, values in pd.Series(np.arange(len(keys_test))).groupby(keys_test)
    }

    for key, all_idx in group_to_indices.items():
        for fold in range(1, N_FOLDS + 1):
            valid_idx = all_idx[fold_id[all_idx] == fold]
            fit_idx = all_idx[fold_id[all_idx] != fold]
            if len(valid_idx) == 0 or len(fit_idx) < max(10, min_group // 2):
                continue
            target = y_log[fit_idx] if target_kind == "direct" else y_log[fit_idx] - base_log[fit_idx]
            local = fit_predict_local_model(
                X[fit_idx],
                target,
                X[valid_idx],
                model_kind=model_kind,
                seed=SEED + fold,
            )
            if target_kind == "direct":
                pred[valid_idx] = blend_base * base_log[valid_idx] + (1.0 - blend_base) * local
            else:
                pred[valid_idx] = base_log[valid_idx] + shrink * local
            covered[valid_idx] = True

        test_idx = test_group_to_indices.get(key)
        if test_idx is None or len(test_idx) == 0:
            continue
        target = y_log[all_idx] if target_kind == "direct" else y_log[all_idx] - base_log[all_idx]
        local_test = fit_predict_local_model(
            X[all_idx],
            target,
            X_test[test_idx],
            model_kind=model_kind,
            seed=SEED + 999,
        )
        if target_kind == "direct":
            pred_test[test_idx] = blend_base * base_test_log[test_idx] + (1.0 - blend_base) * local_test
        else:
            pred_test[test_idx] = base_test_log[test_idx] + shrink * local_test
        covered_test[test_idx] = True

    return pred, pred_test, float(covered.mean()), float(covered_test.mean())


def optimize_stack(
    experts: List[np.ndarray],
    experts_test: List[np.ndarray],
    y_log: np.ndarray,
    names: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    E = np.column_stack(experts)
    ET = np.column_stack(experts_test)
    scores = np.array([np.mean(qerror_from_logs(E[:, i], y_log)) for i in range(E.shape[1])])
    keep = np.argsort(scores)[: min(18, E.shape[1])]
    print("best individual experts:", flush=True)
    for i in keep:
        q = qerror_from_logs(E[:, i], y_log)
        print(
            f"  {i:2d} {names[i]:36s} mean={q.mean():.5f} med={np.median(q):.5f} p95={np.percentile(q,95):.5f}",
            flush=True,
        )

    EE = E[:, keep]
    EET = ET[:, keep]

    def transform(z: np.ndarray, M: np.ndarray) -> np.ndarray:
        w = np.exp(z[: M.shape[1]])
        w = w / w.sum()
        pred = M @ w
        alpha = 0.92 + 0.16 / (1.0 + np.exp(-z[M.shape[1]]))
        beta = 0.25 * np.tanh(z[M.shape[1] + 1])
        return alpha * pred + beta

    def objective(z: np.ndarray) -> float:
        return float(np.mean(qerror_from_logs(transform(z, EE), y_log)))

    result = differential_evolution(
        objective,
        [(-7, 7)] * (EE.shape[1] + 2),
        seed=SEED,
        maxiter=260,
        tol=1e-9,
        polish=True,
        workers=1,
        updating="immediate",
    )
    pred = transform(result.x, EE)
    pred_test = transform(result.x, EET)
    weights = np.exp(result.x[: EE.shape[1]])
    weights = weights / weights.sum()
    print(f"stack score={np.mean(qerror_from_logs(pred, y_log)):.5f}", flush=True)
    for idx, weight in sorted(zip(keep, weights), key=lambda x: -x[1]):
        print(f"  weight {names[idx]:36s} {weight:.5f}", flush=True)
    print("stack percentiles:", np.percentile(qerror_from_logs(pred, y_log), [50, 90, 95, 99, 99.5, 100]), flush=True)
    return pred, pred_test


def main() -> None:
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    col_info = load_col_info()
    keys_train = make_keys(train)
    keys_test = make_keys(test)
    X = build_template_features(train, col_info)
    X_test = build_template_features(test, col_info)
    y_log = np.log1p(train["Cardinality"].astype(float).values)

    # Use the best existing full-pipeline prediction only as a fallback anchor.
    fallback_oof = pd.read_csv(ROOT / "group_expert_oof.csv") if (ROOT / "group_expert_oof.csv").exists() else pd.read_csv(ROOT / "final_oof_predictions.csv")
    fallback_sub = pd.read_csv(ROOT / "submission_group_expert.csv") if (ROOT / "submission_group_expert.csv").exists() else pd.read_csv(ROOT / "submission.csv")
    base_col = "pred_log" if "pred_log" in fallback_oof.columns else "PredLog"
    base_log = fallback_oof[base_col].values.astype(float)
    base_test_log = np.log1p(fallback_sub["Cardinality"].astype(float).values)

    experts = [base_log]
    experts_test = [base_test_log]
    names = ["fallback_group"]

    settings = [
        ("exact", "ridge_poly", "direct", 80, 0.0, 0.45),
        ("exact", "ridge_poly", "direct", 200, 0.0, 0.35),
        ("exact", "huber", "resid", 80, 0.55, 0.0),
        ("exact", "et", "resid", 150, 0.45, 0.0),
        ("exact", "rf", "direct", 200, 0.0, 0.55),
        ("cols", "ridge_poly", "direct", 80, 0.0, 0.40),
        ("cols", "huber", "resid", 80, 0.45, 0.0),
        ("cols", "et", "resid", 150, 0.35, 0.0),
        ("tableop", "ridge_poly", "direct", 80, 0.0, 0.45),
        ("tableop", "huber", "resid", 80, 0.45, 0.0),
        ("ops", "ridge_poly", "direct", 100, 0.0, 0.50),
        ("table_npred", "huber", "resid", 120, 0.35, 0.0),
        ("table", "huber", "resid", 200, 0.25, 0.0),
    ]

    for key_name, model_kind, target_kind, min_group, shrink, blend_base in settings:
        print(
            f"local {key_name=} {model_kind=} {target_kind=} min={min_group}",
            flush=True,
        )
        pred, pred_test, cov, cov_test = local_template_expert(
            keys_train[key_name],
            keys_test[key_name],
            X,
            X_test,
            y_log,
            base_log,
            base_test_log,
            model_kind=model_kind,
            target_kind=target_kind,
            min_group=min_group,
            shrink=shrink,
            blend_base=blend_base,
        )
        q = qerror_from_logs(pred, y_log)
        name = f"{key_name}_{model_kind}_{target_kind}_m{min_group}"
        print(
            f"  {name}: mean={q.mean():.5f} med={np.median(q):.5f} p95={np.percentile(q,95):.5f} "
            f"cov={cov:.3f} test_cov={cov_test:.3f}",
            flush=True,
        )
        experts.append(pred)
        experts_test.append(pred_test)
        names.append(name)

    pred, pred_test = optimize_stack(experts, experts_test, y_log, names)
    q = qerror_from_logs(pred, y_log)
    pd.DataFrame(
        {
            "Id": train["Id"].values,
            "Cardinality": train["Cardinality"].values,
            "PredLog": pred,
            "QError": q,
        }
    ).to_csv(ROOT / "hierarchical_template_oof.csv", index=False)
    submission = pd.DataFrame(
        {
            "Id": test["Id"].values,
            "Cardinality": np.rint(np.maximum(np.expm1(pred_test), 1.0)).astype(np.int64),
        }
    )
    submission.to_csv(ROOT / "submission_hierarchical_template.csv", index=False)
    print(
        "submission stats:",
        np.percentile(submission["Cardinality"], [0, 1, 5, 25, 50, 75, 95, 99, 100]),
        submission["Cardinality"].mean(),
        flush=True,
    )


if __name__ == "__main__":
    main()
