"""
Template-local KNN/interpolation experiments.

This is intentionally separate from the main final pipeline so we can try a
different model family without destabilizing the known-good submission.csv.
It uses the current best OOF/test predictions as a fallback and only replaces
or corrects predictions where a query has enough same-template neighbours.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.model_selection import KFold

from strong_model import qerror_from_logs


ROOT = Path(__file__).resolve().parent
SEED = 42
N_FOLDS = 5


def load_stats() -> Dict[str, Tuple[float, float, float, float]]:
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


def make_signature(row: pd.Series, mode: str) -> str:
    tables = "" if pd.isna(row["Tables"]) else str(row["Tables"])
    joins = "" if pd.isna(row["Join Conditions"]) else str(row["Join Conditions"])
    preds = parse_predicates(row["Predicates"])
    if mode == "shape":
        pred_part = "|".join(f"{col}{op}" for col, op, _ in preds)
    elif mode == "cols":
        pred_part = "|".join(col for col, _, _ in preds)
    elif mode == "tableop":
        pred_part = "|".join(f"{col.split('.')[0]}.{op}" for col, op, _ in preds)
    elif mode == "ops":
        pred_part = "|".join(op for _, op, _ in preds)
    else:
        raise ValueError(mode)
    return f"{tables}||{joins}||{pred_part}"


def make_vector(row: pd.Series, col_info: Dict[str, Tuple[float, float, float, float]]) -> np.ndarray:
    values: List[float] = []
    for col, op, val in parse_predicates(row["Predicates"]):
        cmin, cmax, _, nunique = col_info.get(col, (0.0, 1.0, 1.0, 1.0))
        crange = max(cmax - cmin, 1.0)
        norm = (val - cmin) / crange
        log_norm = np.log1p(max(val, 0.0)) / np.log1p(max(cmax, 1.0))
        if op == "=":
            sel = 1.0 / max(nunique, 1.0)
        elif op == "<":
            sel = np.clip(norm, 1e-7, 1.0)
        elif op == ">":
            sel = np.clip(1.0 - norm, 1e-7, 1.0)
        else:
            sel = 1.0
        values.extend([norm, log_norm, np.log(sel)])
    if not values:
        return np.zeros(1, dtype=np.float32)
    return np.array(values, dtype=np.float32)


def weighted_knn(
    X_query: np.ndarray,
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    k: int,
    temp: float,
) -> np.ndarray:
    if len(X_fit) == 0:
        return np.zeros(len(X_query), dtype=np.float32)
    k = min(k, len(X_fit))
    # Squared Euclidean distance. Groups are small enough that a dense matrix is OK.
    q2 = np.sum(X_query * X_query, axis=1, keepdims=True)
    f2 = np.sum(X_fit * X_fit, axis=1)[None, :]
    dist2 = np.maximum(q2 + f2 - 2.0 * (X_query @ X_fit.T), 0.0)
    if k < X_fit.shape[0]:
        part = np.argpartition(dist2, kth=k - 1, axis=1)[:, :k]
        d = np.take_along_axis(dist2, part, axis=1)
        vals = y_fit[part]
    else:
        d = dist2
        vals = np.broadcast_to(y_fit, d.shape)
    weights = np.exp(-np.sqrt(d) / max(temp, 1e-6))
    weights_sum = np.maximum(weights.sum(axis=1), 1e-12)
    return ((weights * vals).sum(axis=1) / weights_sum).astype(np.float32)


def make_knn_expert(
    train: pd.DataFrame,
    test: pd.DataFrame,
    vectors_train: List[np.ndarray],
    vectors_test: List[np.ndarray],
    y_log: np.ndarray,
    base_log: np.ndarray,
    base_test_log: np.ndarray,
    *,
    mode: str,
    target: str,
    k: int,
    temp: float,
    min_group: int,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    keys = np.array([make_signature(row, mode) for _, row in train.iterrows()])
    test_keys = np.array([make_signature(row, mode) for _, row in test.iterrows()])
    pred = base_log.copy()
    pred_test = base_test_log.copy()
    covered = np.zeros(len(train), dtype=bool)
    covered_test = np.zeros(len(test), dtype=bool)
    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    group_to_indices: Dict[str, np.ndarray] = {
        key: idx.values for key, idx in pd.Series(np.arange(len(keys))).groupby(keys)
    }
    test_group_to_indices: Dict[str, np.ndarray] = {
        key: idx.values for key, idx in pd.Series(np.arange(len(test_keys))).groupby(test_keys)
    }

    fold_id = np.zeros(len(train), dtype=np.int16)
    for fold, (_, valid_idx) in enumerate(folds.split(train), start=1):
        fold_id[valid_idx] = fold

    for key, all_idx in group_to_indices.items():
        if len(all_idx) < min_group:
            continue
        dim = len(vectors_train[all_idx[0]])
        if dim == 0:
            continue
        if any(len(vectors_train[i]) != dim for i in all_idx):
            continue
        X_all = np.vstack([vectors_train[i] for i in all_idx])
        target_all = y_log[all_idx] if target == "y" else y_log[all_idx] - base_log[all_idx]

        for fold in range(1, N_FOLDS + 1):
            valid_mask = fold_id[all_idx] == fold
            if not valid_mask.any():
                continue
            fit_mask = ~valid_mask
            if fit_mask.sum() < max(2, min_group // 2):
                continue
            valid_local = np.where(valid_mask)[0]
            fit_local = np.where(fit_mask)[0]
            vals = weighted_knn(
                X_all[valid_local],
                X_all[fit_local],
                target_all[fit_local],
                k=k,
                temp=temp,
            )
            valid_global = all_idx[valid_local]
            pred[valid_global] = vals if target == "y" else base_log[valid_global] + vals
            covered[valid_global] = True

        test_idx = test_group_to_indices.get(key)
        if test_idx is None or len(test_idx) == 0:
            continue
        if any(len(vectors_test[i]) != dim for i in test_idx):
            continue
        vals = weighted_knn(
            np.vstack([vectors_test[i] for i in test_idx]),
            X_all,
            target_all,
            k=k,
            temp=temp,
        )
        pred_test[test_idx] = vals if target == "y" else base_test_log[test_idx] + vals
        covered_test[test_idx] = True

    return pred, pred_test, float(covered.mean()), float(covered_test.mean())


def main() -> None:
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    oof = pd.read_csv(ROOT / "final_oof_predictions.csv")
    current_submission = pd.read_csv(ROOT / "submission.csv")
    col_info = load_stats()

    y_log = np.log1p(train["Cardinality"].astype(float).values)
    base_log = oof["PredLog"].values.astype(float)
    base_test_log = np.log1p(current_submission["Cardinality"].astype(float).values)

    vectors_train = [make_vector(row, col_info) for _, row in train.iterrows()]
    vectors_test = [make_vector(row, col_info) for _, row in test.iterrows()]

    settings = [
        ("shape", "resid", 5, 0.03, 8),
        ("shape", "resid", 20, 0.08, 8),
        ("shape", "resid", 50, 0.15, 8),
        ("shape", "y", 10, 0.05, 10),
        ("shape", "y", 50, 0.15, 10),
        ("cols", "resid", 20, 0.08, 15),
        ("cols", "resid", 80, 0.20, 15),
        ("tableop", "resid", 20, 0.08, 15),
        ("ops", "resid", 50, 0.15, 25),
    ]

    experts = [base_log]
    experts_test = [base_test_log]
    names = ["base"]

    for mode, target, k, temp, min_group in settings:
        print(f"knn {mode=} {target=} k={k} temp={temp} min_group={min_group}", flush=True)
        pred, pred_test, cov, cov_test = make_knn_expert(
            train,
            test,
            vectors_train,
            vectors_test,
            y_log,
            base_log,
            base_test_log,
            mode=mode,
            target=target,
            k=k,
            temp=temp,
            min_group=min_group,
        )
        qe = qerror_from_logs(pred, y_log)
        name = f"knn_{mode}_{target}_k{k}_t{temp}_m{min_group}"
        print(
            f"  {name}: mean={qe.mean():.5f}, median={np.median(qe):.5f}, "
            f"p95={np.percentile(qe, 95):.5f}, cov={cov:.3f}, test_cov={cov_test:.3f}",
            flush=True,
        )
        experts.append(pred)
        experts_test.append(pred_test)
        names.append(name)

    E = np.column_stack(experts)
    ET = np.column_stack(experts_test)
    scores = np.array([np.mean(qerror_from_logs(E[:, i], y_log)) for i in range(E.shape[1])])
    keep = np.argsort(scores)[: min(10, E.shape[1])]
    print("best individual experts:", flush=True)
    for i in keep:
        print(f"  {i:2d} {names[i]:35s} {scores[i]:.5f}", flush=True)

    EE = E[:, keep]
    EET = ET[:, keep]

    def objective(z: np.ndarray) -> float:
        w = np.exp(z)
        w = w / w.sum()
        pred = EE @ w
        return float(np.mean(qerror_from_logs(pred, y_log)))

    result = differential_evolution(
        objective,
        [(-6, 6)] * len(keep),
        seed=123,
        maxiter=220,
        tol=1e-8,
        polish=True,
        workers=1,
        updating="immediate",
    )
    weights = np.exp(result.x)
    weights = weights / weights.sum()
    pred = EE @ weights
    pred_test = EET @ weights
    qe = qerror_from_logs(pred, y_log)
    print(f"blend mean={qe.mean():.5f}", flush=True)
    for idx, weight in sorted(zip(keep, weights), key=lambda x: -x[1]):
        print(f"  {names[idx]:35s} weight={weight:.5f} indiv={scores[idx]:.5f}", flush=True)
    print("blend percentiles:", np.percentile(qe, [50, 90, 95, 99, 99.5, 100]), flush=True)

    pd.DataFrame(
        {
            "Id": train["Id"].values,
            "Cardinality": train["Cardinality"].values,
            "PredLog": pred,
            "QError": qe,
        }
    ).to_csv(ROOT / "template_knn_oof.csv", index=False)
    submission = pd.DataFrame(
        {
            "Id": test["Id"].values,
            "Cardinality": np.rint(np.maximum(np.expm1(pred_test), 1.0)).astype(np.int64),
        }
    )
    submission.to_csv(ROOT / "submission_template_knn.csv", index=False)
    print(
        "submission stats:",
        np.percentile(submission["Cardinality"], [0, 1, 5, 25, 50, 75, 95, 99, 100]),
        submission["Cardinality"].mean(),
        flush=True,
    )


if __name__ == "__main__":
    main()
