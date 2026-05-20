"""
Local residual forest experiment.

This is a deliberately different algorithm from the recent expert gates:
within each `table + predicate columns` family, fit a small ExtraTrees residual
model over normalized predicate coordinates. The current best family-fallback
prediction remains the anchor, and the forest correction is heavily shrunk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold

from strong_model import qerror_from_logs


ROOT = Path(__file__).resolve().parent
SEED = 331
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
    parts = [part.strip() for part in str(value).split(",")]
    out: List[Tuple[str, str, float]] = []
    for i in range(0, len(parts) - 2, 3):
        try:
            val = float(parts[i + 2])
        except ValueError:
            val = 0.0
        out.append((parts[i], parts[i + 1], val))
    return out


def key_and_vector(row: dict, col_info: Dict[str, Tuple[float, float, float, float]]) -> Tuple[str, np.ndarray]:
    preds = parse_predicates(row["Predicates"])
    table = "" if pd.isna(row["Tables"]) else str(row["Tables"])
    cols: List[str] = []
    feats: List[float] = []
    sels: List[float] = []
    for col, op, val in preds:
        cols.append(col)
        cmin, cmax, card, nunique = col_info.get(col, (0.0, 1.0, 1.0, 1.0))
        crange = max(cmax - cmin, 1.0)
        norm = (val - cmin) / crange
        clipped = float(np.clip(norm, 0.0, 1.0))
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
                float(norm),
                clipped,
                float(np.log1p(max(val, 0.0)) / np.log1p(max(cmax, 1.0))),
                float(np.log(max(sel, 1e-12))),
                op_code,
                float(np.log1p(max(card, 1.0))),
                float(np.log1p(max(nunique, 1.0))),
            ]
        )
    feats.extend(
        [
            float(len(preds)),
            float(sum(op == "=" for _, op, _ in preds)),
            float(sum(op == "<" for _, op, _ in preds)),
            float(sum(op == ">" for _, op, _ in preds)),
            float(sum(np.log(max(sel, 1e-12)) for sel in sels)) if sels else 0.0,
        ]
    )
    return f"{table}||{'|'.join(cols)}", np.asarray(feats, dtype=np.float32)


def load_submission_log(test: pd.DataFrame, filename: str) -> np.ndarray:
    sub = pd.read_csv(ROOT / filename).set_index("Id").reindex(test["Id"].values)
    return np.log1p(np.maximum(sub["Cardinality"].astype(float).values, 1.0))


def local_forest_expert(
    train_keys: np.ndarray,
    test_keys: np.ndarray,
    train_vecs: List[np.ndarray],
    test_vecs: List[np.ndarray],
    y_log: np.ndarray,
    base_log: np.ndarray,
    base_test_log: np.ndarray,
    *,
    min_group: int,
    shrink: float,
    min_leaf: int,
    max_depth: int,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    pred = base_log.copy()
    pred_test = base_test_log.copy()
    covered = np.zeros(len(train_keys), dtype=bool)
    covered_test = np.zeros(len(test_keys), dtype=bool)

    fold_id = np.zeros(len(train_keys), dtype=np.int16)
    for fold, (_, valid_idx) in enumerate(KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(train_keys), start=1):
        fold_id[valid_idx] = fold

    train_groups = {
        key: values.values
        for key, values in pd.Series(np.arange(len(train_keys))).groupby(train_keys)
        if len(values) >= min_group
    }
    test_groups = {key: values.values for key, values in pd.Series(np.arange(len(test_keys))).groupby(test_keys)}
    residual = np.clip(y_log - base_log, -1.0, 1.0)

    for key, all_idx in train_groups.items():
        dim = len(train_vecs[all_idx[0]])
        if any(len(train_vecs[idx]) != dim for idx in all_idx):
            continue
        x_all = np.vstack([train_vecs[idx] for idx in all_idx])
        r_all = residual[all_idx]
        for fold in range(1, N_FOLDS + 1):
            valid_mask = fold_id[all_idx] == fold
            if not valid_mask.any():
                continue
            fit_mask = ~valid_mask
            if fit_mask.sum() < max(30, min_group // 2):
                continue
            model = ExtraTreesRegressor(
                n_estimators=160,
                max_depth=max_depth,
                min_samples_leaf=min_leaf,
                random_state=SEED + fold,
                n_jobs=-1,
            )
            model.fit(x_all[fit_mask], r_all[fit_mask])
            valid_idx = all_idx[valid_mask]
            pred[valid_idx] = base_log[valid_idx] + shrink * model.predict(x_all[valid_mask])
            covered[valid_idx] = True

        test_idx = test_groups.get(key)
        if test_idx is None or len(test_idx) == 0:
            continue
        if any(len(test_vecs[idx]) != dim for idx in test_idx):
            continue
        model = ExtraTreesRegressor(
            n_estimators=180,
            max_depth=max_depth,
            min_samples_leaf=min_leaf,
            random_state=SEED + 999,
            n_jobs=-1,
        )
        model.fit(x_all, r_all)
        pred_test[test_idx] = base_test_log[test_idx] + shrink * model.predict(np.vstack([test_vecs[idx] for idx in test_idx]))
        covered_test[test_idx] = True

    return pred, pred_test, float(covered.mean()), float(covered_test.mean())


def main() -> None:
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    col_info = load_col_info()
    y_log = np.log1p(np.maximum(train["Cardinality"].astype(float).values, 1.0))
    base_oof = pd.read_csv(ROOT / "meta_gate_family_fallback_oof.csv")
    base_log = base_oof["PredLog"].astype(float).values
    base_test_log = load_submission_log(test, "submission_meta_gate_family_fallback.csv")

    train_pairs = [key_and_vector(row, col_info) for row in train.to_dict("records")]
    test_pairs = [key_and_vector(row, col_info) for row in test.to_dict("records")]
    train_keys = np.asarray([pair[0] for pair in train_pairs], dtype=object)
    test_keys = np.asarray([pair[0] for pair in test_pairs], dtype=object)
    train_vecs = [pair[1] for pair in train_pairs]
    test_vecs = [pair[1] for pair in test_pairs]

    best = (float("inf"), None, None, None)
    for min_group in (80, 120, 200):
        for shrink in (0.03, 0.05, 0.08, 0.12):
            for min_leaf in (20, 50, 100):
                print(f"forest min={min_group} shrink={shrink} leaf={min_leaf}", flush=True)
                pred, pred_test, cov, cov_test = local_forest_expert(
                    train_keys,
                    test_keys,
                    train_vecs,
                    test_vecs,
                    y_log,
                    base_log,
                    base_test_log,
                    min_group=min_group,
                    shrink=shrink,
                    min_leaf=min_leaf,
                    max_depth=6,
                )
                q = qerror_from_logs(pred, y_log)
                print(
                    f"  mean={q.mean():.6f} med={np.median(q):.4f} p95={np.percentile(q, 95):.4f} "
                    f"cov={cov:.3f} test_cov={cov_test:.3f}",
                    flush=True,
                )
                if q.mean() < best[0]:
                    best = (float(q.mean()), pred, pred_test, (min_group, shrink, min_leaf, cov, cov_test))

    score, pred, pred_test, params = best
    q = qerror_from_logs(pred, y_log)
    print(f"best score={score:.6f} params={params}", flush=True)
    pd.DataFrame(
        {
            "Id": train["Id"].values,
            "Cardinality": train["Cardinality"].values,
            "PredLog": pred,
            "QError": q,
        }
    ).to_csv(ROOT / "local_residual_forest_oof.csv", index=False)
    card = np.rint(np.maximum(np.expm1(pred_test), 1.0)).astype(np.int64)
    pd.DataFrame({"Id": test["Id"].values, "Cardinality": card}).to_csv(
        ROOT / "submission_local_residual_forest.csv",
        index=False,
    )
    print(
        "saved submission_local_residual_forest.csv",
        f"range=[{card.min()}, {card.max()}]",
        f"mean={card.mean():.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
