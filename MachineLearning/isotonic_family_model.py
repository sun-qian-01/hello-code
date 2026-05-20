"""
Column-family isotonic residual model.

This is a fresh modeling assumption after the post-processing gates plateaued:
within a `table + predicate columns` family, cardinality should move mostly
monotonically with the product of per-predicate selectivities. We fit a 1D
isotonic residual curve over that selectivity score and apply it only to large
families.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

from strong_model import qerror_from_logs


ROOT = Path(__file__).resolve().parent
SEED = 909
N_FOLDS = 5
MIN_GROUP = 400
SHRINK = 0.35


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


def key_and_score(row: dict, col_info: Dict[str, Tuple[float, float, float, float]]) -> Tuple[str, float]:
    preds = parse_predicates(row["Predicates"])
    table = "" if pd.isna(row["Tables"]) else str(row["Tables"])
    cols: List[str] = []
    score = 0.0
    for col, op, val in preds:
        cols.append(col)
        cmin, cmax, _, nunique = col_info.get(col, (0.0, 1.0, 1.0, 1.0))
        norm = float(np.clip((val - cmin) / max(cmax - cmin, 1.0), 0.0, 1.0))
        if op == "=":
            sel = 1.0 / max(nunique, 1.0)
        elif op == "<":
            sel = max(norm, 1e-7)
        elif op == ">":
            sel = max(1.0 - norm, 1e-7)
        else:
            sel = 1.0
        score += float(np.log(max(sel, 1e-12)))
    return f"{table}||{'|'.join(cols)}", score


def load_submission_log(test: pd.DataFrame, filename: str) -> np.ndarray:
    sub = pd.read_csv(ROOT / filename).set_index("Id").reindex(test["Id"].values)
    return np.log1p(np.maximum(sub["Cardinality"].astype(float).values, 1.0))


def main() -> None:
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    col_info = load_col_info()
    y_log = np.log1p(np.maximum(train["Cardinality"].astype(float).values, 1.0))
    base_log = pd.read_csv(ROOT / "stacked_family_gate_oof.csv")["PredLog"].astype(float).values
    base_test_log = load_submission_log(test, "submission_stacked_family_gate.csv")

    train_pairs = [key_and_score(row, col_info) for row in train.to_dict("records")]
    test_pairs = [key_and_score(row, col_info) for row in test.to_dict("records")]
    train_keys = np.asarray([pair[0] for pair in train_pairs], dtype=object)
    test_keys = np.asarray([pair[0] for pair in test_pairs], dtype=object)
    train_score = np.asarray([pair[1] for pair in train_pairs], dtype=float)
    test_score = np.asarray([pair[1] for pair in test_pairs], dtype=float)

    fold_id = np.zeros(len(train), dtype=np.int16)
    for fold, (_, valid_idx) in enumerate(KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(train), start=1):
        fold_id[valid_idx] = fold

    pred = base_log.copy()
    pred_test = base_test_log.copy()
    covered = np.zeros(len(train), dtype=bool)
    covered_test = np.zeros(len(test), dtype=bool)
    residual = y_log - base_log
    groups = {
        key: values.values
        for key, values in pd.Series(np.arange(len(train_keys))).groupby(train_keys)
        if len(values) >= MIN_GROUP
    }
    test_groups = {key: values.values for key, values in pd.Series(np.arange(len(test_keys))).groupby(test_keys)}

    for key, all_idx in groups.items():
        x_all = train_score[all_idx]
        y_all = residual[all_idx]
        for fold in range(1, N_FOLDS + 1):
            valid_mask = fold_id[all_idx] == fold
            if not valid_mask.any():
                continue
            fit_mask = ~valid_mask
            if fit_mask.sum() < max(20, MIN_GROUP // 2):
                continue
            model = IsotonicRegression(increasing=True, out_of_bounds="clip")
            model.fit(x_all[fit_mask], y_all[fit_mask])
            valid_idx = all_idx[valid_mask]
            pred[valid_idx] = base_log[valid_idx] + SHRINK * model.predict(x_all[valid_mask])
            covered[valid_idx] = True

        test_idx = test_groups.get(key)
        if test_idx is None or len(test_idx) == 0:
            continue
        model = IsotonicRegression(increasing=True, out_of_bounds="clip")
        model.fit(x_all, y_all)
        pred_test[test_idx] = base_test_log[test_idx] + SHRINK * model.predict(test_score[test_idx])
        covered_test[test_idx] = True

    q = qerror_from_logs(pred, y_log)
    print(
        "isotonic_family",
        f"mean={q.mean():.6f}",
        f"med={np.median(q):.4f}",
        f"p95={np.percentile(q, 95):.4f}",
        f"coverage={covered.mean():.3f}",
        f"test_coverage={covered_test.mean():.3f}",
        flush=True,
    )
    pd.DataFrame(
        {
            "Id": train["Id"].values,
            "Cardinality": train["Cardinality"].values,
            "PredLog": pred,
            "QError": q,
        }
    ).to_csv(ROOT / "isotonic_family_oof.csv", index=False)
    card = np.rint(np.maximum(np.expm1(pred_test), 1.0)).astype(np.int64)
    pd.DataFrame({"Id": test["Id"].values, "Cardinality": card}).to_csv(
        ROOT / "submission_isotonic_family.csv",
        index=False,
    )
    print(
        "saved submission_isotonic_family.csv",
        f"range=[{card.min()}, {card.max()}]",
        f"mean={card.mean():.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
