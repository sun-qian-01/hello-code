"""
Value-level residual target encoding.

The isotonic family model improved online, but it only uses column min/max
selectivity and therefore treats equality predicates as uniform. This script
adds a different signal: repeated concrete predicate values such as
`mk.keyword_id = 123` or `ci.person_id = 456` get an out-of-fold residual
correction learned from training rows where the same value appeared.

The correction is intentionally simple and cross-fitted. For each fold, token
means are fit on the training side only, smoothed toward zero residual, then
applied to validation rows. Test predictions use maps fit on all training rows.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from strong_model import qerror_from_logs


ROOT = Path(__file__).resolve().parent
SEED = 1111
N_FOLDS = 5


TokenRows = List[List[str]]


def parse_predicates(value: object) -> List[Tuple[str, str, str]]:
    if pd.isna(value) or str(value).strip() == "":
        return []
    parts = [part.strip() for part in str(value).split(",")]
    return [(parts[i], parts[i + 1], parts[i + 2]) for i in range(0, len(parts) - 2, 3)]


def norm(value: object) -> str:
    return "" if pd.isna(value) else str(value)


def make_token_rows(frame: pd.DataFrame, mode: str) -> TokenRows:
    rows: TokenRows = []
    for row in frame.to_dict("records"):
        table = norm(row["Tables"])
        preds = parse_predicates(row["Predicates"])
        raw = [f"{col}{op}{val}" for col, op, val in preds]
        tokens: List[str] = []

        for col, op, val in preds:
            if mode == "eq_col" and op == "=":
                tokens.append(f"EC|{col}={val}")
            elif mode == "eq_table" and op == "=":
                tokens.append(f"ET|{table}|{col}={val}")
            elif mode == "all_col":
                tokens.append(f"AC|{col}{op}{val}")
            elif mode == "all_table":
                tokens.append(f"AT|{table}|{col}{op}{val}")

        if mode == "pair_table":
            for left, right in combinations(raw, 2):
                tokens.append(f"PT|{table}|{left}&{right}")

        rows.append(tokens)
    return rows


def make_column_eq_rows(frame: pd.DataFrame, column: str) -> TokenRows:
    rows: TokenRows = []
    for row in frame.to_dict("records"):
        tokens = [
            f"{col}={val}"
            for col, op, val in parse_predicates(row["Predicates"])
            if op == "=" and col == column
        ]
        rows.append(tokens)
    return rows


def fit_token_map(
    rows: TokenRows,
    residual: np.ndarray,
    indices: Iterable[int],
    *,
    smooth: float,
    min_count: int,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    sums: DefaultDict[str, float] = defaultdict(float)
    counts: DefaultDict[str, int] = defaultdict(int)
    for idx in indices:
        for token in rows[int(idx)]:
            sums[token] += float(residual[int(idx)])
            counts[token] += 1

    effects: Dict[str, float] = {}
    reliability: Dict[str, float] = {}
    for token, count in counts.items():
        if count < min_count:
            continue
        effects[token] = sums[token] / (count + smooth)
        reliability[token] = count / (count + smooth)
    return effects, reliability


def apply_token_map(
    rows: TokenRows,
    effects: Dict[str, float],
    reliability: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    correction = np.zeros(len(rows), dtype=np.float64)
    covered = np.zeros(len(rows), dtype=bool)
    for idx, tokens in enumerate(rows):
        total = 0.0
        weight = 0.0
        for token in tokens:
            if token not in effects:
                continue
            rel = reliability[token]
            total += rel * effects[token]
            weight += rel
        if weight > 0.0:
            correction[idx] = total / weight
            covered[idx] = True
    return correction, covered


def crossfit_correction(
    train_rows: TokenRows,
    test_rows: TokenRows,
    residual: np.ndarray,
    *,
    smooth: float,
    min_count: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    train_corr = np.zeros(len(train_rows), dtype=np.float64)
    train_cov = np.zeros(len(train_rows), dtype=bool)
    for fit_idx, valid_idx in folds.split(train_rows):
        effects, reliability = fit_token_map(
            train_rows,
            residual,
            fit_idx,
            smooth=smooth,
            min_count=min_count,
        )
        valid_rows = [train_rows[int(idx)] for idx in valid_idx]
        fold_corr, fold_cov = apply_token_map(valid_rows, effects, reliability)
        train_corr[valid_idx] = fold_corr
        train_cov[valid_idx] = fold_cov

    effects, reliability = fit_token_map(
        train_rows,
        residual,
        range(len(train_rows)),
        smooth=smooth,
        min_count=min_count,
    )
    test_corr, test_cov = apply_token_map(test_rows, effects, reliability)
    return train_corr, test_corr, train_cov, test_cov


def load_oof_log(train: pd.DataFrame, filename: str) -> np.ndarray:
    frame = pd.read_csv(ROOT / filename).set_index("Id").reindex(train["Id"].values)
    if "PredLog" not in frame.columns:
        raise ValueError(f"{filename} has no PredLog column")
    if frame["PredLog"].isna().any():
        raise ValueError(f"{filename} is not aligned to train Ids")
    return frame["PredLog"].astype(float).values


def load_submission_log(test: pd.DataFrame, filename: str) -> np.ndarray:
    frame = pd.read_csv(ROOT / filename).set_index("Id").reindex(test["Id"].values)
    if frame["Cardinality"].isna().any():
        raise ValueError(f"{filename} is not aligned to test Ids")
    return np.log1p(np.maximum(frame["Cardinality"].astype(float).values, 1.0))


def save_oof(train: pd.DataFrame, name: str, pred: np.ndarray, y_log: np.ndarray) -> None:
    q = qerror_from_logs(pred, y_log)
    pd.DataFrame(
        {
            "Id": train["Id"].values,
            "Cardinality": train["Cardinality"].values,
            "PredLog": pred,
            "QError": q,
        }
    ).to_csv(ROOT / f"{name}_oof.csv", index=False)
    print(
        name,
        f"mean={q.mean():.6f}",
        f"med={np.median(q):.4f}",
        f"p95={np.percentile(q, 95):.4f}",
        flush=True,
    )


def save_submission(test: pd.DataFrame, name: str, pred_test: np.ndarray) -> None:
    card = np.rint(np.maximum(np.expm1(pred_test), 1.0)).astype(np.int64)
    pd.DataFrame({"Id": test["Id"].values, "Cardinality": card}).to_csv(
        ROOT / f"submission_{name}.csv",
        index=False,
    )
    print(
        f"saved submission_{name}.csv",
        f"range=[{card.min()}, {card.max()}]",
        f"mean={card.mean():.1f}",
        flush=True,
    )


def weighted_sum(parts: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    out = np.zeros_like(parts[0], dtype=np.float64)
    for part, weight in zip(parts, weights):
        out += float(weight) * part
    return out


def main() -> None:
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    y_log = np.log1p(np.maximum(train["Cardinality"].astype(float).values, 1.0))

    base_log = load_oof_log(train, "isotonic_stack_gate_oof.csv")
    base_test_log = load_submission_log(test, "submission_isotonic_stack_gate.csv")
    residual = np.clip(y_log - base_log, -2.0, 2.0)
    base_q = qerror_from_logs(base_log, y_log)
    print(
        "base_isotonic_stack",
        f"mean={base_q.mean():.6f}",
        f"med={np.median(base_q):.4f}",
        f"p95={np.percentile(base_q, 95):.4f}",
        flush=True,
    )

    configs = {
        "eq_col": ("eq_col", 5.0, 2),
        "eq_table": ("eq_table", 5.0, 2),
        "pair_table": ("pair_table", 5.0, 2),
    }
    train_corrs: Dict[str, np.ndarray] = {}
    test_corrs: Dict[str, np.ndarray] = {}
    for name, (mode, smooth, min_count) in configs.items():
        train_rows = make_token_rows(train, mode)
        test_rows = make_token_rows(test, mode)
        train_corr, test_corr, train_cov, test_cov = crossfit_correction(
            train_rows,
            test_rows,
            residual,
            smooth=smooth,
            min_count=min_count,
        )
        train_corrs[name] = np.clip(train_corr, -1.0, 1.0)
        test_corrs[name] = np.clip(test_corr, -1.0, 1.0)
        print(
            name,
            f"train_coverage={train_cov.mean():.3f}",
            f"test_coverage={test_cov.mean():.3f}",
            f"mean_abs_corr={np.mean(np.abs(train_corr)):.4f}",
            flush=True,
        )

    candidates = {
        # Most conservative: only concrete equality-value hotness.
        "value_te_eq_conservative": (["eq_col"], [0.40]),
        # Stronger equality-value model; good local gain with less capacity than pairs.
        "value_te_eq": (["eq_col"], [0.80]),
        # Local best-style model, still using only transparent value/pair residual maps.
        "value_te_pair_boost": (["eq_col", "pair_table"], [0.80, 0.50]),
        # More aggressive blend that was near the best in local searches.
        "value_te_pair_full": (["eq_col", "eq_table", "pair_table"], [0.90, 0.10, 0.60]),
    }

    for name, (parts, weights) in candidates.items():
        train_delta = weighted_sum([train_corrs[part] for part in parts], weights)
        test_delta = weighted_sum([test_corrs[part] for part in parts], weights)
        pred = base_log + train_delta
        pred_test = base_test_log + test_delta
        save_oof(train, name, pred, y_log)
        save_submission(test, name, pred_test)

    # Column-specific equality maps let us keep the useful high-cardinality ID
    # corrections while avoiding low-cardinality columns that were not stable in
    # OOF, such as t.kind_id and mc.company_type_id.
    eq_columns = (
        "t.kind_id",
        "t.production_year",
        "mk.keyword_id",
        "mc.company_type_id",
        "ci.role_id",
        "mc.company_id",
        "ci.person_id",
        "mi.info_type_id",
        "mi_idx.info_type_id",
    )
    column_train_corrs: Dict[str, np.ndarray] = {}
    column_test_corrs: Dict[str, np.ndarray] = {}
    for column in eq_columns:
        train_rows = make_column_eq_rows(train, column)
        test_rows = make_column_eq_rows(test, column)
        train_corr, test_corr, train_cov, test_cov = crossfit_correction(
            train_rows,
            test_rows,
            residual,
            smooth=5.0,
            min_count=2,
        )
        column_train_corrs[column] = np.clip(train_corr, -1.0, 1.0)
        column_test_corrs[column] = np.clip(test_corr, -1.0, 1.0)
        print(
            f"column={column}",
            f"train_coverage={train_cov.mean():.3f}",
            f"test_coverage={test_cov.mean():.3f}",
            f"mean_abs_corr={np.mean(np.abs(train_corr)):.4f}",
            flush=True,
        )

    column_candidates: Dict[str, Dict[str, float]] = {
        # Strong and compact: only high-cardinality entity IDs.
        "value_te_id_heavy": {
            "mk.keyword_id": 0.80,
            "mc.company_id": 0.80,
            "ci.person_id": 0.80,
        },
        # Adds the low-cardinality columns that showed reliable OOF wins.
        "value_te_column_core": {
            "t.production_year": 0.90,
            "mk.keyword_id": 1.00,
            "ci.role_id": 0.50,
            "mc.company_id": 1.00,
            "ci.person_id": 0.70,
            "mi.info_type_id": 0.70,
        },
        # Near local-best but still hand-capped; useful as an aggressive probe.
        "value_te_column_capped": {
            "t.production_year": 1.20,
            "mk.keyword_id": 1.20,
            "ci.role_id": 0.50,
            "mc.company_id": 1.20,
            "ci.person_id": 1.20,
            "mi.info_type_id": 0.70,
        },
    }

    for name, weights in column_candidates.items():
        train_delta = weighted_sum(
            [column_train_corrs[column] for column in eq_columns],
            [weights.get(column, 0.0) for column in eq_columns],
        )
        test_delta = weighted_sum(
            [column_test_corrs[column] for column in eq_columns],
            [weights.get(column, 0.0) for column in eq_columns],
        )
        pred = base_log + train_delta
        pred_test = base_test_log + test_delta
        save_oof(train, name, pred, y_log)
        save_submission(test, name, pred_test)


if __name__ == "__main__":
    main()
