"""
Conservative table-level gate for the isotonic family model.

`isotonic_family_model.py` applies the monotone selectivity residual correction
to every sufficiently large column family. This gate keeps the stacked-family
prediction as the default and switches an entire table-combination group to the
isotonic prediction only when the training side of the fold shows a clear OOF
Q-error win. It reproduces the previously generated `isotonic_stack_gate` files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from strong_model import qerror_from_logs


ROOT = Path(__file__).resolve().parent
SEED = 707
N_FOLDS = 5
MIN_GROUP = 800
MARGIN = 0.01


def group_indices(values: np.ndarray, indices: Iterable[int]) -> Dict[str, np.ndarray]:
    groups: Dict[str, list[int]] = {}
    for idx in indices:
        groups.setdefault(str(values[int(idx)]), []).append(int(idx))
    return {key: np.asarray(group, dtype=np.int32) for key, group in groups.items()}


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


def route(
    train_tables: np.ndarray,
    test_tables: np.ndarray,
    base_log: np.ndarray,
    isotonic_log: np.ndarray,
    base_test_log: np.ndarray,
    isotonic_test_log: np.ndarray,
    y_log: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    q_base = qerror_from_logs(base_log, y_log)
    q_iso = qerror_from_logs(isotonic_log, y_log)

    pred = base_log.copy()
    for fit_idx, valid_idx in KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(base_log):
        fold_pred = base_log[valid_idx].copy()
        fit_groups = group_indices(train_tables, fit_idx)
        valid_groups = group_indices(train_tables, valid_idx)
        for key, valid_group in valid_groups.items():
            fit_group = fit_groups.get(key)
            if fit_group is None or len(fit_group) < MIN_GROUP:
                continue
            if q_iso[fit_group].mean() + MARGIN < q_base[fit_group].mean():
                fold_pred[np.isin(valid_idx, valid_group)] = isotonic_log[valid_group]
        pred[valid_idx] = fold_pred

    pred_test = base_test_log.copy()
    test_choice = np.zeros(len(test_tables), dtype=np.int8)
    train_groups = group_indices(train_tables, range(len(train_tables)))
    test_groups = group_indices(test_tables, range(len(test_tables)))
    for key, test_idx in test_groups.items():
        fit_group = train_groups.get(key)
        if fit_group is None or len(fit_group) < MIN_GROUP:
            continue
        if q_iso[fit_group].mean() + MARGIN < q_base[fit_group].mean():
            pred_test[test_idx] = isotonic_test_log[test_idx]
            test_choice[test_idx] = 1

    choices = {"stacked": int((test_choice == 0).sum()), "isotonic": int((test_choice == 1).sum())}
    return pred, pred_test, choices


def main() -> None:
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    y_log = np.log1p(np.maximum(train["Cardinality"].astype(float).values, 1.0))
    train_tables = train["Tables"].fillna("").astype(str).values
    test_tables = test["Tables"].fillna("").astype(str).values

    base_log = load_oof_log(train, "stacked_family_gate_oof.csv")
    isotonic_log = load_oof_log(train, "isotonic_family_oof.csv")
    base_test_log = load_submission_log(test, "submission_stacked_family_gate.csv")
    isotonic_test_log = load_submission_log(test, "submission_isotonic_family.csv")

    pred, pred_test, choices = route(
        train_tables,
        test_tables,
        base_log,
        isotonic_log,
        base_test_log,
        isotonic_test_log,
        y_log,
    )
    q = qerror_from_logs(pred, y_log)
    print(
        "isotonic_stack_gate",
        f"mean={q.mean():.6f}",
        f"med={np.median(q):.4f}",
        f"p95={np.percentile(q, 95):.4f}",
        f"choices={choices}",
        flush=True,
    )
    pd.DataFrame(
        {
            "Id": train["Id"].values,
            "Cardinality": train["Cardinality"].values,
            "PredLog": pred,
            "QError": q,
        }
    ).to_csv(ROOT / "isotonic_stack_gate_oof.csv", index=False)
    card = np.rint(np.maximum(np.expm1(pred_test), 1.0)).astype(np.int64)
    pd.DataFrame({"Id": test["Id"].values, "Cardinality": card}).to_csv(
        ROOT / "submission_isotonic_stack_gate.csv",
        index=False,
    )
    print(
        "saved submission_isotonic_stack_gate.csv",
        f"range=[{card.min()}, {card.max()}]",
        f"mean={card.mean():.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
