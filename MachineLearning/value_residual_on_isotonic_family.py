"""
Value-level residual target encoding on the online-validated isotonic base.

`value_residual_target_encoding.py` uses the locally stronger
`isotonic_stack_gate` base. This companion script preserves the latest
online-validated base, `submission_isotonic_family.csv`, and adds the same
concrete equality-value residual signal on top of it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from strong_model import qerror_from_logs
from value_residual_target_encoding import (
    crossfit_correction,
    load_oof_log,
    load_submission_log,
    make_column_eq_rows,
    make_token_rows,
    save_oof,
    save_submission,
    weighted_sum,
)


ROOT = Path(__file__).resolve().parent


def main() -> None:
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    y_log = np.log1p(np.maximum(train["Cardinality"].astype(float).values, 1.0))

    base_log = load_oof_log(train, "isotonic_family_oof.csv")
    base_test_log = load_submission_log(test, "submission_isotonic_family.csv")
    residual = np.clip(y_log - base_log, -2.0, 2.0)
    base_q = qerror_from_logs(base_log, y_log)
    print(
        "base_isotonic_family",
        f"mean={base_q.mean():.6f}",
        f"med={np.median(base_q):.4f}",
        f"p95={np.percentile(base_q, 95):.4f}",
        flush=True,
    )

    train_rows = make_token_rows(train, "eq_col")
    test_rows = make_token_rows(test, "eq_col")
    eq_train, eq_test, eq_cov, eq_test_cov = crossfit_correction(
        train_rows,
        test_rows,
        residual,
        smooth=5.0,
        min_count=2,
    )
    eq_train = np.clip(eq_train, -1.0, 1.0)
    eq_test = np.clip(eq_test, -1.0, 1.0)
    print(
        "eq_col",
        f"train_coverage={eq_cov.mean():.3f}",
        f"test_coverage={eq_test_cov.mean():.3f}",
        f"mean_abs_corr={np.mean(np.abs(eq_train)):.4f}",
        flush=True,
    )

    for name, shrink in {
        "value_te_eq_conservative_on_isotonic": 0.40,
        "value_te_eq_on_isotonic": 0.80,
    }.items():
        pred = base_log + shrink * eq_train
        pred_test = base_test_log + shrink * eq_test
        save_oof(train, name, pred, y_log)
        save_submission(test, name, pred_test)

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
        "value_te_id_heavy_on_isotonic": {
            "mk.keyword_id": 0.80,
            "mc.company_id": 0.80,
            "ci.person_id": 0.80,
        },
        "value_te_column_core_on_isotonic": {
            "t.production_year": 0.90,
            "mk.keyword_id": 1.00,
            "ci.role_id": 0.50,
            "mc.company_id": 1.00,
            "ci.person_id": 0.70,
            "mi.info_type_id": 0.70,
        },
        "value_te_column_capped_on_isotonic": {
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
