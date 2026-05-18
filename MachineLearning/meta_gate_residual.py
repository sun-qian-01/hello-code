"""
Lightweight residual meta-model on top of the column-family gate.

The gate chooses the best existing expert by template family. This script keeps
that strong discrete prediction and learns a tiny, heavily shrunk residual from
the existing OOF prediction coordinates plus simple query-structure features.

The model is intentionally conservative:

  * base prediction is submission_column_family_gate_shape50_cols80_affine;
  * target is only the base residual, clipped to [-1, 1] in log space;
  * Huber residual predictions are shrunk by 0.05 before applying.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from strong_model import qerror_from_logs


ROOT = Path(__file__).resolve().parent
SEED = 991
N_FOLDS = 5


EXPERT_FILES: Tuple[Tuple[str, str, str], ...] = (
    ("gate50", "column_family_gate_shape50_cols80_oof.csv", "submission_column_family_gate_shape50_cols80.csv"),
    ("gate_aff", "column_family_gate_shape50_cols80_affine_oof.csv", "submission_column_family_gate_shape50_cols80_affine.csv"),
    ("latent", "latent_final_blend_oof.csv", "submission_latent_final_blend.csv"),
    ("lowrank", "lowrank_latent_oof.csv", "submission_lowrank_latent.csv"),
    ("grid", "grid_surface_oof.csv", "submission_grid_surface.csv"),
    ("rewrite", "oof_rewrite_angle_affine_oof.csv", "submission_rewrite_angle_affine.csv"),
    ("monotone", "monotone_template_oof.csv", "submission_monotone_template.csv"),
    ("group", "group_expert_oof.csv", "submission_group_expert.csv"),
    ("shape400", "shape_local_oof_400.csv", "submission_shape_local_400.csv"),
)


def prediction_column(frame: pd.DataFrame) -> str:
    for col in ("PredLog", "pred_log", "PredLogCalibrated", "PredLogRaw"):
        if col in frame.columns:
            return col
    raise ValueError(f"no prediction-log column found in {list(frame.columns)}")


def load_predictions(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[List[str], np.ndarray, np.ndarray]:
    names: List[str] = []
    oof_logs: List[np.ndarray] = []
    test_logs: List[np.ndarray] = []
    for name, oof_file, sub_file in EXPERT_FILES:
        oof = pd.read_csv(ROOT / oof_file)
        sub = pd.read_csv(ROOT / sub_file)
        col = prediction_column(oof)
        oof_aligned = oof.set_index("Id").reindex(train["Id"].values)
        sub_aligned = sub.set_index("Id").reindex(test["Id"].values)
        if oof_aligned[col].isna().any() or sub_aligned["Cardinality"].isna().any():
            raise ValueError(f"{name} predictions are not aligned")
        names.append(name)
        oof_logs.append(oof_aligned[col].astype(float).values)
        test_logs.append(np.log1p(np.maximum(sub_aligned["Cardinality"].astype(float).values, 1.0)))
    return names, np.column_stack(oof_logs), np.column_stack(test_logs)


def parse_query_features(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in frame.to_dict("records"):
        parts = [] if pd.isna(row["Predicates"]) or str(row["Predicates"]).strip() == "" else [
            part.strip() for part in str(row["Predicates"]).split(",")
        ]
        values: List[float] = []
        ops: List[str] = []
        for i in range(0, len(parts) - 2, 3):
            ops.append(parts[i + 1])
            try:
                values.append(float(parts[i + 2]))
            except ValueError:
                values.append(0.0)
        rows.append(
            {
                "table": "" if pd.isna(row["Tables"]) else str(row["Tables"]),
                "npred": len(ops),
                "neq": sum(op == "=" for op in ops),
                "nlt": sum(op == "<" for op in ops),
                "ngt": sum(op == ">" for op in ops),
                "mean_val": float(np.mean(values)) if values else 0.0,
                "max_val": float(max(values)) if values else 0.0,
                "min_val": float(min(values)) if values else 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_meta_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    oof_logs: np.ndarray,
    test_logs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    train_query = parse_query_features(train)
    test_query = parse_query_features(test)
    numeric_cols = ["npred", "neq", "nlt", "ngt", "mean_val", "max_val", "min_val"]
    table_dummies = pd.get_dummies(
        pd.concat([train_query["table"], test_query["table"]], ignore_index=True),
        prefix="table",
    )

    train_parts = [
        oof_logs,
        oof_logs.max(axis=1, keepdims=True) - oof_logs.min(axis=1, keepdims=True),
        oof_logs.std(axis=1, keepdims=True),
        np.abs(oof_logs[:, 0:1] - oof_logs[:, 1:2]),
        np.abs(oof_logs[:, 1:2] - oof_logs[:, 2:3]),
        np.abs(oof_logs[:, 1:2] - oof_logs[:, 3:4]),
        train_query[numeric_cols].values.astype(float),
        table_dummies.iloc[: len(train)].values.astype(float),
    ]
    test_parts = [
        test_logs,
        test_logs.max(axis=1, keepdims=True) - test_logs.min(axis=1, keepdims=True),
        test_logs.std(axis=1, keepdims=True),
        np.abs(test_logs[:, 0:1] - test_logs[:, 1:2]),
        np.abs(test_logs[:, 1:2] - test_logs[:, 2:3]),
        np.abs(test_logs[:, 1:2] - test_logs[:, 3:4]),
        test_query[numeric_cols].values.astype(float),
        table_dummies.iloc[len(train) :].values.astype(float),
    ]
    return np.hstack(train_parts), np.hstack(test_parts)


def main() -> None:
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    y_log = np.log1p(np.maximum(train["Cardinality"].astype(float).values, 1.0))
    names, oof_logs, test_logs = load_predictions(train, test)
    base_idx = names.index("gate_aff")
    base = oof_logs[:, base_idx]
    base_test = test_logs[:, base_idx]
    residual = y_log - base
    X, X_test = build_meta_features(train, test, oof_logs, test_logs)

    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    pred = np.zeros(len(train), dtype=np.float64)
    test_preds: List[np.ndarray] = []
    for fold, (fit_idx, valid_idx) in enumerate(folds.split(X), start=1):
        model = make_pipeline(
            StandardScaler(),
            HuberRegressor(alpha=0.1, epsilon=1.2, max_iter=700, tol=1e-5),
        )
        model.fit(X[fit_idx], np.clip(residual[fit_idx], -1.0, 1.0))
        pred[valid_idx] = base[valid_idx] + 0.05 * model.predict(X[valid_idx])
        test_preds.append(base_test + 0.05 * model.predict(X_test))
        print(f"fold {fold} done", flush=True)

    pred_test = np.column_stack(test_preds).mean(axis=1)
    q = qerror_from_logs(pred, y_log)
    print(
        "meta_gate_huber_resid",
        f"mean={q.mean():.6f}",
        f"med={np.median(q):.4f}",
        f"p95={np.percentile(q, 95):.4f}",
        f"p99={np.percentile(q, 99):.4f}",
        flush=True,
    )

    pd.DataFrame(
        {
            "Id": train["Id"].values,
            "Cardinality": train["Cardinality"].values,
            "PredLog": pred,
            "QError": q,
        }
    ).to_csv(ROOT / "meta_gate_huber_resid_oof.csv", index=False)
    card = np.rint(np.maximum(np.expm1(pred_test), 1.0)).astype(np.int64)
    pd.DataFrame({"Id": test["Id"].values, "Cardinality": card}).to_csv(
        ROOT / "submission_meta_gate_huber_resid.csv",
        index=False,
    )
    print(
        "saved submission_meta_gate_huber_resid.csv",
        f"range=[{card.min()}, {card.max()}]",
        f"mean={card.mean():.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
