"""
HistGradientBoosting residual plus family-level expert gate.

This is a different algorithmic route from the recent hard expert gates:

1. Train a global sklearn HistGradientBoosting residual model on top of the
   current best family-fallback prediction.
2. Treat that hist residual prediction as one more expert.
3. For each `table + predicate columns` family, choose among a small expert set
   using out-of-fold training-side Q-error.

The standalone hist residual slightly improves tail percentiles but not mean
Q-error. Its value comes from being a different expert for family-level routing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder

from strong_model import qerror_from_logs


ROOT = Path(__file__).resolve().parent
SEED = 442
N_FOLDS = 5


BASE_EXPERTS: Tuple[Tuple[str, str, str], ...] = (
    ("base", "meta_gate_family_fallback_oof.csv", "submission_meta_gate_family_fallback.csv"),
    ("meta", "meta_gate_huber_resid_oof.csv", "submission_meta_gate_huber_resid.csv"),
    ("aff", "column_family_gate_shape50_cols80_affine_oof.csv", "submission_column_family_gate_shape50_cols80_affine.csv"),
    ("latent", "latent_final_blend_oof.csv", "submission_latent_final_blend.csv"),
    ("lowrank", "lowrank_latent_oof.csv", "submission_lowrank_latent.csv"),
    ("grid", "grid_surface_oof.csv", "submission_grid_surface.csv"),
    ("rewrite", "oof_rewrite_angle_affine_oof.csv", "submission_rewrite_angle_affine.csv"),
)


def prediction_column(frame: pd.DataFrame) -> str:
    for col in ("PredLog", "pred_log", "PredLogCalibrated", "PredLogRaw"):
        if col in frame.columns:
            return col
    raise ValueError(f"no prediction column found in {list(frame.columns)}")


def load_experts(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[List[str], np.ndarray, np.ndarray]:
    names: List[str] = []
    oof_logs: List[np.ndarray] = []
    test_logs: List[np.ndarray] = []
    for name, oof_file, sub_file in BASE_EXPERTS:
        oof = pd.read_csv(ROOT / oof_file).set_index("Id").reindex(train["Id"].values)
        sub = pd.read_csv(ROOT / sub_file).set_index("Id").reindex(test["Id"].values)
        col = prediction_column(oof)
        if oof[col].isna().any() or sub["Cardinality"].isna().any():
            raise ValueError(f"{name} predictions are not aligned")
        names.append(name)
        oof_logs.append(oof[col].astype(float).values)
        test_logs.append(np.log1p(np.maximum(sub["Cardinality"].astype(float).values, 1.0)))
    return names, np.column_stack(oof_logs), np.column_stack(test_logs)


def parse_features(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in frame.to_dict("records"):
        parts = [] if pd.isna(row["Predicates"]) or str(row["Predicates"]).strip() == "" else [
            part.strip() for part in str(row["Predicates"]).split(",")
        ]
        cols: List[str] = []
        ops: List[str] = []
        vals: List[float] = []
        for i in range(0, len(parts) - 2, 3):
            cols.append(parts[i])
            ops.append(parts[i + 1])
            try:
                vals.append(float(parts[i + 2]))
            except ValueError:
                vals.append(0.0)
        table = "" if pd.isna(row["Tables"]) else str(row["Tables"])
        join = "" if pd.isna(row["Join Conditions"]) else str(row["Join Conditions"])
        rows.append(
            {
                "table": table,
                "cols": f"{table}||{'|'.join(cols)}",
                "shape": f"{table}||{join}||{'|'.join(col + op for col, op in zip(cols, ops))}",
                "npred": len(cols),
                "neq": sum(op == "=" for op in ops),
                "nlt": sum(op == "<" for op in ops),
                "ngt": sum(op == ">" for op in ops),
                "mean_val": float(np.mean(vals)) if vals else 0.0,
                "min_val": float(min(vals)) if vals else 0.0,
                "max_val": float(max(vals)) if vals else 0.0,
                "sum_log_val": float(sum(np.log1p(max(val, 0.0)) for val in vals)),
            }
        )
    return pd.DataFrame(rows)


def build_hist_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    oof_logs: np.ndarray,
    test_logs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    train_features = parse_features(train)
    test_features = parse_features(test)
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    train_cats = enc.fit_transform(train_features[["table", "cols", "shape"]])
    test_cats = enc.transform(test_features[["table", "cols", "shape"]])
    numeric_cols = ["npred", "neq", "nlt", "ngt", "mean_val", "min_val", "max_val", "sum_log_val"]

    x_train = np.column_stack(
        [
            oof_logs,
            oof_logs.std(axis=1),
            oof_logs.max(axis=1) - oof_logs.min(axis=1),
            np.abs(oof_logs[:, 0] - oof_logs[:, 3]),
            np.abs(oof_logs[:, 0] - oof_logs[:, 1]),
            train_features[numeric_cols].values,
            train_cats,
        ]
    )
    x_test = np.column_stack(
        [
            test_logs,
            test_logs.std(axis=1),
            test_logs.max(axis=1) - test_logs.min(axis=1),
            np.abs(test_logs[:, 0] - test_logs[:, 3]),
            np.abs(test_logs[:, 0] - test_logs[:, 1]),
            test_features[numeric_cols].values,
            test_cats,
        ]
    )
    return x_train, x_test, train_features, test_features


def fit_hist_residual(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_log: np.ndarray,
    base_log: np.ndarray,
    base_test_log: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    residual = np.clip(y_log - base_log, -1.5, 1.5)
    raw = np.zeros(len(x_train), dtype=np.float64)
    raw_test: List[np.ndarray] = []
    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (fit_idx, valid_idx) in enumerate(folds.split(x_train), start=1):
        model = HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.05,
            max_iter=180,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            min_samples_leaf=45,
            random_state=100 + fold,
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=15,
        )
        model.fit(x_train[fit_idx], residual[fit_idx])
        raw[valid_idx] = model.predict(x_train[valid_idx])
        raw_test.append(model.predict(x_test))
        print(f"hist fold {fold} done", flush=True)

    shrink = 0.02
    return base_log + shrink * raw, base_test_log + shrink * np.column_stack(raw_test).mean(axis=1)


def group_indices(keys: np.ndarray, indices: np.ndarray) -> Dict[str, np.ndarray]:
    groups: Dict[str, List[int]] = {}
    for idx in indices:
        groups.setdefault(str(keys[idx]), []).append(int(idx))
    return {key: np.asarray(values, dtype=np.int32) for key, values in groups.items()}


def family_gate(
    names: Sequence[str],
    expert_logs: np.ndarray,
    expert_test_logs: np.ndarray,
    y_log: np.ndarray,
    train_cols: np.ndarray,
    test_cols: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    allowed_names = ("base", "hist", "aff", "latent")
    allowed = np.asarray([names.index(name) for name in allowed_names], dtype=np.int32)
    default = names.index("base")
    min_group = 120
    margin = 0.02

    pred = np.zeros(len(expert_logs), dtype=np.float64)
    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=505)
    for fold, (fit_idx, valid_idx) in enumerate(folds.split(expert_logs), start=1):
        fold_pred = expert_logs[valid_idx, default].copy()
        fit_groups = group_indices(train_cols, fit_idx)
        valid_groups = group_indices(train_cols, valid_idx)
        for key, valid_group_idx in valid_groups.items():
            fit_group_idx = fit_groups.get(key)
            if fit_group_idx is None or len(fit_group_idx) < min_group:
                continue
            scores = {
                idx: float(np.mean(qerror_from_logs(expert_logs[fit_group_idx, idx], y_log[fit_group_idx])))
                for idx in allowed
            }
            best = min(scores, key=scores.get)
            if scores[best] + margin < scores[default]:
                fold_pred[np.isin(valid_idx, valid_group_idx)] = expert_logs[valid_group_idx, best]
        pred[valid_idx] = fold_pred
        print(f"gate fold {fold} done", flush=True)

    pred_test = expert_test_logs[:, default].copy()
    test_choice = np.array([names[default]] * len(test_cols), dtype=object)
    train_groups = group_indices(train_cols, np.arange(len(train_cols), dtype=np.int32))
    test_groups = group_indices(test_cols, np.arange(len(test_cols), dtype=np.int32))
    for key, test_idx in test_groups.items():
        train_idx = train_groups.get(key)
        if train_idx is None or len(train_idx) < min_group:
            continue
        scores = {
            idx: float(np.mean(qerror_from_logs(expert_logs[train_idx, idx], y_log[train_idx])))
            for idx in allowed
        }
        best = min(scores, key=scores.get)
        if scores[best] + margin < scores[default]:
            pred_test[test_idx] = expert_test_logs[test_idx, best]
            test_choice[test_idx] = names[best]
    return pred, pred_test, pd.Series(test_choice).value_counts().to_dict()


def main() -> None:
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    y_log = np.log1p(np.maximum(train["Cardinality"].astype(float).values, 1.0))
    names, oof_logs, test_logs = load_experts(train, test)
    x_train, x_test, train_features, test_features = build_hist_features(train, test, oof_logs, test_logs)
    hist_oof, hist_test = fit_hist_residual(x_train, x_test, y_log, oof_logs[:, 0], test_logs[:, 0])

    hist_q = qerror_from_logs(hist_oof, y_log)
    print(
        "hist_residual",
        f"mean={hist_q.mean():.6f}",
        f"med={np.median(hist_q):.4f}",
        f"p95={np.percentile(hist_q, 95):.4f}",
        flush=True,
    )
    pd.DataFrame(
        {
            "Id": train["Id"].values,
            "Cardinality": train["Cardinality"].values,
            "PredLog": hist_oof,
            "QError": hist_q,
        }
    ).to_csv(ROOT / "hist_residual_oof.csv", index=False)
    hist_card = np.rint(np.maximum(np.expm1(hist_test), 1.0)).astype(np.int64)
    pd.DataFrame({"Id": test["Id"].values, "Cardinality": hist_card}).to_csv(
        ROOT / "submission_hist_residual.csv",
        index=False,
    )

    names = [*names, "hist"]
    expert_logs = np.column_stack([oof_logs, hist_oof])
    expert_test_logs = np.column_stack([test_logs, hist_test])
    pred, pred_test, choices = family_gate(
        names,
        expert_logs,
        expert_test_logs,
        y_log,
        train_features["cols"].values.astype(object),
        test_features["cols"].values.astype(object),
    )
    q = qerror_from_logs(pred, y_log)
    print(
        "hist_family_gate",
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
    ).to_csv(ROOT / "hist_family_gate_oof.csv", index=False)
    card = np.rint(np.maximum(np.expm1(pred_test), 1.0)).astype(np.int64)
    pd.DataFrame({"Id": test["Id"].values, "Cardinality": card}).to_csv(
        ROOT / "submission_hist_family_gate.csv",
        index=False,
    )
    print(
        "saved submission_hist_family_gate.csv",
        f"range=[{card.min()}, {card.max()}]",
        f"mean={card.mean():.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
