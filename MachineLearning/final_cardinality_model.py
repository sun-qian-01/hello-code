"""
Final competition pipeline for SQL cardinality estimation.

Run:
    python final_cardinality_model.py

Outputs:
    submission.csv              main submission
    submission_no_smallblend.csv conservative regression-only fallback
    final_oof_predictions.csv   local OOF diagnostics

The model intentionally avoids leakage-heavy target encodings. It uses parsed
SQL features, four LightGBM regressors, four small-cardinality classifiers, and
a leakage-free OOF calibration that optimizes mean Q-Error directly.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.model_selection import KFold

from strong_model import (
    TABLES,
    add_frequency_and_code_features,
    make_feature_frames,
    qerror_from_logs,
)


ROOT = Path(__file__).resolve().parent
N_FOLDS = 5
SEED = 42
SMALL_THRESHOLDS = np.array([1.0, 5.0, 20.0, 100.0])


def load_features() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    stats = pd.read_csv(ROOT / "column_min_max_vals.csv")

    col_info = {
        row["name"]: (
            float(row["min"]),
            float(row["max"]),
            float(row["cardinality"]),
            float(row["num_unique_values"]),
        )
        for _, row in stats.iterrows()
    }
    table_rows = {table: col_info[f"{table}.id"][2] for table in TABLES}

    X_train, cat_train = make_feature_frames(train, col_info, list(col_info.keys()), table_rows)
    X_test, cat_test = make_feature_frames(test, col_info, list(col_info.keys()), table_rows)
    X_train, X_test, _ = add_frequency_and_code_features(X_train, X_test, cat_train, cat_test)
    y = train["Cardinality"].astype(float).values
    return train, test, X_train, X_test, y


def train_regressors(
    X: pd.DataFrame, X_test: pd.DataFrame, y_card: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    y_log = np.log1p(y_card)
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    base_params: Dict[str, object] = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.03,
        "num_leaves": 255,
        "max_depth": 12,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 30,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "num_threads": -1,
    }

    small_weight = np.where(y_card <= 100, 5.0, np.where(y_card <= 1000, 2.0, 1.0))
    small_weight /= small_weight.mean()
    invlog_weight = 1.0 / np.maximum(np.log1p(y_card), 1.0)
    invlog_weight /= invlog_weight.mean()

    configs: List[Tuple[str, Dict[str, object], np.ndarray | None]] = [
        ("base", base_params, None),
        (
            "wide",
            {
                **base_params,
                "learning_rate": 0.025,
                "num_leaves": 383,
                "max_depth": 13,
                "feature_fraction": 0.75,
                "bagging_fraction": 0.85,
                "min_data_in_leaf": 35,
                "lambda_l2": 2.0,
            },
            None,
        ),
        ("small_weight", {**base_params, "learning_rate": 0.028}, small_weight),
        ("invlog_weight", {**base_params, "learning_rate": 0.028}, invlog_weight),
    ]

    oof_columns: List[np.ndarray] = []
    test_columns: List[np.ndarray] = []
    names: List[str] = []

    for model_id, (name, params, weights) in enumerate(configs):
        print(f"regressor: {name}", flush=True)
        oof = np.zeros(len(X), dtype=np.float32)
        test_fold_preds: List[np.ndarray] = []
        iters: List[int] = []
        for fold, (fit_idx, valid_idx) in enumerate(kfold.split(X), start=1):
            fold_params = dict(params)
            fold_params["seed"] = 7000 + model_id * 100 + fold
            dtrain = lgb.Dataset(
                X.iloc[fit_idx],
                label=y_log[fit_idx],
                weight=None if weights is None else weights[fit_idx],
            )
            dvalid = lgb.Dataset(X.iloc[valid_idx], label=y_log[valid_idx])
            model = lgb.train(
                fold_params,
                dtrain,
                num_boost_round=2200,
                valid_sets=[dvalid],
                callbacks=[
                    lgb.early_stopping(120, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            oof[valid_idx] = model.predict(X.iloc[valid_idx], num_iteration=model.best_iteration)
            test_fold_preds.append(model.predict(X_test, num_iteration=model.best_iteration))
            iters.append(int(model.best_iteration))
        print(
            f"  iters={iters}, OOF mean Q={np.mean(qerror_from_logs(oof, y_log)):.5f}",
            flush=True,
        )
        names.append(name)
        oof_columns.append(oof)
        test_columns.append(np.column_stack(test_fold_preds).mean(axis=1).astype(np.float32))

    return np.column_stack(oof_columns), np.column_stack(test_columns), names


def train_small_classifiers(
    X: pd.DataFrame, X_test: pd.DataFrame, y_card: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_columns: List[np.ndarray] = []
    test_columns: List[np.ndarray] = []

    for thr in SMALL_THRESHOLDS:
        label = (y_card <= thr).astype(np.int32)
        pos = int(label.sum())
        neg = len(label) - pos
        params: Dict[str, object] = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.035,
            "num_leaves": 127,
            "max_depth": 10,
            "feature_fraction": 0.82,
            "bagging_fraction": 0.82,
            "bagging_freq": 1,
            "min_data_in_leaf": 35,
            "lambda_l1": 0.05,
            "lambda_l2": 1.5,
            "scale_pos_weight": min(12.0, max(1.0, neg / max(pos, 1))),
            "verbose": -1,
            "num_threads": -1,
        }

        print(f"small-cardinality classifier <= {int(thr)}", flush=True)
        oof = np.zeros(len(X), dtype=np.float32)
        test_fold_preds: List[np.ndarray] = []
        iters: List[int] = []
        for fold, (fit_idx, valid_idx) in enumerate(kfold.split(X), start=1):
            fold_params = dict(params)
            fold_params["seed"] = 5000 + int(thr) * 10 + fold
            dtrain = lgb.Dataset(X.iloc[fit_idx], label=label[fit_idx])
            dvalid = lgb.Dataset(X.iloc[valid_idx], label=label[valid_idx])
            model = lgb.train(
                fold_params,
                dtrain,
                num_boost_round=1800,
                valid_sets=[dvalid],
                callbacks=[
                    lgb.early_stopping(120, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            oof[valid_idx] = model.predict(X.iloc[valid_idx], num_iteration=model.best_iteration)
            test_fold_preds.append(model.predict(X_test, num_iteration=model.best_iteration))
            iters.append(int(model.best_iteration))
        print(f"  positive_rate={label.mean():.4f}, iters={iters}", flush=True)
        oof_columns.append(oof)
        test_columns.append(np.column_stack(test_fold_preds).mean(axis=1).astype(np.float32))

    return np.column_stack(oof_columns), np.column_stack(test_columns)


def optimize_blend(
    reg_oof: np.ndarray, small_oof: np.ndarray, y_log: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float, float, np.ndarray, float]:
    def transform(z: np.ndarray, reg_matrix: np.ndarray, small_matrix: np.ndarray) -> np.ndarray:
        raw_weights = np.exp(z[: reg_matrix.shape[1]])
        weights = raw_weights / raw_weights.sum()
        beta = np.tanh(z[reg_matrix.shape[1]]) * 1.0
        alpha = 0.85 + 0.30 / (1.0 + np.exp(-z[reg_matrix.shape[1] + 1]))
        small_coeff = 1.0 / (1.0 + np.exp(-z[reg_matrix.shape[1] + 2 :]))

        pred = alpha * (reg_matrix @ weights) + beta
        for j, coeff in enumerate(small_coeff):
            target = np.log1p(SMALL_THRESHOLDS[j])
            shrink = small_matrix[:, j] * coeff
            pred = (1.0 - shrink) * pred + shrink * target
        return pred

    def objective(z: np.ndarray) -> float:
        pred = transform(z, reg_oof, small_oof)
        return float(np.mean(qerror_from_logs(pred, y_log)))

    bounds = [(-4, 4)] * reg_oof.shape[1] + [(-4, 4), (-4, 4)] + [(-8, 8)] * len(SMALL_THRESHOLDS)
    result = differential_evolution(
        objective,
        bounds,
        seed=SEED,
        maxiter=250,
        tol=1e-7,
        polish=True,
        workers=1,
        updating="immediate",
    )

    z = result.x
    raw_weights = np.exp(z[: reg_oof.shape[1]])
    weights = raw_weights / raw_weights.sum()
    beta = float(np.tanh(z[reg_oof.shape[1]]) * 1.0)
    alpha = float(0.85 + 0.30 / (1.0 + np.exp(-z[reg_oof.shape[1] + 1])))
    small_coeff = 1.0 / (1.0 + np.exp(-z[reg_oof.shape[1] + 2 :]))
    best_score = float(result.fun)
    return z, weights, alpha, beta, small_coeff, best_score


def apply_blend(
    z: np.ndarray, reg_matrix: np.ndarray, small_matrix: np.ndarray
) -> np.ndarray:
    raw_weights = np.exp(z[: reg_matrix.shape[1]])
    weights = raw_weights / raw_weights.sum()
    beta = np.tanh(z[reg_matrix.shape[1]]) * 1.0
    alpha = 0.85 + 0.30 / (1.0 + np.exp(-z[reg_matrix.shape[1] + 1]))
    small_coeff = 1.0 / (1.0 + np.exp(-z[reg_matrix.shape[1] + 2 :]))

    pred = alpha * (reg_matrix @ weights) + beta
    for j, coeff in enumerate(small_coeff):
        target = np.log1p(SMALL_THRESHOLDS[j])
        shrink = small_matrix[:, j] * coeff
        pred = (1.0 - shrink) * pred + shrink * target
    return pred


def main() -> None:
    start = time.time()
    print("building parsed features", flush=True)
    train, test, X, X_test, y_card = load_features()
    y_log = np.log1p(y_card)
    print(f"features: train={X.shape}, test={X_test.shape}", flush=True)

    reg_oof, reg_test, reg_names = train_regressors(X, X_test, y_card)
    small_oof, small_test = train_small_classifiers(X, X_test, y_card)

    print("optimizing OOF blend for mean Q-Error", flush=True)
    z, weights, alpha, beta, small_coeff, best_score = optimize_blend(reg_oof, small_oof, y_log)
    oof_pred = apply_blend(z, reg_oof, small_oof)
    oof_q = qerror_from_logs(oof_pred, y_log)

    no_small_z = z.copy()
    no_small_z[reg_oof.shape[1] + 2 :] = -20.0
    no_small_test_log = apply_blend(no_small_z, reg_test, small_test)
    final_test_log = apply_blend(z, reg_test, small_test)

    final_card = np.rint(np.maximum(np.expm1(final_test_log), 1.0)).astype(np.int64)
    fallback_card = np.rint(np.maximum(np.expm1(no_small_test_log), 1.0)).astype(np.int64)

    submission = pd.DataFrame({"Id": test["Id"].values, "Cardinality": final_card})
    submission.to_csv(ROOT / "submission.csv", index=False)
    pd.DataFrame({"Id": test["Id"].values, "Cardinality": fallback_card}).to_csv(
        ROOT / "submission_no_smallblend.csv", index=False
    )

    oof = pd.DataFrame(
        {
            "Id": train["Id"].values,
            "Cardinality": y_card,
            "PredLog": oof_pred,
            "PredCard": np.maximum(np.expm1(oof_pred), 1.0),
            "QError": oof_q,
        }
    )
    for idx, name in enumerate(reg_names):
        oof[f"reg_{name}"] = reg_oof[:, idx]
    for idx, thr in enumerate(SMALL_THRESHOLDS):
        oof[f"p_le_{int(thr)}"] = small_oof[:, idx]
    oof.to_csv(ROOT / "final_oof_predictions.csv", index=False)

    print("done", flush=True)
    print(f"OOF mean Q-Error: {best_score:.5f}", flush=True)
    print(
        "OOF percentiles:",
        np.percentile(oof_q, [50, 90, 95, 99, 99.5, 100]),
        flush=True,
    )
    print(f"regression weights: {dict(zip(reg_names, weights.round(4)))}", flush=True)
    print(f"alpha={alpha:.4f}, beta={beta:.4f}, small_coeff={small_coeff.round(4)}", flush=True)
    print(
        f"submission range=[{final_card.min():,}, {final_card.max():,}], "
        f"mean={final_card.mean():,.0f}",
        flush=True,
    )
    print(f"elapsed={time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
