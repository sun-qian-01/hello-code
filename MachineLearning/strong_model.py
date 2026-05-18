"""
Stronger SQL cardinality estimator.

This script is built for the competition layout in this folder:
train.csv, test.csv, column_min_max_vals.csv, sample_submission.csv.

Main differences from the earlier scripts:
  * uses log(Cardinality), which matches Q-Error directly;
  * adds query-template target encodings with leakage-safe folds;
  * adds uniform join/selectivity estimates and richer predicate features;
  * trains a small LightGBM ensemble with early stopping on mean Q-Error;
  * applies a global log-space calibration learned from OOF predictions.
"""

from __future__ import annotations

import math
import time
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.model_selection import KFold


warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "train.csv"
TEST_PATH = ROOT / "test.csv"
STATS_PATH = ROOT / "column_min_max_vals.csv"
OUT_PATH = ROOT / "submission.csv"
OOF_PATH = ROOT / "strong_oof_predictions.csv"

SEED = 2026
N_FOLDS = 5
EARLY_STOPPING_ROUNDS = 200
NUM_BOOST_ROUND = 5000

TABLES = ["t", "mc", "ci", "mi", "mi_idx", "mk"]
OP_CODE = {"=": 0, "<": 1, ">": 2}

CATEGORY_COLS = [
    "table_combo",
    "join_combo",
    "pred_sig",
    "pred_cols_sig",
    "pred_ops_sig",
    "colops_sorted_sig",
    "table_pred_sig",
    "table_cols_sig",
    "shape_sig",
    "table_pred_count_sig",
]


def qerror_from_logs(pred_log: np.ndarray, true_log: np.ndarray) -> np.ndarray:
    pred_card = np.maximum(np.expm1(pred_log), 1.0)
    true_card = np.maximum(np.expm1(true_log), 1.0)
    return np.maximum(pred_card / true_card, true_card / pred_card)


def lgb_qerror(preds: np.ndarray, dataset: lgb.Dataset) -> Tuple[str, float, bool]:
    labels = dataset.get_label()
    return "mean_qerror", float(np.mean(qerror_from_logs(preds, labels))), False


def optimal_offset(pred_log: np.ndarray, true_log: np.ndarray) -> Tuple[float, float]:
    """Find additive log1p offset minimizing mean Q-Error."""

    def objective(offset: float) -> float:
        return float(np.mean(qerror_from_logs(pred_log + offset, true_log)))

    result = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
    offset = float(result.x)
    score = float(result.fun)
    return offset, score


def optimal_affine_calibration(
    pred_log: np.ndarray, true_log: np.ndarray
) -> Tuple[float, float, float]:
    """Tune pred = alpha * pred + beta on OOF data for Q-Error."""
    best_alpha = 1.0
    best_beta, best_score = optimal_offset(pred_log, true_log)

    # Keep this conservative. Large alpha changes can overfit the public split.
    for alpha in np.linspace(0.92, 1.08, 65):
        beta, score = optimal_offset(alpha * pred_log, true_log)
        if score < best_score:
            best_alpha = float(alpha)
            best_beta = float(beta)
            best_score = float(score)
    return best_alpha, best_beta, best_score


def parse_tables(value: object) -> List[str]:
    if pd.isna(value) or str(value).strip() == "":
        return []
    aliases: List[str] = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        aliases.append(part.split()[-1])
    return aliases


def parse_joins(value: object) -> List[str]:
    if pd.isna(value) or str(value).strip() == "":
        return []
    joins: List[str] = []
    for part in str(value).split(","):
        part = part.strip().replace(" ", "")
        if part:
            joins.append(part)
    return joins


def parse_predicates(value: object) -> List[Tuple[str, str, float]]:
    if pd.isna(value) or str(value).strip() == "":
        return []
    parts = [p.strip() for p in str(value).split(",")]
    preds: List[Tuple[str, str, float]] = []
    for i in range(0, len(parts) - 2, 3):
        col = parts[i]
        op = parts[i + 1]
        try:
            val = float(parts[i + 2])
        except ValueError:
            continue
        preds.append((col, op, val))
    return preds


def safe_log1p(x: float) -> float:
    return float(np.log1p(max(float(x), 0.0)))


def predicate_selectivity(
    col: str, op: str, val: float, col_info: Dict[str, Tuple[float, float, float, float]]
) -> float:
    cmin, cmax, _, nunique = col_info.get(col, (0.0, 1.0, 1.0, 1.0))
    crange = max(cmax - cmin, 1.0)
    norm = (val - cmin) / crange
    if op == "=":
        sel = 1.0 / max(nunique, 1.0)
    elif op == "<":
        sel = norm
    elif op == ">":
        sel = 1.0 - norm
    else:
        sel = 1.0
    return float(np.clip(sel, 1e-7, 1.0))


def make_feature_frames(
    df: pd.DataFrame,
    col_info: Dict[str, Tuple[float, float, float, float]],
    all_cols: Sequence[str],
    table_rows: Dict[str, float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, float]] = []
    cats: List[Dict[str, str]] = []

    col_to_id = {col: i + 1 for i, col in enumerate(all_cols)}
    table_to_id = {table: i + 1 for i, table in enumerate(TABLES)}

    for row_dict in df.to_dict("records"):
        tables = parse_tables(row_dict.get("Tables", ""))
        joins = parse_joins(row_dict.get("Join Conditions", ""))
        preds = parse_predicates(row_dict.get("Predicates", ""))

        feat: Dict[str, float] = {}
        cat: Dict[str, str] = {}

        cols = [col for col, _, _ in preds]
        ops = [op for _, op, _ in preds]
        colops = [f"{col}{op}" for col, op, _ in preds]

        table_combo = "|".join(tables) if tables else "none"
        join_combo = "|".join(sorted(joins)) if joins else "none"
        pred_sig = "|".join(colops) if colops else "none"
        pred_cols_sig = "|".join(cols) if cols else "none"
        pred_ops_sig = "|".join(ops) if ops else "none"
        colops_sorted_sig = "|".join(sorted(colops)) if colops else "none"

        cat["table_combo"] = table_combo
        cat["join_combo"] = join_combo
        cat["pred_sig"] = pred_sig
        cat["pred_cols_sig"] = pred_cols_sig
        cat["pred_ops_sig"] = pred_ops_sig
        cat["colops_sorted_sig"] = colops_sorted_sig
        cat["table_pred_sig"] = f"{table_combo}||{pred_sig}"
        cat["table_cols_sig"] = f"{table_combo}||{pred_cols_sig}"
        cat["shape_sig"] = f"{table_combo}||{join_combo}||{pred_sig}"
        cat["table_pred_count_sig"] = f"{table_combo}||{len(preds)}"

        feat["num_tables"] = float(len(tables))
        feat["num_joins"] = float(len(joins))
        feat["num_predicates"] = float(len(preds))
        feat["num_eq"] = float(sum(op == "=" for op in ops))
        feat["num_lt"] = float(sum(op == "<" for op in ops))
        feat["num_gt"] = float(sum(op == ">" for op in ops))
        feat["num_range"] = feat["num_lt"] + feat["num_gt"]
        feat["has_equality"] = float(feat["num_eq"] > 0)
        feat["has_range"] = float(feat["num_range"] > 0)
        feat["preds_per_table"] = feat["num_predicates"] / max(feat["num_tables"], 1.0)
        feat["tables_x_preds"] = feat["num_tables"] * feat["num_predicates"]
        feat["eq_x_range"] = feat["num_eq"] * feat["num_range"]
        feat["lt_x_gt"] = feat["num_lt"] * feat["num_gt"]
        feat["tables_string_len"] = float(len(str(row_dict.get("Tables", ""))))
        feat["joins_string_len"] = float(len(str(row_dict.get("Join Conditions", ""))))
        feat["predicates_string_len"] = float(len(str(row_dict.get("Predicates", ""))))

        table_mask = 0
        table_sel = {table: 1.0 for table in TABLES}
        table_pred_count = {table: 0 for table in TABLES}
        table_eq_count = {table: 0 for table in TABLES}
        table_lt_count = {table: 0 for table in TABLES}
        table_gt_count = {table: 0 for table in TABLES}
        pred_sels: List[float] = []

        pred_by_col: Dict[str, Tuple[str, float, float]] = {}
        for col, op, val in preds:
            table = col.split(".")[0] if "." in col else ""
            sel = predicate_selectivity(col, op, val, col_info)
            pred_sels.append(sel)
            pred_by_col[col] = (op, val, sel)
            if table in table_sel:
                table_sel[table] *= sel
                table_pred_count[table] += 1
                table_eq_count[table] += int(op == "=")
                table_lt_count[table] += int(op == "<")
                table_gt_count[table] += int(op == ">")

        for table in TABLES:
            if table in tables:
                table_mask |= 1 << TABLES.index(table)
            has_table = float(table in tables)
            rows_count = table_rows.get(table, 1.0)
            filtered_rows = rows_count * table_sel[table] if has_table else 0.0
            feat[f"has_table_{table}"] = has_table
            feat[f"table_{table}_rows_log"] = safe_log1p(rows_count) * has_table
            feat[f"table_{table}_pred_count"] = float(table_pred_count[table])
            feat[f"table_{table}_eq_count"] = float(table_eq_count[table])
            feat[f"table_{table}_lt_count"] = float(table_lt_count[table])
            feat[f"table_{table}_gt_count"] = float(table_gt_count[table])
            feat[f"table_{table}_sel"] = table_sel[table] if has_table else 0.0
            feat[f"table_{table}_sel_log"] = math.log(max(table_sel[table], 1e-12)) if has_table else 0.0
            feat[f"table_{table}_filtered_log"] = safe_log1p(filtered_rows)
            feat[f"table_{table}_has_predicate"] = float(table_pred_count[table] > 0)

        feat["table_mask"] = float(table_mask)

        query_tables = [table for table in tables if table in table_rows]
        filtered_by_table = {
            table: table_rows[table] * table_sel[table] for table in query_tables
        }
        if query_tables:
            filtered_values = list(filtered_by_table.values())
            sels = [table_sel[table] for table in query_tables]
            feat["max_filtered_table_log"] = safe_log1p(max(filtered_values))
            feat["min_filtered_table_log"] = safe_log1p(min(filtered_values))
            feat["sum_filtered_table_log"] = safe_log1p(sum(filtered_values))
            feat["product_filtered_tables_log"] = float(
                sum(math.log(max(v, 1e-12)) for v in filtered_values)
            )
            feat["selectivity_product_log"] = float(
                sum(math.log(max(sel, 1e-12)) for sel in sels)
            )
            feat["selectivity_min"] = float(min(sels))
            feat["selectivity_max"] = float(max(sels))
            feat["selectivity_mean"] = float(np.mean(sels))
        else:
            feat["max_filtered_table_log"] = 0.0
            feat["min_filtered_table_log"] = 0.0
            feat["sum_filtered_table_log"] = 0.0
            feat["product_filtered_tables_log"] = 0.0
            feat["selectivity_product_log"] = 0.0
            feat["selectivity_min"] = 0.0
            feat["selectivity_max"] = 0.0
            feat["selectivity_mean"] = 0.0

        if len(query_tables) == 1:
            join_est = filtered_by_table[query_tables[0]]
            join_unfiltered = table_rows[query_tables[0]]
        elif "t" in query_tables:
            title_rows = table_rows["t"]
            join_est = table_rows["t"] * table_sel["t"]
            join_unfiltered = table_rows["t"]
            for table in query_tables:
                if table == "t":
                    continue
                join_est *= (table_rows[table] * table_sel[table]) / max(title_rows, 1.0)
                join_unfiltered *= table_rows[table] / max(title_rows, 1.0)
        elif query_tables:
            denom = max(max(table_rows[t] for t in query_tables), 1.0) ** max(len(query_tables) - 1, 0)
            join_est = float(np.prod([filtered_by_table[t] for t in query_tables])) / denom
            join_unfiltered = float(np.prod([table_rows[t] for t in query_tables])) / denom
        else:
            join_est = 1.0
            join_unfiltered = 1.0

        feat["uniform_join_est_log"] = safe_log1p(join_est)
        feat["uniform_join_unfiltered_log"] = safe_log1p(join_unfiltered)
        feat["uniform_join_selectivity_log"] = math.log(max(join_est / max(join_unfiltered, 1.0), 1e-12))

        if pred_sels:
            feat["predicate_sel_min"] = float(min(pred_sels))
            feat["predicate_sel_max"] = float(max(pred_sels))
            feat["predicate_sel_mean"] = float(np.mean(pred_sels))
            feat["predicate_sel_product_log"] = float(
                sum(math.log(max(sel, 1e-12)) for sel in pred_sels)
            )
        else:
            feat["predicate_sel_min"] = 1.0
            feat["predicate_sel_max"] = 1.0
            feat["predicate_sel_mean"] = 1.0
            feat["predicate_sel_product_log"] = 0.0

        for i in range(6):
            if i < len(preds):
                col, op, val = preds[i]
                table = col.split(".")[0] if "." in col else ""
                cmin, cmax, ccard, nunique = col_info.get(col, (0.0, 1.0, 1.0, 1.0))
                crange = max(cmax - cmin, 1.0)
                norm = (val - cmin) / crange
                sel = predicate_selectivity(col, op, val, col_info)
                feat[f"slot_{i}_present"] = 1.0
                feat[f"slot_{i}_col_id"] = float(col_to_id.get(col, 0))
                feat[f"slot_{i}_table_id"] = float(table_to_id.get(table, 0))
                feat[f"slot_{i}_op_code"] = float(OP_CODE.get(op, -1))
                feat[f"slot_{i}_val"] = float(val)
                feat[f"slot_{i}_val_log"] = safe_log1p(val)
                feat[f"slot_{i}_val_norm"] = float(norm)
                feat[f"slot_{i}_val_norm_clipped"] = float(np.clip(norm, 0.0, 1.0))
                feat[f"slot_{i}_selectivity"] = sel
                feat[f"slot_{i}_selectivity_log"] = math.log(max(sel, 1e-12))
                feat[f"slot_{i}_num_unique_log"] = safe_log1p(nunique)
                feat[f"slot_{i}_col_card_log"] = safe_log1p(ccard)
            else:
                feat[f"slot_{i}_present"] = 0.0
                feat[f"slot_{i}_col_id"] = 0.0
                feat[f"slot_{i}_table_id"] = 0.0
                feat[f"slot_{i}_op_code"] = -1.0
                feat[f"slot_{i}_val"] = -1.0
                feat[f"slot_{i}_val_log"] = 0.0
                feat[f"slot_{i}_val_norm"] = -1.0
                feat[f"slot_{i}_val_norm_clipped"] = -1.0
                feat[f"slot_{i}_selectivity"] = 1.0
                feat[f"slot_{i}_selectivity_log"] = 0.0
                feat[f"slot_{i}_num_unique_log"] = 0.0
                feat[f"slot_{i}_col_card_log"] = 0.0

        for col in all_cols:
            cmin, cmax, ccard, nunique = col_info[col]
            crange = max(cmax - cmin, 1.0)
            prefix = col.replace(".", "_")
            used = col in pred_by_col
            feat[f"{prefix}_used"] = float(used)
            feat[f"{prefix}_card_log_if_used"] = safe_log1p(ccard) if used else 0.0
            feat[f"{prefix}_unique_log_if_used"] = safe_log1p(nunique) if used else 0.0
            feat[f"{prefix}_op_code"] = -1.0
            feat[f"{prefix}_value"] = -1.0
            feat[f"{prefix}_value_log"] = 0.0
            feat[f"{prefix}_value_norm"] = -1.0
            feat[f"{prefix}_value_norm_clip"] = -1.0
            feat[f"{prefix}_selectivity"] = 1.0
            feat[f"{prefix}_selectivity_log"] = 0.0
            feat[f"{prefix}_has_eq"] = 0.0
            feat[f"{prefix}_has_lt"] = 0.0
            feat[f"{prefix}_has_gt"] = 0.0
            feat[f"{prefix}_eq_value_norm"] = -1.0
            feat[f"{prefix}_lt_value_norm"] = -1.0
            feat[f"{prefix}_gt_value_norm"] = -1.0
            if used:
                op, val, sel = pred_by_col[col]
                norm = (val - cmin) / crange
                feat[f"{prefix}_op_code"] = float(OP_CODE.get(op, -1))
                feat[f"{prefix}_value"] = float(val)
                feat[f"{prefix}_value_log"] = safe_log1p(val)
                feat[f"{prefix}_value_norm"] = float(norm)
                feat[f"{prefix}_value_norm_clip"] = float(np.clip(norm, 0.0, 1.0))
                feat[f"{prefix}_selectivity"] = sel
                feat[f"{prefix}_selectivity_log"] = math.log(max(sel, 1e-12))
                feat[f"{prefix}_has_eq"] = float(op == "=")
                feat[f"{prefix}_has_lt"] = float(op == "<")
                feat[f"{prefix}_has_gt"] = float(op == ">")
                if op == "=":
                    feat[f"{prefix}_eq_value_norm"] = float(norm)
                elif op == "<":
                    feat[f"{prefix}_lt_value_norm"] = float(norm)
                elif op == ">":
                    feat[f"{prefix}_gt_value_norm"] = float(norm)

        for i, t1 in enumerate(TABLES):
            for t2 in TABLES[i + 1 :]:
                feat[f"both_{t1}_{t2}"] = feat[f"has_table_{t1}"] * feat[f"has_table_{t2}"]
                feat[f"pred_count_{t1}_{t2}"] = (
                    feat[f"table_{t1}_pred_count"] + feat[f"table_{t2}_pred_count"]
                ) * feat[f"both_{t1}_{t2}"]

        rows.append(feat)
        cats.append(cat)

    base = pd.DataFrame(rows).astype(np.float32)
    cat_frame = pd.DataFrame(cats).astype(str)
    return base, cat_frame


def add_frequency_and_code_features(
    base_train: pd.DataFrame,
    base_test: pd.DataFrame,
    cat_train: pd.DataFrame,
    cat_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    base_train = base_train.copy()
    base_test = base_test.copy()
    categorical_features: List[str] = []

    for col in CATEGORY_COLS:
        combined = pd.concat([cat_train[col], cat_test[col]], axis=0, ignore_index=True)
        codes, _ = pd.factorize(combined, sort=True)
        train_codes = codes[: len(cat_train)] + 1
        test_codes = codes[len(cat_train) :] + 1
        code_name = f"code_{col}"
        base_train[code_name] = train_codes.astype(np.int32)
        base_test[code_name] = test_codes.astype(np.int32)
        categorical_features.append(code_name)

        all_counts = combined.value_counts()
        train_counts = cat_train[col].value_counts()
        test_counts = cat_test[col].value_counts()

        base_train[f"freq_all_{col}_log"] = np.log1p(cat_train[col].map(all_counts).fillna(0)).astype(np.float32)
        base_test[f"freq_all_{col}_log"] = np.log1p(cat_test[col].map(all_counts).fillna(0)).astype(np.float32)
        base_train[f"freq_train_{col}_log"] = np.log1p(cat_train[col].map(train_counts).fillna(0)).astype(np.float32)
        base_test[f"freq_train_{col}_log"] = np.log1p(cat_test[col].map(train_counts).fillna(0)).astype(np.float32)
        base_train[f"freq_test_{col}_log"] = np.log1p(cat_train[col].map(test_counts).fillna(0)).astype(np.float32)
        base_test[f"freq_test_{col}_log"] = np.log1p(cat_test[col].map(test_counts).fillna(0)).astype(np.float32)

    return base_train, base_test, categorical_features


def add_target_encoding_features(
    base_part: pd.DataFrame,
    cat_part: pd.DataFrame,
    fit_cats: pd.DataFrame,
    fit_y_log: np.ndarray,
    *,
    part_y_log: np.ndarray | None = None,
    use_leave_one_out: bool = False,
    smooth_values: Sequence[float] = (5.0, 30.0),
) -> pd.DataFrame:
    result = base_part.copy()
    prior = float(np.mean(fit_y_log))
    prior_std = float(np.std(fit_y_log))

    for col in CATEGORY_COLS:
        tmp = pd.DataFrame({"cat": fit_cats[col].values, "y": fit_y_log})
        grouped = tmp.groupby("cat")["y"]
        count_map = grouped.count()
        sum_map = grouped.sum()
        mean_map = grouped.mean()
        median_map = grouped.median()
        std_map = grouped.std().fillna(prior_std)

        part_values = cat_part[col]
        counts = part_values.map(count_map).fillna(0).astype(np.float32).values
        sums = part_values.map(sum_map).fillna(0.0).astype(np.float32).values

        if use_leave_one_out:
            if part_y_log is None:
                raise ValueError("part_y_log is required for leave-one-out target encoding")
            loo_counts = np.maximum(counts - 1.0, 0.0)
            loo_sums = sums - part_y_log.astype(np.float32)
            for smooth in smooth_values:
                enc = (loo_sums + prior * smooth) / np.maximum(loo_counts + smooth, 1e-6)
                enc = np.where(loo_counts > 0, enc, prior)
                result[f"te_{col}_mean_s{int(smooth)}"] = enc.astype(np.float32)
        else:
            for smooth in smooth_values:
                enc = (sums + prior * smooth) / np.maximum(counts + smooth, 1e-6)
                result[f"te_{col}_mean_s{int(smooth)}"] = enc.astype(np.float32)

        result[f"te_{col}_count_log"] = np.log1p(counts).astype(np.float32)
        result[f"te_{col}_mean_raw"] = part_values.map(mean_map).fillna(prior).astype(np.float32).values
        result[f"te_{col}_median"] = part_values.map(median_map).fillna(prior).astype(np.float32).values
        result[f"te_{col}_std"] = part_values.map(std_map).fillna(prior_std).astype(np.float32).values

    return result


def align_columns(*frames: pd.DataFrame) -> List[pd.DataFrame]:
    cols = frames[0].columns
    for frame in frames[1:]:
        cols = cols.intersection(frame.columns)
    return [frame.loc[:, cols].copy() for frame in frames]


def train_one_model(
    params: Dict[str, object],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    categorical_features: Sequence[str],
    fold: int,
    model_name: str,
) -> lgb.Booster:
    # The code_* features are high-cardinality query-template ids. Treating
    # them as numeric IDs plus smoothed target encodings was more stable than
    # native categorical splits in local validation.
    dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain, free_raw_data=False)
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dvalid],
        valid_names=["valid"],
        feval=lgb_qerror,
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, first_metric_only=True, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    valid_pred = model.predict(X_valid, num_iteration=model.best_iteration)
    score = float(np.mean(qerror_from_logs(valid_pred, y_valid)))
    print(
        f"    fold {fold} {model_name}: iter={model.best_iteration:4d}, "
        f"valid_q={score:.5f}"
    )
    return model


def main() -> None:
    start = time.time()
    print("=" * 72)
    print("Load data")
    print("=" * 72)
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    stats_df = pd.read_csv(STATS_PATH)

    col_info = {
        row["name"]: (
            float(row["min"]),
            float(row["max"]),
            float(row["cardinality"]),
            float(row["num_unique_values"]),
        )
        for _, row in stats_df.iterrows()
    }
    all_cols = list(col_info.keys())
    table_rows = {table: col_info[f"{table}.id"][2] for table in TABLES}

    y = train_df["Cardinality"].astype(float).values
    y_log = np.log1p(np.maximum(y, 1.0)).astype(np.float32)
    print(f"train={len(train_df):,}, test={len(test_df):,}, columns={len(all_cols)}")

    print("\n" + "=" * 72)
    print("Build base features")
    print("=" * 72)
    base_train, cat_train = make_feature_frames(train_df, col_info, all_cols, table_rows)
    base_test, cat_test = make_feature_frames(test_df, col_info, all_cols, table_rows)
    base_train, base_test, categorical_features = add_frequency_and_code_features(
        base_train, base_test, cat_train, cat_test
    )
    print(f"base_train={base_train.shape}, base_test={base_test.shape}")

    common_base_cols = base_train.columns.intersection(base_test.columns)
    base_train = base_train.loc[:, common_base_cols].copy()
    base_test = base_test.loc[:, common_base_cols].copy()
    categorical_features = [c for c in categorical_features if c in common_base_cols]

    configs: List[Tuple[str, Dict[str, object]]] = [
        (
            "rmse_deep",
            {
                "objective": "regression",
                "metric": "rmse",
                "boosting_type": "gbdt",
                "learning_rate": 0.025,
                "num_leaves": 255,
                "max_depth": 12,
                "min_data_in_leaf": 25,
                "feature_fraction": 0.86,
                "bagging_fraction": 0.86,
                "bagging_freq": 1,
                "lambda_l1": 0.15,
                "lambda_l2": 1.0,
                "cat_smooth": 20.0,
                "verbose": -1,
                "num_threads": -1,
                "seed": SEED,
            },
        ),
        (
            "rmse_wide",
            {
                "objective": "regression",
                "metric": "rmse",
                "boosting_type": "gbdt",
                "learning_rate": 0.022,
                "num_leaves": 383,
                "max_depth": 13,
                "min_data_in_leaf": 35,
                "feature_fraction": 0.78,
                "bagging_fraction": 0.82,
                "bagging_freq": 1,
                "lambda_l1": 0.05,
                "lambda_l2": 2.5,
                "cat_smooth": 35.0,
                "verbose": -1,
                "num_threads": -1,
                "seed": SEED + 17,
            },
        ),
        (
            "l1_stable",
            {
                "objective": "regression_l1",
                "metric": "mae",
                "boosting_type": "gbdt",
                "learning_rate": 0.02,
                "num_leaves": 191,
                "max_depth": 11,
                "min_data_in_leaf": 45,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.82,
                "bagging_freq": 1,
                "lambda_l1": 0.2,
                "lambda_l2": 1.5,
                "cat_smooth": 25.0,
                "verbose": -1,
                "num_threads": -1,
                "seed": SEED + 29,
            },
        ),
    ]

    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_by_config = {name: np.zeros(len(train_df), dtype=np.float32) for name, _ in configs}
    test_preds_by_config = {name: [] for name, _ in configs}
    best_iterations = {name: [] for name, _ in configs}

    print("\n" + "=" * 72)
    print("Cross-validated training")
    print("=" * 72)
    for fold, (fit_idx, valid_idx) in enumerate(kfold.split(train_df), start=1):
        print(f"\nFold {fold}/{N_FOLDS}")
        fit_cats = cat_train.iloc[fit_idx].reset_index(drop=True)
        valid_cats = cat_train.iloc[valid_idx].reset_index(drop=True)

        X_fit_base = base_train.iloc[fit_idx].reset_index(drop=True)
        X_valid_base = base_train.iloc[valid_idx].reset_index(drop=True)
        X_test_base = base_test.reset_index(drop=True)
        y_fit = y_log[fit_idx]
        y_valid = y_log[valid_idx]

        X_fit = add_target_encoding_features(
            X_fit_base,
            fit_cats,
            fit_cats,
            y_fit,
            part_y_log=y_fit,
            use_leave_one_out=True,
        )
        X_valid = add_target_encoding_features(
            X_valid_base,
            valid_cats,
            fit_cats,
            y_fit,
            use_leave_one_out=False,
        )
        X_test = add_target_encoding_features(
            X_test_base,
            cat_test.reset_index(drop=True),
            fit_cats,
            y_fit,
            use_leave_one_out=False,
        )
        X_fit, X_valid, X_test = align_columns(X_fit, X_valid, X_test)

        for config_id, (name, base_params) in enumerate(configs):
            params = dict(base_params)
            params["seed"] = int(params["seed"]) + fold * 101
            model = train_one_model(
                params,
                X_fit,
                y_fit,
                X_valid,
                y_valid,
                categorical_features,
                fold,
                name,
            )
            valid_pred = model.predict(X_valid, num_iteration=model.best_iteration).astype(np.float32)
            test_pred = model.predict(X_test, num_iteration=model.best_iteration).astype(np.float32)
            oof_by_config[name][valid_idx] = valid_pred
            test_preds_by_config[name].append(test_pred)
            best_iterations[name].append(int(model.best_iteration))

    print("\n" + "=" * 72)
    print("OOF scores")
    print("=" * 72)
    for name, _ in configs:
        q = qerror_from_logs(oof_by_config[name], y_log)
        print(
            f"{name:14s}: mean={q.mean():.5f}, median={np.median(q):.5f}, "
            f"p95={np.percentile(q, 95):.5f}, p99={np.percentile(q, 99):.5f}"
        )

    oof_stack = np.column_stack([oof_by_config[name] for name, _ in configs])
    oof_ensemble = oof_stack.mean(axis=1)
    alpha, beta, calibrated_score = optimal_affine_calibration(oof_ensemble, y_log)
    raw_score = float(np.mean(qerror_from_logs(oof_ensemble, y_log)))
    print(f"ensemble raw     : mean={raw_score:.5f}")
    print(f"ensemble calibrated: alpha={alpha:.4f}, beta={beta:.4f}, mean={calibrated_score:.5f}")

    oof_out = pd.DataFrame(
        {
            "Id": train_df["Id"].values,
            "Cardinality": y,
            "PredLogRaw": oof_ensemble,
            "PredLogCalibrated": alpha * oof_ensemble + beta,
        }
    )
    oof_out.to_csv(OOF_PATH, index=False)

    cv_test_stack = []
    for name, _ in configs:
        cv_test_stack.append(np.column_stack(test_preds_by_config[name]).mean(axis=1))
    cv_test_log = np.column_stack(cv_test_stack).mean(axis=1)

    print("\n" + "=" * 72)
    print("Full-data models")
    print("=" * 72)
    full_test_logs = []
    X_full = add_target_encoding_features(
        base_train.reset_index(drop=True),
        cat_train.reset_index(drop=True),
        cat_train.reset_index(drop=True),
        y_log,
        part_y_log=y_log,
        use_leave_one_out=True,
    )
    X_full_test = add_target_encoding_features(
        base_test.reset_index(drop=True),
        cat_test.reset_index(drop=True),
        cat_train.reset_index(drop=True),
        y_log,
        use_leave_one_out=False,
    )
    X_full, X_full_test = align_columns(X_full, X_full_test)

    for name, base_params in configs:
        params = dict(base_params)
        params["seed"] = int(params["seed"]) + 999
        mean_iter = int(np.mean(best_iterations[name]))
        train_rounds = max(300, int(mean_iter * 1.06))
        dtrain_full = lgb.Dataset(
            X_full,
            label=y_log,
            free_raw_data=False,
        )
        print(f"  {name}: full rounds={train_rounds} (cv mean iter={mean_iter})")
        full_params = dict(params)
        full_params["metric"] = "None"
        model = lgb.train(
            full_params,
            dtrain_full,
            num_boost_round=train_rounds,
            callbacks=[lgb.log_evaluation(period=0)],
        )
        full_test_logs.append(model.predict(X_full_test, num_iteration=train_rounds).astype(np.float32))

    full_test_log = np.column_stack(full_test_logs).mean(axis=1)

    # The CV ensemble has honest calibration; the full models use all rows and
    # slightly stronger template statistics. A moderate blend is usually safer
    # than replacing the CV ensemble entirely.
    test_log_raw = 0.62 * cv_test_log + 0.38 * full_test_log
    test_log = alpha * test_log_raw + beta
    test_card = np.maximum(np.expm1(test_log), 1.0)
    test_card = np.rint(test_card).astype(np.int64)

    submission = pd.DataFrame({"Id": test_df["Id"].values, "Cardinality": test_card})
    submission.to_csv(OUT_PATH, index=False)

    alt_cv = np.maximum(np.expm1(alpha * cv_test_log + beta), 1.0)
    pd.DataFrame(
        {"Id": test_df["Id"].values, "Cardinality": np.rint(alt_cv).astype(np.int64)}
    ).to_csv(ROOT / "submission_cv_only.csv", index=False)

    print("\n" + "=" * 72)
    print("Saved")
    print("=" * 72)
    print(f"submission: {OUT_PATH}")
    print(f"cv-only alternative: {ROOT / 'submission_cv_only.csv'}")
    print(
        f"prediction range=[{test_card.min():,}, {test_card.max():,}], "
        f"mean={test_card.mean():,.0f}"
    )
    print(f"elapsed={time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
