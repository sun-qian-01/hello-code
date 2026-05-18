"""
Join-distribution inspired estimator.

This script moves away from pure text regression and builds a physical
cardinality estimate for the JOB-like star schema:

    title t -- movie_id -- {mc, ci, mi, mi_idx, mk}

For a query with title and several child tables, the raw estimate is based on
table-level selectivities and child rows-per-title fanouts. Then small
calibration models learn how to correct the physical estimate by table combo
and predicate structure. The resulting expert is blended with the best current
template model.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.model_selection import KFold

from strong_model import TABLES, qerror_from_logs


ROOT = Path(__file__).resolve().parent
SEED = 2042
N_FOLDS = 5
CHILD_TABLES = ["mc", "ci", "mi", "mi_idx", "mk"]


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


def parse_tables(value: object) -> List[str]:
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [part.strip().split()[-1] for part in str(value).split(",") if part.strip()]


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


def table_combo(tables: List[str]) -> str:
    return "|".join(tables) if tables else "none"


def pred_signature(preds: List[Tuple[str, str, float]], mode: str) -> str:
    if mode == "colop":
        return "|".join(f"{c}{o}" for c, o, _ in preds) or "none"
    if mode == "cols":
        return "|".join(c for c, _, _ in preds) or "none"
    if mode == "ops":
        return "|".join(o for _, o, _ in preds) or "none"
    if mode == "tableop":
        return "|".join(f"{c.split('.')[0]}{o}" for c, o, _ in preds) or "none"
    raise ValueError(mode)


def selectivity(col: str, op: str, val: float, col_info: Dict[str, Tuple[float, float, float, float]]) -> float:
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
    return float(np.clip(sel, 1e-9, 1.0))


def make_physical_features(df: pd.DataFrame, col_info: Dict[str, Tuple[float, float, float, float]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    table_rows = {t: col_info[f"{t}.id"][2] for t in TABLES}
    title_rows = table_rows["t"]
    # Number of distinct movies represented by child tables from stats.
    child_movie_unique = {
        "mc": col_info["mc.movie_id"][3],
        "ci": col_info["ci.movie_id"][3],
        "mi": col_info["mi.movie_id"][3],
        "mi_idx": col_info["mi_idx.movie_id"][3],
        "mk": col_info["mk.movie_id"][3],
    }
    child_fanout = {t: table_rows[t] / max(child_movie_unique[t], 1.0) for t in CHILD_TABLES}
    child_coverage = {t: child_movie_unique[t] / title_rows for t in CHILD_TABLES}

    rows: List[Dict[str, float]] = []
    cats: List[Dict[str, str]] = []
    for row in df.to_dict("records"):
        tables = parse_tables(row["Tables"])
        preds = parse_predicates(row["Predicates"])
        table_sel = {t: 1.0 for t in TABLES}
        table_pred_count = {t: 0 for t in TABLES}
        table_eq = {t: 0 for t in TABLES}
        table_lt = {t: 0 for t in TABLES}
        table_gt = {t: 0 for t in TABLES}
        pred_sels: List[float] = []
        values_norm: List[float] = []

        for col, op, val in preds:
            t = col.split(".")[0]
            sel = selectivity(col, op, val, col_info)
            pred_sels.append(sel)
            if t in table_sel:
                table_sel[t] *= sel
                table_pred_count[t] += 1
                table_eq[t] += int(op == "=")
                table_lt[t] += int(op == "<")
                table_gt[t] += int(op == ">")
            cmin, cmax, _, _ = col_info.get(col, (0.0, 1.0, 1.0, 1.0))
            values_norm.append((val - cmin) / max(cmax - cmin, 1.0))

        has_title = "t" in tables
        child_tables = [t for t in CHILD_TABLES if t in tables]
        title_sel = table_sel["t"] if has_title else 1.0

        if has_title:
            # Expected number of title movies after title predicates.
            movie_base = title_rows * title_sel
            raw_join = movie_base
            for child in child_tables:
                # coverage says how often title movie has at least one child row;
                # fanout gives rows per represented movie. Predicate selectivity
                # applies to child rows.
                raw_join *= child_coverage[child] * child_fanout[child] * table_sel[child]
        elif len(tables) == 1:
            only = tables[0]
            raw_join = table_rows.get(only, 1.0) * table_sel.get(only, 1.0)
            movie_base = raw_join
        else:
            # Rare non-title multi-table fallback.
            raw_join = 1.0
            for t in tables:
                raw_join *= table_rows.get(t, 1.0) * table_sel.get(t, 1.0)
            raw_join /= title_rows ** max(len(tables) - 1, 0)
            movie_base = raw_join

        # Alternative estimates to let the calibration model learn correlation.
        raw_ind = 1.0
        for t in tables:
            raw_ind *= table_rows.get(t, 1.0) * table_sel.get(t, 1.0)
        if len(tables) > 1:
            raw_ind /= title_rows ** (len(tables) - 1)
        raw_min = min([table_rows.get(t, 1.0) * table_sel.get(t, 1.0) for t in tables] or [1.0])
        raw_max = max([table_rows.get(t, 1.0) * table_sel.get(t, 1.0) for t in tables] or [1.0])

        feat: Dict[str, float] = {
            "num_tables": float(len(tables)),
            "num_children": float(len(child_tables)),
            "num_predicates": float(len(preds)),
            "raw_star_log": math.log1p(max(raw_join, 1e-9)),
            "raw_ind_log": math.log1p(max(raw_ind, 1e-9)),
            "raw_min_log": math.log1p(max(raw_min, 1e-9)),
            "raw_max_log": math.log1p(max(raw_max, 1e-9)),
            "movie_base_log": math.log1p(max(movie_base, 1e-9)),
            "table_sel_product_log": sum(math.log(max(table_sel[t], 1e-12)) for t in tables),
            "pred_sel_product_log": sum(math.log(max(s, 1e-12)) for s in pred_sels) if pred_sels else 0.0,
            "pred_sel_min": min(pred_sels) if pred_sels else 1.0,
            "pred_sel_mean": float(np.mean(pred_sels)) if pred_sels else 1.0,
            "value_norm_min": min(values_norm) if values_norm else 0.0,
            "value_norm_max": max(values_norm) if values_norm else 0.0,
            "value_norm_mean": float(np.mean(values_norm)) if values_norm else 0.0,
        }
        for t in TABLES:
            present = float(t in tables)
            filtered = table_rows[t] * table_sel[t] if t in table_rows else 1.0
            feat[f"has_{t}"] = present
            feat[f"{t}_sel_log"] = math.log(max(table_sel[t], 1e-12)) if present else 0.0
            feat[f"{t}_filtered_log"] = math.log1p(max(filtered, 1e-9)) if present else 0.0
            feat[f"{t}_pred_count"] = float(table_pred_count[t])
            feat[f"{t}_eq"] = float(table_eq[t])
            feat[f"{t}_lt"] = float(table_lt[t])
            feat[f"{t}_gt"] = float(table_gt[t])
            if t in child_fanout:
                feat[f"{t}_fanout_log"] = math.log1p(child_fanout[t]) * present
                feat[f"{t}_coverage"] = child_coverage[t] * present

        rows.append(feat)
        cats.append(
            {
                "table_combo": table_combo(tables),
                "shape": f"{table_combo(tables)}||{pred_signature(preds, 'colop')}",
                "cols": f"{table_combo(tables)}||{pred_signature(preds, 'cols')}",
                "ops": f"{table_combo(tables)}||{pred_signature(preds, 'ops')}",
                "tableop": f"{table_combo(tables)}||{pred_signature(preds, 'tableop')}",
            }
        )
    return pd.DataFrame(rows).astype(np.float32), pd.DataFrame(cats).astype(str)


def add_category_codes(X_train: pd.DataFrame, X_test: pd.DataFrame, C_train: pd.DataFrame, C_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X_train = X_train.copy()
    X_test = X_test.copy()
    for col in C_train.columns:
        combined = pd.concat([C_train[col], C_test[col]], ignore_index=True)
        codes, _ = pd.factorize(combined, sort=True)
        X_train[f"code_{col}"] = (codes[: len(C_train)] + 1).astype(np.int32)
        X_test[f"code_{col}"] = (codes[len(C_train) :] + 1).astype(np.int32)
        vc = C_train[col].value_counts()
        X_train[f"count_{col}_log"] = np.log1p(C_train[col].map(vc).fillna(0)).astype(np.float32)
        X_test[f"count_{col}_log"] = np.log1p(C_test[col].map(vc).fillna(0)).astype(np.float32)
    return X_train, X_test


def target_encode(
    X_part: pd.DataFrame,
    C_part: pd.DataFrame,
    C_fit: pd.DataFrame,
    y_fit: np.ndarray,
    *,
    smooth: float = 20.0,
) -> pd.DataFrame:
    X_part = X_part.copy()
    prior = float(np.mean(y_fit))
    for col in C_fit.columns:
        tmp = pd.DataFrame({"k": C_fit[col].values, "y": y_fit})
        g = tmp.groupby("k").y
        cnt = g.size()
        sm = g.sum()
        c = C_part[col].map(cnt).fillna(0).values.astype(float)
        s = C_part[col].map(sm).fillna(0).values.astype(float)
        X_part[f"te_{col}"] = ((s + prior * smooth) / (c + smooth)).astype(np.float32)
    return X_part


def train_physical_expert(
    X: pd.DataFrame,
    X_test: pd.DataFrame,
    C: pd.DataFrame,
    C_test: pd.DataFrame,
    y_log: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X), dtype=np.float32)
    test_preds: List[np.ndarray] = []
    params = {
        "objective": "regression_l1",
        "metric": "mae",
        "learning_rate": 0.025,
        "num_leaves": 127,
        "max_depth": 9,
        "min_data_in_leaf": 35,
        "feature_fraction": 0.86,
        "bagging_fraction": 0.86,
        "bagging_freq": 1,
        "lambda_l1": 0.2,
        "lambda_l2": 3.0,
        "verbose": -1,
        "num_threads": -1,
    }
    for fold, (fit_idx, valid_idx) in enumerate(folds.split(X), start=1):
        X_fit = target_encode(X.iloc[fit_idx].reset_index(drop=True), C.iloc[fit_idx].reset_index(drop=True), C.iloc[fit_idx].reset_index(drop=True), y_log[fit_idx])
        X_valid = target_encode(X.iloc[valid_idx].reset_index(drop=True), C.iloc[valid_idx].reset_index(drop=True), C.iloc[fit_idx].reset_index(drop=True), y_log[fit_idx])
        X_t = target_encode(X_test.reset_index(drop=True), C_test.reset_index(drop=True), C.iloc[fit_idx].reset_index(drop=True), y_log[fit_idx])
        cols = X_fit.columns.intersection(X_valid.columns).intersection(X_t.columns)
        dtrain = lgb.Dataset(X_fit[cols], label=y_log[fit_idx])
        dvalid = lgb.Dataset(X_valid[cols], label=y_log[valid_idx])
        model = lgb.train(
            {**params, "seed": SEED + fold},
            dtrain,
            num_boost_round=2500,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)],
        )
        oof[valid_idx] = model.predict(X_valid[cols], num_iteration=model.best_iteration)
        test_preds.append(model.predict(X_t[cols], num_iteration=model.best_iteration))
        q = qerror_from_logs(oof[valid_idx], y_log[valid_idx])
        print(f"  fold {fold}: iter={model.best_iteration}, q={q.mean():.5f}", flush=True)
    return oof, np.column_stack(test_preds).mean(axis=1).astype(np.float32)


def optimize_blend(experts: List[np.ndarray], experts_test: List[np.ndarray], y_log: np.ndarray, names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    E = np.column_stack(experts)
    ET = np.column_stack(experts_test)
    scores = np.array([np.mean(qerror_from_logs(E[:, i], y_log)) for i in range(E.shape[1])])
    for i, name in enumerate(names):
        q = qerror_from_logs(E[:, i], y_log)
        print(f"{name}: mean={q.mean():.5f}, med={np.median(q):.5f}, p95={np.percentile(q,95):.5f}", flush=True)
    keep = np.argsort(scores)[: len(scores)]

    def transform(z: np.ndarray, M: np.ndarray) -> np.ndarray:
        w = np.exp(z[: M.shape[1]])
        w = w / w.sum()
        p = M @ w
        alpha = 0.93 + 0.14 / (1 + np.exp(-z[M.shape[1]]))
        beta = 0.2 * np.tanh(z[M.shape[1] + 1])
        return alpha * p + beta

    def obj(z: np.ndarray) -> float:
        return float(np.mean(qerror_from_logs(transform(z, E[:, keep]), y_log)))

    result = differential_evolution(
        obj,
        [(-7, 7)] * (len(keep) + 2),
        seed=SEED,
        maxiter=260,
        tol=1e-9,
        polish=True,
        workers=1,
        updating="immediate",
    )
    pred = transform(result.x, E[:, keep])
    pred_test = transform(result.x, ET[:, keep])
    w = np.exp(result.x[: len(keep)])
    w = w / w.sum()
    print("blend:", np.mean(qerror_from_logs(pred, y_log)), dict(zip([names[i] for i in keep], w)), flush=True)
    print("pct:", np.percentile(qerror_from_logs(pred, y_log), [50, 90, 95, 99, 99.5, 100]), flush=True)
    return pred, pred_test


def main() -> None:
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    col_info = load_col_info()
    y_log = np.log1p(train["Cardinality"].astype(float).values)

    print("building physical features", flush=True)
    X, C = make_physical_features(train, col_info)
    X_test, C_test = make_physical_features(test, col_info)
    X, X_test = add_category_codes(X, X_test, C, C_test)

    print("training physical expert", flush=True)
    phys_oof, phys_test = train_physical_expert(X, X_test, C, C_test, y_log)
    pd.DataFrame({"Id": train["Id"], "Cardinality": train["Cardinality"], "PredLog": phys_oof, "QError": qerror_from_logs(phys_oof, y_log)}).to_csv(ROOT / "join_physical_oof.csv", index=False)
    pd.DataFrame({"Id": test["Id"], "Cardinality": np.rint(np.maximum(np.expm1(phys_test), 1.0)).astype(np.int64)}).to_csv(ROOT / "submission_join_physical_only.csv", index=False)

    experts = [phys_oof]
    experts_test = [phys_test]
    names = ["physical"]
    for name, oof_file, sub_file, col in [
        ("rewrite", "oof_rewrite_angle_affine_oof.csv", "submission_rewrite_angle_affine.csv", "PredLog"),
        ("hier", "hierarchical_template_oof.csv", "submission_hierarchical_template.csv", "PredLog"),
        ("group", "group_expert_oof.csv", "submission_group_expert.csv", "pred_log"),
        ("final", "final_oof_predictions.csv", "submission.csv", "PredLog"),
    ]:
        if (ROOT / oof_file).exists() and (ROOT / sub_file).exists():
            o = pd.read_csv(ROOT / oof_file)
            s = pd.read_csv(ROOT / sub_file)
            experts.append(o[col].values.astype(float))
            experts_test.append(np.log1p(s["Cardinality"].astype(float).values))
            names.append(name)

    pred, pred_test = optimize_blend(experts, experts_test, y_log, names)
    q = qerror_from_logs(pred, y_log)
    pd.DataFrame({"Id": train["Id"], "Cardinality": train["Cardinality"], "PredLog": pred, "QError": q}).to_csv(ROOT / "join_distribution_blend_oof.csv", index=False)
    sub = pd.DataFrame({"Id": test["Id"], "Cardinality": np.rint(np.maximum(np.expm1(pred_test), 1.0)).astype(np.int64)})
    sub.to_csv(ROOT / "submission_join_distribution_blend.csv", index=False)
    print("submission stats:", np.percentile(sub["Cardinality"], [0, 1, 5, 50, 95, 99, 100]), sub["Cardinality"].mean(), flush=True)


if __name__ == "__main__":
    main()
