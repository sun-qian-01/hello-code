"""
Exact-shape residual surface models.

The value target-encoding route is now online-confirmed, while broad pair-token
residuals have only marginal gains. This script tries a different local model:
within each exact predicate shape, fit a low-dimensional continuous residual
surface over normalized predicate values. It then gates the surface by exact
shape using only OOF evidence, so unstable shapes fall back to the current
online-best prediction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from strong_model import qerror_from_logs


ROOT = Path(__file__).resolve().parent
SEED = 5151
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


def norm(value: object) -> str:
    return "" if pd.isna(value) else str(value)


def key_and_vec(row: dict, col_info: Dict[str, Tuple[float, float, float, float]]) -> Tuple[str, np.ndarray]:
    table = norm(row["Tables"])
    join = norm(row["Join Conditions"])
    preds = parse_predicates(row["Predicates"])
    shape = "|".join(f"{col}{op}" for col, op, _ in preds)
    key = f"{table}||{join}||{shape}"
    vals: List[float] = []
    sel_logs: List[float] = []
    for col, op, val in preds:
        cmin, cmax, _, nunique = col_info.get(col, (0.0, 1.0, 1.0, 1.0))
        denom = max(cmax - cmin, 1.0)
        x = (val - cmin) / denom
        xc = float(np.clip(x, 0.0, 1.0))
        logx = float(np.log1p(max(val, 0.0)) / np.log1p(max(cmax, 1.0)))
        if op == "=":
            sel = 1.0 / max(nunique, 1.0)
        elif op == "<":
            sel = max(xc, 1e-7)
        elif op == ">":
            sel = max(1.0 - xc, 1e-7)
        else:
            sel = 1.0
        log_sel = float(np.log(max(sel, 1e-12)))
        sel_logs.append(log_sel)
        vals.extend([x, xc, logx, log_sel])
    vals.extend(
        [
            float(len(preds)),
            float(np.sum(sel_logs)) if sel_logs else 0.0,
            float(np.min(sel_logs)) if sel_logs else 0.0,
            float(np.max(sel_logs)) if sel_logs else 0.0,
        ]
    )
    return key, np.asarray(vals, dtype=np.float32)


def load_oof(train: pd.DataFrame, filename: str) -> np.ndarray:
    frame = pd.read_csv(ROOT / filename).set_index("Id").reindex(train["Id"].values)
    return frame["PredLog"].astype(float).values


def load_sub(test: pd.DataFrame, filename: str) -> np.ndarray:
    frame = pd.read_csv(ROOT / filename).set_index("Id").reindex(test["Id"].values)
    return np.log1p(np.maximum(frame["Cardinality"].astype(float).values, 1.0))


def build_model(alpha: float, degree: int):
    steps = [("scale", StandardScaler())]
    if degree == 2:
        steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))
        steps.append(("scale2", StandardScaler()))
    steps.append(("ridge", Ridge(alpha=alpha, random_state=SEED)))
    return make_pipeline(*[step for _, step in steps])


def fit_surface(
    train_keys: np.ndarray,
    test_keys: np.ndarray,
    train_vecs: List[np.ndarray],
    test_vecs: List[np.ndarray],
    y_log: np.ndarray,
    base_log: np.ndarray,
    base_test_log: np.ndarray,
    *,
    min_group: int,
    alpha: float,
    degree: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    residual = np.clip(y_log - base_log, -1.5, 1.5)
    corr = np.zeros(len(train_keys), dtype=np.float64)
    corr_test = np.zeros(len(test_keys), dtype=np.float64)
    covered = np.zeros(len(train_keys), dtype=bool)
    covered_test = np.zeros(len(test_keys), dtype=bool)

    fold_id = np.zeros(len(train_keys), dtype=np.int16)
    for fold, (_, valid_idx) in enumerate(KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(train_keys), start=1):
        fold_id[valid_idx] = fold

    groups = {
        key: values.values
        for key, values in pd.Series(np.arange(len(train_keys))).groupby(train_keys)
        if len(values) >= min_group
    }
    test_groups = {key: values.values for key, values in pd.Series(np.arange(len(test_keys))).groupby(test_keys)}

    for key, all_idx in groups.items():
        dim = len(train_vecs[all_idx[0]])
        if any(len(train_vecs[idx]) != dim for idx in all_idx):
            continue
        X_all = np.vstack([train_vecs[idx] for idx in all_idx])
        y_all = residual[all_idx]

        for fold in range(1, N_FOLDS + 1):
            valid_mask = fold_id[all_idx] == fold
            if not valid_mask.any():
                continue
            fit_mask = ~valid_mask
            if fit_mask.sum() < max(20, min_group // 2):
                continue
            model = build_model(alpha, degree)
            model.fit(X_all[fit_mask], y_all[fit_mask])
            valid_idx = all_idx[valid_mask]
            corr[valid_idx] = model.predict(X_all[valid_mask])
            covered[valid_idx] = True

        test_idx = test_groups.get(key)
        if test_idx is None or len(test_idx) == 0:
            continue
        if any(len(test_vecs[idx]) != dim for idx in test_idx):
            continue
        model = build_model(alpha, degree)
        model.fit(X_all, y_all)
        corr_test[test_idx] = model.predict(np.vstack([test_vecs[idx] for idx in test_idx]))
        covered_test[test_idx] = True

    return np.clip(corr, -1.0, 1.0), np.clip(corr_test, -1.0, 1.0), covered, covered_test


def gated_prediction(
    keys: np.ndarray,
    base_log: np.ndarray,
    cand_log: np.ndarray,
    y_log: np.ndarray,
    *,
    min_group: int,
    margin: float,
) -> Tuple[np.ndarray, Dict[str, bool]]:
    q_base = qerror_from_logs(base_log, y_log)
    q_cand = qerror_from_logs(cand_log, y_log)
    pred = base_log.copy()
    decisions: Dict[str, bool] = {}
    for key, values in pd.Series(np.arange(len(keys))).groupby(keys):
        idx = values.values
        use = len(idx) >= min_group and q_cand[idx].mean() + margin < q_base[idx].mean()
        decisions[str(key)] = bool(use)
        if use:
            pred[idx] = cand_log[idx]
    return pred, decisions


def apply_test_gate(
    test_keys: np.ndarray,
    base_test_log: np.ndarray,
    cand_test_log: np.ndarray,
    decisions: Dict[str, bool],
) -> np.ndarray:
    pred = base_test_log.copy()
    for key, values in pd.Series(np.arange(len(test_keys))).groupby(test_keys):
        if decisions.get(str(key), False):
            pred[values.values] = cand_test_log[values.values]
    return pred


def main() -> None:
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    col_info = load_col_info()
    y_log = np.log1p(np.maximum(train["Cardinality"].astype(float).values, 1.0))
    base_log = load_oof(train, "value_te_capped_pair02_on_isotonic_oof.csv")
    base_test_log = load_sub(test, "submission_value_te_capped_pair02_on_isotonic.csv")

    train_pairs = [key_and_vec(row, col_info) for row in train.to_dict("records")]
    test_pairs = [key_and_vec(row, col_info) for row in test.to_dict("records")]
    train_keys = np.asarray([pair[0] for pair in train_pairs], dtype=object)
    test_keys = np.asarray([pair[0] for pair in test_pairs], dtype=object)
    train_vecs = [pair[1] for pair in train_pairs]
    test_vecs = [pair[1] for pair in test_pairs]

    base_q = qerror_from_logs(base_log, y_log)
    print(
        "base_pair02",
        f"mean={base_q.mean():.6f}",
        f"med={np.median(base_q):.4f}",
        f"p95={np.percentile(base_q, 95):.4f}",
        flush=True,
    )

    results = []
    for min_group in (50, 80, 150, 300):
        for degree in (1, 2):
            for alpha in (1.0, 10.0, 100.0, 1000.0):
                print(f"surface min={min_group} degree={degree} alpha={alpha:g}", flush=True)
                corr, corr_test, covered, covered_test = fit_surface(
                    train_keys,
                    test_keys,
                    train_vecs,
                    test_vecs,
                    y_log,
                    base_log,
                    base_test_log,
                    min_group=min_group,
                    alpha=alpha,
                    degree=degree,
                )
                for shrink in (0.05, 0.10, 0.18, 0.30):
                    cand = base_log + shrink * corr
                    cand_test = base_test_log + shrink * corr_test
                    q = qerror_from_logs(cand, y_log)
                    print(
                        f"  shrink={shrink:g}",
                        f"mean={q.mean():.6f}",
                        f"p95={np.percentile(q, 95):.4f}",
                        f"cov={covered.mean():.3f}",
                        f"test={covered_test.mean():.3f}",
                        flush=True,
                    )
                    for gate_margin in (0.0, 0.01, 0.03):
                        gated, decisions = gated_prediction(
                            train_keys,
                            base_log,
                            cand,
                            y_log,
                            min_group=min_group,
                            margin=gate_margin,
                        )
                        gated_test = apply_test_gate(test_keys, base_test_log, cand_test, decisions)
                        gq = qerror_from_logs(gated, y_log)
                        used_test = sum(decisions.get(str(k), False) for k in pd.unique(test_keys))
                        name = (
                            f"shape_surface_m{min_group}_d{degree}_a{int(alpha)}_"
                            f"s{str(shrink).replace('.', 'p')}_g{str(gate_margin).replace('.', 'p')}"
                        )
                        results.append((float(gq.mean()), float(np.percentile(gq, 95)), name, gated, gated_test, used_test))

    results.sort(key=lambda item: item[0])
    print("best gated", flush=True)
    for row in results[:12]:
        print(row[:3] + (row[5],), flush=True)

    for mean_score, _, name, pred, pred_test, _ in results[:5]:
        q = qerror_from_logs(pred, y_log)
        pd.DataFrame(
            {
                "Id": train["Id"].values,
                "Cardinality": train["Cardinality"].values,
                "PredLog": pred,
                "QError": q,
            }
        ).to_csv(ROOT / f"{name}_oof.csv", index=False)
        card = np.rint(np.maximum(np.expm1(pred_test), 1.0)).astype(np.int64)
        pd.DataFrame({"Id": test["Id"].values, "Cardinality": card}).to_csv(
            ROOT / f"submission_{name}.csv",
            index=False,
        )
        print(
            "saved",
            name,
            f"mean={mean_score:.6f}",
            f"range=[{card.min()}, {card.max()}]",
            f"sub_mean={card.mean():.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
