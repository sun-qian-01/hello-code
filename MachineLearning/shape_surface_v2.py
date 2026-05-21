"""
Shape-surface v2: per-shape expert selection.

The first exact-shape residual surface transferred online. This version keeps
the same modeling family but lets each exact shape choose among several surface
experts and shrink strengths using OOF Q-error. It is still anchored to the
current online-best `shape_surface_m50_d2_a10_s0p3_g0p0` prediction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from strong_model import qerror_from_logs


ROOT = Path(__file__).resolve().parent
SEED = 6262
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


def key_vec(row: dict, col_info: Dict[str, Tuple[float, float, float, float]]) -> Tuple[str, np.ndarray]:
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


def model_factory(kind: str, alpha: float, degree: int):
    steps = [("scale", StandardScaler())]
    if degree == 2:
        steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))
        steps.append(("scale2", StandardScaler()))
    if kind == "ridge":
        steps.append(("model", Ridge(alpha=alpha, random_state=SEED)))
    elif kind == "huber":
        steps.append(("model", HuberRegressor(alpha=alpha, epsilon=1.2, max_iter=300)))
    else:
        raise ValueError(kind)
    return make_pipeline(*[step for _, step in steps])


def fit_corr(
    train_keys: np.ndarray,
    test_keys: np.ndarray,
    train_vecs: List[np.ndarray],
    test_vecs: List[np.ndarray],
    residual: np.ndarray,
    *,
    min_group: int,
    kind: str,
    alpha: float,
    degree: int,
) -> Tuple[np.ndarray, np.ndarray]:
    corr = np.zeros(len(train_keys), dtype=np.float64)
    corr_test = np.zeros(len(test_keys), dtype=np.float64)
    fold_id = np.zeros(len(train_keys), dtype=np.int16)
    for fold, (_, valid_idx) in enumerate(KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(train_keys), start=1):
        fold_id[valid_idx] = fold

    groups = {
        key: values.values
        for key, values in pd.Series(np.arange(len(train_keys))).groupby(train_keys)
        if len(values) >= min_group
    }
    test_groups = {key: values.values for key, values in pd.Series(np.arange(len(test_keys))).groupby(test_keys)}

    for key, idx in groups.items():
        dim = len(train_vecs[idx[0]])
        if any(len(train_vecs[i]) != dim for i in idx):
            continue
        X = np.vstack([train_vecs[i] for i in idx])
        y = residual[idx]
        for fold in range(1, N_FOLDS + 1):
            valid_mask = fold_id[idx] == fold
            if not valid_mask.any():
                continue
            fit_mask = ~valid_mask
            if fit_mask.sum() < max(20, min_group // 2):
                continue
            model = model_factory(kind, alpha, degree)
            try:
                model.fit(X[fit_mask], y[fit_mask])
                corr[idx[valid_mask]] = model.predict(X[valid_mask])
            except Exception:
                continue
        test_idx = test_groups.get(key)
        if test_idx is not None and len(test_idx):
            if any(len(test_vecs[i]) != dim for i in test_idx):
                continue
            model = model_factory(kind, alpha, degree)
            try:
                model.fit(X, y)
                corr_test[test_idx] = model.predict(np.vstack([test_vecs[i] for i in test_idx]))
            except Exception:
                continue
    return np.clip(corr, -1.0, 1.0), np.clip(corr_test, -1.0, 1.0)


def per_shape_gate(
    keys: np.ndarray,
    test_keys: np.ndarray,
    y_log: np.ndarray,
    candidates: List[Tuple[str, np.ndarray, np.ndarray]],
    base_log: np.ndarray,
    base_test_log: np.ndarray,
    *,
    min_group: int,
    margin: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    all_train = [base_log] + [cand[1] for cand in candidates]
    all_test = [base_test_log] + [cand[2] for cand in candidates]
    qmat = np.column_stack([qerror_from_logs(pred, y_log) for pred in all_train])
    pred = base_log.copy()
    pred_test = base_test_log.copy()
    decisions: Dict[str, int] = {}

    for key, values in pd.Series(np.arange(len(keys))).groupby(keys):
        idx = values.values
        if len(idx) < min_group:
            decisions[str(key)] = 0
            continue
        means = qmat[idx].mean(axis=0)
        best = int(np.argmin(means))
        if best != 0 and means[best] + margin < means[0]:
            pred[idx] = all_train[best][idx]
            decisions[str(key)] = best
        else:
            decisions[str(key)] = 0

    for key, values in pd.Series(np.arange(len(test_keys))).groupby(test_keys):
        chosen = decisions.get(str(key), 0)
        if chosen:
            pred_test[values.values] = all_test[chosen][values.values]

    return pred, pred_test, decisions


def save_candidate(train: pd.DataFrame, test: pd.DataFrame, name: str, pred: np.ndarray, pred_test: np.ndarray, y_log: np.ndarray) -> None:
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
    pd.DataFrame({"Id": test["Id"].values, "Cardinality": card}).to_csv(ROOT / f"submission_{name}.csv", index=False)
    print(
        "saved",
        name,
        f"mean={q.mean():.6f}",
        f"med={np.median(q):.4f}",
        f"p95={np.percentile(q, 95):.4f}",
        f"range=[{card.min()}, {card.max()}]",
        f"sub_mean={card.mean():.1f}",
        flush=True,
    )


def main() -> None:
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    col_info = load_col_info()
    y_log = np.log1p(np.maximum(train["Cardinality"].astype(float).values, 1.0))

    # Anchor to the current online-best shape surface.
    base_log = load_oof(train, "shape_surface_m50_d2_a10_s0p3_g0p0_oof.csv")
    base_test_log = load_sub(test, "submission_shape_surface_m50_d2_a10_s0p3_g0p0.csv")
    lower_log = load_oof(train, "value_te_capped_pair02_on_isotonic_oof.csv")
    residual = np.clip(y_log - lower_log, -1.5, 1.5)

    train_pairs = [key_vec(row, col_info) for row in train.to_dict("records")]
    test_pairs = [key_vec(row, col_info) for row in test.to_dict("records")]
    train_keys = np.asarray([pair[0] for pair in train_pairs], dtype=object)
    test_keys = np.asarray([pair[0] for pair in test_pairs], dtype=object)
    train_vecs = [pair[1] for pair in train_pairs]
    test_vecs = [pair[1] for pair in test_pairs]

    bq = qerror_from_logs(base_log, y_log)
    print("base_shape_g0", f"mean={bq.mean():.6f}", f"p95={np.percentile(bq,95):.4f}", flush=True)

    raw_candidates: List[Tuple[str, np.ndarray, np.ndarray]] = []
    specs = [
        ("ridge", 1.0, 2),
        ("ridge", 3.0, 2),
        ("ridge", 10.0, 2),
        ("ridge", 30.0, 2),
        ("ridge", 100.0, 2),
        ("ridge", 10.0, 1),
        ("huber", 0.0001, 1),
        ("huber", 0.001, 1),
    ]
    for kind, alpha, degree in specs:
        print(f"fit {kind} alpha={alpha:g} degree={degree}", flush=True)
        corr, corr_test = fit_corr(
            train_keys,
            test_keys,
            train_vecs,
            test_vecs,
            residual,
            min_group=50,
            kind=kind,
            alpha=alpha,
            degree=degree,
        )
        for shrink in (0.20, 0.30, 0.40, 0.55, 0.75):
            pred = lower_log + shrink * corr
            pred_test = load_sub(test, "submission_value_te_capped_pair02_on_isotonic.csv") + shrink * corr_test
            q = qerror_from_logs(pred, y_log)
            name = f"{kind}_a{str(alpha).replace('.', 'p')}_d{degree}_s{str(shrink).replace('.', 'p')}"
            print(name, f"mean={q.mean():.6f}", f"p95={np.percentile(q,95):.4f}", flush=True)
            raw_candidates.append((name, pred, pred_test))

    results = []
    for margin in (0.0, 0.005, 0.01, 0.02):
        pred, pred_test, decisions = per_shape_gate(
            train_keys,
            test_keys,
            y_log,
            raw_candidates,
            base_log,
            base_test_log,
            min_group=50,
            margin=margin,
        )
        q = qerror_from_logs(pred, y_log)
        used_rows = int(np.sum(~np.isclose(pred, base_log)))
        used_test = int(np.sum(~np.isclose(pred_test, base_test_log)))
        name = f"shape_surface_v2_gate_g{str(margin).replace('.', 'p')}"
        print(
            name,
            f"mean={q.mean():.6f}",
            f"p95={np.percentile(q,95):.4f}",
            f"train_changed={used_rows}",
            f"test_changed={used_test}",
            flush=True,
        )
        results.append((float(q.mean()), name, pred, pred_test))

    results.sort(key=lambda item: item[0])
    for _, name, pred, pred_test in results[:4]:
        save_candidate(train, test, name, pred, pred_test, y_log)


if __name__ == "__main__":
    main()
