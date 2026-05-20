"""
Sparse hashed residual model on top of the current online-best submission.

The value target-encoding route has transferred online, but pair-token tweaks
are now only marginal. This script changes the residual model family: instead
of independent token means, it fits a regularized sparse linear model over a
large hashed feature space containing template, exact-value, binned-value, and
pairwise predicate tokens.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from strong_model import qerror_from_logs


ROOT = Path(__file__).resolve().parent
SEED = 4242
N_FOLDS = 5
N_FEATURES = 2**19


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


def parse_predicates(value: object) -> List[Tuple[str, str, float, str]]:
    if pd.isna(value) or str(value).strip() == "":
        return []
    parts = [part.strip() for part in str(value).split(",")]
    out: List[Tuple[str, str, float, str]] = []
    for i in range(0, len(parts) - 2, 3):
        raw = parts[i + 2]
        try:
            val = float(raw)
        except ValueError:
            val = 0.0
        out.append((parts[i], parts[i + 1], val, raw))
    return out


def norm(value: object) -> str:
    return "" if pd.isna(value) else str(value)


def bin_id(x: float, n_bins: int) -> int:
    return int(np.clip(np.floor(float(np.clip(x, 0.0, 1.0)) * n_bins), 0, n_bins - 1))


def row_features(row: dict, col_info: Dict[str, Tuple[float, float, float, float]]) -> Dict[str, float]:
    table = norm(row["Tables"])
    join = norm(row["Join Conditions"])
    preds = parse_predicates(row["Predicates"])
    cols = "|".join(col for col, _, _, _ in preds)
    shape = "|".join(f"{col}{op}" for col, op, _, _ in preds)
    ops = "|".join(op for _, op, _, _ in preds)
    tableops = "|".join(f"{col.split('.')[0]}{op}" for col, op, _, _ in preds)

    feats: Dict[str, float] = {
        f"T={table}": 1.0,
        f"J={join}": 1.0,
        f"COLS={table}|{cols}": 1.0,
        f"SHAPE={table}|{join}|{shape}": 1.0,
        f"OPS={table}|{ops}": 1.0,
        f"TOPS={table}|{tableops}": 1.0,
        f"NP={len(preds)}": 1.0,
    }

    exact_tokens: List[str] = []
    bin_tokens: List[str] = []
    sel_sum = 0.0
    eq_count = lt_count = gt_count = 0
    for col, op, val, raw in preds:
        cmin, cmax, _, nunique = col_info.get(col, (0.0, 1.0, 1.0, 1.0))
        denom = max(cmax - cmin, 1.0)
        x = (val - cmin) / denom
        xc = float(np.clip(x, 0.0, 1.0))
        logx = float(np.log1p(max(val, 0.0)) / np.log1p(max(cmax, 1.0)))
        if op == "=":
            sel = 1.0 / max(nunique, 1.0)
            eq_count += 1
        elif op == "<":
            sel = max(xc, 1e-7)
            lt_count += 1
        elif op == ">":
            sel = max(1.0 - xc, 1e-7)
            gt_count += 1
        else:
            sel = 1.0
        log_sel = float(np.log(max(sel, 1e-12)))
        sel_sum += log_sel

        prefix = f"{table}|{col}{op}"
        short = f"{col}{op}"
        exact = f"{col}{op}{raw}"
        exact_tokens.append(exact)
        for n_bins in (8, 16, 32, 64):
            b = bin_id(xc, n_bins)
            lb = bin_id(logx, n_bins)
            feats[f"B{n_bins}={prefix}:{b}"] = 1.0
            feats[f"LB{n_bins}={prefix}:{lb}"] = 1.0
            bin_tokens.append(f"{short}:{b}")
        feats[f"P={prefix}"] = 1.0
        feats[f"PV={prefix}:{raw}"] = 1.0
        feats[f"V={col}{op}{raw}"] = 1.0
        feats[f"OPV={op}:{raw}"] = 1.0
        feats[f"NUM_X={prefix}"] = xc
        feats[f"NUM_LOGX={prefix}"] = logx
        feats[f"NUM_SEL={prefix}"] = log_sel

    feats["NUM_SEL_SUM"] = sel_sum
    feats["NUM_EQ"] = float(eq_count)
    feats["NUM_LT"] = float(lt_count)
    feats["NUM_GT"] = float(gt_count)

    for i in range(len(exact_tokens)):
        for j in range(i + 1, len(exact_tokens)):
            feats[f"PAIRV={table}|{exact_tokens[i]}&{exact_tokens[j]}"] = 1.0
    for i in range(len(bin_tokens)):
        for j in range(i + 1, len(bin_tokens)):
            feats[f"PAIRB={table}|{bin_tokens[i]}&{bin_tokens[j]}"] = 1.0

    return feats


def make_matrix(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
    info = load_col_info()
    rows_train = [row_features(row, info) for row in train.to_dict("records")]
    rows_test = [row_features(row, info) for row in test.to_dict("records")]
    hasher = FeatureHasher(n_features=N_FEATURES, input_type="dict", alternate_sign=False)
    X = hasher.transform(rows_train).tocsr().astype(np.float32)
    Xt = hasher.transform(rows_test).tocsr().astype(np.float32)
    return X, Xt


def load_oof(train: pd.DataFrame, filename: str) -> np.ndarray:
    frame = pd.read_csv(ROOT / filename).set_index("Id").reindex(train["Id"].values)
    return frame["PredLog"].astype(float).values


def load_sub(test: pd.DataFrame, filename: str) -> np.ndarray:
    frame = pd.read_csv(ROOT / filename).set_index("Id").reindex(test["Id"].values)
    return np.log1p(np.maximum(frame["Cardinality"].astype(float).values, 1.0))


def density_weights(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    def key(frame: pd.DataFrame) -> pd.Series:
        rows = []
        for row in frame.to_dict("records"):
            preds = parse_predicates(row["Predicates"])
            table = norm(row["Tables"])
            join = norm(row["Join Conditions"])
            shape = "|".join(f"{col}{op}" for col, op, _, _ in preds)
            rows.append(f"{table}||{join}||{shape}")
        return pd.Series(rows)

    train_key = key(train)
    test_key = key(test)
    tr_count = train_key.value_counts()
    te_count = test_key.value_counts()
    weights = train_key.map(te_count).fillna(0.0).values / train_key.map(tr_count).values
    weights = weights.astype(np.float64)
    weights = weights / max(weights.mean(), 1e-12)
    return np.clip(weights, 0.05, 20.0)


def main() -> None:
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    y_log = np.log1p(np.maximum(train["Cardinality"].astype(float).values, 1.0))
    base_log = load_oof(train, "value_te_capped_pair02_on_isotonic_oof.csv")
    base_test_log = load_sub(test, "submission_value_te_capped_pair02_on_isotonic.csv")
    residual = np.clip(y_log - base_log, -1.5, 1.5)
    weights = density_weights(train, test)

    print("building hashed matrix", flush=True)
    X, Xt = make_matrix(train, test)
    print(f"matrix train={X.shape} nnz={X.nnz} test_nnz={Xt.nnz}", flush=True)
    base_q = qerror_from_logs(base_log, y_log)
    print(
        "base_pair02",
        f"mean={base_q.mean():.6f}",
        f"weighted={np.average(base_q, weights=weights):.6f}",
        f"p95={np.percentile(base_q, 95):.4f}",
        flush=True,
    )

    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    candidates: List[Tuple[str, np.ndarray, np.ndarray]] = []

    for alpha in (30.0, 100.0, 300.0, 1000.0):
        oof = np.zeros(len(train), dtype=np.float64)
        test_preds: List[np.ndarray] = []
        for fold, (fit_idx, valid_idx) in enumerate(folds.split(X), start=1):
            model = Ridge(alpha=alpha, solver="lsqr", max_iter=2000, random_state=SEED + fold)
            model.fit(X[fit_idx], residual[fit_idx], sample_weight=weights[fit_idx])
            oof[valid_idx] = model.predict(X[valid_idx])
            test_preds.append(model.predict(Xt))
        corr_test = np.column_stack(test_preds).mean(axis=1)
        for shrink in (0.03, 0.06, 0.10, 0.16, 0.24):
            pred = base_log + shrink * np.clip(oof, -1.0, 1.0)
            predt = base_test_log + shrink * np.clip(corr_test, -1.0, 1.0)
            q = qerror_from_logs(pred, y_log)
            print(
                f"ridge_a{alpha:g}_s{shrink:g}",
                f"mean={q.mean():.6f}",
                f"weighted={np.average(q, weights=weights):.6f}",
                f"p95={np.percentile(q, 95):.4f}",
                flush=True,
            )
            candidates.append((f"hashed_ridge_a{int(alpha)}_s{str(shrink).replace('.', 'p')}", pred, predt))

    # A second, more robust online-learning family. It is weaker locally in many
    # cases but can regularize exact-value collisions differently from Ridge.
    scaler = StandardScaler(with_mean=False)
    Xs = scaler.fit_transform(X)
    Xts = scaler.transform(Xt)
    for alpha in (1e-5, 3e-5, 1e-4):
        oof = np.zeros(len(train), dtype=np.float64)
        test_preds = []
        for fold, (fit_idx, valid_idx) in enumerate(folds.split(Xs), start=1):
            model = SGDRegressor(
                loss="huber",
                epsilon=0.08,
                penalty="l2",
                alpha=alpha,
                learning_rate="invscaling",
                eta0=0.02,
                max_iter=2500,
                tol=1e-4,
                random_state=SEED + fold,
                average=True,
            )
            model.fit(Xs[fit_idx], residual[fit_idx], sample_weight=weights[fit_idx])
            oof[valid_idx] = model.predict(Xs[valid_idx])
            test_preds.append(model.predict(Xts))
        corr_test = np.column_stack(test_preds).mean(axis=1)
        for shrink in (0.01, 0.02, 0.04, 0.08):
            pred = base_log + shrink * np.clip(oof, -1.0, 1.0)
            predt = base_test_log + shrink * np.clip(corr_test, -1.0, 1.0)
            q = qerror_from_logs(pred, y_log)
            print(
                f"sgd_a{alpha:g}_s{shrink:g}",
                f"mean={q.mean():.6f}",
                f"weighted={np.average(q, weights=weights):.6f}",
                f"p95={np.percentile(q, 95):.4f}",
                flush=True,
            )
            candidates.append((f"hashed_sgd_a{alpha:g}_s{str(shrink).replace('.', 'p').replace('-', 'm')}", pred, predt))

    ranked = []
    for name, pred, predt in candidates:
        q = qerror_from_logs(pred, y_log)
        ranked.append((float(np.average(q, weights=weights)), float(q.mean()), name, pred, predt))
    ranked.sort()
    print("best weighted candidates", flush=True)
    for item in ranked[:8]:
        print(item[:3], flush=True)

    for _, _, name, pred, predt in ranked[:4]:
        q = qerror_from_logs(pred, y_log)
        pd.DataFrame(
            {
                "Id": train["Id"].values,
                "Cardinality": train["Cardinality"].values,
                "PredLog": pred,
                "QError": q,
            }
        ).to_csv(ROOT / f"{name}_oof.csv", index=False)
        card = np.rint(np.maximum(np.expm1(predt), 1.0)).astype(np.int64)
        pd.DataFrame({"Id": test["Id"].values, "Cardinality": card}).to_csv(
            ROOT / f"submission_{name}.csv",
            index=False,
        )
        print(
            "saved",
            name,
            f"mean={q.mean():.6f}",
            f"weighted={np.average(q, weights=weights):.6f}",
            f"range=[{card.min()}, {card.max()}]",
            f"sub_mean={card.mean():.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
