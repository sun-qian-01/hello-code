"""
Stacked family gate over the best post-processing experts.

This is a cheap second-stage router after introducing the HistGradientBoosting
residual expert. It keeps `hist_family_gate` as the default and only switches
large, stable families to another expert when the training-side Q-error says so.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from strong_model import qerror_from_logs


ROOT = Path(__file__).resolve().parent
SEED = 707
N_FOLDS = 5

EXPERTS: Tuple[Tuple[str, str, str], ...] = (
    ("hist_family", "hist_family_gate_oof.csv", "submission_hist_family_gate.csv"),
    ("base", "meta_gate_family_fallback_oof.csv", "submission_meta_gate_family_fallback.csv"),
    ("hist", "hist_residual_oof.csv", "submission_hist_residual.csv"),
    ("meta", "meta_gate_huber_resid_oof.csv", "submission_meta_gate_huber_resid.csv"),
    ("tiny", "meta_gate_huber_tiny_oof.csv", "submission_meta_gate_huber_tiny.csv"),
    ("aff", "column_family_gate_shape50_cols80_affine_oof.csv", "submission_column_family_gate_shape50_cols80_affine.csv"),
    ("latent", "latent_final_blend_oof.csv", "submission_latent_final_blend.csv"),
    ("lowrank", "lowrank_latent_oof.csv", "submission_lowrank_latent.csv"),
)

DEFAULT = "hist_family"
ALLOWED = ("hist_family", "base", "hist", "aff", "latent", "tiny", "lowrank")
LEVELS = ("cols", "tableop")
MIN_GROUPS = (400, 400)
MARGINS = (0.01, 0.01)


def prediction_column(frame: pd.DataFrame) -> str:
    for col in ("PredLog", "pred_log", "PredLogCalibrated", "PredLogRaw"):
        if col in frame.columns:
            return col
    raise ValueError(f"no prediction column found in {list(frame.columns)}")


def load_experts(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[List[str], np.ndarray, np.ndarray]:
    names: List[str] = []
    oof_logs: List[np.ndarray] = []
    test_logs: List[np.ndarray] = []
    for name, oof_file, sub_file in EXPERTS:
        oof_path = ROOT / oof_file
        sub_path = ROOT / sub_file
        if not oof_path.exists() or not sub_path.exists():
            print(f"skip {name}: missing {oof_file} or {sub_file}", flush=True)
            continue
        oof = pd.read_csv(oof_path).set_index("Id").reindex(train["Id"].values)
        sub = pd.read_csv(sub_path).set_index("Id").reindex(test["Id"].values)
        col = prediction_column(oof)
        if oof[col].isna().any() or sub["Cardinality"].isna().any():
            raise ValueError(f"{name} predictions are not aligned")
        names.append(name)
        oof_logs.append(oof[col].astype(float).values)
        test_logs.append(np.log1p(np.maximum(sub["Cardinality"].astype(float).values, 1.0)))
    return names, np.column_stack(oof_logs), np.column_stack(test_logs)


def make_keys(frame: pd.DataFrame) -> Dict[str, np.ndarray]:
    rows = {level: [] for level in ("table", "cols", "shape", "tableop", "ops")}
    for row in frame.to_dict("records"):
        parts = [] if pd.isna(row["Predicates"]) or str(row["Predicates"]).strip() == "" else [
            part.strip() for part in str(row["Predicates"]).split(",")
        ]
        cols: List[str] = []
        shape: List[str] = []
        tableop: List[str] = []
        ops: List[str] = []
        for i in range(0, len(parts) - 2, 3):
            col = parts[i]
            op = parts[i + 1]
            cols.append(col)
            shape.append(col + op)
            tableop.append((col.split(".")[0] if "." in col else col) + op)
            ops.append(op)
        table = "" if pd.isna(row["Tables"]) else str(row["Tables"])
        join = "" if pd.isna(row["Join Conditions"]) else str(row["Join Conditions"])
        rows["table"].append(table)
        rows["cols"].append(f"{table}||{'|'.join(cols)}")
        rows["shape"].append(f"{table}||{join}||{'|'.join(shape)}")
        rows["tableop"].append(f"{table}||{'|'.join(tableop)}")
        rows["ops"].append(f"{table}||{'|'.join(ops)}")
    return {key: np.asarray(value, dtype=object) for key, value in rows.items()}


def group_indices(values: np.ndarray, indices: np.ndarray) -> Dict[str, np.ndarray]:
    groups: Dict[str, List[int]] = {}
    for idx in indices:
        groups.setdefault(str(values[idx]), []).append(int(idx))
    return {key: np.asarray(group, dtype=np.int32) for key, group in groups.items()}


def choose_expert(
    qmat: np.ndarray,
    fit_idx: np.ndarray,
    allowed: np.ndarray,
    default: int,
    margin: float,
) -> int:
    means = qmat[np.ix_(fit_idx, allowed)].mean(axis=0)
    best = int(allowed[int(np.argmin(means))])
    default_score = float(qmat[fit_idx, default].mean())
    return best if float(means.min()) + margin < default_score else default


def route(
    names: Sequence[str],
    oof_logs: np.ndarray,
    test_logs: np.ndarray,
    train_keys: Dict[str, np.ndarray],
    test_keys: Dict[str, np.ndarray],
    y_log: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    default = names.index(DEFAULT)
    allowed = np.asarray([names.index(name) for name in ALLOWED if name in names], dtype=np.int32)
    qmat = np.column_stack([qerror_from_logs(oof_logs[:, i], y_log) for i in range(oof_logs.shape[1])])
    pred = np.zeros(len(oof_logs), dtype=np.float64)
    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for fold, (fit_idx, valid_idx) in enumerate(folds.split(oof_logs), start=1):
        fold_pred = oof_logs[valid_idx, default].copy()
        for level, min_group, margin in zip(LEVELS, MIN_GROUPS, MARGINS):
            fit_groups = group_indices(train_keys[level], fit_idx)
            valid_groups = group_indices(train_keys[level], valid_idx)
            for key, valid_group_idx in valid_groups.items():
                fit_group_idx = fit_groups.get(key)
                if fit_group_idx is None or len(fit_group_idx) < min_group:
                    continue
                chosen = choose_expert(qmat, fit_group_idx, allowed, default, margin)
                if chosen != default:
                    fold_pred[np.isin(valid_idx, valid_group_idx)] = oof_logs[valid_group_idx, chosen]
        pred[valid_idx] = fold_pred
        print(f"fold {fold} done", flush=True)

    pred_test = test_logs[:, default].copy()
    test_choice = np.full(len(test_logs), default, dtype=np.int32)
    for level, min_group, margin in zip(LEVELS, MIN_GROUPS, MARGINS):
        train_groups = group_indices(train_keys[level], np.arange(len(oof_logs), dtype=np.int32))
        test_groups = group_indices(test_keys[level], np.arange(len(test_logs), dtype=np.int32))
        for key, test_idx in test_groups.items():
            fit_idx = train_groups.get(key)
            if fit_idx is None or len(fit_idx) < min_group:
                continue
            chosen = choose_expert(qmat, fit_idx, allowed, default, margin)
            if chosen != default:
                pred_test[test_idx] = test_logs[test_idx, chosen]
                test_choice[test_idx] = chosen

    choices = {names[idx]: int((test_choice == idx).sum()) for idx in np.unique(test_choice)}
    return pred, pred_test, choices


def main() -> None:
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    y_log = np.log1p(np.maximum(train["Cardinality"].astype(float).values, 1.0))
    names, oof_logs, test_logs = load_experts(train, test)
    train_keys = make_keys(train)
    test_keys = make_keys(test)
    pred, pred_test, choices = route(names, oof_logs, test_logs, train_keys, test_keys, y_log)
    q = qerror_from_logs(pred, y_log)
    print(
        "stacked_family_gate",
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
    ).to_csv(ROOT / "stacked_family_gate_oof.csv", index=False)
    card = np.rint(np.maximum(np.expm1(pred_test), 1.0)).astype(np.int64)
    pd.DataFrame({"Id": test["Id"].values, "Cardinality": card}).to_csv(
        ROOT / "submission_stacked_family_gate.csv",
        index=False,
    )
    print(
        "saved submission_stacked_family_gate.csv",
        f"range=[{card.min()}, {card.max()}]",
        f"mean={card.mean():.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
