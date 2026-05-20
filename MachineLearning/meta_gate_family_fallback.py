"""
Family-level fallback from the meta residual model to the latent model.

Reflection from the online result:

The tiny Huber residual model helps on average, but some column families still
prefer the simpler latent prediction. Rather than adding more model capacity,
this script uses a very cheap second pass:

  * default to `submission_meta_gate_huber_resid`
  * back off to `submission_latent_final_blend` for `table + predicate columns`
    families where latent is reliably better in the training side of each fold

This keeps the stronger meta model for most rows while removing some of its
least stable corrections.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from strong_model import qerror_from_logs


ROOT = Path(__file__).resolve().parent
SEED = 202
N_FOLDS = 5
MIN_GROUP = 80
MARGIN = 0.001


def column_family_keys(frame: pd.DataFrame) -> np.ndarray:
    keys: List[str] = []
    for row in frame.to_dict("records"):
        parts = [] if pd.isna(row["Predicates"]) or str(row["Predicates"]).strip() == "" else [
            part.strip() for part in str(row["Predicates"]).split(",")
        ]
        cols = [parts[i] for i in range(0, len(parts) - 2, 3)]
        table = "" if pd.isna(row["Tables"]) else str(row["Tables"])
        keys.append(f"{table}||{'|'.join(cols)}")
    return np.asarray(keys, dtype=object)


def group_indices(keys: np.ndarray, indices: np.ndarray) -> Dict[str, np.ndarray]:
    groups: Dict[str, List[int]] = {}
    for idx in indices:
        groups.setdefault(str(keys[idx]), []).append(int(idx))
    return {key: np.asarray(values, dtype=np.int32) for key, values in groups.items()}


def load_log_predictions(frame: pd.DataFrame, oof_file: str, sub_file: str) -> Tuple[np.ndarray, np.ndarray]:
    oof = pd.read_csv(ROOT / oof_file).set_index("Id").reindex(frame["Id"].values)
    if "PredLog" not in oof.columns:
        raise ValueError(f"{oof_file} is missing PredLog")
    pred = oof["PredLog"].astype(float).values
    sub = pd.read_csv(ROOT / sub_file).set_index("Id")
    return pred, np.log1p(np.maximum(sub["Cardinality"].astype(float).values, 1.0))


def load_test_log_predictions(test: pd.DataFrame, sub_file: str) -> np.ndarray:
    sub = pd.read_csv(ROOT / sub_file).set_index("Id").reindex(test["Id"].values)
    return np.log1p(np.maximum(sub["Cardinality"].astype(float).values, 1.0))


def main() -> None:
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    y_log = np.log1p(np.maximum(train["Cardinality"].astype(float).values, 1.0))

    meta = pd.read_csv(ROOT / "meta_gate_huber_resid_oof.csv").set_index("Id").reindex(train["Id"].values)["PredLog"].astype(float).values
    latent = pd.read_csv(ROOT / "latent_final_blend_oof.csv").set_index("Id").reindex(train["Id"].values)["PredLog"].astype(float).values
    meta_test = load_test_log_predictions(test, "submission_meta_gate_huber_resid.csv")
    latent_test = load_test_log_predictions(test, "submission_latent_final_blend.csv")

    train_keys = column_family_keys(train)
    test_keys = column_family_keys(test)

    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    pred = np.zeros(len(train), dtype=np.float64)
    fallback_rows = 0
    for fold, (fit_idx, valid_idx) in enumerate(folds.split(train), start=1):
        fold_pred = meta[valid_idx].copy()
        fit_groups = group_indices(train_keys, fit_idx)
        valid_groups = group_indices(train_keys, valid_idx)
        for key, valid_group_idx in valid_groups.items():
            fit_group_idx = fit_groups.get(key)
            if fit_group_idx is None or len(fit_group_idx) < MIN_GROUP:
                continue
            meta_score = float(np.mean(qerror_from_logs(meta[fit_group_idx], y_log[fit_group_idx])))
            latent_score = float(np.mean(qerror_from_logs(latent[fit_group_idx], y_log[fit_group_idx])))
            if latent_score + MARGIN < meta_score:
                fold_pred[np.isin(valid_idx, valid_group_idx)] = latent[valid_group_idx]
                fallback_rows += len(valid_group_idx)
        pred[valid_idx] = fold_pred
        print(f"fold {fold} done", flush=True)

    pred_test = meta_test.copy()
    test_choice = np.array(["meta"] * len(test), dtype=object)
    train_groups = group_indices(train_keys, np.arange(len(train), dtype=np.int32))
    test_groups = group_indices(test_keys, np.arange(len(test), dtype=np.int32))
    for key, test_idx in test_groups.items():
        train_idx = train_groups.get(key)
        if train_idx is None or len(train_idx) < MIN_GROUP:
            continue
        meta_score = float(np.mean(qerror_from_logs(meta[train_idx], y_log[train_idx])))
        latent_score = float(np.mean(qerror_from_logs(latent[train_idx], y_log[train_idx])))
        if latent_score + MARGIN < meta_score:
            pred_test[test_idx] = latent_test[test_idx]
            test_choice[test_idx] = "latent"

    q = qerror_from_logs(pred, y_log)
    print(
        "meta_gate_family_fallback",
        f"mean={q.mean():.6f}",
        f"med={np.median(q):.4f}",
        f"p95={np.percentile(q, 95):.4f}",
        f"p99={np.percentile(q, 99):.4f}",
        f"fallback_rows={fallback_rows}",
        f"test_choices={pd.Series(test_choice).value_counts().to_dict()}",
        flush=True,
    )

    pd.DataFrame(
        {
            "Id": train["Id"].values,
            "Cardinality": train["Cardinality"].values,
            "PredLog": pred,
            "QError": q,
        }
    ).to_csv(ROOT / "meta_gate_family_fallback_oof.csv", index=False)
    card = np.rint(np.maximum(np.expm1(pred_test), 1.0)).astype(np.int64)
    pd.DataFrame({"Id": test["Id"].values, "Cardinality": card}).to_csv(
        ROOT / "submission_meta_gate_family_fallback.csv",
        index=False,
    )
    print(
        "saved submission_meta_gate_family_fallback.csv",
        f"range=[{card.min()}, {card.max()}]",
        f"mean={card.mean():.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
