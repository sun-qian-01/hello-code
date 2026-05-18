"""
Column-family expert gate.

The existing best predictions are extremely close globally, but they win on
different predicate-column families. This script uses a second-level, leakage-
checked gate:

  * group queries by table combo + ordered predicate columns;
  * within each outer fold, choose the best existing expert for a group using
    only the fold's training side;
  * apply that choice to the validation side and to matching test families.

It writes two candidates:

  * submission_column_family_gate.csv      - wider/aggressive expert pool
  * submission_column_family_gate_core.csv - conservative strong-expert pool
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parent
SEED = 765
N_FOLDS = 5


@dataclass(frozen=True)
class ExpertSpec:
    name: str
    oof_file: str
    sub_file: str


EXPERTS: Tuple[ExpertSpec, ...] = (
    ExpertSpec("rewrite", "oof_rewrite_angle_affine_oof.csv", "submission_rewrite_angle_affine.csv"),
    ExpertSpec("grid", "grid_surface_oof.csv", "submission_grid_surface.csv"),
    ExpertSpec("monotone", "monotone_template_oof.csv", "submission_monotone_template.csv"),
    ExpertSpec("lowrank", "lowrank_latent_oof.csv", "submission_lowrank_latent.csv"),
    ExpertSpec("latent", "latent_final_blend_oof.csv", "submission_latent_final_blend.csv"),
    ExpertSpec("final", "final_oof_predictions.csv", "submission.csv"),
    ExpertSpec("group", "group_expert_oof.csv", "submission_group_expert.csv"),
    ExpertSpec("resid", "residual_blend_oof.csv", "submission_residual_blend.csv"),
    ExpertSpec("shape400", "shape_local_oof_400.csv", "submission_shape_local_400.csv"),
)


VARIANTS = {
    # Best nested-CV result in local tests: around 3.68062.
    "full": {
        "allowed": ("latent", "lowrank", "grid", "rewrite", "monotone", "group", "resid", "shape400", "final"),
        "min_group": 80,
        "margin": 0.005,
        "levels": ("column_family",),
        "oof_file": "column_family_gate_oof.csv",
        "sub_file": "submission_column_family_gate.csv",
    },
    # More conservative: only strong late-stage experts, around 3.68616 nested-CV.
    "core": {
        "allowed": ("latent", "lowrank", "grid", "rewrite", "monotone"),
        "min_group": 200,
        "margin": 0.0025,
        "levels": ("column_family",),
        "oof_file": "column_family_gate_core_oof.csv",
        "sub_file": "submission_column_family_gate_core.csv",
    },
    # Slightly better nested-CV than the full pool, while avoiding the weakest
    # old submissions. Shape choices are applied first and column-family choices
    # may override them when that broader family is more reliable.
    "shape_cols": {
        "allowed": ("latent", "lowrank", "grid", "rewrite", "monotone", "group", "shape400"),
        "min_group": 80,
        "margin": 0.0025,
        "levels": ("exact_shape", "column_family"),
        "oof_file": "column_family_gate_shape_cols_oof.csv",
        "sub_file": "submission_column_family_gate_shape_cols.csv",
    },
    # Best two-level gate found in nested-CV sweeps: exact shape first when it
    # has at least 50 training rows, then broader column family with a small
    # improvement margin.
    "shape50_cols80": {
        "allowed": ("latent", "lowrank", "grid", "rewrite", "monotone", "group", "shape400"),
        "levels": ("exact_shape", "column_family"),
        "min_groups": (50, 80),
        "margins": (0.0, 0.0025),
        "oof_file": "column_family_gate_shape50_cols80_oof.csv",
        "sub_file": "submission_column_family_gate_shape50_cols80.csv",
    },
    "shape50_cols80_affine": {
        "allowed": ("latent", "lowrank", "grid", "rewrite", "monotone", "group", "shape400"),
        "levels": ("exact_shape", "column_family"),
        "min_groups": (50, 80),
        "margins": (0.0, 0.0025),
        "affine": True,
        "oof_file": "column_family_gate_shape50_cols80_affine_oof.csv",
        "sub_file": "submission_column_family_gate_shape50_cols80_affine.csv",
    },
}


def prediction_column(frame: pd.DataFrame) -> str:
    for col in ("PredLog", "pred_log", "PredLogCalibrated", "PredLogRaw"):
        if col in frame.columns:
            return col
    raise ValueError(f"no prediction-log column found in {list(frame.columns)}")


def qerror_from_logs(pred_log: np.ndarray, true_log: np.ndarray) -> np.ndarray:
    pred_card = np.maximum(np.expm1(pred_log), 1.0)
    true_card = np.maximum(np.expm1(true_log), 1.0)
    return np.maximum(pred_card / true_card, true_card / pred_card)


def parse_predicate_columns(value: object) -> List[str]:
    if pd.isna(value) or str(value).strip() == "":
        return []
    parts = [part.strip() for part in str(value).split(",")]
    return [parts[i] for i in range(0, len(parts) - 2, 3)]


def column_family_keys(frame: pd.DataFrame) -> np.ndarray:
    keys: List[str] = []
    for row in frame.to_dict("records"):
        table = "" if pd.isna(row["Tables"]) else str(row["Tables"])
        cols = parse_predicate_columns(row["Predicates"])
        keys.append(f"{table}||{'|'.join(cols)}")
    return np.asarray(keys, dtype=object)


def exact_shape_keys(frame: pd.DataFrame) -> np.ndarray:
    keys: List[str] = []
    for row in frame.to_dict("records"):
        table = "" if pd.isna(row["Tables"]) else str(row["Tables"])
        join = "" if pd.isna(row["Join Conditions"]) else str(row["Join Conditions"])
        if pd.isna(row["Predicates"]) or str(row["Predicates"]).strip() == "":
            shape = "none"
        else:
            parts = [part.strip() for part in str(row["Predicates"]).split(",")]
            shape = "|".join(f"{parts[i]}{parts[i + 1]}" for i in range(0, len(parts) - 2, 3)) or "none"
        keys.append(f"{table}||{join}||{shape}")
    return np.asarray(keys, dtype=object)


def load_experts(
    train: pd.DataFrame,
    test: pd.DataFrame,
    specs: Sequence[ExpertSpec],
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    names: List[str] = []
    oof_logs: List[np.ndarray] = []
    test_logs: List[np.ndarray] = []

    for spec in specs:
        oof_path = ROOT / spec.oof_file
        sub_path = ROOT / spec.sub_file
        if not oof_path.exists() or not sub_path.exists():
            print(f"skip {spec.name}: missing {oof_path.name} or {sub_path.name}", flush=True)
            continue

        oof = pd.read_csv(oof_path)
        sub = pd.read_csv(sub_path)
        oof_col = prediction_column(oof)

        if "Id" not in oof.columns or "Id" not in sub.columns:
            raise ValueError(f"{spec.name} is missing Id column")
        if "Cardinality" not in sub.columns:
            raise ValueError(f"{spec.name} submission is missing Cardinality column")

        oof_aligned = oof.set_index("Id").reindex(train["Id"].values)
        sub_aligned = sub.set_index("Id").reindex(test["Id"].values)
        if oof_aligned[oof_col].isna().any() or sub_aligned["Cardinality"].isna().any():
            raise ValueError(f"{spec.name} predictions do not align with train/test IDs")

        names.append(spec.name)
        oof_logs.append(oof_aligned[oof_col].astype(float).values)
        test_logs.append(np.log1p(np.maximum(sub_aligned["Cardinality"].astype(float).values, 1.0)))

    if not names:
        raise RuntimeError("no experts were loaded")
    return names, np.column_stack(oof_logs), np.column_stack(test_logs)


def qerror_matrix(oof_logs: np.ndarray, y_log: np.ndarray) -> np.ndarray:
    pred_card = np.maximum(np.expm1(oof_logs), 1.0)
    true_card = np.maximum(np.expm1(y_log), 1.0)
    return np.maximum(pred_card / true_card[:, None], true_card[:, None] / pred_card)


def group_indices(keys: np.ndarray, indices: np.ndarray) -> Dict[str, np.ndarray]:
    grouped: Dict[str, List[int]] = {}
    for idx in indices:
        grouped.setdefault(str(keys[idx]), []).append(int(idx))
    return {key: np.asarray(values, dtype=np.int32) for key, values in grouped.items()}


def choose_default(qmat: np.ndarray, fit_idx: np.ndarray, allowed: np.ndarray) -> int:
    means = qmat[np.ix_(fit_idx, allowed)].mean(axis=0)
    return int(allowed[int(np.argmin(means))])


def choose_group_expert(
    qmat: np.ndarray,
    idx: np.ndarray,
    allowed: np.ndarray,
    default_expert: int,
    margin: float,
) -> int:
    means = qmat[np.ix_(idx, allowed)].mean(axis=0)
    best = int(allowed[int(np.argmin(means))])
    default_score = float(qmat[idx, default_expert].mean())
    return best if float(means.min()) + margin < default_score else default_expert


def crossfit_gate(
    oof_logs: np.ndarray,
    qmat: np.ndarray,
    key_levels: Sequence[np.ndarray],
    allowed: np.ndarray,
    *,
    min_groups: Sequence[int],
    margins: Sequence[float],
) -> Tuple[np.ndarray, List[Tuple[str, str, int, int]]]:
    pred = np.zeros(oof_logs.shape[0], dtype=np.float64)
    choices: List[Tuple[str, str, int, int]] = []
    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for fold, (fit_idx, valid_idx) in enumerate(folds.split(oof_logs), start=1):
        default = choose_default(qmat, fit_idx, allowed)
        pred[valid_idx] = oof_logs[valid_idx, default]

        for keys, min_group, margin in zip(key_levels, min_groups, margins):
            fit_groups = group_indices(keys, fit_idx)
            valid_groups = group_indices(keys, valid_idx)
            for key, valid_group_idx in valid_groups.items():
                fit_group_idx = fit_groups.get(key)
                if fit_group_idx is None or len(fit_group_idx) < min_group:
                    continue
                chosen = choose_group_expert(qmat, fit_group_idx, allowed, default, margin)
                if chosen != default:
                    pred[valid_group_idx] = oof_logs[valid_group_idx, chosen]
                    choices.append((key, str(chosen), fold, len(fit_group_idx)))

    return pred, choices


def fit_full_gate(
    test_logs: np.ndarray,
    qmat: np.ndarray,
    train_key_levels: Sequence[np.ndarray],
    test_key_levels: Sequence[np.ndarray],
    allowed: np.ndarray,
    *,
    min_groups: Sequence[int],
    margins: Sequence[float],
) -> Tuple[np.ndarray, Dict[str, int]]:
    all_idx = np.arange(qmat.shape[0], dtype=np.int32)
    default = choose_default(qmat, all_idx, allowed)
    pred_test = test_logs[:, default].copy()
    choices: Dict[str, int] = {}

    for train_keys, test_keys, min_group, margin in zip(train_key_levels, test_key_levels, min_groups, margins):
        train_groups = group_indices(train_keys, all_idx)
        test_groups = group_indices(test_keys, np.arange(len(test_keys), dtype=np.int32))
        for key, test_idx in test_groups.items():
            train_idx = train_groups.get(key)
            if train_idx is None or len(train_idx) < min_group:
                continue
            chosen = choose_group_expert(qmat, train_idx, allowed, default, margin)
            if chosen != default:
                pred_test[test_idx] = test_logs[test_idx, chosen]
                choices[key] = chosen
    return pred_test, choices


def expert_summary(names: Sequence[str], qmat: np.ndarray) -> None:
    print("individual experts:", flush=True)
    for idx, name in sorted(enumerate(names), key=lambda item: qmat[:, item[0]].mean()):
        q = qmat[:, idx]
        print(
            f"  {name:10s} mean={q.mean():.6f} med={np.median(q):.4f} "
            f"p95={np.percentile(q, 95):.4f}",
            flush=True,
        )


def fit_affine(pred_log: np.ndarray, y_log: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
    def transform(z: np.ndarray, values: np.ndarray) -> np.ndarray:
        alpha = 0.9 + 0.2 / (1.0 + np.exp(-z[0]))
        beta = 0.3 * np.tanh(z[1])
        return alpha * values + beta

    def objective(z: np.ndarray) -> float:
        return float(qerror_from_logs(transform(z, pred_log), y_log).mean())

    result = differential_evolution(
        objective,
        [(-7.0, 7.0), (-7.0, 7.0)],
        seed=SEED + 404,
        maxiter=100,
        tol=1e-9,
        polish=True,
        workers=1,
        updating="immediate",
    )
    alpha = 0.9 + 0.2 / (1.0 + np.exp(-result.x[0]))
    beta = 0.3 * np.tanh(result.x[1])
    return alpha * pred_log + beta, float(alpha), float(beta), float(result.fun)


def save_variant(
    variant_name: str,
    config: dict,
    names: Sequence[str],
    oof_logs: np.ndarray,
    test_logs: np.ndarray,
    qmat: np.ndarray,
    train: pd.DataFrame,
    test: pd.DataFrame,
    key_sets: Dict[str, Tuple[np.ndarray, np.ndarray]],
    y_log: np.ndarray,
) -> None:
    allowed = np.asarray([names.index(name) for name in config["allowed"] if name in names], dtype=np.int32)
    if len(allowed) == 0:
        raise RuntimeError(f"{variant_name} has no available experts")
    level_names = tuple(config.get("levels", ("column_family",)))
    train_key_levels = [key_sets[name][0] for name in level_names]
    test_key_levels = [key_sets[name][1] for name in level_names]
    if "min_groups" in config:
        min_groups = tuple(int(value) for value in config["min_groups"])
    else:
        min_groups = (int(config["min_group"]),) * len(level_names)
    if "margins" in config:
        margins = tuple(float(value) for value in config["margins"])
    else:
        margins = (float(config["margin"]),) * len(level_names)
    if len(min_groups) != len(level_names) or len(margins) != len(level_names):
        raise ValueError(f"{variant_name} has mismatched levels/min_groups/margins")

    pred_oof, fold_choices = crossfit_gate(
        oof_logs,
        qmat,
        train_key_levels,
        allowed,
        min_groups=min_groups,
        margins=margins,
    )
    pred_test, test_choices = fit_full_gate(
        test_logs,
        qmat,
        train_key_levels,
        test_key_levels,
        allowed,
        min_groups=min_groups,
        margins=margins,
    )

    if config.get("affine", False):
        pred_oof, alpha, beta, affine_score = fit_affine(pred_oof, y_log)
        pred_test = alpha * pred_test + beta
        print(
            f"{variant_name} affine alpha={alpha:.8f} beta={beta:.8f} "
            f"score={affine_score:.6f}",
            flush=True,
        )

    q = qerror_from_logs(pred_oof, y_log)
    print(
        f"{variant_name}: mean={q.mean():.6f} med={np.median(q):.4f} "
        f"p95={np.percentile(q, 95):.4f} levels={level_names} "
        f"min_groups={min_groups} margins={margins} "
        f"fold_overrides={len(fold_choices)} test_groups={len(test_choices)}",
        flush=True,
    )

    choice_counts = pd.Series([names[idx] for idx in test_choices.values()]).value_counts()
    if not choice_counts.empty:
        print(f"{variant_name} test override experts:", choice_counts.to_dict(), flush=True)

    pd.DataFrame(
        {
            "Id": train["Id"].values,
            "Cardinality": train["Cardinality"].values,
            "PredLog": pred_oof,
            "QError": q,
        }
    ).to_csv(ROOT / str(config["oof_file"]), index=False)

    card = np.rint(np.maximum(np.expm1(pred_test), 1.0)).astype(np.int64)
    pd.DataFrame({"Id": test["Id"].values, "Cardinality": card}).to_csv(
        ROOT / str(config["sub_file"]),
        index=False,
    )
    print(
        f"saved {config['sub_file']} range=[{card.min()}, {card.max()}] mean={card.mean():.1f}",
        flush=True,
    )


def main() -> None:
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    y_log = np.log1p(np.maximum(train["Cardinality"].astype(float).values, 1.0))

    names, oof_logs, test_logs = load_experts(train, test, EXPERTS)
    qmat = qerror_matrix(oof_logs, y_log)
    key_sets = {
        "column_family": (column_family_keys(train), column_family_keys(test)),
        "exact_shape": (exact_shape_keys(train), exact_shape_keys(test)),
    }

    expert_summary(names, qmat)
    train_keys, test_keys = key_sets["column_family"]
    print(
        f"families: train={len(np.unique(train_keys))} test={len(np.unique(test_keys))} "
        f"covered_test_rows={np.isin(test_keys, train_keys).sum()}/{len(test_keys)}",
        flush=True,
    )
    train_shapes, test_shapes = key_sets["exact_shape"]
    print(
        f"shapes: train={len(np.unique(train_shapes))} test={len(np.unique(test_shapes))} "
        f"covered_test_rows={np.isin(test_shapes, train_shapes).sum()}/{len(test_shapes)}",
        flush=True,
    )

    for variant_name, config in VARIANTS.items():
        save_variant(
            variant_name,
            config,
            names,
            oof_logs,
            test_logs,
            qmat,
            train,
            test,
            key_sets,
            y_log,
        )


if __name__ == "__main__":
    main()
