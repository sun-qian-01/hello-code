import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


BASES = [
    ("shape_surface_m50_d2_a10_s0p3_g0p0", "shape_surface_m50_d2_a10_s0p3_g0p0_oof.csv", "submission_shape_surface_m50_d2_a10_s0p3_g0p0.csv"),
    ("shape_surface_v2_gate_g0p01", "shape_surface_v2_gate_g0p01_oof.csv", "submission_shape_surface_v2_gate_g0p01.csv"),
    ("shape_surface_v2_gate_g0p005", "shape_surface_v2_gate_g0p005_oof.csv", "submission_shape_surface_v2_gate_g0p005_min2.csv"),
    ("shape_surface_v2_gate_g0p0", "shape_surface_v2_gate_g0p0_oof.csv", "submission_shape_surface_v2_gate_g0p0_min2.csv"),
]


def qerror(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.maximum(np.asarray(y_pred, dtype=np.float64), 1.0)
    return np.maximum(y_true / y_pred, y_pred / y_true)


def load_physical_counts(train, test):
    spec = importlib.util.spec_from_file_location("pcs", ROOT / "physical_count_single.py")
    pcs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pcs)
    counter = pcs.SingleCounter()
    train_counts = pcs.compute_counts(train, counter)
    test_counts = pcs.compute_counts(test, counter)
    return train_counts, test_counts


def main():
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    train_counts, test_counts = load_physical_counts(train, test)
    y = train["Cardinality"].astype(float).to_numpy()

    for name, oof_file, sub_file in BASES:
        oof_path = ROOT / oof_file
        sub_path = ROOT / sub_file
        if not oof_path.exists() or not sub_path.exists():
            print("missing", name, oof_path.exists(), sub_path.exists())
            continue
        oof = pd.read_csv(oof_path)
        pred = np.exp(oof["PredLog"].astype(float).to_numpy())
        base_q = qerror(y, pred).mean()
        for idx, count in train_counts.items():
            pred[idx] = max(1.0, count)
        hybrid_q = qerror(y, pred).mean()

        sub = pd.read_csv(sub_path)
        sub_pred = sub["Cardinality"].astype(float).to_numpy(copy=True)
        for idx, count in test_counts.items():
            sub_pred[idx] = max(1.0, count)
        out = sub.copy()
        out["Cardinality"] = np.maximum(1, np.rint(sub_pred)).astype(np.int64)
        out_path = ROOT / f"submission_physical_single_on_{name}.csv"
        out.to_csv(out_path, index=False)

        print(
            name,
            f"base_oof={base_q:.8f}",
            f"physical_oof={hybrid_q:.8f}",
            f"delta={base_q - hybrid_q:.8f}",
            f"changed={len(test_counts)}",
            f"out={out_path.name}",
            flush=True,
        )


if __name__ == "__main__":
    main()
