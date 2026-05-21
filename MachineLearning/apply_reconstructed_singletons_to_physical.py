import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
BASE_OOF = ROOT / "shape_surface_m50_d2_a10_s0p3_g0p0_oof.csv"
BASE_SUBS = [
    ("physical_single_base", ROOT / "submission_physical_single_on_shape_surface_m50_d2_a10_s0p3_g0p0.csv"),
    ("physical_single_v2_g01", ROOT / "submission_physical_single_on_shape_surface_v2_gate_g0p01.csv"),
    ("physical_single_v2_g005", ROOT / "submission_physical_single_on_shape_surface_v2_gate_g0p005.csv"),
    ("physical_single_v2_g0", ROOT / "submission_physical_single_on_shape_surface_v2_gate_g0p0.csv"),
]


def qerror(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.maximum(np.asarray(y_pred, dtype=np.float64), 1.0)
    return np.maximum(y_true / y_pred, y_pred / y_true)


def load_modules():
    ps = importlib.util.spec_from_file_location("pcs", ROOT / "physical_count_single.py")
    pcs = importlib.util.module_from_spec(ps)
    ps.loader.exec_module(pcs)
    rs = importlib.util.spec_from_file_location("recon", ROOT / "single_table_distribution_reconstruct.py")
    recon = importlib.util.module_from_spec(rs)
    rs.loader.exec_module(recon)
    return pcs, recon


def main():
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    pcs, recon = load_modules()

    counter = pcs.SingleCounter()
    physical_train = pcs.compute_counts(train, counter)
    physical_test = pcs.compute_counts(test, counter)

    oracles = recon.build_oracles(train)
    recon_train_all = recon.predict_supported(train, oracles)
    recon_test_all = recon.predict_supported(test, oracles)

    col_q = {}
    for key in oracles:
        rows = []
        vals = []
        for idx, row in train.iterrows():
            sq = recon.single_col_query(row)
            if sq is None or sq[:2] != key:
                continue
            if idx in recon_train_all:
                rows.append(idx)
                vals.append(recon_train_all[idx])
        if rows:
            col_q[key] = qerror(train.loc[rows, "Cardinality"], vals).mean()
    safe_keys = {key for key, score in col_q.items() if score <= 1.02}

    recon_train = {}
    for idx, val in recon_train_all.items():
        sq = recon.single_col_query(train.loc[idx])
        if sq is not None and sq[:2] in safe_keys:
            recon_train[idx] = val
    recon_test = {}
    for idx, val in recon_test_all.items():
        sq = recon.single_col_query(test.loc[idx])
        if sq is not None and sq[:2] in safe_keys:
            recon_test[idx] = val

    base_oof = pd.read_csv(BASE_OOF)
    pred = np.exp(base_oof["PredLog"].astype(float).to_numpy())
    y = train["Cardinality"].astype(float).to_numpy()
    base_q = qerror(y, pred).mean()
    for idx, val in recon_train.items():
        pred[idx] = max(1.0, val)
    for idx, val in physical_train.items():
        pred[idx] = max(1.0, val)
    hybrid_q = qerror(y, pred).mean()
    print(
        f"base_oof={base_q:.8f}",
        f"combined_oof={hybrid_q:.8f}",
        f"physical_train={len(physical_train)}",
        f"recon_train={len(recon_train)}",
        f"union_train={len(set(physical_train) | set(recon_train))}",
        f"physical_test={len(physical_test)}",
        f"recon_test={len(recon_test)}",
        f"union_test={len(set(physical_test) | set(recon_test))}",
    )

    for name, sub_path in BASE_SUBS:
        if not sub_path.exists():
            continue
        sub = pd.read_csv(sub_path)
        sub_pred = sub["Cardinality"].astype(float).to_numpy(copy=True)
        changed_extra = 0
        for idx, val in recon_test.items():
            old = sub_pred[idx]
            sub_pred[idx] = max(1.0, val)
            if old != sub_pred[idx]:
                changed_extra += 1
        sub["Cardinality"] = np.maximum(1, np.rint(sub_pred)).astype(np.int64)
        out = ROOT / f"submission_{name}_plus_reconstructed_singletons.csv"
        sub.to_csv(out, index=False)
        print("wrote", out.name, "extra_changed", changed_extra)


if __name__ == "__main__":
    main()
