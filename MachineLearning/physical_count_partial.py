import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "job_data" / "partial_extract"
BASE_SUB = ROOT / "submission_shape_surface_m50_d2_a10_s0p3_g0p0.csv"
BASE_OOF = ROOT / "shape_surface_m50_d2_a10_s0p3_g0p0_oof.csv"


TABLE_FILES = {
    "ci": DATA / "cast_info.csv",
    "mc": DATA / "movie_companies.csv",
}

COL_INDEX = {
    "ci": {"person_id": 1, "movie_id": 2, "role_id": 6},
    "mc": {"movie_id": 1, "company_id": 2, "company_type_id": 3},
}

PRED_COLS = {
    "ci": ["person_id", "role_id"],
    "mc": ["company_id", "company_type_id"],
}


def qerror(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.maximum(np.asarray(y_pred, dtype=np.float64), 1.0)
    return np.maximum(y_true / y_pred, y_pred / y_true)


def parse_tables(s):
    if not isinstance(s, str) or not s:
        return []
    aliases = []
    for part in s.split(","):
        bits = part.strip().split()
        if bits:
            aliases.append(bits[-1])
    return aliases


def parse_predicates(s):
    if not isinstance(s, str) or not s or pd.isna(s):
        return []
    toks = s.split(",")
    preds = []
    for i in range(0, len(toks), 3):
        if i + 2 >= len(toks):
            continue
        full_col, op, val = toks[i], toks[i + 1], toks[i + 2]
        alias, col = full_col.split(".", 1)
        preds.append((alias, col, op, int(val)))
    return preds


def pred_key(preds, alias):
    return tuple((col, op, val) for a, col, op, val in preds if a == alias)


def supported_query(row):
    aliases = set(parse_tables(row["Tables"]))
    preds = parse_predicates(row.get("Predicates", ""))
    if not aliases <= {"t", "ci", "mc"}:
        return False
    for alias, col, _op, _val in preds:
        if alias == "t":
            return False
        if alias not in {"ci", "mc"} or col not in PRED_COLS[alias]:
            return False
    return any(a in aliases for a in ("ci", "mc"))


def _count_lines(path):
    n = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            n += chunk.count(b"\n")
    return n


def read_table(alias):
    path = TABLE_FILES[alias]
    n = _count_lines(path)
    arrays = {name: np.empty(n, dtype=np.int32) for name in COL_INDEX[alias]}
    with path.open("rb") as f:
        for i, line in enumerate(f):
            parts = line.rstrip(b"\r\n").split(b",")
            if alias == "ci":
                arrays["person_id"][i] = int(parts[1])
                arrays["movie_id"][i] = int(parts[2])
                arrays["role_id"][i] = int(parts[-1])
            elif alias == "mc":
                arrays["movie_id"][i] = int(parts[1])
                arrays["company_id"][i] = int(parts[2])
                arrays["company_type_id"][i] = int(parts[3])
            else:
                raise ValueError(alias)
    return pd.DataFrame(arrays)


def apply_predicates(df, key):
    mask = np.ones(len(df), dtype=bool)
    for col, op, val in key:
        arr = df[col].to_numpy()
        if op == "=":
            mask &= arr == val
        elif op == "<":
            mask &= arr < val
        elif op == ">":
            mask &= arr > val
        else:
            raise ValueError(f"unsupported op {op}")
    return mask


def build_exact_counts(rows, table_dfs):
    need_keys = {alias: set() for alias in table_dfs}
    for _idx, row in rows.iterrows():
        aliases = set(parse_tables(row["Tables"]))
        preds = parse_predicates(row.get("Predicates", ""))
        for alias in aliases:
            if alias in need_keys:
                need_keys[alias].add(pred_key(preds, alias))

    single_counts = {alias: {} for alias in table_dfs}
    movie_counts = {alias: {} for alias in table_dfs}
    for alias, df in table_dfs.items():
        movie_id = df["movie_id"].to_numpy()
        for key in sorted(need_keys[alias]):
            mask = apply_predicates(df, key)
            single_counts[alias][key] = int(mask.sum())
            mids, counts = np.unique(movie_id[mask], return_counts=True)
            movie_counts[alias][key] = (mids.astype(np.int32), counts.astype(np.int32))

    out = {}
    for idx, row in rows.iterrows():
        aliases = set(parse_tables(row["Tables"]))
        preds = parse_predicates(row.get("Predicates", ""))
        child_aliases = [a for a in ("ci", "mc") if a in aliases]
        if len(child_aliases) == 1 and "t" not in aliases:
            alias = child_aliases[0]
            out[idx] = single_counts[alias][pred_key(preds, alias)]
            continue
        if set(child_aliases) == {"ci", "mc"}:
            ci_mids, ci_counts = movie_counts["ci"][pred_key(preds, "ci")]
            mc_mids, mc_counts = movie_counts["mc"][pred_key(preds, "mc")]
            common, ci_idx, mc_idx = np.intersect1d(ci_mids, mc_mids, assume_unique=True, return_indices=True)
            del common
            out[idx] = int(np.dot(ci_counts[ci_idx].astype(np.int64), mc_counts[mc_idx].astype(np.int64)))
            continue
        # title joined with one child and no title predicate means one title row per movie_id.
        if len(child_aliases) == 1:
            alias = child_aliases[0]
            out[idx] = single_counts[alias][pred_key(preds, alias)]
            continue
        raise RuntimeError(f"unsupported row after filter: {row.to_dict()}")
    return out


def optimize_blend(train, exact_by_idx, base_oof):
    idxs = np.array(sorted(exact_by_idx), dtype=np.int64)
    exact = np.array([exact_by_idx[int(i)] for i in idxs], dtype=np.float64)
    y = train.loc[idxs, "Cardinality"].to_numpy(dtype=np.float64)
    base_map = base_oof.set_index("Id")["Cardinality"].astype(float)
    base = train.loc[idxs, "Id"].map(base_map).to_numpy(dtype=np.float64)

    print(f"supported rows: {len(idxs)}")
    print(f"exact-only mean qerror: {qerror(y, exact).mean():.6f}")
    print(f"base on supported mean qerror: {qerror(y, base).mean():.6f}")
    print(f"exact/base median ratio: {np.median(exact / np.maximum(base, 1)):.6f}")

    best = None
    for scale in np.linspace(0.75, 1.25, 101):
        scaled = np.maximum(1.0, exact * scale)
        qe = qerror(y, scaled).mean()
        if best is None or qe < best[0]:
            best = (qe, float(scale), 1.0, 0.0)
        for w in np.linspace(0.1, 0.9, 17):
            blend = np.exp(w * np.log(scaled) + (1.0 - w) * np.log(np.maximum(base, 1.0)))
            qe = qerror(y, blend).mean()
            if qe < best[0]:
                best = (qe, float(scale), float(w), 1.0 - float(w))
    print(f"best supported qerror: {best[0]:.6f} scale={best[1]:.4f} exact_w={best[2]:.3f} base_w={best[3]:.3f}")
    return best


def apply_submission(test, exact_by_idx, best):
    base = pd.read_csv(BASE_SUB)
    pred = base["Cardinality"].astype(float).to_numpy()
    scale, exact_w, base_w = best[1], best[2], best[3]
    changed = 0
    for idx, exact in exact_by_idx.items():
        exact_scaled = max(1.0, float(exact) * scale)
        if base_w == 0:
            new_pred = exact_scaled
        else:
            new_pred = math.exp(exact_w * math.log(exact_scaled) + base_w * math.log(max(1.0, pred[idx])))
        pred[idx] = new_pred
        changed += 1
    out = base.copy()
    out["Cardinality"] = np.maximum(1, np.rint(pred)).astype(np.int64)
    path = ROOT / "submission_physical_partial_ci_mc.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path.name}, changed {changed} rows")


def main():
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    table_dfs = {alias: read_table(alias) for alias in TABLE_FILES}
    print({alias: len(df) for alias, df in table_dfs.items()})

    train_supported = train[train.apply(supported_query, axis=1)]
    test_supported = test[test.apply(supported_query, axis=1)]
    train_exact = build_exact_counts(train_supported, table_dfs)
    test_exact = build_exact_counts(test_supported, table_dfs)

    base_oof = pd.read_csv(BASE_OOF)
    best = optimize_blend(train, train_exact, base_oof)

    base_pred = pd.read_csv(BASE_OOF).set_index("Id")["Cardinality"].astype(float)
    blended = base_pred.reindex(train["Id"]).to_numpy()
    for idx, exact in train_exact.items():
        scaled = max(1.0, exact * best[1])
        if best[3] == 0:
            blended[idx] = scaled
        else:
            blended[idx] = math.exp(best[2] * math.log(scaled) + best[3] * math.log(max(1.0, blended[idx])))
    print(f"full train hybrid qerror: {qerror(train['Cardinality'], blended).mean():.6f}")

    apply_submission(test, test_exact, best)


if __name__ == "__main__":
    main()
