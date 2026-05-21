from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "job_data" / "partial_extract"
BASE_SUB = ROOT / "submission_shape_surface_m50_d2_a10_s0p3_g0p0.csv"
BASE_OOF = ROOT / "shape_surface_m50_d2_a10_s0p3_g0p0_oof.csv"


def qerror(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.maximum(np.asarray(y_pred, dtype=np.float64), 1.0)
    return np.maximum(y_true / y_pred, y_pred / y_true)


def parse_tables(s):
    return [part.strip().split()[-1] for part in str(s).split(",") if part.strip()]


def parse_predicates(s):
    if not isinstance(s, str) or not s or pd.isna(s):
        return []
    toks = s.split(",")
    out = []
    for i in range(0, len(toks), 3):
        if i + 2 >= len(toks):
            continue
        alias, col = toks[i].split(".", 1)
        out.append((alias, col, toks[i + 1], int(toks[i + 2])))
    return tuple(out)


def supported_single(row):
    aliases = set(parse_tables(row["Tables"]))
    if aliases not in ({"ci"}, {"mc"}, {"t", "ci"}, {"t", "mc"}):
        return None
    child = "ci" if "ci" in aliases else "mc"
    preds = parse_predicates(row.get("Predicates", ""))
    child_preds = []
    for alias, col, op, val in preds:
        if alias == "t":
            return None
        if child == "ci" and (alias != "ci" or col not in {"person_id", "role_id"}):
            return None
        if child == "mc" and (alias != "mc" or col not in {"company_id", "company_type_id"}):
            return None
        child_preds.append((col, op, val))
    return child, tuple(child_preds)


def bounds_from_key(key, high_col, low_col, low_values):
    lo = -10**18
    hi = 10**18
    allowed = set(low_values)
    for col, op, val in key:
        if col == high_col:
            if op == "=":
                lo = max(lo, val)
                hi = min(hi, val + 1)
            elif op == "<":
                hi = min(hi, val)
            elif op == ">":
                lo = max(lo, val + 1)
            else:
                raise ValueError(op)
        elif col == low_col:
            if op == "=":
                allowed &= {val}
            elif op == "<":
                allowed &= {x for x in low_values if x < val}
            elif op == ">":
                allowed &= {x for x in low_values if x > val}
            else:
                raise ValueError(op)
        else:
            raise ValueError(col)
    return lo, hi, sorted(allowed)


def count_lines(path):
    total = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            total += chunk.count(b"\n")
    return total


def build_needed(train, test):
    needed = {"ci": set(), "mc": set()}
    rows = []
    for split, df in (("train", train), ("test", test)):
        for idx, row in df.iterrows():
            supported = supported_single(row)
            if supported is None:
                continue
            alias, key = supported
            needed[alias].add(key)
            rows.append((split, idx, alias, key))
    return needed, rows


def scan_alias(alias, keys):
    if alias == "ci":
        path = DATA / "cast_info.csv"
        n = count_lines(path)
        high = np.empty(n, dtype=np.int32)
        low = np.empty(n, dtype=np.int16)
        with path.open("rb") as f:
            for line_no, line in enumerate(f, 1):
                c1 = line.find(b",")
                c2 = line.find(b",", c1 + 1)
                high[line_no - 1] = int(line[c1 + 1 : c2])
                last = line.rfind(b",")
                low[line_no - 1] = int(line[last + 1 :].strip())
        buckets = {v: np.sort(high[low == v]) for v in range(1, 12)}
        high_col, low_col, low_values = "person_id", "role_id", range(1, 12)
    elif alias == "mc":
        path = DATA / "movie_companies.csv"
        n = count_lines(path)
        high = np.empty(n, dtype=np.int32)
        low = np.empty(n, dtype=np.int16)
        with path.open("rb") as f:
            for line_no, line in enumerate(f, 1):
                c1 = line.find(b",")
                c2 = line.find(b",", c1 + 1)
                c3 = line.find(b",", c2 + 1)
                c4 = line.find(b",", c3 + 1)
                high[line_no - 1] = int(line[c2 + 1 : c3])
                low[line_no - 1] = int(line[c3 + 1 : c4])
        buckets = {v: np.sort(high[low == v]) for v in range(1, 3)}
        high_col, low_col, low_values = "company_id", "company_type_id", range(1, 3)
    else:
        raise ValueError(alias)
    counts = {}
    for key in keys:
        lo, hi, allowed = bounds_from_key(key, high_col, low_col, low_values)
        if lo >= hi or not allowed:
            counts[key] = 0
            continue
        total = 0
        for v in allowed:
            arr = buckets[v]
            total += int(np.searchsorted(arr, hi, side="left") - np.searchsorted(arr, lo, side="left"))
        counts[key] = total
    print(f"{alias} counted {len(keys)} predicate keys from {n} rows")
    return counts


def main():
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    needed, rows = build_needed(train, test)
    print({k: len(v) for k, v in needed.items()})
    counts_by_alias = {alias: scan_alias(alias, sorted(keys)) for alias, keys in needed.items()}

    train_counts = {}
    test_counts = {}
    for split, idx, alias, key in rows:
        if split == "train":
            train_counts[idx] = counts_by_alias[alias][key]
        else:
            test_counts[idx] = counts_by_alias[alias][key]

    base_oof_raw = pd.read_csv(BASE_OOF)
    base_oof_raw["BasePred"] = np.exp(base_oof_raw["PredLog"].astype(float))
    base_oof = base_oof_raw.set_index("Id")["BasePred"].astype(float)
    base_train = train["Id"].map(base_oof).to_numpy(dtype=np.float64)
    idxs = np.array(sorted(train_counts), dtype=np.int64)
    exact = np.array([train_counts[int(i)] for i in idxs], dtype=np.float64)
    y = train.loc[idxs, "Cardinality"].to_numpy(dtype=np.float64)
    base = base_train[idxs]
    print(f"supported train rows {len(idxs)}, test rows {len(test_counts)}")
    print(f"exact supported qerror {qerror(y, exact).mean():.8f}")
    print(f"base supported qerror {qerror(y, base).mean():.8f}")

    hybrid = base_train.copy()
    for idx, count in train_counts.items():
        hybrid[idx] = max(1.0, count)
    print(f"full train hybrid qerror {qerror(train['Cardinality'], hybrid).mean():.8f}")

    sub = pd.read_csv(BASE_SUB)
    pred = sub["Cardinality"].astype(float).to_numpy(copy=True)
    for idx, count in test_counts.items():
        pred[idx] = max(1.0, count)
    sub["Cardinality"] = np.maximum(1, np.rint(pred)).astype(np.int64)
    out = ROOT / "submission_physical_stream_single_ci_mc.csv"
    sub.to_csv(out, index=False)
    print(f"wrote {out.name}, changed {len(test_counts)} rows")


if __name__ == "__main__":
    main()
