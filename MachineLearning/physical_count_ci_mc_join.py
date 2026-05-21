import bisect
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "job_data" / "partial_extract"
BASE_SUB = ROOT / "submission_shape_surface_m50_d2_a10_s0p3_g0p0.csv"
BASE_OOF = ROOT / "shape_surface_m50_d2_a10_s0p3_g0p0_oof.csv"
SINGLE_SUB = ROOT / "submission_physical_single_ci_mc.csv"


def qerror(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.maximum(np.asarray(y_pred, dtype=np.float64), 1.0)
    return np.maximum(y_true / y_pred, y_pred / y_true)


def parse_tables(s):
    return set(part.strip().split()[-1] for part in str(s).split(",") if part.strip())


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
    return out


def key_for(preds, alias):
    return tuple((col, op, val) for a, col, op, val in preds if a == alias)


def supported_join(row):
    aliases = parse_tables(row["Tables"])
    if aliases != {"t", "ci", "mc"}:
        return None
    preds = parse_predicates(row.get("Predicates", ""))
    for alias, col, _op, _val in preds:
        if alias == "t":
            return None
        if alias == "ci" and col not in {"person_id", "role_id"}:
            return None
        if alias == "mc" and col not in {"company_id", "company_type_id"}:
            return None
        if alias not in {"ci", "mc"}:
            return None
    return key_for(preds, "ci"), key_for(preds, "mc")


def match(vals, key):
    for col, op, val in key:
        got = vals[col]
        if op == "=":
            if got != val:
                return False
        elif op == "<":
            if got >= val:
                return False
        elif op == ">":
            if got <= val:
                return False
        else:
            raise ValueError(op)
    return True


def build_movie_counts(alias, keys):
    keys = list(keys)
    accum = {key: {} for key in keys}
    if alias == "ci":
        path = DATA / "cast_info.csv"
        with path.open("rb") as f:
            for line_no, line in enumerate(f, 1):
                c1 = line.find(b",")
                c2 = line.find(b",", c1 + 1)
                c3 = line.find(b",", c2 + 1)
                person_id = int(line[c1 + 1 : c2])
                movie_id = int(line[c2 + 1 : c3])
                last = line.rfind(b",")
                role_id = int(line[last + 1 :].strip())
                vals = {"person_id": person_id, "role_id": role_id}
                for key in keys:
                    if match(vals, key):
                        d = accum[key]
                        d[movie_id] = d.get(movie_id, 0) + 1
                if line_no % 5_000_000 == 0:
                    print(f"{alias} scanned {line_no}", flush=True)
    elif alias == "mc":
        path = DATA / "movie_companies.csv"
        with path.open("rb") as f:
            for line_no, line in enumerate(f, 1):
                c1 = line.find(b",")
                c2 = line.find(b",", c1 + 1)
                c3 = line.find(b",", c2 + 1)
                c4 = line.find(b",", c3 + 1)
                movie_id = int(line[c1 + 1 : c2])
                company_id = int(line[c2 + 1 : c3])
                company_type_id = int(line[c3 + 1 : c4])
                vals = {"company_id": company_id, "company_type_id": company_type_id}
                for key in keys:
                    if match(vals, key):
                        d = accum[key]
                        d[movie_id] = d.get(movie_id, 0) + 1
                if line_no % 1_000_000 == 0:
                    print(f"{alias} scanned {line_no}", flush=True)
    else:
        raise ValueError(alias)

    out = {}
    for key, d in accum.items():
        items = sorted(d.items())
        mids = [m for m, _c in items]
        counts = [c for _m, c in items]
        out[key] = (mids, counts)
    print(f"{alias} built movie-count vectors for {len(keys)} keys", flush=True)
    return out


def dot_join(left, right):
    lmids, lcnt = left
    rmids, rcnt = right
    total = 0
    i = j = 0
    while i < len(lmids) and j < len(rmids):
        a = lmids[i]
        b = rmids[j]
        if a == b:
            total += lcnt[i] * rcnt[j]
            i += 1
            j += 1
        elif a < b:
            i += 1
        else:
            j += 1
    return total


def main():
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    needed = {"ci": set(), "mc": set()}
    rows = []
    for split, df in (("train", train), ("test", test)):
        for idx, row in df.iterrows():
            supported = supported_join(row)
            if supported is None:
                continue
            ci_key, mc_key = supported
            needed["ci"].add(ci_key)
            needed["mc"].add(mc_key)
            rows.append((split, idx, ci_key, mc_key))
    print({k: len(v) for k, v in needed.items()}, "rows", len(rows), flush=True)

    ci_counts = build_movie_counts("ci", needed["ci"])
    mc_counts = build_movie_counts("mc", needed["mc"])

    train_counts = {}
    test_counts = {}
    for split, idx, ci_key, mc_key in rows:
        count = dot_join(ci_counts[ci_key], mc_counts[mc_key])
        if split == "train":
            train_counts[idx] = count
        else:
            test_counts[idx] = count

    base_oof_raw = pd.read_csv(BASE_OOF)
    base_pred = np.exp(base_oof_raw["PredLog"].astype(float).to_numpy())
    single_sub_exists = SINGLE_SUB.exists()
    idxs = np.array(sorted(train_counts), dtype=np.int64)
    exact = np.array([train_counts[int(i)] for i in idxs], dtype=np.float64)
    y = train.loc[idxs, "Cardinality"].to_numpy(dtype=np.float64)
    print(f"join train rows {len(idxs)}, test rows {len(test_counts)}")
    print(f"join exact qerror {qerror(y, exact).mean():.8f}")
    print(f"join base qerror {qerror(y, base_pred[idxs]).mean():.8f}")

    hybrid = base_pred.copy()
    for idx, count in train_counts.items():
        hybrid[idx] = max(1.0, count)
    print(f"full train with join-only qerror {qerror(train['Cardinality'], hybrid).mean():.8f}")

    if single_sub_exists:
        sub = pd.read_csv(SINGLE_SUB)
    else:
        sub = pd.read_csv(BASE_SUB)
    pred = sub["Cardinality"].astype(float).to_numpy(copy=True)
    for idx, count in test_counts.items():
        pred[idx] = max(1.0, count)
    sub["Cardinality"] = np.maximum(1, np.rint(pred)).astype(np.int64)
    out = ROOT / "submission_physical_single_join_ci_mc.csv"
    sub.to_csv(out, index=False)
    print(f"wrote {out.name}, changed/additional join rows {len(test_counts)}", flush=True)


if __name__ == "__main__":
    main()
