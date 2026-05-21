from pathlib import Path
import bisect

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "job_data" / "partial_extract"
CACHE = ROOT / "job_data" / "cache"
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
    return out


def bounds_from_preds(preds, high_col, low_col, low_values):
    lo = -10**18
    hi = 10**18
    allowed = set(low_values)
    for _alias, col, op, val in preds:
        if col == high_col:
            if op == "=":
                lo = max(lo, val)
                hi = min(hi, val + 1)
            elif op == ">":
                lo = max(lo, val + 1)
            elif op == "<":
                hi = min(hi, val)
            else:
                raise ValueError(op)
        elif col == low_col:
            if op == "=":
                allowed &= {val}
            elif op == ">":
                allowed &= {x for x in low_values if x > val}
            elif op == "<":
                allowed &= {x for x in low_values if x < val}
            else:
                raise ValueError(op)
        else:
            raise ValueError(col)
    return int(lo), int(hi), sorted(allowed)


def count_lines(path):
    total = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            total += chunk.count(b"\n")
    return total


def ensure_ci_cache():
    CACHE.mkdir(parents=True, exist_ok=True)
    done = CACHE / "ci_single_done.txt"
    if done.exists():
        return
    path = DATA / "cast_info.csv"
    n = count_lines(path)
    person = np.empty(n, dtype=np.int32)
    role = np.empty(n, dtype=np.int16)
    with path.open("rb") as f:
        for i, line in enumerate(f):
            c1 = line.find(b",")
            c2 = line.find(b",", c1 + 1)
            person[i] = int(line[c1 + 1 : c2])
            last = line.rfind(b",")
            role[i] = int(line[last + 1 :].strip())
    for r in range(1, 12):
        vals = np.sort(person[role == r])
        np.save(CACHE / f"ci_person_role_{r}.npy", vals)
    done.write_text(str(n), encoding="ascii")


def ensure_mc_cache():
    CACHE.mkdir(parents=True, exist_ok=True)
    done = CACHE / "mc_single_done.txt"
    if done.exists():
        return
    path = DATA / "movie_companies.csv"
    n = count_lines(path)
    company = np.empty(n, dtype=np.int32)
    company_type = np.empty(n, dtype=np.int16)
    with path.open("rb") as f:
        for i, line in enumerate(f):
            c1 = line.find(b",")
            c2 = line.find(b",", c1 + 1)
            c3 = line.find(b",", c2 + 1)
            c4 = line.find(b",", c3 + 1)
            company[i] = int(line[c2 + 1 : c3])
            company_type[i] = int(line[c3 + 1 : c4])
    for t in range(1, 3):
        vals = np.sort(company[company_type == t])
        np.save(CACHE / f"mc_company_type_{t}.npy", vals)
    done.write_text(str(n), encoding="ascii")


class SingleCounter:
    def __init__(self):
        ensure_ci_cache()
        ensure_mc_cache()
        print("loading ci/mc sorted count caches", flush=True)
        self.ci = {r: np.load(CACHE / f"ci_person_role_{r}.npy", mmap_mode="r").tolist() for r in range(1, 12)}
        self.mc = {t: np.load(CACHE / f"mc_company_type_{t}.npy", mmap_mode="r").tolist() for t in range(1, 3)}
        self.memo = {}

    def count_alias(self, alias, preds):
        memo_key = (alias, tuple(preds))
        if memo_key in self.memo:
            return self.memo[memo_key]
        if alias == "ci":
            lo, hi, allowed = bounds_from_preds(preds, "person_id", "role_id", range(1, 12))
            tables = self.ci
        elif alias == "mc":
            lo, hi, allowed = bounds_from_preds(preds, "company_id", "company_type_id", range(1, 3))
            tables = self.mc
        else:
            raise ValueError(alias)
        if not allowed or lo >= hi:
            return 0
        total = 0
        for key in allowed:
            arr = tables[key]
            total += bisect.bisect_left(arr, hi) - bisect.bisect_left(arr, lo)
        self.memo[memo_key] = total
        return total


def supported_single(row):
    aliases = set(parse_tables(row["Tables"]))
    if aliases not in ({"ci"}, {"mc"}, {"t", "ci"}, {"t", "mc"}):
        return None
    preds = parse_predicates(row.get("Predicates", ""))
    child = "ci" if "ci" in aliases else "mc"
    for alias, col, _op, _val in preds:
        if alias == "t":
            return None
        if child == "ci" and (alias != "ci" or col not in {"person_id", "role_id"}):
            return None
        if child == "mc" and (alias != "mc" or col not in {"company_id", "company_type_id"}):
            return None
    return child


def compute_counts(df, counter):
    out = {}
    for idx, row in df.iterrows():
        child = supported_single(row)
        if child is None:
            continue
        preds = [p for p in parse_predicates(row.get("Predicates", "")) if p[0] == child]
        out[idx] = counter.count_alias(child, preds)
        if len(out) % 2000 == 0:
            print(f"computed {len(out)} supported rows", flush=True)
    return out


def optimize_and_write(train, test, train_counts, test_counts):
    base_oof_raw = pd.read_csv(BASE_OOF)
    base_oof_raw["BasePred"] = np.exp(base_oof_raw["PredLog"].astype(float))
    base_oof = base_oof_raw.set_index("Id")["BasePred"].astype(float)
    base_train = train["Id"].map(base_oof).to_numpy(dtype=np.float64)
    idxs = np.array(sorted(train_counts), dtype=np.int64)
    exact = np.array([train_counts[int(i)] for i in idxs], dtype=np.float64)
    y = train.loc[idxs, "Cardinality"].to_numpy(dtype=np.float64)
    base = base_train[idxs]

    print(f"single/title-child supported train rows: {len(idxs)}")
    print(f"exact mean qerror on supported: {qerror(y, exact).mean():.8f}")
    print(f"base mean qerror on supported: {qerror(y, base).mean():.8f}")

    best = (10**9, 1.0, 1.0)
    for scale in np.linspace(0.90, 1.10, 81):
        pred = np.maximum(1.0, exact * scale)
        qe = qerror(y, pred).mean()
        if qe < best[0]:
            best = (qe, float(scale), 1.0)
        for w in np.linspace(0.25, 0.95, 15):
            blend = np.exp(w * np.log(pred) + (1 - w) * np.log(np.maximum(base, 1.0)))
            qe = qerror(y, blend).mean()
            if qe < best[0]:
                best = (qe, float(scale), float(w))
    print(f"best supported qerror: {best[0]:.8f}, scale={best[1]:.4f}, exact_weight={best[2]:.3f}")

    hybrid = base_train.copy()
    for idx, exact_count in train_counts.items():
        exact_scaled = max(1.0, exact_count * best[1])
        if best[2] >= 0.999:
            hybrid[idx] = exact_scaled
        else:
            hybrid[idx] = np.exp(best[2] * np.log(exact_scaled) + (1 - best[2]) * np.log(max(1.0, hybrid[idx])))
    print(f"full train hybrid mean qerror: {qerror(train['Cardinality'], hybrid).mean():.8f}")

    sub = pd.read_csv(BASE_SUB)
    pred = sub["Cardinality"].astype(float).to_numpy(copy=True)
    for idx, exact_count in test_counts.items():
        exact_scaled = max(1.0, exact_count * best[1])
        if best[2] >= 0.999:
            pred[idx] = exact_scaled
        else:
            pred[idx] = np.exp(best[2] * np.log(exact_scaled) + (1 - best[2]) * np.log(max(1.0, pred[idx])))
    sub["Cardinality"] = np.maximum(1, np.rint(pred)).astype(np.int64)
    out = ROOT / "submission_physical_single_ci_mc.csv"
    sub.to_csv(out, index=False)
    print(f"wrote {out.name}, changed {len(test_counts)} rows")


def main():
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    counter = SingleCounter()
    print("counting train", flush=True)
    train_counts = compute_counts(train, counter)
    print("counting test", flush=True)
    test_counts = compute_counts(test, counter)
    optimize_and_write(train, test, train_counts, test_counts)


if __name__ == "__main__":
    main()
