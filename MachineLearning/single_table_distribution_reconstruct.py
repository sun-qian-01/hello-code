from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
BASE_SUB = ROOT / "submission_physical_single_on_shape_surface_m50_d2_a10_s0p3_g0p0.csv"
BASE_OOF = ROOT / "shape_surface_m50_d2_a10_s0p3_g0p0_oof.csv"


TABLE_CARD = {
    "t": 2528312,
    "ci": 36244344,
    "mc": 2609129,
    "mk": 4523930,
    "mi": 14835720,
    "mi_idx": 1380035,
}


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


def single_col_query(row):
    aliases = parse_tables(row["Tables"])
    if len(aliases) != 1:
        return None
    alias = aliases[0]
    preds = parse_predicates(row.get("Predicates", ""))
    if not preds:
        return None
    cols = {(a, c) for a, c, _o, _v in preds}
    if len(cols) != 1:
        return None
    a, col = next(iter(cols))
    if a != alias:
        return None
    return alias, col, tuple((op, val) for _a, _c, op, val in preds)


def interval_from_preds(total, preds):
    lo = -10**18
    hi = 10**18
    exact_values = None
    for op, val in preds:
        if op == "=":
            vals = {val}
            exact_values = vals if exact_values is None else exact_values & vals
        elif op == "<":
            hi = min(hi, val)
        elif op == ">":
            lo = max(lo, val + 1)
        else:
            raise ValueError(op)
    if exact_values is not None:
        vals = [v for v in exact_values if lo <= v < hi]
        return ("values", vals)
    return ("range", lo, hi)


class DistributionOracle:
    def __init__(self, total, min_val, max_val):
        self.total = int(total)
        self.min_val = int(min_val)
        self.max_val = int(max_val)
        self.lt = {}
        self.gt = {}
        self.eq = {}

    def add(self, op, val, count):
        val = int(val)
        count = int(count)
        if op == "<":
            self.lt[val] = count
        elif op == ">":
            self.gt[val] = count
        elif op == "=":
            self.eq[val] = count

    def finalize(self):
        # Convert >v to <=v, hence <(v+1).
        for v, c in self.gt.items():
            self.lt.setdefault(v + 1, self.total - c)
        self.lt.setdefault(self.min_val, 0)
        self.lt.setdefault(self.max_val + 1, self.total)
        for v, c in self.eq.items():
            if v in self.lt and (v + 1) not in self.lt:
                self.lt[v + 1] = self.lt[v] + c
            if (v + 1) in self.lt and v not in self.lt:
                self.lt[v] = self.lt[v + 1] - c
        xs = sorted(self.lt)
        ys = [self.lt[x] for x in xs]
        # Enforce monotone CDF in case duplicate observations disagree slightly.
        ys = np.maximum.accumulate(np.asarray(ys, dtype=np.float64))
        self.xs = np.asarray(xs, dtype=np.float64)
        self.ys = np.asarray(ys, dtype=np.float64)

    def cdf_lt(self, x):
        if x in self.lt:
            return float(self.lt[x])
        return float(np.interp(float(x), self.xs, self.ys))

    def count_range(self, lo, hi):
        if lo >= hi:
            return 0.0
        lo = max(lo, self.min_val)
        hi = min(hi, self.max_val + 1)
        if lo >= hi:
            return 0.0
        return max(0.0, self.cdf_lt(hi) - self.cdf_lt(lo))

    def predict(self, preds):
        kind = interval_from_preds(self.total, preds)
        if kind[0] == "values":
            vals = kind[1]
            total = 0.0
            for v in vals:
                if v in self.eq:
                    total += self.eq[v]
                else:
                    total += self.count_range(v, v + 1)
            return total
        _tag, lo, hi = kind
        return self.count_range(lo, hi)


def build_oracles(train):
    stats = pd.read_csv(ROOT / "column_min_max_vals.csv").set_index("name")
    grouped = defaultdict(list)
    for idx, row in train.iterrows():
        sq = single_col_query(row)
        if sq is None:
            continue
        alias, col, preds = sq
        if len(preds) != 1:
            continue
        op, val = preds[0]
        grouped[(alias, col)].append((op, val, int(row["Cardinality"])))

    oracles = {}
    for (alias, col), obs in grouped.items():
        full = f"{alias}.{col}"
        if full not in stats.index or alias not in TABLE_CARD:
            continue
        row = stats.loc[full]
        oracle = DistributionOracle(TABLE_CARD[alias], int(row["min"]), int(row["max"]))
        for op, val, count in obs:
            oracle.add(op, val, count)
        oracle.finalize()
        oracles[(alias, col)] = oracle
        print((alias, col), "obs", len(obs), "cdf_points", len(oracle.xs), "eq", len(oracle.eq))
    return oracles


def predict_supported(df, oracles):
    pred = {}
    for idx, row in df.iterrows():
        sq = single_col_query(row)
        if sq is None:
            continue
        alias, col, preds = sq
        oracle = oracles.get((alias, col))
        if oracle is None:
            continue
        pred[idx] = oracle.predict(preds)
    return pred


def main():
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    oracles = build_oracles(train)
    train_pred = predict_supported(train, oracles)
    test_pred = predict_supported(test, oracles)

    idxs = np.array(sorted(train_pred), dtype=int)
    y = train.loc[idxs, "Cardinality"].to_numpy(dtype=float)
    p = np.array([train_pred[i] for i in idxs], dtype=float)
    print("supported train", len(idxs), "test", len(test_pred), "qerror", qerror(y, p).mean(), "p95", np.percentile(qerror(y, p), 95))

    base_oof = pd.read_csv(BASE_OOF)
    base = np.exp(base_oof["PredLog"].astype(float).to_numpy())
    base_q = qerror(train["Cardinality"], base).mean()
    # Only apply rows where the reconstructed single-column distribution is very reliable.
    reliable = {}
    for idx, val in train_pred.items():
        qe = qerror([train.loc[idx, "Cardinality"]], [val])[0]
        if qe <= 1.02:
            reliable[idx] = val
    hybrid = base.copy()
    for idx, val in reliable.items():
        hybrid[idx] = max(1.0, val)
    print("train reliable", len(reliable), "base", base_q, "hybrid", qerror(train["Cardinality"], hybrid).mean())

    # For test, use only columns whose training reconstruction is near exact globally.
    col_q = {}
    for key in oracles:
        rows = []
        vals = []
        for idx, row in train.iterrows():
            sq = single_col_query(row)
            if sq is None or sq[:2] != key:
                continue
            if idx in train_pred:
                rows.append(idx)
                vals.append(train_pred[idx])
        if rows:
            col_q[key] = qerror(train.loc[rows, "Cardinality"], vals).mean()
    safe_keys = {key for key, score in col_q.items() if score <= 1.02}
    print("safe_keys", safe_keys, col_q)

    sub = pd.read_csv(BASE_SUB)
    sub_pred = sub["Cardinality"].astype(float).to_numpy(copy=True)
    changed = 0
    for idx, val in test_pred.items():
        row = test.loc[idx]
        sq = single_col_query(row)
        if sq is None or sq[:2] not in safe_keys:
            continue
        sub_pred[idx] = max(1.0, val)
        changed += 1
    sub["Cardinality"] = np.maximum(1, np.rint(sub_pred)).astype(np.int64)
    out = ROOT / "submission_physical_single_plus_reconstructed_singletons.csv"
    sub.to_csv(out, index=False)
    print("wrote", out.name, "changed_reconstructed", changed)


if __name__ == "__main__":
    main()
