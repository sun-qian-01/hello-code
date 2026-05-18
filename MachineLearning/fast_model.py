"""
SQL Cardinality Estimation - Fast Feature Engineering + LightGBM
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import warnings
import time

warnings.filterwarnings('ignore')

# ============================================================
# 1. 加载
# ============================================================
print("=" * 60)
print("1. 加载数据")
print("=" * 60)

train_df = pd.read_csv("c:/Users/sunqi/Desktop/CODE/hello-code/MachineLearning/train.csv")
test_df = pd.read_csv("c:/Users/sunqi/Desktop/CODE/hello-code/MachineLearning/test.csv")
col_stats = pd.read_csv("c:/Users/sunqi/Desktop/CODE/hello-code/MachineLearning/column_min_max_vals.csv")

col_info = {}
for _, row in col_stats.iterrows():
    col_info[row['name']] = (row['min'], row['max'], row['cardinality'], row['num_unique_values'])

ALL_TABLES = ['t', 'mc', 'ci', 'mi', 'mi_idx', 'mk']
ALL_COLS = list(col_info.keys())

# 表主键基数
table_pk_card = {}
for t in ALL_TABLES:
    pk = f'{t}.id'
    table_pk_card[t] = col_info[pk][2] if pk in col_info else 1

print(f"训练集: {len(train_df)}, 测试集: {len(test_df)}, 列统计: {len(col_info)}")

# ============================================================
# 2. 特征提取
# ============================================================
print("\n" + "=" * 60)
print("2. 快速特征提取")
print("=" * 60)

def extract(df, name):
    t0 = time.time()
    N = len(df)
    feats = {}

    tables_col = df['Tables'].fillna('')
    joins_col = df['Join Conditions'].fillna('')
    preds_col = df['Predicates'].fillna('')

    # --- 表数量 ---
    feats['num_tables'] = tables_col.apply(lambda x: len([s for s in x.split(',') if s.strip()]) if x else 0).values.astype(float)

    # --- 连接数量 ---
    feats['num_joins'] = joins_col.apply(lambda x: len(x.split(',')) if x else 0).values.astype(float)

    # --- 谓词数量 ---
    feats['num_predicates'] = preds_col.apply(lambda x: max(len(x.split(',')) // 3, 0) if x else 0).values.astype(float)

    # --- 表存在性 ---
    for t in ALL_TABLES:
        feats[f'has_table_{t}'] = tables_col.str.contains(rf'\b{t}\b', regex=True).astype(float).values

    # --- 表基数 (log) ---
    for t in ALL_TABLES:
        feats[f'{t}_cardinality'] = np.full(N, np.log1p(table_pk_card.get(t, 1)))

    # --- 解析谓词 ---
    def parse_preds(s):
        if not s:
            return {}
        parts = [p.strip() for p in s.split(',')]
        d = {}
        for i in range(0, len(parts) - 2, 3):
            col = parts[i]
            op = parts[i + 1]
            try:
                val = float(parts[i + 2])
            except ValueError:
                continue
            d.setdefault(col, []).append((op, val))
        return d

    parsed = preds_col.apply(parse_preds)

    # --- 为每个列创建特征 ---
    for col_name in ALL_COLS:
        cmin, cmax, ccard, cnunique = col_info[col_name]
        crange = max(cmax - cmin, 1)

        used = np.zeros(N); pred_count = np.zeros(N)
        has_eq = np.zeros(N); eq_val = np.zeros(N); eq_val_norm = np.zeros(N); eq_sel = np.ones(N)
        has_lt = np.zeros(N); lt_val = np.zeros(N); lt_val_norm = np.zeros(N); lt_sel = np.ones(N)
        has_gt = np.zeros(N); gt_val = np.zeros(N); gt_val_norm = np.zeros(N); gt_sel = np.ones(N)

        for i, pm in enumerate(parsed):
            if col_name not in pm:
                continue
            ops = pm[col_name]
            used[i] = 1
            pred_count[i] = len(ops)
            for op, val in ops:
                if op == '=':
                    has_eq[i] = 1; eq_val[i] = val
                    eq_val_norm[i] = (val - cmin) / crange
                    eq_sel[i] = 1.0 / max(cnunique, 1)
                elif op == '<':
                    has_lt[i] = 1; lt_val[i] = val
                    lt_val_norm[i] = (val - cmin) / crange
                    lt_sel[i] = max((val - cmin) / crange, 0.0001)
                elif op == '>':
                    has_gt[i] = 1; gt_val[i] = val
                    gt_val_norm[i] = (val - cmin) / crange
                    gt_sel[i] = max((cmax - val) / crange, 0.0001)

        prefix = col_name
        feats[f'{prefix}_used'] = used; feats[f'{prefix}_pred_count'] = pred_count
        feats[f'{prefix}_has_eq'] = has_eq; feats[f'{prefix}_eq_val'] = eq_val
        feats[f'{prefix}_eq_val_norm'] = eq_val_norm; feats[f'{prefix}_eq_sel'] = eq_sel
        feats[f'{prefix}_has_lt'] = has_lt; feats[f'{prefix}_lt_val'] = lt_val
        feats[f'{prefix}_lt_val_norm'] = lt_val_norm; feats[f'{prefix}_lt_sel'] = lt_sel
        feats[f'{prefix}_has_gt'] = has_gt; feats[f'{prefix}_gt_val'] = gt_val
        feats[f'{prefix}_gt_val_norm'] = gt_val_norm; feats[f'{prefix}_gt_sel'] = gt_sel

    # --- 操作符计数、选择性、启发式估计 ---
    eq_counts = np.zeros(N); lt_counts = np.zeros(N); gt_counts = np.zeros(N)
    overall_sel = np.zeros(N); heuristic = np.zeros(N)

    # 解析表的辅助
    def parse_tables(s):
        if not s:
            return []
        return [x.strip().split()[-1] for x in s.split(',') if x.strip()]

    tables_list = tables_col.apply(parse_tables)

    for i in range(N):
        pm = parsed.iloc[i]
        tables = tables_list.iloc[i]

        table_sel = {}
        for col, ops in pm.items():
            tn = col.split('.')[0] if '.' in col else ''
            table_sel.setdefault(tn, 1.0)
            cmin2, cmax2, _, cnunique2 = col_info.get(col, (0, 1, 1, 1))
            crange2 = max(cmax2 - cmin2, 1)
            for op, val in ops:
                if op == '=':
                    eq_counts[i] += 1
                    table_sel[tn] *= 1.0 / max(cnunique2, 1)
                elif op == '<':
                    lt_counts[i] += 1
                    table_sel[tn] *= max((val - cmin2) / crange2, 0.0001)
                elif op == '>':
                    gt_counts[i] += 1
                    table_sel[tn] *= max((cmax2 - val) / crange2, 0.0001)

        sel = 1.0
        for v in table_sel.values():
            sel *= v
        overall_sel[i] = np.log1p(sel * 1000)

        if tables and tables[0]:
            mt = tables[0]
            bc = table_pk_card.get(mt, 1)
            ms = table_sel.get(mt, 1.0)
            hest = bc * ms
            if len(tables) > 1:
                for jt in tables[1:]:
                    if jt:
                        jc = table_pk_card.get(jt, 1)
                        hest *= min(1.0, jc / max(bc, 1))
            heuristic[i] = np.log1p(max(hest, 0.01))

    feats['eq_count'] = eq_counts; feats['lt_count'] = lt_counts; feats['gt_count'] = gt_counts
    feats['overall_selectivity'] = overall_sel; feats['heuristic_est_log'] = heuristic

    result = pd.DataFrame(feats)
    print(f"  {name} 特征提取: {time.time()-t0:.1f}s, 形状 {result.shape}")
    return result

X_train = extract(train_df, "训练")
X_test = extract(test_df, "测试")
y_train = train_df['Cardinality'].values.astype(float)

common_cols = X_train.columns.intersection(X_test.columns)
X_train = X_train[common_cols]
X_test = X_test[common_cols]
print(f"特征数: {len(common_cols)}")

# ============================================================
# 3. 训练
# ============================================================
print("\n" + "=" * 60)
print("3. 模型训练")
print("=" * 60)

y_train_log = np.log1p(y_train)

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train_log, test_size=0.1, random_state=42)
print(f"训练: {len(X_tr)}, 验证: {len(X_val)}")

params = {
    'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
    'num_leaves': 255, 'learning_rate': 0.03, 'feature_fraction': 0.7,
    'bagging_fraction': 0.7, 'bagging_freq': 5, 'min_data_in_leaf': 30,
    'lambda_l1': 0.5, 'lambda_l2': 0.5, 'verbose': -1, 'seed': 42,
    'num_threads': -1, 'max_depth': 12,
}

dtrain = lgb.Dataset(X_tr, label=y_tr)
dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

print("训练中...")
model = lgb.train(params, dtrain, num_boost_round=3000,
                  valid_sets=[dtrain, dval], valid_names=['train', 'valid'],
                  callbacks=[lgb.early_stopping(stopping_rounds=150),
                             lgb.log_evaluation(period=200)])

# ============================================================
# 4. 评估
# ============================================================
print("\n" + "=" * 60)
print("4. 验证集评估")
print("=" * 60)

yp_log = model.predict(X_val)
yp = np.expm1(yp_log)
yt = np.expm1(y_val)
qe = np.maximum(yp / yt, yt / yp)
print(f"Mean Q-Error:     {np.mean(qe):.4f}")
print(f"Median Q-Error:   {np.median(qe):.4f}")
print(f"90th: {np.percentile(qe, 90):.4f}, 95th: {np.percentile(qe, 95):.4f}, 99th: {np.percentile(qe, 99):.4f}")

# ============================================================
# 5. 重要性
# ============================================================
print("\n" + "=" * 60)
print("5. Top 20 特征重要性")
print("=" * 60)
imp = model.feature_importance(importance_type='gain')
for feat, sc in sorted(zip(X_train.columns, imp), key=lambda x: -x[1])[:20]:
    print(f"  {feat:45s} {sc:15.0f}")

# ============================================================
# 6. 生成提交文件
# ============================================================
print("\n" + "=" * 60)
print("6. 生成提交文件")
print("=" * 60)

ytest_log = model.predict(X_test)
ytest = np.expm1(ytest_log)
ytest = np.maximum(ytest, 1)
ytest = np.round(ytest).astype(int)

sub = pd.DataFrame({'Id': test_df['Id'], 'Cardinality': ytest})
sub.to_csv("c:/Users/sunqi/Desktop/CODE/hello-code/MachineLearning/submission.csv", index=False)

print(f"已保存 submission.csv ({len(sub)} 条)")
print(f"基数范围: [{ytest.min():,}, {ytest.max():,}], 均值: {ytest.mean():,.0f}")
print(f"训练集基数范围: [{y_train.min():,.0f}, {y_train.max():,.0f}], 均值: {y_train.mean():,.0f}")
print("\n完成!")
