"""
SQL Cardinality Estimation - 超参数优化版
尝试多种配置找到最佳 Q-Error
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import warnings
import time

warnings.filterwarnings('ignore')

print("=" * 60)
print("加载数据 & 特征提取")
print("=" * 60)

train_df = pd.read_csv("c:/Users/sunqi/Desktop/CODE/hello-code/MachineLearning/train.csv")
test_df = pd.read_csv("c:/Users/sunqi/Desktop/CODE/hello-code/MachineLearning/test.csv")
col_stats = pd.read_csv("c:/Users/sunqi/Desktop/CODE/hello-code/MachineLearning/column_min_max_vals.csv")

col_info = {}
for _, row in col_stats.iterrows():
    col_info[row['name']] = (row['min'], row['max'], row['cardinality'], row['num_unique_values'])

ALL_TABLES = ['t', 'mc', 'ci', 'mi', 'mi_idx', 'mk']
ALL_COLS = list(col_info.keys())

table_pk_card = {}
for t in ALL_TABLES:
    pk = f'{t}.id'
    table_pk_card[t] = col_info[pk][2] if pk in col_info else 1

# 复用 fast_model 的特征提取
def extract(df, name):
    t0 = time.time()
    N = len(df)
    feats = {}
    tables_col = df['Tables'].fillna('')
    joins_col = df['Join Conditions'].fillna('')
    preds_col = df['Predicates'].fillna('')

    feats['num_tables'] = tables_col.apply(lambda x: len([s for s in x.split(',') if s.strip()]) if x else 0).values.astype(float)
    feats['num_joins'] = joins_col.apply(lambda x: len(x.split(',')) if x else 0).values.astype(float)
    feats['num_predicates'] = preds_col.apply(lambda x: max(len(x.split(',')) // 3, 0) if x else 0).values.astype(float)

    for t in ALL_TABLES:
        feats[f'has_table_{t}'] = tables_col.str.contains(rf'\b{t}\b', regex=True).astype(float).values

    for t in ALL_TABLES:
        feats[f'{t}_cardinality'] = np.full(N, np.log1p(table_pk_card.get(t, 1)))

    def parse_preds(s):
        if not s: return {}
        parts = [p.strip() for p in s.split(',')]
        d = {}
        for i in range(0, len(parts) - 2, 3):
            col = parts[i]; op = parts[i + 1]
            try:
                val = float(parts[i + 2])
            except ValueError:
                continue
            d.setdefault(col, []).append((op, val))
        return d

    parsed = preds_col.apply(parse_preds)

    for col_name in ALL_COLS:
        cmin, cmax, ccard, cnunique = col_info[col_name]
        crange = max(cmax - cmin, 1)
        used = np.zeros(N); pred_count = np.zeros(N)
        has_eq = np.zeros(N); eq_val = np.zeros(N); eq_val_norm = np.zeros(N); eq_sel = np.ones(N)
        has_lt = np.zeros(N); lt_val = np.zeros(N); lt_val_norm = np.zeros(N); lt_sel = np.ones(N)
        has_gt = np.zeros(N); gt_val = np.zeros(N); gt_val_norm = np.zeros(N); gt_sel = np.ones(N)

        for i, pm in enumerate(parsed):
            if col_name not in pm: continue
            ops = pm[col_name]; used[i] = 1; pred_count[i] = len(ops)
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

        p = col_name
        feats[f'{p}_used'] = used; feats[f'{p}_pred_count'] = pred_count
        feats[f'{p}_has_eq'] = has_eq; feats[f'{p}_eq_val'] = eq_val
        feats[f'{p}_eq_val_norm'] = eq_val_norm; feats[f'{p}_eq_sel'] = eq_sel
        feats[f'{p}_has_lt'] = has_lt; feats[f'{p}_lt_val'] = lt_val
        feats[f'{p}_lt_val_norm'] = lt_val_norm; feats[f'{p}_lt_sel'] = lt_sel
        feats[f'{p}_has_gt'] = has_gt; feats[f'{p}_gt_val'] = gt_val
        feats[f'{p}_gt_val_norm'] = gt_val_norm; feats[f'{p}_gt_sel'] = gt_sel

    eq_counts = np.zeros(N); lt_counts = np.zeros(N); gt_counts = np.zeros(N)
    overall_sel = np.zeros(N); heuristic = np.zeros(N)

    def parse_tables(s):
        if not s: return []
        return [x.strip().split()[-1] for x in s.split(',') if x.strip()]
    tables_list = tables_col.apply(parse_tables)

    for i in range(N):
        pm = parsed.iloc[i]; tables = tables_list.iloc[i]
        table_sel = {}
        for col, ops in pm.items():
            tn = col.split('.')[0] if '.' in col else ''
            table_sel.setdefault(tn, 1.0)
            cmin2, cmax2, _, cnunique2 = col_info.get(col, (0, 1, 1, 1))
            crange2 = max(cmax2 - cmin2, 1)
            for op, val in ops:
                if op == '=': eq_counts[i] += 1; table_sel[tn] *= 1.0 / max(cnunique2, 1)
                elif op == '<': lt_counts[i] += 1; table_sel[tn] *= max((val - cmin2) / crange2, 0.0001)
                elif op == '>': gt_counts[i] += 1; table_sel[tn] *= max((cmax2 - val) / crange2, 0.0001)
        sel = 1.0
        for v in table_sel.values(): sel *= v
        overall_sel[i] = np.log1p(sel * 1000)
        if tables and tables[0]:
            mt = tables[0]; bc = table_pk_card.get(mt, 1); ms = table_sel.get(mt, 1.0)
            hest = bc * ms
            if len(tables) > 1:
                for jt in tables[1:]:
                    if jt: jc = table_pk_card.get(jt, 1); hest *= min(1.0, jc / max(bc, 1))
            heuristic[i] = np.log1p(max(hest, 0.01))

    feats['eq_count'] = eq_counts; feats['lt_count'] = lt_counts; feats['gt_count'] = gt_counts
    feats['overall_selectivity'] = overall_sel; feats['heuristic_est_log'] = heuristic
    result = pd.DataFrame(feats)
    print(f"  {name}: {time.time()-t0:.1f}s, {result.shape}")
    return result

X_train_full = extract(train_df, "训练")
X_test = extract(test_df, "测试")
y_train_full = train_df['Cardinality'].values.astype(float)

common_cols = X_train_full.columns.intersection(X_test.columns)
X_train_full = X_train_full[common_cols]
X_test = X_test[common_cols]

# 划分训练/验证
y_train_log = np.log1p(y_train_full)
X_tr, X_val, y_tr, y_val = train_test_split(X_train_full, y_train_log, test_size=0.1, random_state=42)

print(f"\n特征数: {len(common_cols)}")

# ============================================================
# 尝试多种配置
# ============================================================
configs = [
    # (name, params)
    ("Base (RMSE, nl=255, lr=0.03)", {
        'objective': 'regression', 'metric': 'rmse',
        'num_leaves': 255, 'learning_rate': 0.03, 'max_depth': 12,
    }),
    ("RMSE, nl=127, lr=0.02", {
        'objective': 'regression', 'metric': 'rmse',
        'num_leaves': 127, 'learning_rate': 0.02, 'max_depth': 10,
    }),
    ("MAE, nl=255, lr=0.03", {
        'objective': 'regression_l1', 'metric': 'mae',
        'num_leaves': 255, 'learning_rate': 0.03, 'max_depth': 12,
    }),
    ("Huber, nl=255, lr=0.03", {
        'objective': 'huber', 'metric': 'rmse', 'alpha': 1.0,
        'num_leaves': 255, 'learning_rate': 0.03, 'max_depth': 12,
    }),
    ("RMSE, nl=511, lr=0.02", {
        'objective': 'regression', 'metric': 'rmse',
        'num_leaves': 511, 'learning_rate': 0.02, 'max_depth': 14,
    }),
    ("RMSE, nl=255, lr=0.05", {
        'objective': 'regression', 'metric': 'rmse',
        'num_leaves': 255, 'learning_rate': 0.05, 'max_depth': 12,
    }),
]

common_params = {
    'boosting_type': 'gbdt', 'feature_fraction': 0.7,
    'bagging_fraction': 0.7, 'bagging_freq': 5, 'min_data_in_leaf': 30,
    'lambda_l1': 0.5, 'lambda_l2': 0.5, 'verbose': -1, 'seed': 42,
    'num_threads': -1,
}

dtrain = lgb.Dataset(X_tr, label=y_tr)
dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

best_model = None
best_qe = float('inf')
best_name = ""

print("\n" + "=" * 60)
print("超参数搜索")
print("=" * 60)

for name, extra_params in configs:
    params = {**common_params, **extra_params}
    print(f"\n--- {name} ---")
    
    model = lgb.train(params, dtrain, num_boost_round=3000,
                      valid_sets=[dval], valid_names=['valid'],
                      callbacks=[lgb.early_stopping(stopping_rounds=150),
                                 lgb.log_evaluation(period=500)])
    
    yp_log = model.predict(X_val)
    yp = np.expm1(yp_log); yt = np.expm1(y_val)
    qe = np.maximum(yp / yt, yt / yp)
    mqe = np.mean(qe)
    print(f"  => Mean Q-Error: {mqe:.4f}  (best_iter: {model.best_iteration})")
    
    if mqe < best_qe:
        best_qe = mqe
        best_model = model
        best_name = name

print(f"\n{'='*60}")
print(f"最佳: {best_name} -> Q-Error: {best_qe:.4f}")
print(f"{'='*60}")

# ============================================================
# 使用最佳模型预测测试集
# ============================================================
print("\n使用最佳模型生成提交文件...")
ytest_log = best_model.predict(X_test)
ytest = np.expm1(ytest_log)
ytest = np.maximum(ytest, 1)
ytest = np.round(ytest).astype(int)

sub = pd.DataFrame({'Id': test_df['Id'], 'Cardinality': ytest})
sub.to_csv("c:/Users/sunqi/Desktop/CODE/hello-code/MachineLearning/submission.csv", index=False)
print(f"已保存 ({len(sub)} 条)")
print(f"基数: [{ytest.min():,}, {ytest.max():,}], μ={ytest.mean():,.0f}")
print("\n完成!")
