"""
SQL Cardinality Estimation - 改进版
更好的连接选择性估计 + 交叉特征
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import warnings
import time

warnings.filterwarnings('ignore')

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

# 表行数
table_row_counts = {
    't': col_info['t.id'][2],
    'mc': col_info['mc.id'][2],
    'ci': col_info['ci.id'][2],
    'mi': col_info['mi.id'][2],
    'mi_idx': col_info['mi_idx.id'][2],
    'mk': col_info['mk.id'][2],
}

# 外键关系：子表.列 -> 父表.列
# 从 schema 和连接条件推断
FK_RELATIONS = {
    'mc.movie_id': ('t.id', table_row_counts['t']),
    'ci.movie_id': ('t.id', table_row_counts['t']),
    'mi.movie_id': ('t.id', table_row_counts['t']),
    'mi_idx.movie_id': ('t.id', table_row_counts['t']),
    'mk.movie_id': ('t.id', table_row_counts['t']),
}

# 主键唯一值 = 行数（对于 id 列）
PK_UNIQUE = {col: col_info[col][3] for col in ALL_COLS if col.endswith('.id')}

print(f"训练: {len(train_df)}, 测试: {len(test_df)}")

# ============================================================
# 2. 改进特征提取
# ============================================================
print("\n" + "=" * 60)
print("2. 改进特征提取")
print("=" * 60)

def extract_v2(df, name):
    t0 = time.time()
    N = len(df)
    feats = {}

    tables_col = df['Tables'].fillna('')
    joins_col = df['Join Conditions'].fillna('')
    preds_col = df['Predicates'].fillna('')

    # --- 基础计数 ---
    feats['num_tables'] = tables_col.apply(lambda x: len([s for s in x.split(',') if s.strip()]) if x else 0).values.astype(float)
    feats['num_joins'] = joins_col.apply(lambda x: len(x.split(',')) if x else 0).values.astype(float)
    feats['num_predicates'] = preds_col.apply(lambda x: max(len(x.split(',')) // 3, 0) if x else 0).values.astype(float)

    # --- 表存在性 ---
    for t in ALL_TABLES:
        feats[f'has_table_{t}'] = tables_col.str.contains(rf'\b{t}\b', regex=True).astype(float).values

    # --- 表基数 (log) ---
    for t in ALL_TABLES:
        feats[f'{t}_cardinality'] = np.full(N, np.log1p(table_pk_card.get(t, 1)))

    # --- 解析表名列表 ---
    def parse_tables(s):
        if not s: return []
        return [x.strip().split()[-1] for x in s.split(',') if x.strip()]
    tables_list = tables_col.apply(parse_tables)

    # --- 解析谓词 ---
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

    # --- 解析连接条件 ---
    def parse_joins(s):
        if not s: return []
        result = []
        for j in s.split(','):
            j = j.strip()
            if '=' in j:
                parts = j.split('=')
                result.append((parts[0].strip(), parts[1].strip()))
        return result
    joins_parsed = joins_col.apply(parse_joins)

    # --- 为每个列创建特征 ---
    for col_name in ALL_COLS:
        cmin, cmax, ccard, cnunique = col_info[col_name]
        crange = max(cmax - cmin, 1)

        used = np.zeros(N); pred_count = np.zeros(N)
        has_eq = np.zeros(N); eq_val = np.zeros(N); eq_val_norm = np.zeros(N); eq_sel = np.ones(N)
        has_lt = np.zeros(N); lt_val = np.zeros(N); lt_val_norm = np.zeros(N); lt_sel = np.ones(N)
        has_gt = np.zeros(N); gt_val = np.zeros(N); gt_val_norm = np.zeros(N); gt_sel = np.ones(N)

        for i, pm in enumerate(parsed):
            if col_name not in pm: continue
            ops = pm[col_name]
            used[i] = 1; pred_count[i] = len(ops)
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

    # --- 统计和启发式估计 ---
    eq_counts = np.zeros(N); lt_counts = np.zeros(N); gt_counts = np.zeros(N)
    overall_sel = np.zeros(N); heuristic = np.zeros(N)
    # 改进的启发式估计 v2
    heuristic_v2 = np.zeros(N)
    # 最大表基数
    max_table_card = np.zeros(N)
    # 连接因子
    join_factor = np.zeros(N)

    for i in range(N):
        pm = parsed.iloc[i]
        tables = tables_list.iloc[i]
        joins = joins_parsed.iloc[i]

        # 最大表基数
        max_tc = 0
        for t in tables:
            if t in table_row_counts:
                max_tc = max(max_tc, table_row_counts[t])
        max_table_card[i] = np.log1p(max_tc)

        # 每表选择性
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

        # 启发式估计 v1
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

        # 启发式估计 v2：更好的连接处理
        if tables:
            # 找到"驱动表"：有最多谓词的表，或第一个表
            main_table = tables[0]
            # 使用连接图计算更精确的基数
            # 基本思路：从驱动表开始，依次连接其他表
            base_card = table_row_counts.get(main_table, 1)
            main_sel = table_sel.get(main_table, 1.0)
            est = base_card * main_sel

            # 处理每个额外的表
            jf = 1.0  # 连接因子
            for jt in tables[1:]:
                jt_sel = table_sel.get(jt, 1.0)
                jt_rows = table_row_counts.get(jt, 1)

                # 查找连接条件中涉及 jt 的 FK
                fk_match = False
                for left, right in joins:
                    lt = left.split('.')[0] if '.' in left else ''
                    rt = right.split('.')[0] if '.' in right else ''

                    if lt == jt and f'{left}' in FK_RELATIONS:
                        # jt 通过 FK 连接到父表
                        parent_pk, parent_rows = FK_RELATIONS[left]
                        fk_match = True
                        # 连接不会增加行数（FK 连接）
                        pass
                    elif rt == jt and f'{right}' in FK_RELATIONS:
                        parent_pk, parent_rows = FK_RELATIONS[right]
                        fk_match = True

                if not fk_match:
                    # 没有 FK，使用启发式
                    jf *= 1.0  # 保守估计

                # 乘以表的过滤选择性
                est *= jt_sel

            est *= jf
            heuristic_v2[i] = np.log1p(max(est, 0.01))
            join_factor[i] = np.log1p(jf)
        else:
            heuristic_v2[i] = 0
            join_factor[i] = 0

    feats['eq_count'] = eq_counts; feats['lt_count'] = lt_counts; feats['gt_count'] = gt_counts
    feats['overall_selectivity'] = overall_sel
    feats['heuristic_est_log'] = heuristic
    feats['heuristic_v2_log'] = heuristic_v2
    feats['max_table_card_log'] = max_table_card
    feats['join_factor_log'] = join_factor

    # --- 交叉特征 ---
    feats['eq_x_lt'] = eq_counts * lt_counts
    feats['eq_x_gt'] = eq_counts * gt_counts
    feats['preds_per_table'] = feats['num_predicates'] / np.maximum(feats['num_tables'], 1)

    # --- 表组合特征 ---
    for t1 in ALL_TABLES:
        for t2 in ALL_TABLES:
            if t1 < t2:
                feats[f'both_{t1}_{t2}'] = feats[f'has_table_{t1}'] * feats[f'has_table_{t2}']

    result = pd.DataFrame(feats)
    print(f"  {name}: {time.time()-t0:.1f}s, {result.shape}")
    return result

X_train = extract_v2(train_df, "训练特征")
X_test = extract_v2(test_df, "测试特征")
y_train = train_df['Cardinality'].values.astype(float)

common_cols = X_train.columns.intersection(X_test.columns)
X_train = X_train[common_cols]
X_test = X_test[common_cols]
print(f"特征数: {len(common_cols)}")

# ============================================================
# 3. 训练
# ============================================================
print("\n" + "=" * 60)
print("3. LightGBM 训练")
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
# 4. 验证
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
print(f"90th: {np.percentile(qe, 90):.4f}  95th: {np.percentile(qe, 95):.4f}  99th: {np.percentile(qe, 99):.4f}")

# ============================================================
# 5. 特征重要性
# ============================================================
print("\n" + "=" * 60)
print("5. Top 25 特征重要性")
print("=" * 60)
imp = model.feature_importance(importance_type='gain')
for feat, sc in sorted(zip(X_train.columns, imp), key=lambda x: -x[1])[:25]:
    print(f"  {feat:50s} {sc:15.0f}")

# ============================================================
# 6. 生成提交
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
print(f"已保存 ({len(sub)} 条)")
print(f"基数: [{ytest.min():,}, {ytest.max():,}], μ={ytest.mean():,.0f}")
print(f"训练集 μ={y_train.mean():,.0f}")
print("\n完成!")
