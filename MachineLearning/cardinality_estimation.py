"""
SQL Cardinality Estimation - Feature Engineering + LightGBM Model
评估指标: Mean Q-Error
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import warnings
import re

warnings.filterwarnings('ignore')

# ============================================================
# 1. 加载数据
# ============================================================
print("=" * 60)
print("1. 加载数据")
print("=" * 60)

train_df = pd.read_csv("c:/Users/sunqi/Desktop/CODE/hello-code/MachineLearning/train.csv")
test_df = pd.read_csv("c:/Users/sunqi/Desktop/CODE/hello-code/MachineLearning/test.csv")
col_stats = pd.read_csv("c:/Users/sunqi/Desktop/CODE/hello-code/MachineLearning/column_min_max_vals.csv")

# 构建列统计字典
col_stats_dict = {}
for _, row in col_stats.iterrows():
    col_stats_dict[row['name']] = {
        'min': row['min'],
        'max': row['max'],
        'cardinality': row['cardinality'],
        'num_unique_values': row['num_unique_values']
    }

# 所有可能的表名
ALL_TABLES = ['t', 'mc', 'ci', 'mi', 'mi_idx', 'mk']

# 所有可能出现在谓词中的列（从 column_min_max_vals 中）
ALL_PRED_COLUMNS = list(col_stats_dict.keys())

print(f"训练集大小: {len(train_df)}")
print(f"测试集大小: {len(test_df)}")
print(f"列统计数: {len(col_stats_dict)}")

# ============================================================
# 2. 特征工程函数
# ============================================================
def parse_tables(tables_str):
    """解析表名，返回表名列表"""
    if pd.isna(tables_str) or tables_str == '':
        return []
    tables = []
    for part in tables_str.split(','):
        part = part.strip()
        # 提取表别名，如 "title t" -> "t"
        match = re.search(r'\s(\w+)$', part)
        if match:
            tables.append(match.group(1))
        else:
            tables.append(part)
    return tables

def parse_joins(joins_str):
    """解析连接条件"""
    if pd.isna(joins_str) or joins_str == '':
        return []
    return [j.strip() for j in joins_str.split(',')]

def parse_predicates(pred_str):
    """解析谓词，返回 (列名, 操作符, 值) 三元组列表"""
    if pd.isna(pred_str) or pred_str == '':
        return []
    parts = [p.strip() for p in pred_str.split(',')]
    predicates = []
    i = 0
    while i + 2 < len(parts):
        col = parts[i]
        op = parts[i + 1]
        try:
            val = float(parts[i + 2])
        except ValueError:
            val = float('nan')
        predicates.append((col, op, val))
        i += 3
    return predicates

def extract_features(df, col_stats_dict):
    """从数据框中提取特征"""
    features_list = []
    
    for idx, row in df.iterrows():
        feats = {}
        
        # --- 基本信息 ---
        tables = parse_tables(row.get('Tables', ''))
        joins = parse_joins(row.get('Join Conditions', ''))
        predicates = parse_predicates(row.get('Predicates', ''))
        
        feats['num_tables'] = len(tables)
        feats['num_joins'] = len(joins)
        feats['num_predicates'] = len(predicates)
        
        # --- 表存在性特征 ---
        for t in ALL_TABLES:
            feats[f'has_table_{t}'] = 1 if t in tables else 0
        
        # --- 谓词特征：为每个可能的列创建特征 ---
        # 先按列分组谓词
        pred_map = {}  # col -> [(op, val), ...]
        for col, op, val in predicates:
            if col not in pred_map:
                pred_map[col] = []
            pred_map[col].append((op, val))
        
        # 为每个已知列创建特征
        for col_name in ALL_PRED_COLUMNS:
            stats = col_stats_dict.get(col_name, {})
            cmin = stats.get('min', 0)
            cmax = stats.get('max', 1)
            ccard = stats.get('cardinality', 1)
            cnunique = stats.get('num_unique_values', 1)
            crange = cmax - cmin if cmax > cmin else 1
            
            if col_name in pred_map:
                ops_vals = pred_map[col_name]
                feats[f'{col_name}_used'] = 1
                feats[f'{col_name}_pred_count'] = len(ops_vals)
                
                # 分别处理每种操作符
                eq_vals = [v for op, v in ops_vals if op == '=']
                lt_vals = [v for op, v in ops_vals if op == '<']
                gt_vals = [v for op, v in ops_vals if op == '>']
                
                # 等值特征
                if eq_vals:
                    feats[f'{col_name}_has_eq'] = 1
                    feats[f'{col_name}_eq_val'] = eq_vals[0]
                    feats[f'{col_name}_eq_val_norm'] = (eq_vals[0] - cmin) / crange
                    # 等值选择性估计
                    feats[f'{col_name}_eq_sel'] = 1.0 / max(cnunique, 1)
                else:
                    feats[f'{col_name}_has_eq'] = 0
                    feats[f'{col_name}_eq_val'] = 0
                    feats[f'{col_name}_eq_val_norm'] = 0
                    feats[f'{col_name}_eq_sel'] = 1.0
                
                # 小于特征
                if lt_vals:
                    feats[f'{col_name}_has_lt'] = 1
                    feats[f'{col_name}_lt_val'] = lt_vals[0]
                    feats[f'{col_name}_lt_val_norm'] = (lt_vals[0] - cmin) / crange
                    feats[f'{col_name}_lt_sel'] = max((lt_vals[0] - cmin) / crange, 0.0001)
                else:
                    feats[f'{col_name}_has_lt'] = 0
                    feats[f'{col_name}_lt_val'] = 0
                    feats[f'{col_name}_lt_val_norm'] = 0
                    feats[f'{col_name}_lt_sel'] = 1.0
                
                # 大于特征
                if gt_vals:
                    feats[f'{col_name}_has_gt'] = 1
                    feats[f'{col_name}_gt_val'] = gt_vals[0]
                    feats[f'{col_name}_gt_val_norm'] = (gt_vals[0] - cmin) / crange
                    feats[f'{col_name}_gt_sel'] = max((cmax - gt_vals[0]) / crange, 0.0001)
                else:
                    feats[f'{col_name}_has_gt'] = 0
                    feats[f'{col_name}_gt_val'] = 0
                    feats[f'{col_name}_gt_val_norm'] = 0
                    feats[f'{col_name}_gt_sel'] = 1.0
            else:
                feats[f'{col_name}_used'] = 0
                feats[f'{col_name}_pred_count'] = 0
                feats[f'{col_name}_has_eq'] = 0
                feats[f'{col_name}_eq_val'] = 0
                feats[f'{col_name}_eq_val_norm'] = 0
                feats[f'{col_name}_eq_sel'] = 1.0
                feats[f'{col_name}_has_lt'] = 0
                feats[f'{col_name}_lt_val'] = 0
                feats[f'{col_name}_lt_val_norm'] = 0
                feats[f'{col_name}_lt_sel'] = 1.0
                feats[f'{col_name}_has_gt'] = 0
                feats[f'{col_name}_gt_val'] = 0
                feats[f'{col_name}_gt_val_norm'] = 0
                feats[f'{col_name}_gt_sel'] = 1.0
        
        # --- 综合选择性估计 ---
        # 为每个表计算选择性
        table_selectivity = {}
        for col, op, val in predicates:
            # 提取表名
            table_name = col.split('.')[0] if '.' in col else ''
            if table_name not in table_selectivity:
                table_selectivity[table_name] = 1.0
            
            stats = col_stats_dict.get(col, {})
            cmin = stats.get('min', 0)
            cmax = stats.get('max', 1)
            crange = cmax - cmin if cmax > cmin else 1
            cnunique = stats.get('num_unique_values', 1)
            
            if op == '=':
                table_selectivity[table_name] *= 1.0 / max(cnunique, 1)
            elif op == '<':
                table_selectivity[table_name] *= max((val - cmin) / crange, 0.0001)
            elif op == '>':
                table_selectivity[table_name] *= max((cmax - val) / crange, 0.0001)
        
        # 总体选择性（所有谓词的乘积）
        overall_sel = 1.0
        for sel in table_selectivity.values():
            overall_sel *= sel
        feats['overall_selectivity'] = np.log1p(overall_sel * 1000)  # log scale
        
        # 每个表的基数估计
        for t in ALL_TABLES:
            # 获取该表的主键基数
            pk_col = f'{t}.id'
            if pk_col in col_stats_dict:
                feats[f'{t}_cardinality'] = np.log1p(col_stats_dict[pk_col]['cardinality'])
            else:
                feats[f'{t}_cardinality'] = 0
        
        # --- 启发式基数估计 ---
        heuristic_est = 1.0
        if tables:
            # 使用主表的基数作为基础
            main_table = tables[0]
            pk_col = f'{main_table}.id'
            if pk_col in col_stats_dict:
                base_card = col_stats_dict[pk_col]['cardinality']
            else:
                base_card = 1
            
            # 乘以选择性
            if main_table in table_selectivity:
                heuristic_est = base_card * table_selectivity[main_table]
            else:
                heuristic_est = base_card
            
            # 如果有连接，乘以连接因子
            if len(tables) > 1:
                # 简单的连接因子估计
                for jt in tables[1:]:
                    j_pk = f'{jt}.id'
                    if j_pk in col_stats_dict:
                        # 假设连接是外键连接
                        heuristic_est *= min(1.0, col_stats_dict[j_pk]['cardinality'] / max(base_card, 1))
        
        feats['heuristic_est_log'] = np.log1p(max(heuristic_est, 0.01))
        
        # --- 连接特征 ---
        # 从连接条件中提取涉及的表
        join_tables = set()
        for join_str in joins:
            for part in join_str.split('='):
                part = part.strip()
                if '.' in part:
                    join_tables.add(part.split('.')[0])
        feats['join_table_count'] = len(join_tables)
        
        # --- 谓词操作符分布 ---
        eq_count = sum(1 for _, op, _ in predicates if op == '=')
        lt_count = sum(1 for _, op, _ in predicates if op == '<')
        gt_count = sum(1 for _, op, _ in predicates if op == '>')
        feats['eq_count'] = eq_count
        feats['lt_count'] = lt_count
        feats['gt_count'] = gt_count
        
        features_list.append(feats)
    
    return pd.DataFrame(features_list)

# ============================================================
# 3. 特征提取
# ============================================================
print("\n" + "=" * 60)
print("2. 特征提取")
print("=" * 60)

print("提取训练特征...")
X_train = extract_features(train_df, col_stats_dict)
y_train = train_df['Cardinality'].values.astype(float)

print("提取测试特征...")
X_test = extract_features(test_df, col_stats_dict)

print(f"训练特征维度: {X_train.shape}")
print(f"测试特征维度: {X_test.shape}")

# 确保训练集和测试集列一致
common_cols = X_train.columns.intersection(X_test.columns)
X_train = X_train[common_cols]
X_test = X_test[common_cols]

print(f"共同特征数: {len(common_cols)}")

# ============================================================
# 4. 模型训练
# ============================================================
print("\n" + "=" * 60)
print("3. 模型训练")
print("=" * 60)

# 目标值取对数（基数通常跨越多个数量级）
y_train_log = np.log1p(y_train)

# 划分训练/验证集
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train_log, test_size=0.1, random_state=42
)

print(f"训练集: {len(X_tr)}, 验证集: {len(X_val)}")

# 创建 LightGBM Dataset
train_data = lgb.Dataset(X_tr, label=y_tr)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# 模型参数
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 127,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_data_in_leaf': 20,
    'min_sum_hessian_in_leaf': 1e-3,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'verbose': -1,
    'seed': 42,
    'num_threads': -1,
}

print("开始训练 LightGBM 模型...")
model = lgb.train(
    params,
    train_data,
    num_boost_round=2000,
    valid_sets=[train_data, val_data],
    callbacks=[
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(period=200)
    ]
)

# ============================================================
# 5. 验证集评估
# ============================================================
print("\n" + "=" * 60)
print("4. 验证集评估")
print("=" * 60)

y_val_pred_log = model.predict(X_val)
y_val_pred = np.expm1(y_val_pred_log)
y_val_true = np.expm1(y_val)

# 计算 Q-Error
q_errors = np.maximum(y_val_pred / y_val_true, y_val_true / y_val_pred)
mean_q_error = np.mean(q_errors)
median_q_error = np.median(q_errors)
p90_q_error = np.percentile(q_errors, 90)
p95_q_error = np.percentile(q_errors, 95)
p99_q_error = np.percentile(q_errors, 99)

print(f"Mean Q-Error:    {mean_q_error:.4f}")
print(f"Median Q-Error:  {median_q_error:.4f}")
print(f"90th Q-Error:    {p90_q_error:.4f}")
print(f"95th Q-Error:    {p95_q_error:.4f}")
print(f"99th Q-Error:    {p99_q_error:.4f}")

# ============================================================
# 6. 测试集预测
# ============================================================
print("\n" + "=" * 60)
print("5. 测试集预测")
print("=" * 60)

y_test_pred_log = model.predict(X_test)
y_test_pred = np.expm1(y_test_pred_log)

# 确保预测值 > 0
y_test_pred = np.maximum(y_test_pred, 1)
y_test_pred = np.round(y_test_pred).astype(int)

# ============================================================
# 7. 生成提交文件
# ============================================================
print("\n" + "=" * 60)
print("6. 生成提交文件")
print("=" * 60)

submission = pd.DataFrame({
    'Id': test_df['Id'],
    'Cardinality': y_test_pred
})

output_path = "c:/Users/sunqi/Desktop/CODE/hello-code/MachineLearning/submission.csv"
submission.to_csv(output_path, index=False)
print(f"提交文件已保存到: {output_path}")
print(f"预测样本数: {len(submission)}")

# 显示一些统计信息
print(f"\n预测基数统计:")
print(f"  Min:      {y_test_pred.min():,}")
print(f"  Max:      {y_test_pred.max():,}")
print(f"  Mean:     {y_test_pred.mean():,.0f}")
print(f"  Median:   {np.median(y_test_pred):,.0f}")

# 对比训练集真实值
print(f"\n训练集真实基数统计:")
print(f"  Min:      {y_train.min():,.0f}")
print(f"  Max:      {y_train.max():,.0f}")
print(f"  Mean:     {y_train.mean():,.0f}")
print(f"  Median:   {np.median(y_train):,.0f}")

print("\n" + "=" * 60)
print("完成!")
print("=" * 60)
