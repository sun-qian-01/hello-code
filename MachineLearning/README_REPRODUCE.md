# Reproduce Compliant Cross-Fit Blend

主提交文件：

```text
submission_compliant_crossfit_blend_mean.csv
```

复现命令：

```powershell
cd MachineLearning
Expand-Archive source_code.zip -DestinationPath source_code -Force
cd source_code
pip install -r requirements.txt
python reproduce_compliant_crossfit_blend.py
```

流程：

1. 检查核心数据文件和基础 OOF 专家是否存在。
2. 如缺少 KNN residual 中间结果，运行 `shape_knn_residual_v2.py`。
3. 如缺少嵌套路由中间结果，运行 `compliant_expert_router.py`。
4. 如缺少最终提交，运行 `compliant_crossfit_blend.py`。
5. 校验最终提交行数、列名、ID 顺序和正整数预测。

`source_code.zip` 内包含源码、给定数据文件、基础专家 OOF/提交和最终融合所需中间产物。该流程只使用给定训练集、测试集查询字段、schema、列统计信息以及由这些文件派生的 OOF 预测。
