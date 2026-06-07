# AI 工具使用说明

本项目使用 AI 编程助手辅助完成代码阅读、实验设计、脚本编写、结果整理和报告撰写。AI 工具使用方式如下。

## 使用方式

1. 读取 `train.csv`、`test.csv`、`schematext.sql`、`column_min_max_vals.csv` 和已有脚本，理解任务、数据结构和评价指标。
2. 辅助实现 SQL 字段解析函数，将 `Tables`、`Join Conditions`、`Predicates` 转换为模板 key 和数值特征。
3. 辅助设计并实现 train-only 模型，包括 exact-shape residual surface、value target encoding、KNN residual、nested expert router 和 cross-fitted blend。
4. 辅助运行本地 OOF 评估，检查 Mean Q-Error、分位数、测试 shape 分布加权指标和提交文件格式。
5. 根据实验结果整理最终报告和复现说明。

## 关键流程

AI 首先帮助比较了全局回归、模板专家、目标编码和局部残差曲面等路线。随着实验推进，发现测试集 exact shape 覆盖率很高，因此重点转向模板内建模和专家路由。最后，AI 辅助实现 `compliant_crossfit_blend.py`，使用外层 5 折交叉验证优化融合权重，避免直接在全量 OOF 上调参造成过拟合。

## 人工检查与思考过程

人工主要完成以下工作：

- 判断作业规范要求不能使用测试答案或网上额外数据，因此最终报告选择合规的 `submission_compliant_crossfit_blend_mean.csv`；
- 检查线上反馈和本地 OOF 是否一致，避免只根据单次本地提升选择不稳定方案；
- 确认最终提交文件格式合法；
- 要求报告包含任务理解、数据处理、特征设计、模型训练、实验结果和误差分析；
- 对 AI 生成的报告内容进行方向修正，确保以合规提交为主。

## 合规性说明

最终报告主方法只使用给定训练集、测试集查询字段、schema 和列统计信息，不使用测试答案，也不使用网上 JOB/IMDb 原始表作为训练或预测数据。历史探索中曾尝试外部数据路线，但该路线未作为本报告主方案。
