# Fin 数据集合成与评估任务指引

目标：针对 `TabPFN/sync_data_proj/datasets/Fin_data` 下的每个金融数据集，复用我们在 EHR 任务中的流程，分别用 TabPFGen 以及其他合成方法生成数据，再用统一的 ML 模型在真实测试集上评估并记录指标。

## 预备
1) 浏览 `Fin_data` 目录，确认每个子数据集路径、文件名（假设有 `dataset.csv` 和数据字典/说明文件）。  
2) 阅读对应的字段说明（如 *.names / README），确定特征列、标签列、类型（整数/浮点/类别）以及缺失值标记规则（如 `?`、`0` 等）。

## 清洗 + 划分
1) 加载真实集 `dataset.csv`。  
2) 将缺失标记统一为 NaN，按类型填充：连续/计数列用中位数；类别/布尔列用众数；必要时将 0 视为缺失（依字段说明）。  
3) 强制列类型：类别/布尔→int，计数→int，连续→float。记录每列的 min/max 以便后续裁剪。  
4) 确认标签分布，保留标签列为 int/离散类。  
5) 随机划分 80% 作为训练（供 TabPFGen 生成、以及直接训练 ML 基线），20% 作为测试（ML 基线和合成数据模型共同使用）。保持划分种子固定（如 42），并确保分层抽样保持类分布。
6) 保存训练集和测试集到/home/zhu11/TabPFN/sync_data_proj/datasets/Fin_data/{dataset_name}/{tr_te}.csv

## TabPFGen 合成（基于训练集 80%）
1) 使用训练集 80% 作为输入；设置 `n_sgld_steps=1000`（或更高视资源）、device='cuda'，seed=42。  
2) 按类均衡初始化并生成 3 个规模（基于训练集行数）：
   - 100%：`/home/zhu11/TabPFN/sync_data_proj/datasets/Fin_data/{dataset_name}/synthetic_100.csv`（=训练集大小）  
   - 200%：`/home/zhu11/TabPFN/sync_data_proj/datasets/Fin_data/{dataset_name}/synthetic_200.csv`  
   - 1000%：`/home/zhu11/TabPFN/sync_data_proj/datasets/Fin_data/{dataset_name}/synthetic_1000.csv`  
3) 生成后裁剪到真实数据 min/max，并恢复整数列取整。保存到对应路径（覆盖已有）。


## 评估模型
对每个合成集（TabPFGen 三份 ），使用统一模型在训练集（真实 80% 或合成集）上训练，并在同一真实测试集（20% 保留）上测试：
1) 多分类：LabelEncoder 处理标签，宏平均指标；二分类：直接 0/1。  
2) 模型：logistic（带 StandardScaler）、xgboost、lightgbm、catboost、mlp（标准化）。  
3) 指标：accuracy、precision_macro、recall_macro、f1_macro、roc_auc_macro_ovr（类别缺失则 nan）、log_loss。

## 记录
1) 将 TabPFGen 三份的结果追加到 `TabPFN/sync_data_proj/datasets/document/fin_metric.md`（新增 Fin 数据集小节，按 100/200/1000 表格列出）。  
2) 若有特殊清洗/字段约定（如某些 0 视为缺失），在文档中简要备注，方便复现。

## 交付
1) 确认所有合成文件存在且行数/标签分布正确。  
2) 确认 metric 文档更新完整无遗漏。  
3) 保留评估脚本（如临时脚本）或删除临时文件，保证仓库干净。***
