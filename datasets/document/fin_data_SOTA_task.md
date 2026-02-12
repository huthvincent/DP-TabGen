# Fin 数据集 SOTA 合成与评估执行手册

目标：对 `TabPFN/sync_data_proj/datasets/Fin_data` 下的每个数据集，复用 Arrhythmia SOTA 流程，用多种合成器在训练集 80% 上生成合成数据，再在真实测试集 20% 上评估并记录指标。

## 数据集
- `australian_credit_approval`
- `german_credit`
- （如有新增，同样遵循本流程）

## 通用设置
- 环境：`conda run -n sync_data python ...`
- 随机种子：`42`
- 划分：对 `data.csv` 做分层 80/20（label 列为 `label`），得到 `train.csv`、`test.csv`，保存在数据集目录。若已存在，可复用；否则先生成。
- 预处理：`?` 视为 NaN；特征列用中位数填充，标签用众数；数值列强制 numeric，标签转 int。

## 合成器与输出命名（基于 80% 训练集，样本数 = 训练集行数）
- SDV GaussianCopula → `sync_gaussian_copula.csv`
- SDV CTGAN（示例：epochs=50, batch_size=64, pac=1, cuda=False）→ `sync_sdv_ctgan.csv`
- SDV TVAE（示例：epochs=50, batch_size=64, cuda=False）→ `sync_sdv_tvae.csv`
- SynthCity DDPM（示例：n_iter=120, batch_size=128, device=cpu; 若 torch 无 RMSNorm，先补丁）→ `sync_synthcity_ddpm.csv`
- TabPFNConditionalGenerator（如可用；若 TabPFN 不可用，走 sklearn fallback）→ `sync_tabpfn.csv`
- 以上文件均放在对应数据集目录（例如 `.../australian_credit_approval/sync_sdv_ctgan.csv`）。

## 生成要点
1) 仅使用训练集 80% 作为输入，样本数与训练集等量。
2) 列名、类型与清洗后的训练集保持一致。
3) 记录合成集的标签集合。

## 评估
1) 模型组：logistic（带标准化）、xgboost、lightgbm、catboost、mlp（标准化）。
2) 测试集：真实 20%（`test.csv`），但仅保留“真实测试标签 ∩ 合成集标签”的样本；若合成集缺类，AUC 记 `nan`，log_loss 需传入 labels 以避免报错。
3) 指标：accuracy、precision_macro、recall_macro、f1_macro、roc_auc_macro_ovr（缺类则 nan）、log_loss。
4) 训练数据：每个合成集分别训练；测试统一用真实 20% 子集。

## 记录
- 将结果写入 `TabPFN/sync_data_proj/datasets/document/SOTA_fin_metric.md`，按数据集 → 合成方法 → 模型表格展示，注明缺失类别/AUC=nan 情况。
- 在表格前简要说明合成器超参/设备（CPU/GPU）和生成样本数。

## 参考代码片段（伪代码）
```python
# 划分
train, test = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

# 示例：SDV CTGAN
meta = SingleTableMetadata(); meta.detect_from_dataframe(train); meta.update_column("label", sdtype="categorical")
ctgan = CTGANSynthesizer(meta, epochs=50, batch_size=64, pac=1, cuda=False)
ctgan.fit(train)
syn = ctgan.sample(num_rows=len(train))
syn.to_csv("sync_sdv_ctgan.csv", index=False)
```

## 注意事项
- SynthCity DDPM 在 torch<2.3 时需补 `torch.nn.RMSNorm` 再导入 synthcity。
- TabPFN 可能不可用时自动回退 sklearn，不要中断流程。
- 若评估时报类别不一致，先做标签交集过滤再算指标。
