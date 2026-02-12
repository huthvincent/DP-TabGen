# path: README.md
# TabPFN 条件合成数据最小实践

本项目提供一套可直接落地的 DP-TabPFN 数据生成流水线，涵盖环境部署、配置管理、命令行工具以及可视化测试。核心思路：对每一列建立 DP-TabPFN 条件模型，采用 Gibbs 式迭代采样，在保证类别/整数特性的同时近似还原联合分布。

## 目录结构
- `scripts/bootstrap.sh`：一键部署（创建/复用 `tabpfn` conda 环境，自动安装 GPU/CPU 版 PyTorch、TabPFN 与依赖）。
- `src/generator.py`：生成器核心，负责类型识别、编码、TabPFN/回退建模、残差自助与迭代采样。
- `src/cli_generate.py`：命令行入口，支持 YAML 配置或参数覆盖。
- `config/config.yaml`：包含中文注释的参数模板，覆盖路径、采样规模、GPU、种子、裁剪、分布保持等。
- `sync_data_proj/test_pipeline.py`：端到端测试，构造混合类型数据、运行合成器、输出统计 + 直方图/柱状图对比。
- `outputs/`、`sync_data_proj/*.csv` 等运行后生成的产物（首次运行会自动创建目录）。

## 环境准备
1. **确保已有 Anaconda/Miniconda**，并在服务器上具备必要的 CUDA 驱动（若需 GPU）。
2. 运行脚本：
   ```bash
   bash scripts/bootstrap.sh
   ```
   - 自动检测 `nvidia-smi`，有 GPU 时安装 `torch` cu118 版本；否则安装 CPU 轮子。
   - 先尝试 `pip install git+https://github.com/automl/TabPFN.git`，失败后回退到 PyPI 包。
   - 安装 `pandas/numpy/scikit-learn/pyyaml/matplotlib/tqdm` 等依赖。
   - 任一步失败立即退出，避免半成品环境。

## 使用方式
### 方式一：完全依赖配置
```bash
conda run -n tabpfn python src/cli_generate.py --config config/config.yaml
```
配置文件中的 `original_data_path/output_data_path/sample_count` 等项会直接生效。

### 方式二：命令行即时传参
```bash
conda run -n tabpfn python src/cli_generate.py \
  --input input.csv \
  --output synth.csv \
  --n 1000 \
  --use-gpu \
  --seed 123 \
  --num-gibbs-rounds 5
```
- `--cpu-only` 可强制禁用 GPU。
- `--target-column label --preserve-label-distribution` 可在生成后重采样指定列以保持边际比率。
- `--integer-columns age count` 可手动声明整数列，防止被误判成连续变量。

### 快速测试与可视化
```bash
conda run -n tabpfn python sync_data_proj/test_pipeline.py
```
- 自动构造数值 + 类别混合数据 `sync_data_proj/input_test.csv`。
- 调用 CLI 生成 `sync_data_proj/output_test.csv`。
- 输出简单统计对比，并生成 `sync_data_proj/distribution_compare.png`（按列绘制直方图/柱状图）。

## 配置项说明（摘自 `config/config.yaml`）
- `original_data_path`：原始 CSV 路径，可被 `--input` 覆盖。
- `output_data_path`：结果输出路径，可被 `--output` 覆盖。
- `sample_count`：目标样本数；默认=原始行数；可用 `--n` 改写。
- `use_gpu`：`true` 时优先走 GPU；若检测不到 CUDA，会打印告警并自动退回 CPU。
- `seed`：全局随机种子。
- `num_gibbs_rounds`：每轮对所有列随机排序依次条件采样；建议 3~5 轮平衡耗时与精度。
- `batch_size`：模型预测批大小；若显存紧张可调小。
- `clip_quantile_low/high`：连续/整数列按分位裁剪，防止极端噪点。
- `integer_columns`：强制视为整数的列名/索引列表，留空表示自动识别。
- `target_column` + `preserve_label_distribution`：指定列后可保持其边际分布（适合分类标签）。

## GPU/CPU 切换与回退策略
- CLI/配置中的 `use_gpu` 控制 TabPFN 实例化时的 `device`。若发现 `torch.cuda.is_available()` 为假，会自动降级并打印 WARNING。
- `TabPFNClassifier/Regressor` 未安装或初始化失败时，代码会降级到 `RandomForestClassifier/Regressor`，同时发出显式警告，确保流程不断档。

## 常见问题 (FAQ)
1. **列类型误判**：若整数编码的类别被识别为连续列，可在配置 `integer_columns` 中加入列名或索引；必要时先把该列转换为字符串。
2. **内存/显存占用过高**：调小 `batch_size` 或 `sample_count`；也可减少 `num_gibbs_rounds`。
3. **随机性复现**：固定 `seed`，并在 CLI 中显式传入同样的 `--seed` 即可复现。
4. **极端值偏差**：通过 `clip_quantile_*` 限制生成范围；若仍需原样保留，可把上下分位设置为 0/1。
5. **TabPFN 回退到 sklearn**：检查 bootstrap 日志确保 TabPFN 安装成功；若 GPU 显存不足，可在 CPU 模式下复现。

## 运行示例
```bash
# 1) 一键部署（Ubuntu，已安装 Anaconda）
bash scripts/bootstrap.sh

# 2) 使用配置运行
conda run -n tabpfn python src/cli_generate.py --config config/config.yaml

# 3) 直接传参运行
conda run -n tabpfn python src/cli_generate.py \
  --input input.csv --output synth.csv --n 1000 --use-gpu

# 4) 测试与可视化
conda run -n tabpfn python sync_data_proj/test_pipeline.py
```

