# Fin 数据集 SOTA 合成集 → 真实测试指标

说明：在训练集 80%（seed=42，分层）上生成等量合成数据；测试集为真实 20%，先取“真实标签 ∩ 合成标签”子集，再评估 logistic（标准化）、xgboost、lightgbm、catboost、mlp（标准化）；缺类时 AUC 记为 nan，log_loss 传入 labels 以避免报错。

## australian_credit_approval（train=552, test=138, seed=42）
- 清洗：'?→NaN'，特征列中位数填充，标签众数填充并转 int，保持原始整型列取整。
- 合成器（样本数=训练行数，CPU）：
  - sync_gaussian_copula.csv — SDV GaussianCopula (default, seed=42, CPU)；标签 {0:319, 1:233}
  - sync_sdv_ctgan.csv — SDV CTGAN (epochs=50, batch_size=64, pac=1, cuda=False, seed=42)；标签 {0:302, 1:250}
  - sync_sdv_tvae.csv — SDV TVAE (epochs=50, batch_size=64, cuda=False, seed=42)；标签 {0:333, 1:219}
  - sync_synthcity_ddpm.csv — SynthCity DDPM (ddpm, n_iter=120, batch_size=128, device=cpu, seed=42)；标签 {0:317, 1:235}
  - sync_tabpfn.csv — TabPFNConditionalGenerator (sample_count=train_rows, seed=42, num_gibbs_rounds=3, batch_size=256, clip_quantile_low/high=0.01/0.99, preserve_label_distribution=True, use_gpu=False)；标签 {0:306, 1:246}

### sync_gaussian_copula.csv
| model | accuracy | precision_macro | recall_macro | f1_macro | roc_auc_macro_ovr | log_loss |
| --- | --- | --- | --- | --- | --- | --- |
| logistic | 0.7681 | 0.7744 | 0.7547 | 0.7582 | 0.8725 | 0.4811 |
| xgboost | 0.7101 | 0.7195 | 0.6909 | 0.6914 | 0.8041 | 0.6446 |
| lightgbm | 0.7101 | 0.7280 | 0.6875 | 0.6864 | 0.7965 | 0.7481 |
| catboost | 0.7464 | 0.7594 | 0.7284 | 0.7310 | 0.8209 | 0.5906 |
| mlp | 0.7029 | 0.6987 | 0.6980 | 0.6983 | 0.7837 | 1.4258 |

### sync_sdv_ctgan.csv
| model | accuracy | precision_macro | recall_macro | f1_macro | roc_auc_macro_ovr | log_loss |
| --- | --- | --- | --- | --- | --- | --- |
| logistic | 0.5870 | 0.6391 | 0.5379 | 0.4596 | 0.3717 | 0.7125 |
| xgboost | 0.5797 | 0.5693 | 0.5655 | 0.5642 | 0.5574 | 0.8627 |
| lightgbm | 0.5725 | 0.5604 | 0.5556 | 0.5527 | 0.5301 | 1.0081 |
| catboost | 0.5870 | 0.5763 | 0.5617 | 0.5511 | 0.5546 | 0.7982 |
| mlp | 0.5435 | 0.5204 | 0.5160 | 0.4993 | 0.4903 | 2.2747 |

### sync_sdv_tvae.csv
| model | accuracy | precision_macro | recall_macro | f1_macro | roc_auc_macro_ovr | log_loss |
| --- | --- | --- | --- | --- | --- | --- |
| logistic | 0.7246 | 0.7288 | 0.7311 | 0.7244 | 0.8484 | 1.0803 |
| xgboost | 0.8261 | 0.8247 | 0.8288 | 0.8252 | 0.9023 | 0.5271 |
| lightgbm | 0.8333 | 0.8326 | 0.8370 | 0.8326 | 0.9042 | 1.0469 |
| catboost | 0.8333 | 0.8326 | 0.8370 | 0.8326 | 0.9146 | 0.4919 |
| mlp | 0.7464 | 0.7435 | 0.7455 | 0.7441 | 0.8158 | 4.2292 |

### sync_synthcity_ddpm.csv
| model | accuracy | precision_macro | recall_macro | f1_macro | roc_auc_macro_ovr | log_loss |
| --- | --- | --- | --- | --- | --- | --- |
| logistic | 0.7681 | 0.7667 | 0.7701 | 0.7669 | 0.8437 | 0.5718 |
| xgboost | 0.7174 | 0.7180 | 0.7042 | 0.7061 | 0.8050 | 0.8130 |
| lightgbm | 0.7464 | 0.7435 | 0.7455 | 0.7441 | 0.8536 | 0.7320 |
| catboost | 0.7246 | 0.7212 | 0.7175 | 0.7187 | 0.7974 | 0.7056 |
| mlp | 0.7681 | 0.7651 | 0.7667 | 0.7657 | 0.8529 | 1.1360 |

### sync_tabpfn.csv
| model | accuracy | precision_macro | recall_macro | f1_macro | roc_auc_macro_ovr | log_loss |
| --- | --- | --- | --- | --- | --- | --- |
| logistic | 0.5797 | 0.5720 | 0.5433 | 0.5102 | 0.5323 | 0.6912 |
| xgboost | 0.5725 | 0.5649 | 0.5641 | 0.5642 | 0.5667 | 0.8126 |
| lightgbm | 0.5652 | 0.5593 | 0.5593 | 0.5593 | 0.5650 | 0.9307 |
| catboost | 0.5797 | 0.5718 | 0.5706 | 0.5707 | 0.5548 | 0.7657 |
| mlp | 0.6159 | 0.6112 | 0.6116 | 0.6114 | 0.5902 | 2.5861 |

## german_credit（train=800, test=200, seed=42）
- 清洗：'?→NaN'，特征列中位数填充，标签众数填充并转 int，保持原始整型列取整。
- 合成器（样本数=训练行数，CPU）：
  - sync_gaussian_copula.csv — SDV GaussianCopula (default, seed=42, CPU)；标签 {0:566, 1:234}
  - sync_sdv_ctgan.csv — SDV CTGAN (epochs=50, batch_size=64, pac=1, cuda=False, seed=42)；标签 {0:506, 1:294}
  - sync_sdv_tvae.csv — SDV TVAE (epochs=50, batch_size=64, cuda=False, seed=42)；标签 {0:694, 1:106}
  - sync_synthcity_ddpm.csv — SynthCity DDPM (ddpm, n_iter=120, batch_size=128, device=cpu, seed=42)；标签 {0:726, 1:74}
  - sync_tabpfn.csv — TabPFNConditionalGenerator (sample_count=train_rows, seed=42, num_gibbs_rounds=3, batch_size=256, clip_quantile_low/high=0.01/0.99, preserve_label_distribution=True, use_gpu=False)；标签 {0:560, 1:240}

### sync_gaussian_copula.csv
| model | accuracy | precision_macro | recall_macro | f1_macro | roc_auc_macro_ovr | log_loss |
| --- | --- | --- | --- | --- | --- | --- |
| logistic | 0.6850 | 0.5175 | 0.5036 | 0.4483 | 0.6405 | 0.5880 |
| xgboost | 0.6550 | 0.5381 | 0.5250 | 0.5167 | 0.5411 | 0.7688 |
| lightgbm | 0.6450 | 0.5200 | 0.5131 | 0.5026 | 0.5537 | 0.8546 |
| catboost | 0.6650 | 0.5270 | 0.5131 | 0.4907 | 0.5548 | 0.6814 |
| mlp | 0.6100 | 0.5133 | 0.5119 | 0.5110 | 0.5185 | 2.1510 |

### sync_sdv_ctgan.csv
| model | accuracy | precision_macro | recall_macro | f1_macro | roc_auc_macro_ovr | log_loss |
| --- | --- | --- | --- | --- | --- | --- |
| logistic | 0.7100 | 0.6579 | 0.5357 | 0.4978 | 0.6045 | 0.6130 |
| xgboost | 0.5450 | 0.4989 | 0.4988 | 0.4949 | 0.4638 | 0.9215 |
| lightgbm | 0.5450 | 0.4989 | 0.4988 | 0.4949 | 0.4986 | 0.9997 |
| catboost | 0.5350 | 0.4548 | 0.4536 | 0.4541 | 0.4969 | 0.7904 |
| mlp | 0.5200 | 0.4560 | 0.4524 | 0.4530 | 0.4810 | 2.3349 |

### sync_sdv_tvae.csv
| model | accuracy | precision_macro | recall_macro | f1_macro | roc_auc_macro_ovr | log_loss |
| --- | --- | --- | --- | --- | --- | --- |
| logistic | 0.7300 | 0.6773 | 0.6738 | 0.6754 | 0.7510 | 0.9087 |
| xgboost | 0.7250 | 0.6681 | 0.6560 | 0.6608 | 0.7163 | 0.8487 |
| lightgbm | 0.7250 | 0.6671 | 0.6512 | 0.6571 | 0.7195 | 1.4458 |
| catboost | 0.7450 | 0.6959 | 0.6940 | 0.6950 | 0.7492 | 0.7713 |
| mlp | 0.7600 | 0.7476 | 0.6333 | 0.6445 | 0.7396 | 5.0411 |

### sync_synthcity_ddpm.csv
| model | accuracy | precision_macro | recall_macro | f1_macro | roc_auc_macro_ovr | log_loss |
| --- | --- | --- | --- | --- | --- | --- |
| logistic | 0.6900 | 0.5621 | 0.5167 | 0.4749 | 0.6317 | 1.6540 |
| xgboost | 0.6900 | 0.6050 | 0.5738 | 0.5751 | 0.6480 | 1.2543 |
| lightgbm | 0.6950 | 0.6069 | 0.5631 | 0.5592 | 0.6398 | 1.3642 |
| catboost | 0.7050 | 0.6270 | 0.5750 | 0.5737 | 0.6983 | 0.8242 |
| mlp | 0.6850 | 0.4926 | 0.4988 | 0.4353 | 0.5986 | 2.9111 |

### sync_tabpfn.csv
| model | accuracy | precision_macro | recall_macro | f1_macro | roc_auc_macro_ovr | log_loss |
| --- | --- | --- | --- | --- | --- | --- |
| logistic | 0.6800 | 0.3469 | 0.4857 | 0.4048 | 0.4189 | 0.6567 |
| xgboost | 0.6050 | 0.4122 | 0.4512 | 0.4198 | 0.4544 | 0.8614 |
| lightgbm | 0.6100 | 0.4516 | 0.4690 | 0.4495 | 0.4746 | 0.9609 |
| catboost | 0.6500 | 0.4953 | 0.4976 | 0.4726 | 0.4367 | 0.7929 |
| mlp | 0.5750 | 0.4340 | 0.4488 | 0.4371 | 0.4185 | 2.6858 |
