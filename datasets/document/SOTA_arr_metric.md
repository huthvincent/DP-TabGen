# Arrhythmia Synthetic (Other Generators) → Real Test Metrics

说明：在不同合成集上训练（logistic, xgboost, lightgbm, catboost, mlp），在真实测试集 `EHR_datasets/arrhythmia/dataset.csv` 上评估；宏平均 precision/recall/f1 与 OVR AUC，log_loss；若生成集缺少某些真实类别，则仅在训练集中存在的类别上评估，缺失 AUC 记为 `nan`。

## sync_gaussian_copula.csv
| model | accuracy | precision_macro | recall_macro | f1_macro | roc_auc_macro_ovr | log_loss |
| --- | --- | --- | --- | --- | --- | --- |
| logistic | 0.1018 | 0.0482 | 0.0693 | 0.0321 | 0.4565 | 14.0267 |
| xgboost | 0.4978 | 0.0847 | 0.0848 | 0.0715 | 0.5900 | 1.8083 |
| lightgbm | 0.4978 | 0.0737 | 0.0819 | 0.0691 | 0.5224 | 3.0262 |
| catboost | 0.5442 | 0.0860 | 0.0814 | 0.0637 | 0.6790 | 1.6423 |
| mlp | 0.2566 | 0.0670 | 0.0889 | 0.0685 | 0.3832 | 8.9166 |

## sync_sdv_ctgan.csv
| model | accuracy | precision_macro | recall_macro | f1_macro | roc_auc_macro_ovr | log_loss |
| --- | --- | --- | --- | --- | --- | --- |
| logistic | 0.3556 | 0.0587 | 0.0728 | 0.0639 | 0.3945 | 5.1095 |
| xgboost | 0.5467 | 0.0873 | 0.0850 | 0.0622 | 0.5948 | 2.1031 |
| lightgbm | 0.5400 | 0.0661 | 0.0840 | 0.0615 | 0.5224 | 4.2682 |
| catboost | 0.5422 | 0.0453 | 0.0830 | 0.0586 | 0.6772 | 1.6272 |
| mlp | 0.2756 | 0.0472 | 0.0524 | 0.0484 | 0.3735 | 4.3319 |

## sync_sdv_tvae.csv
| model | accuracy | precision_macro | recall_macro | f1_macro | roc_auc_macro_ovr | log_loss |
| --- | --- | --- | --- | --- | --- | --- |
| logistic | 0.7220 | 0.4598 | 0.4665 | 0.4624 | nan | 2.1486 |
| xgboost | 0.8305 | 0.4153 | 0.5000 | 0.4537 | nan | 0.9403 |
| lightgbm | 0.7458 | 0.5149 | 0.5127 | 0.5128 | nan | 2.0694 |
| catboost | 0.8305 | 0.4153 | 0.5000 | 0.4537 | nan | 0.7786 |
| mlp | 0.8305 | 0.4153 | 0.5000 | 0.4537 | nan | 5.8112 |

## sync_synthcity_ddpm.csv
| model | accuracy | precision_macro | recall_macro | f1_macro | roc_auc_macro_ovr | log_loss |
| --- | --- | --- | --- | --- | --- | --- |
| logistic | 0.1049 | 0.4780 | 0.4916 | 0.1024 | nan | 1.4574 |
| xgboost | 0.7079 | 0.5197 | 0.5512 | 0.4965 | nan | 0.6202 |
| lightgbm | 0.6966 | 0.5242 | 0.5658 | 0.4978 | nan | 0.7517 |
| catboost | 0.2772 | 0.5196 | 0.5441 | 0.2638 | nan | 0.8884 |
| mlp | 0.0824 | 0.2896 | 0.4793 | 0.0772 | nan | 2.7787 |

## sync_tabpfn.csv
| model | accuracy | precision_macro | recall_macro | f1_macro | roc_auc_macro_ovr | log_loss |
| --- | --- | --- | --- | --- | --- | --- |
| logistic | 0.3164 | 0.0727 | 0.0738 | 0.0715 | 0.4321 | 5.7485 |
| xgboost | 0.5243 | 0.0722 | 0.0808 | 0.0645 | 0.5583 | 2.1439 |
| lightgbm | 0.5221 | 0.0957 | 0.0800 | 0.0647 | 0.5300 | 4.1833 |
| catboost | 0.5376 | 0.0570 | 0.0775 | 0.0566 | 0.6060 | 1.6839 |
| mlp | 0.4159 | 0.0696 | 0.0768 | 0.0721 | 0.4481 | 5.6303 |
