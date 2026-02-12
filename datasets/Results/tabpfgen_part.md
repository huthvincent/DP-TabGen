# TabPFGen Pipeline Results

## australian_credit_approval
- path: `/home/zhu11/TabPFN/sync_data_proj/datasets/Fin_data/australian_credit_approval`
- data: `/home/zhu11/TabPFN/sync_data_proj/datasets/Fin_data/australian_credit_approval/data.csv`
- samples: 690
- TabPFGen: {'n_sgld_steps': 600, 'sgld_step_size': 0.01, 'sgld_noise_scale': 0.005, 'jitter': 0.01, 'synthetic_factor': 1.0, 'energy_subsample': 2048}
- seed: 42

| generator | scheme | model | accuracy | precision | recall | f1 | roc_auc | log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | real | logistic | 0.8710 ± 0.0268 | 0.8433 ± 0.0268 | 0.8732 ± 0.0596 | 0.8571 ± 0.0318 | 0.9292 ± 0.0208 | 0.3380 ± 0.0548 |
| baseline | real | xgboost | 0.8725 ± 0.0392 | 0.8552 ± 0.0599 | 0.8633 ± 0.0569 | 0.8578 ± 0.0426 | 0.9314 ± 0.0214 | 0.3802 ± 0.0910 |
| baseline | real | lightgbm | 0.8565 ± 0.0433 | 0.8419 ± 0.0576 | 0.8372 ± 0.0680 | 0.8381 ± 0.0504 | 0.9278 ± 0.0208 | 0.5814 ± 0.1572 |
| baseline | real | catboost | 0.8681 ± 0.0349 | 0.8514 ± 0.0574 | 0.8568 ± 0.0542 | 0.8526 ± 0.0381 | 0.9341 ± 0.0223 | 0.3556 ± 0.0800 |
| baseline | real | mlp | 0.8261 ± 0.0407 | 0.8028 ± 0.0424 | 0.8080 ± 0.0708 | 0.8045 ± 0.0489 | 0.9006 ± 0.0316 | 1.2799 ± 0.3489 |
| tabpfgen | synthetic | logistic | 0.8681 ± 0.0338 | 0.8490 ± 0.0523 | 0.8602 ± 0.0587 | 0.8530 ± 0.0370 | 0.9245 ± 0.0299 | 0.3481 ± 0.0749 |
| tabpfgen | synthetic | xgboost | 0.8739 ± 0.0451 | 0.8672 ± 0.0604 | 0.8502 ± 0.0819 | 0.8563 ± 0.0536 | 0.9290 ± 0.0297 | 0.4066 ± 0.1211 |
| tabpfgen | synthetic | lightgbm | 0.8609 ± 0.0506 | 0.8462 ± 0.0667 | 0.8438 ± 0.0808 | 0.8431 ± 0.0588 | 0.9232 ± 0.0330 | 0.7163 ± 0.2920 |
| tabpfgen | synthetic | catboost | 0.8739 ± 0.0385 | 0.8636 ± 0.0464 | 0.8535 ± 0.0755 | 0.8567 ± 0.0465 | 0.9363 ± 0.0220 | 0.3701 ± 0.0921 |
| tabpfgen | synthetic | mlp | 0.8304 ± 0.0485 | 0.8150 ± 0.0549 | 0.8051 ± 0.0979 | 0.8072 ± 0.0574 | 0.8890 ± 0.0357 | 1.3552 ± 0.3259 |

## heart
- path: `/home/zhu11/TabPFN/sync_data_proj/datasets/EHR_datasets/heart`
- data: `/home/zhu11/TabPFN/sync_data_proj/datasets/EHR_datasets/heart/dataset.csv`
- samples: 303
- TabPFGen: {'n_sgld_steps': 600, 'sgld_step_size': 0.01, 'sgld_noise_scale': 0.005, 'jitter': 0.01, 'synthetic_factor': 1.0, 'energy_subsample': 2048}
- seed: 42

| generator | scheme | model | accuracy | precision | recall | f1 | roc_auc | log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | real | logistic | 0.8316 ± 0.0554 | 0.8304 ± 0.0719 | 0.7984 ± 0.0745 | 0.8126 ± 0.0613 | 0.9112 ± 0.0199 | 0.3782 ± 0.0415 |
| baseline | real | xgboost | 0.8252 ± 0.0175 | 0.8349 ± 0.0530 | 0.7770 ± 0.0296 | 0.8034 ± 0.0129 | 0.8905 ± 0.0178 | 0.5070 ± 0.0733 |
| baseline | real | lightgbm | 0.8217 ± 0.0295 | 0.8308 ± 0.0653 | 0.7770 ± 0.0586 | 0.7999 ± 0.0307 | 0.8903 ± 0.0142 | 0.6184 ± 0.1338 |
| baseline | real | catboost | 0.8184 ± 0.0392 | 0.8172 ± 0.0697 | 0.7839 ± 0.0458 | 0.7986 ± 0.0413 | 0.8944 ± 0.0254 | 0.4872 ± 0.1032 |
| baseline | real | mlp | 0.8017 ± 0.0495 | 0.7874 ± 0.0672 | 0.7841 ± 0.0669 | 0.7838 ± 0.0533 | 0.8618 ± 0.0252 | 1.1675 ± 0.2692 |
| tabpfgen | synthetic | logistic | 0.8185 ± 0.0554 | 0.7982 ± 0.0682 | 0.8132 ± 0.0767 | 0.8040 ± 0.0594 | 0.8976 ± 0.0378 | 0.4104 ± 0.0806 |
| tabpfgen | synthetic | xgboost | 0.8119 ± 0.0318 | 0.8040 ± 0.0448 | 0.7841 ± 0.0506 | 0.7927 ± 0.0333 | 0.8893 ± 0.0175 | 0.5334 ± 0.1009 |
| tabpfgen | synthetic | lightgbm | 0.8019 ± 0.0265 | 0.8063 ± 0.0459 | 0.7550 ± 0.0830 | 0.7762 ± 0.0371 | 0.8727 ± 0.0109 | 0.8421 ± 0.1970 |
| tabpfgen | synthetic | catboost | 0.7920 ± 0.0324 | 0.7768 ± 0.0444 | 0.7696 ± 0.0552 | 0.7720 ± 0.0371 | 0.8765 ± 0.0079 | 0.5910 ± 0.1300 |
| tabpfgen | synthetic | mlp | 0.8085 ± 0.0329 | 0.7885 ± 0.0339 | 0.7992 ± 0.0811 | 0.7917 ± 0.0424 | 0.8593 ± 0.0180 | 1.2443 ± 0.2038 |

## bank_marketing
- path: `/home/zhu11/TabPFN/sync_data_proj/datasets/Fin_data/bank_marketing`
- data: `/home/zhu11/TabPFN/sync_data_proj/datasets/Fin_data/bank_marketing/data.csv`
- samples: 41188
- TabPFGen: {'n_sgld_steps': 600, 'sgld_step_size': 0.01, 'sgld_noise_scale': 0.005, 'jitter': 0.01, 'synthetic_factor': 1.0, 'energy_subsample': 2048}
- seed: 42

| generator | scheme | model | accuracy | precision | recall | f1 | roc_auc | log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | real | logistic | 0.9102 ± 0.0020 | 0.6661 ± 0.0171 | 0.4075 ± 0.0127 | 0.5056 ± 0.0117 | 0.9296 ± 0.0040 | 0.2137 ± 0.0088 |
| baseline | real | xgboost | 0.9169 ± 0.0018 | 0.6538 ± 0.0114 | 0.5573 ± 0.0114 | 0.6017 ± 0.0085 | 0.9496 ± 0.0022 | 0.1734 ± 0.0036 |
| baseline | real | lightgbm | 0.9176 ± 0.0025 | 0.6594 ± 0.0140 | 0.5563 ± 0.0144 | 0.6034 ± 0.0122 | 0.9505 ± 0.0019 | 0.1718 ± 0.0035 |
| baseline | real | catboost | 0.9177 ± 0.0016 | 0.6661 ± 0.0109 | 0.5414 ± 0.0130 | 0.5972 ± 0.0085 | 0.9500 ± 0.0025 | 0.1725 ± 0.0040 |
| baseline | real | mlp | 0.8903 ± 0.0037 | 0.5137 ± 0.0168 | 0.5125 ± 0.0294 | 0.5125 ± 0.0145 | 0.9079 ± 0.0020 | 0.6726 ± 0.0844 |
| tabpfgen | synthetic | logistic | 0.9108 ± 0.0022 | 0.6646 ± 0.0178 | 0.4218 ± 0.0121 | 0.5159 ± 0.0117 | 0.9253 ± 0.0043 | 0.2170 ± 0.0081 |
| tabpfgen | synthetic | xgboost | 0.8394 ± 0.0130 | 0.3824 ± 0.0238 | 0.6806 ± 0.0163 | 0.4891 ± 0.0176 | 0.8779 ± 0.0070 | 0.4009 ± 0.0268 |
| tabpfgen | synthetic | lightgbm | 0.8367 ± 0.0159 | 0.3784 ± 0.0272 | 0.6879 ± 0.0119 | 0.4878 ± 0.0241 | 0.8795 ± 0.0076 | 0.3987 ± 0.0373 |
| tabpfgen | synthetic | catboost | 0.8550 ± 0.0085 | 0.4124 ± 0.0170 | 0.6685 ± 0.0224 | 0.5097 ± 0.0103 | 0.8869 ± 0.0055 | 0.3548 ± 0.0155 |
| tabpfgen | synthetic | mlp | 0.8951 ± 0.0027 | 0.5382 ± 0.0144 | 0.4901 ± 0.0176 | 0.5127 ± 0.0064 | 0.9035 ± 0.0055 | 0.2945 ± 0.0138 |

## german_credit
- path: `/home/zhu11/TabPFN/sync_data_proj/datasets/Fin_data/german_credit`
- data: `/home/zhu11/TabPFN/sync_data_proj/datasets/Fin_data/german_credit/data.csv`
- samples: 1000
- TabPFGen: {'n_sgld_steps': 600, 'sgld_step_size': 0.01, 'sgld_noise_scale': 0.005, 'jitter': 0.01, 'synthetic_factor': 1.0, 'energy_subsample': 2048}
- seed: 42

| generator | scheme | model | accuracy | precision | recall | f1 | roc_auc | log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | real | logistic | 0.7680 ± 0.0168 | 0.6548 ± 0.0397 | 0.4800 ± 0.0431 | 0.5533 ± 0.0381 | 0.7894 ± 0.0241 | 0.4976 ± 0.0271 |
| baseline | real | xgboost | 0.7800 ± 0.0250 | 0.6723 ± 0.0575 | 0.5233 ± 0.0450 | 0.5879 ± 0.0466 | 0.7871 ± 0.0220 | 0.5636 ± 0.0466 |
| baseline | real | lightgbm | 0.7780 ± 0.0241 | 0.6638 ± 0.0651 | 0.5367 ± 0.0247 | 0.5925 ± 0.0334 | 0.7862 ± 0.0218 | 0.6826 ± 0.0560 |
| baseline | real | catboost | 0.7810 ± 0.0216 | 0.6913 ± 0.0837 | 0.5067 ± 0.0346 | 0.5817 ± 0.0231 | 0.7955 ± 0.0245 | 0.5168 ± 0.0431 |
| baseline | real | mlp | 0.7380 ± 0.0337 | 0.5661 ± 0.0622 | 0.5433 ± 0.0787 | 0.5533 ± 0.0654 | 0.7494 ± 0.0282 | 1.5401 ± 0.3405 |
| tabpfgen | synthetic | logistic | 0.7660 ± 0.0238 | 0.6533 ± 0.0572 | 0.4700 ± 0.0431 | 0.5464 ± 0.0469 | 0.7819 ± 0.0292 | 0.5085 ± 0.0334 |
| tabpfgen | synthetic | xgboost | 0.7480 ± 0.0261 | 0.6177 ± 0.0686 | 0.4333 ± 0.0612 | 0.5067 ± 0.0547 | 0.7687 ± 0.0158 | 0.6366 ± 0.0468 |
| tabpfgen | synthetic | lightgbm | 0.7490 ± 0.0096 | 0.6112 ± 0.0201 | 0.4500 ± 0.0589 | 0.5165 ± 0.0409 | 0.7600 ± 0.0303 | 0.9356 ± 0.1269 |
| tabpfgen | synthetic | catboost | 0.7590 ± 0.0338 | 0.6595 ± 0.0904 | 0.4100 ± 0.0693 | 0.5040 ± 0.0748 | 0.7705 ± 0.0365 | 0.5916 ± 0.0796 |
| tabpfgen | synthetic | mlp | 0.7290 ± 0.0268 | 0.5578 ± 0.0579 | 0.4900 ± 0.0450 | 0.5204 ± 0.0402 | 0.7475 ± 0.0397 | 1.5026 ± 0.2260 |

## polish_companies_bankruptcy
- path: `/home/zhu11/TabPFN/sync_data_proj/datasets/Fin_data/polish_companies_bankruptcy`
- data: `/home/zhu11/TabPFN/sync_data_proj/datasets/Fin_data/polish_companies_bankruptcy/data.csv`
- samples: 43405
- TabPFGen: {'n_sgld_steps': 600, 'sgld_step_size': 0.01, 'sgld_noise_scale': 0.005, 'jitter': 0.01, 'synthetic_factor': 1.0, 'energy_subsample': 2048}
- seed: 42

| generator | scheme | model | accuracy | precision | recall | f1 | roc_auc | log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | real | logistic | 0.9512 ± 0.0007 | 0.2917 ± 0.2357 | 0.0072 ± 0.0061 | 0.0140 ± 0.0119 | 0.6736 ± 0.0125 | 0.1927 ± 0.0022 |
| baseline | real | xgboost | 0.9765 ± 0.0013 | 0.9566 ± 0.0066 | 0.5375 ± 0.0269 | 0.6880 ± 0.0226 | 0.9734 ± 0.0037 | 0.0695 ± 0.0030 |
| baseline | real | lightgbm | 0.9777 ± 0.0017 | 0.9519 ± 0.0173 | 0.5657 ± 0.0352 | 0.7092 ± 0.0283 | 0.9750 ± 0.0034 | 0.0678 ± 0.0035 |
| baseline | real | catboost | 0.9745 ± 0.0013 | 0.9589 ± 0.0102 | 0.4916 ± 0.0270 | 0.6497 ± 0.0243 | 0.9648 ± 0.0064 | 0.0786 ± 0.0037 |
| baseline | real | mlp | 0.9485 ± 0.0047 | 0.4548 ± 0.0791 | 0.2449 ± 0.0364 | 0.3131 ± 0.0208 | 0.8488 ± 0.0133 | 0.2221 ± 0.0237 |
| tabpfgen | synthetic | logistic | 0.9501 ± 0.0005 | 0.1621 ± 0.1213 | 0.0100 ± 0.0087 | 0.0189 ± 0.0162 | 0.6819 ± 0.0214 | 0.2061 ± 0.0050 |
| tabpfgen | synthetic | xgboost | 0.9515 ± 0.0002 | 0.0800 ± 0.1789 | 0.0019 ± 0.0043 | 0.0037 ± 0.0084 | 0.7551 ± 0.0181 | 0.1763 ± 0.0034 |
| tabpfgen | synthetic | lightgbm | 0.9514 ± 0.0003 | 0.1430 ± 0.1431 | 0.0019 ± 0.0020 | 0.0038 ± 0.0039 | 0.7519 ± 0.0171 | 0.1767 ± 0.0029 |
| tabpfgen | synthetic | catboost | 0.9516 ± 0.0001 | 0.2469 ± 0.2577 | 0.0043 ± 0.0052 | 0.0084 ± 0.0101 | 0.7578 ± 0.0202 | 0.1732 ± 0.0029 |
| tabpfgen | synthetic | mlp | 0.9416 ± 0.0063 | 0.1771 ± 0.0629 | 0.0521 ± 0.0249 | 0.0775 ± 0.0323 | 0.6754 ± 0.0321 | 0.2691 ± 0.0337 |

## arrhythmia
- path: `/home/zhu11/TabPFN/sync_data_proj/datasets/EHR_datasets/arrhythmia`
- data: `/home/zhu11/TabPFN/sync_data_proj/datasets/EHR_datasets/arrhythmia/dataset.csv`
- samples: 452
- TabPFGen: {'n_sgld_steps': 600, 'sgld_step_size': 0.01, 'sgld_noise_scale': 0.005, 'jitter': 0.01, 'synthetic_factor': 1.0, 'energy_subsample': 2048}
- seed: 42

| generator | scheme | model | accuracy | precision | recall | f1 | roc_auc | log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | real | logistic | 0.6945 ± 0.0435 | 0.4963 ± 0.0756 | 0.4120 ± 0.0461 | 0.4331 ± 0.0465 | nan ± nan | nan ± nan |
| baseline | real | xgboost | nan ± nan | nan ± nan | nan ± nan | nan ± nan | nan ± nan | nan ± nan |
| baseline | real | lightgbm | 0.7451 ± 0.0488 | 0.5000 ± 0.1355 | 0.4890 ± 0.1074 | 0.4838 ± 0.1200 | nan ± nan | nan ± nan |
| baseline | real | catboost | 0.7495 ± 0.0407 | 0.5197 ± 0.0766 | 0.4568 ± 0.0556 | 0.4714 ± 0.0598 | nan ± nan | nan ± nan |
| baseline | real | mlp | 0.6659 ± 0.0423 | 0.4010 ± 0.0773 | 0.3770 ± 0.0791 | 0.3746 ± 0.0734 | nan ± nan | nan ± nan |
| tabpfgen | synthetic | logistic | 0.6527 ± 0.0710 | 0.4403 ± 0.1148 | 0.3546 ± 0.0935 | 0.3735 ± 0.0963 | nan ± nan | nan ± nan |
| tabpfgen | synthetic | xgboost | nan ± nan | nan ± nan | nan ± nan | nan ± nan | nan ± nan | nan ± nan |
| tabpfgen | synthetic | lightgbm | 0.7011 ± 0.0456 | 0.4507 ± 0.0687 | 0.4318 ± 0.0620 | 0.4236 ± 0.0583 | nan ± nan | nan ± nan |
| tabpfgen | synthetic | catboost | 0.7473 ± 0.0460 | 0.5141 ± 0.0834 | 0.4414 ± 0.0768 | 0.4593 ± 0.0790 | nan ± nan | nan ± nan |
| tabpfgen | synthetic | mlp | 0.6549 ± 0.0344 | 0.3860 ± 0.0634 | 0.3554 ± 0.0664 | 0.3496 ± 0.0604 | nan ± nan | nan ± nan |

## ckd
- path: `/home/zhu11/TabPFN/sync_data_proj/datasets/EHR_datasets/ckd`
- data: `/home/zhu11/TabPFN/sync_data_proj/datasets/EHR_datasets/ckd/dataset.csv`
- samples: 400
- TabPFGen: {'n_sgld_steps': 600, 'sgld_step_size': 0.01, 'sgld_noise_scale': 0.005, 'jitter': 0.01, 'synthetic_factor': 1.0, 'energy_subsample': 2048}
- seed: 42

| generator | scheme | model | accuracy | precision | recall | f1 | roc_auc | log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | real | logistic | 0.9925 ± 0.0112 | 0.9810 ± 0.0281 | 1.0000 ± 0.0000 | 0.9903 ± 0.0145 | 0.9999 ± 0.0003 | 0.0387 ± 0.0119 |
| baseline | real | xgboost | 0.9925 ± 0.0112 | 0.9875 ± 0.0280 | 0.9933 ± 0.0149 | 0.9902 ± 0.0145 | 0.9992 ± 0.0018 | 0.0282 ± 0.0237 |
| baseline | real | lightgbm | 0.9975 ± 0.0056 | 0.9935 ± 0.0144 | 1.0000 ± 0.0000 | 0.9967 ± 0.0073 | 0.9999 ± 0.0003 | 0.0062 ± 0.0119 |
| baseline | real | catboost | 0.9950 ± 0.0112 | 0.9875 ± 0.0280 | 1.0000 ± 0.0000 | 0.9935 ± 0.0144 | 1.0000 ± 0.0000 | 0.0178 ± 0.0255 |
| baseline | real | mlp | 0.9975 ± 0.0056 | 0.9935 ± 0.0144 | 1.0000 ± 0.0000 | 0.9967 ± 0.0073 | 1.0000 ± 0.0000 | 0.0167 ± 0.0108 |
| tabpfgen | synthetic | logistic | 0.9900 ± 0.0105 | 0.9746 ± 0.0263 | 1.0000 ± 0.0000 | 0.9870 ± 0.0135 | 0.9996 ± 0.0006 | 0.0418 ± 0.0141 |
| tabpfgen | synthetic | xgboost | 0.9825 ± 0.0068 | 0.9744 ± 0.0263 | 0.9800 ± 0.0183 | 0.9768 ± 0.0088 | 0.9985 ± 0.0026 | 0.0456 ± 0.0323 |
| tabpfgen | synthetic | lightgbm | 0.9925 ± 0.0112 | 0.9935 ± 0.0144 | 0.9867 ± 0.0298 | 0.9898 ± 0.0153 | 0.9997 ± 0.0006 | 0.0252 ± 0.0365 |
| tabpfgen | synthetic | catboost | 0.9875 ± 0.0125 | 0.9810 ± 0.0281 | 0.9867 ± 0.0298 | 0.9834 ± 0.0167 | 1.0000 ± 0.0000 | 0.0278 ± 0.0238 |
| tabpfgen | synthetic | mlp | 0.9900 ± 0.0056 | 0.9742 ± 0.0144 | 1.0000 ± 0.0000 | 0.9869 ± 0.0073 | 0.9999 ± 0.0003 | 0.0239 ± 0.0131 |

## pima
- path: `/home/zhu11/TabPFN/sync_data_proj/datasets/EHR_datasets/pima`
- data: `/home/zhu11/TabPFN/sync_data_proj/datasets/EHR_datasets/pima/dataset.csv`
- samples: 768
- TabPFGen: {'n_sgld_steps': 600, 'sgld_step_size': 0.01, 'sgld_noise_scale': 0.005, 'jitter': 0.01, 'synthetic_factor': 1.0, 'energy_subsample': 2048}
- seed: 42

| generator | scheme | model | accuracy | precision | recall | f1 | roc_auc | log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | real | logistic | 0.7747 ± 0.0164 | 0.7939 ± 0.0094 | 0.8840 ± 0.0439 | 0.8359 ± 0.0166 | 0.8304 ± 0.0249 | 0.4855 ± 0.0273 |
| baseline | real | xgboost | 0.7434 ± 0.0332 | 0.7878 ± 0.0256 | 0.8300 ± 0.0339 | 0.8081 ± 0.0247 | 0.7984 ± 0.0385 | 0.6451 ± 0.0563 |
| baseline | real | lightgbm | 0.7422 ± 0.0243 | 0.7907 ± 0.0221 | 0.8220 ± 0.0179 | 0.8059 ± 0.0168 | 0.7930 ± 0.0307 | 0.8600 ± 0.0654 |
| baseline | real | catboost | 0.7643 ± 0.0258 | 0.8048 ± 0.0139 | 0.8420 ± 0.0319 | 0.8229 ± 0.0213 | 0.8228 ± 0.0314 | 0.5439 ± 0.0429 |
| baseline | real | mlp | 0.7122 ± 0.0335 | 0.7760 ± 0.0210 | 0.7840 ± 0.0365 | 0.7799 ± 0.0280 | 0.7537 ± 0.0347 | 1.1417 ± 0.2253 |
| tabpfgen | synthetic | logistic | 0.7734 ± 0.0207 | 0.7872 ± 0.0167 | 0.8940 ± 0.0230 | 0.8371 ± 0.0147 | 0.8232 ± 0.0289 | 0.4946 ± 0.0340 |
| tabpfgen | synthetic | xgboost | 0.7304 ± 0.0171 | 0.7792 ± 0.0085 | 0.8180 ± 0.0370 | 0.7977 ± 0.0166 | 0.7915 ± 0.0335 | 0.7166 ± 0.0739 |
| tabpfgen | synthetic | lightgbm | 0.7252 ± 0.0188 | 0.7704 ± 0.0097 | 0.8240 ± 0.0445 | 0.7957 ± 0.0189 | 0.7865 ± 0.0362 | 1.1629 ± 0.1385 |
| tabpfgen | synthetic | catboost | 0.7382 ± 0.0366 | 0.7770 ± 0.0158 | 0.8380 ± 0.0487 | 0.8061 ± 0.0302 | 0.8028 ± 0.0385 | 0.6381 ± 0.0661 |
| tabpfgen | synthetic | mlp | 0.6836 ± 0.0433 | 0.7551 ± 0.0255 | 0.7600 ± 0.0534 | 0.7572 ± 0.0369 | 0.7408 ± 0.0281 | 1.6325 ± 0.4046 |

## uti
- path: `/home/zhu11/TabPFN/sync_data_proj/datasets/EHR_datasets/uti`
- data: `/home/zhu11/TabPFN/sync_data_proj/datasets/EHR_datasets/uti/dataset.csv`
- samples: 120
- TabPFGen: {'n_sgld_steps': 600, 'sgld_step_size': 0.01, 'sgld_noise_scale': 0.005, 'jitter': 0.01, 'synthetic_factor': 1.0, 'energy_subsample': 2048}
- seed: 42

| generator | scheme | model | accuracy | precision | recall | f1 | roc_auc | log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | real | logistic | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0323 ± 0.0042 |
| baseline | real | xgboost | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0254 ± 0.0043 |
| baseline | real | lightgbm | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| baseline | real | catboost | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0016 ± 0.0002 |
| baseline | real | mlp | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0032 ± 0.0003 |
| tabpfgen | synthetic | logistic | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0360 ± 0.0069 |
| tabpfgen | synthetic | xgboost | 0.9750 ± 0.0228 | 1.0000 ± 0.0000 | 0.9400 ± 0.0548 | 0.9684 ± 0.0288 | 1.0000 ± 0.0000 | 0.0825 ± 0.0549 |
| tabpfgen | synthetic | lightgbm | 0.9917 ± 0.0186 | 0.9818 ± 0.0407 | 1.0000 ± 0.0000 | 0.9905 ± 0.0213 | 1.0000 ± 0.0000 | 0.0134 ± 0.0298 |
| tabpfgen | synthetic | catboost | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0046 ± 0.0036 |
| tabpfgen | synthetic | mlp | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0036 ± 0.0005 |

## wdbc
- path: `/home/zhu11/TabPFN/sync_data_proj/datasets/EHR_datasets/wdbc`
- data: `/home/zhu11/TabPFN/sync_data_proj/datasets/EHR_datasets/wdbc/dataset.csv`
- samples: 569
- TabPFGen: {'n_sgld_steps': 600, 'sgld_step_size': 0.01, 'sgld_noise_scale': 0.005, 'jitter': 0.01, 'synthetic_factor': 1.0, 'energy_subsample': 2048}
- seed: 42

| generator | scheme | model | accuracy | precision | recall | f1 | roc_auc | log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | real | logistic | 0.9719 ± 0.0190 | 0.9820 ± 0.0290 | 0.9436 ± 0.0587 | 0.9611 ± 0.0270 | 0.9929 ± 0.0079 | 0.0845 ± 0.0401 |
| baseline | real | xgboost | 0.9613 ± 0.0101 | 0.9582 ± 0.0267 | 0.9385 ± 0.0494 | 0.9471 ± 0.0157 | 0.9943 ± 0.0068 | 0.0938 ± 0.0286 |
| baseline | real | lightgbm | 0.9631 ± 0.0209 | 0.9664 ± 0.0265 | 0.9339 ± 0.0536 | 0.9491 ± 0.0297 | 0.9937 ± 0.0068 | 0.1719 ± 0.1176 |
| baseline | real | catboost | 0.9596 ± 0.0211 | 0.9628 ± 0.0322 | 0.9295 ± 0.0682 | 0.9441 ± 0.0311 | 0.9952 ± 0.0062 | 0.0917 ± 0.0375 |
| baseline | real | mlp | 0.9754 ± 0.0157 | 0.9811 ± 0.0193 | 0.9529 ± 0.0440 | 0.9662 ± 0.0222 | 0.9933 ± 0.0091 | 0.1305 ± 0.0875 |
| tabpfgen | synthetic | logistic | 0.9719 ± 0.0200 | 0.9901 ± 0.0136 | 0.9341 ± 0.0482 | 0.9608 ± 0.0284 | 0.9910 ± 0.0101 | 0.0973 ± 0.0491 |
| tabpfgen | synthetic | xgboost | 0.9490 ± 0.0200 | 0.9347 ± 0.0584 | 0.9340 ± 0.0629 | 0.9317 ± 0.0263 | 0.9919 ± 0.0066 | 0.1225 ± 0.0402 |
| tabpfgen | synthetic | lightgbm | 0.9561 ± 0.0124 | 0.9628 ± 0.0295 | 0.9198 ± 0.0590 | 0.9392 ± 0.0194 | 0.9926 ± 0.0077 | 0.1892 ± 0.1150 |
| tabpfgen | synthetic | catboost | 0.9543 ± 0.0209 | 0.9626 ± 0.0294 | 0.9153 ± 0.0750 | 0.9362 ± 0.0320 | 0.9923 ± 0.0047 | 0.1367 ± 0.0557 |
| tabpfgen | synthetic | mlp | 0.9719 ± 0.0169 | 0.9767 ± 0.0228 | 0.9483 ± 0.0508 | 0.9614 ± 0.0239 | 0.9949 ± 0.0079 | 0.1070 ± 0.0874 |
