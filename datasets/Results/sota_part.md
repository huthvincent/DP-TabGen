# SOTA Generators Pipeline Results

## australian_credit_approval
- path: `/home/zhu11/TabPFN/sync_data_proj/datasets/Fin_data/australian_credit_approval`
- data: `/home/zhu11/TabPFN/sync_data_proj/datasets/Fin_data/australian_credit_approval/data.csv`
- samples: 690
- CTGAN: {'epochs': 50, 'batch_size': 64, 'pac': 1}
- TVAE: {'epochs': 50, 'batch_size': 64}
- SynthCity DDPM: {'n_iter': 120, 'batch_size': 128}
- TabPFN conditional: {'sample_count': None, 'num_gibbs_rounds': 3, 'batch_size': 256, 'clip_quantile_low': 0.01, 'clip_quantile_high': 0.99, 'use_gpu': True}
- seed: 42

| generator | scheme | model | accuracy | precision | recall | f1 | roc_auc | log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gaussian_copula | synthetic | logistic | 0.8130 ± 0.0558 | 0.8565 ± 0.0584 | 0.6970 ± 0.1020 | 0.7660 ± 0.0752 | 0.9068 ± 0.0235 | 0.4484 ± 0.0333 |
| gaussian_copula | synthetic | xgboost | 0.7261 ± 0.0626 | 0.7605 ± 0.0795 | 0.5602 ± 0.1625 | 0.6364 ± 0.1138 | 0.8014 ± 0.0919 | 0.6028 ± 0.1575 |
| gaussian_copula | synthetic | lightgbm | 0.7362 ± 0.0698 | 0.7726 ± 0.0886 | 0.5763 ± 0.1638 | 0.6514 ± 0.1202 | 0.7915 ± 0.0828 | 0.7296 ± 0.2176 |
| gaussian_copula | synthetic | catboost | 0.7580 ± 0.0659 | 0.8075 ± 0.0780 | 0.5925 ± 0.1344 | 0.6794 ± 0.1116 | 0.8223 ± 0.0912 | 0.5274 ± 0.1223 |
| gaussian_copula | synthetic | mlp | 0.6971 ± 0.0873 | 0.6698 ± 0.1097 | 0.6024 ± 0.1619 | 0.6317 ± 0.1384 | 0.7678 ± 0.0967 | 1.5257 ± 0.4432 |
| sdv_ctgan | synthetic | logistic | 0.5768 ± 0.0855 | 0.7284 ± 0.2968 | 0.1846 ± 0.2095 | 0.2414 ± 0.2194 | 0.5950 ± 0.1884 | 0.7109 ± 0.1102 |
| sdv_ctgan | synthetic | xgboost | 0.5768 ± 0.0713 | 0.5427 ± 0.1427 | 0.3582 ± 0.1432 | 0.4189 ± 0.1361 | 0.5643 ± 0.1167 | 0.8815 ± 0.1909 |
| sdv_ctgan | synthetic | lightgbm | 0.5594 ± 0.0963 | 0.5288 ± 0.1810 | 0.3747 ± 0.1092 | 0.4296 ± 0.1159 | 0.5513 ± 0.1148 | 0.9833 ± 0.1668 |
| sdv_ctgan | synthetic | catboost | 0.5594 ± 0.0605 | 0.4926 ± 0.1655 | 0.3026 ± 0.1613 | 0.3593 ± 0.1675 | 0.5735 ± 0.0977 | 0.8133 ± 0.1580 |
| sdv_ctgan | synthetic | mlp | 0.5594 ± 0.1039 | 0.4946 ± 0.1288 | 0.4267 ± 0.1935 | 0.4498 ± 0.1665 | 0.5826 ± 0.1395 | 2.9513 ± 0.9313 |
| sdv_tvae | synthetic | logistic | 0.8362 ± 0.0443 | 0.7972 ± 0.0716 | 0.8603 ± 0.0801 | 0.8240 ± 0.0460 | 0.9030 ± 0.0294 | 0.6779 ± 0.2166 |
| sdv_tvae | synthetic | xgboost | 0.8478 ± 0.0312 | 0.8630 ± 0.0614 | 0.7882 ± 0.0667 | 0.8213 ± 0.0366 | 0.9061 ± 0.0197 | 0.5260 ± 0.1150 |
| sdv_tvae | synthetic | lightgbm | 0.8609 ± 0.0421 | 0.8603 ± 0.0602 | 0.8241 ± 0.0534 | 0.8407 ± 0.0458 | 0.9049 ± 0.0229 | 0.8950 ± 0.2972 |
| sdv_tvae | synthetic | catboost | 0.8348 ± 0.0287 | 0.8240 ± 0.0415 | 0.8014 ± 0.0430 | 0.8119 ± 0.0323 | 0.8960 ± 0.0213 | 0.5462 ± 0.1114 |
| sdv_tvae | synthetic | mlp | 0.7580 ± 0.0544 | 0.7152 ± 0.0889 | 0.7851 ± 0.0365 | 0.7449 ± 0.0401 | 0.7949 ± 0.0579 | 5.9444 ± 1.9321 |
| synthcity_ddpm | synthetic | logistic | 0.7406 ± 0.0757 | 0.7365 ± 0.1304 | 0.7363 ± 0.2410 | 0.7040 ± 0.1141 | 0.8490 ± 0.0622 | 0.7473 ± 0.2853 |
| synthcity_ddpm | synthetic | xgboost | 0.7391 ± 0.0682 | 0.7619 ± 0.1178 | 0.6453 ± 0.2383 | 0.6680 ± 0.1376 | 0.8521 ± 0.0452 | 0.7477 ± 0.1894 |
| synthcity_ddpm | synthetic | lightgbm | 0.7768 ± 0.0585 | 0.7888 ± 0.0624 | 0.6772 ± 0.1014 | 0.7271 ± 0.0829 | 0.8709 ± 0.0375 | 0.7118 ± 0.1745 |
| synthcity_ddpm | synthetic | catboost | 0.7493 ± 0.0563 | 0.7150 ± 0.0853 | 0.7559 ± 0.1237 | 0.7267 ± 0.0636 | 0.8335 ± 0.0572 | 0.7042 ± 0.1751 |
| synthcity_ddpm | synthetic | mlp | 0.7101 ± 0.1007 | 0.6724 ± 0.1379 | 0.7886 ± 0.1729 | 0.7078 ± 0.0890 | 0.8035 ± 0.0931 | 3.1426 ± 2.1984 |

## uti
- path: `/home/zhu11/TabPFN/sync_data_proj/datasets/EHR_datasets/uti`
- data: `/home/zhu11/TabPFN/sync_data_proj/datasets/EHR_datasets/uti/dataset.csv`
- samples: 120
- CTGAN: {'epochs': 50, 'batch_size': 64, 'pac': 1}
- TVAE: {'epochs': 50, 'batch_size': 64}
- SynthCity DDPM: {'n_iter': 120, 'batch_size': 128}
- TabPFN conditional: {'sample_count': None, 'num_gibbs_rounds': 3, 'batch_size': 256, 'clip_quantile_low': 0.01, 'clip_quantile_high': 0.99, 'use_gpu': False}
- seed: 42

| generator | scheme | model | accuracy | precision | recall | f1 | roc_auc | log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gaussian_copula | synthetic | logistic | 0.9250 ± 0.0618 | 0.9333 ± 0.0913 | 0.9000 ± 0.1732 | 0.9031 ± 0.0934 | 0.9714 ± 0.0466 | 0.3839 ± 0.0815 |
| gaussian_copula | synthetic | xgboost | 0.7583 ± 0.1394 | 0.7278 ± 0.1527 | 0.7200 ± 0.2280 | 0.7066 ± 0.1673 | 0.8143 ± 0.1291 | 0.5645 ± 0.2338 |
| gaussian_copula | synthetic | lightgbm | 0.7500 ± 0.0417 | 0.6794 ± 0.0350 | 0.7600 ± 0.1517 | 0.7120 ± 0.0687 | 0.8343 ± 0.1135 | 0.4899 ± 0.1193 |
| gaussian_copula | synthetic | catboost | 0.7333 ± 0.0757 | 0.7220 ± 0.1650 | 0.6800 ± 0.2588 | 0.6661 ± 0.1218 | 0.8257 ± 0.1211 | 0.6103 ± 0.2297 |
| gaussian_copula | synthetic | mlp | 0.7333 ± 0.1369 | 0.6449 ± 0.1547 | 0.7800 ± 0.2775 | 0.6926 ± 0.2061 | 0.7800 ± 0.1797 | 1.4440 ± 0.9028 |

## bank_marketing
- path: `/home/zhu11/TabPFN/sync_data_proj/datasets/Fin_data/bank_marketing`
- data: `/home/zhu11/TabPFN/sync_data_proj/datasets/Fin_data/bank_marketing/data.csv`
- samples: 41188
- CTGAN: {'epochs': 50, 'batch_size': 64, 'pac': 1}
- TVAE: {'epochs': 50, 'batch_size': 64}
- SynthCity DDPM: {'n_iter': 120, 'batch_size': 128}
- TabPFN conditional: {'sample_count': None, 'num_gibbs_rounds': 3, 'batch_size': 256, 'clip_quantile_low': 0.01, 'clip_quantile_high': 0.99, 'use_gpu': True}
- seed: 42

| generator | scheme | model | accuracy | precision | recall | f1 | roc_auc | log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gaussian_copula | synthetic | logistic | 0.8915 ± 0.0059 | 0.6633 ± 0.1701 | 0.0681 ± 0.0559 | 0.1198 ± 0.0894 | 0.9053 ± 0.0222 | 0.2689 ± 0.0139 |
| gaussian_copula | synthetic | xgboost | 0.8885 ± 0.0013 | 0.7666 ± 0.1787 | 0.0293 ± 0.0244 | 0.0542 ± 0.0413 | 0.8168 ± 0.0207 | 0.3041 ± 0.0133 |
| gaussian_copula | synthetic | lightgbm | 0.8884 ± 0.0012 | 0.7161 ± 0.1871 | 0.0287 ± 0.0277 | 0.0525 ± 0.0467 | 0.8314 ± 0.0108 | 0.2973 ± 0.0147 |
| gaussian_copula | synthetic | catboost | 0.8883 ± 0.0007 | 0.7044 ± 0.0881 | 0.0166 ± 0.0166 | 0.0317 ± 0.0307 | 0.8628 ± 0.0155 | 0.2942 ± 0.0108 |
| gaussian_copula | synthetic | mlp | 0.8437 ± 0.0142 | 0.2553 ± 0.0463 | 0.1940 ± 0.0408 | 0.2174 ± 0.0302 | 0.5861 ± 0.0228 | 1.7608 ± 0.3908 |
| sdv_ctgan | synthetic | logistic | 0.9047 ± 0.0131 | 0.6103 ± 0.0833 | 0.5453 ± 0.1471 | 0.5574 ± 0.0459 | 0.9182 ± 0.0088 | 0.2624 ± 0.0446 |
| sdv_ctgan | synthetic | xgboost | 0.8950 ± 0.0080 | 0.5515 ± 0.0500 | 0.4974 ± 0.1584 | 0.5068 ± 0.0656 | 0.8992 ± 0.0121 | 0.2666 ± 0.0255 |
| sdv_ctgan | synthetic | lightgbm | 0.8949 ± 0.0109 | 0.5564 ± 0.0601 | 0.4942 ± 0.1596 | 0.5056 ± 0.0595 | 0.9012 ± 0.0119 | 0.2645 ± 0.0259 |
| sdv_ctgan | synthetic | catboost | 0.8958 ± 0.0092 | 0.5594 ± 0.0517 | 0.4890 ± 0.1599 | 0.5047 ± 0.0609 | 0.9098 ± 0.0085 | 0.2605 ± 0.0284 |
| sdv_ctgan | synthetic | mlp | 0.8367 ± 0.0278 | 0.3623 ± 0.0620 | 0.5155 ± 0.1139 | 0.4139 ± 0.0183 | 0.7818 ± 0.0319 | 0.8941 ± 0.1469 |
| sdv_tvae | synthetic | logistic | 0.8756 ± 0.0341 | 0.5221 ± 0.1504 | 0.4282 ± 0.1687 | 0.4286 ± 0.0535 | 0.8528 ± 0.0209 | 0.4475 ± 0.0269 |
| sdv_tvae | synthetic | xgboost | 0.8923 ± 0.0102 | 0.5919 ± 0.1148 | 0.2955 ± 0.1274 | 0.3687 ± 0.0826 | 0.8487 ± 0.0409 | 0.4725 ± 0.2194 |
| sdv_tvae | synthetic | lightgbm | 0.8914 ± 0.0123 | 0.5764 ± 0.1122 | 0.3172 ± 0.1154 | 0.3871 ± 0.0684 | 0.8522 ± 0.0396 | 0.5574 ± 0.3211 |
| sdv_tvae | synthetic | catboost | 0.8916 ± 0.0094 | 0.5700 ± 0.1011 | 0.3685 ± 0.1628 | 0.4163 ± 0.0960 | 0.8560 ± 0.0486 | 0.4239 ± 0.2019 |
| sdv_tvae | synthetic | mlp | 0.8717 ± 0.0300 | 0.5043 ± 0.1465 | 0.3106 ± 0.1886 | 0.3259 ± 0.1110 | 0.7158 ± 0.1186 | 2.2037 ± 0.8942 |
| synthcity_ddpm | synthetic | logistic | 0.6461 ± 0.2246 | 0.2494 ± 0.1623 | 0.4963 ± 0.2296 | 0.2570 ± 0.0495 | 0.6282 ± 0.0806 | 0.8666 ± 0.4534 |
| synthcity_ddpm | synthetic | xgboost | 0.6322 ± 0.2702 | 0.2338 ± 0.1398 | 0.4959 ± 0.2932 | 0.2534 ± 0.0611 | 0.6421 ± 0.0819 | 1.1549 ± 0.8650 |
| synthcity_ddpm | synthetic | lightgbm | 0.6459 ± 0.2263 | 0.2338 ± 0.1408 | 0.5037 ± 0.2257 | 0.2682 ± 0.0672 | 0.6523 ± 0.0694 | 0.9854 ± 0.5940 |
| synthcity_ddpm | synthetic | catboost | 0.6929 ± 0.1866 | 0.2373 ± 0.1235 | 0.4651 ± 0.2439 | 0.2667 ± 0.0551 | 0.6586 ± 0.0671 | 0.7903 ± 0.4155 |
| synthcity_ddpm | synthetic | mlp | 0.6512 ± 0.1684 | 0.1896 ± 0.0633 | 0.4981 ± 0.1565 | 0.2562 ± 0.0382 | 0.6248 ± 0.0632 | 3.0541 ± 1.8025 |

## german_credit
- path: `/home/zhu11/TabPFN/sync_data_proj/datasets/Fin_data/german_credit`
- data: `/home/zhu11/TabPFN/sync_data_proj/datasets/Fin_data/german_credit/data.csv`
- samples: 1000
- CTGAN: {'epochs': 50, 'batch_size': 64, 'pac': 1}
- TVAE: {'epochs': 50, 'batch_size': 64}
- SynthCity DDPM: {'n_iter': 120, 'batch_size': 128}
- TabPFN conditional: {'sample_count': None, 'num_gibbs_rounds': 3, 'batch_size': 256, 'clip_quantile_low': 0.01, 'clip_quantile_high': 0.99, 'use_gpu': True}
- seed: 42

| generator | scheme | model | accuracy | precision | recall | f1 | roc_auc | log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gaussian_copula | synthetic | logistic | 0.6870 ± 0.0160 | 0.3451 ± 0.2016 | 0.1267 ± 0.0796 | 0.1832 ± 0.1109 | 0.6501 ± 0.0744 | 0.5839 ± 0.0334 |
| gaussian_copula | synthetic | xgboost | 0.6650 ± 0.0302 | 0.4052 ± 0.0658 | 0.2533 ± 0.0908 | 0.3068 ± 0.0821 | 0.5879 ± 0.0474 | 0.7470 ± 0.0660 |
| gaussian_copula | synthetic | lightgbm | 0.6600 ± 0.0212 | 0.3931 ± 0.0569 | 0.2533 ± 0.0628 | 0.3069 ± 0.0618 | 0.5896 ± 0.0406 | 0.8888 ± 0.0986 |
| gaussian_copula | synthetic | catboost | 0.6770 ± 0.0202 | 0.4146 ± 0.0710 | 0.2033 ± 0.0721 | 0.2694 ± 0.0800 | 0.5901 ± 0.0468 | 0.6687 ± 0.0334 |
| gaussian_copula | synthetic | mlp | 0.6140 ± 0.0464 | 0.3488 ± 0.0644 | 0.3133 ± 0.0431 | 0.3284 ± 0.0465 | 0.5472 ± 0.0450 | 2.0201 ± 0.3162 |
| sdv_ctgan | synthetic | logistic | 0.6970 ± 0.0125 | 0.3378 ± 0.3497 | 0.0367 ± 0.0380 | 0.0657 ± 0.0675 | 0.5846 ± 0.0930 | 0.6077 ± 0.0259 |
| sdv_ctgan | synthetic | xgboost | 0.6690 ± 0.0233 | 0.3576 ± 0.0942 | 0.1400 ± 0.0742 | 0.1968 ± 0.0872 | 0.5408 ± 0.0401 | 0.7624 ± 0.0685 |
| sdv_ctgan | synthetic | lightgbm | 0.6580 ± 0.0179 | 0.3355 ± 0.0398 | 0.1567 ± 0.0879 | 0.2045 ± 0.0919 | 0.5473 ± 0.0595 | 0.8766 ± 0.1379 |
| sdv_ctgan | synthetic | catboost | 0.6710 ± 0.0288 | 0.3218 ± 0.1305 | 0.0700 ± 0.0447 | 0.1097 ± 0.0606 | 0.5526 ± 0.0415 | 0.6962 ± 0.0437 |
| sdv_ctgan | synthetic | mlp | 0.6210 ± 0.0284 | 0.3284 ± 0.0242 | 0.2500 ± 0.0825 | 0.2774 ± 0.0554 | 0.5391 ± 0.0697 | 2.0358 ± 0.2041 |
| sdv_tvae | synthetic | logistic | 0.7330 ± 0.0259 | 0.6371 ± 0.1076 | 0.2567 ± 0.0902 | 0.3602 ± 0.0952 | 0.7243 ± 0.0613 | 0.9957 ± 0.2488 |
| sdv_tvae | synthetic | xgboost | 0.6860 ± 0.0152 | 0.4377 ± 0.0688 | 0.1667 ± 0.0441 | 0.2400 ± 0.0525 | 0.6798 ± 0.0481 | 1.2576 ± 0.1616 |
| sdv_tvae | synthetic | lightgbm | 0.6920 ± 0.0091 | 0.4694 ± 0.0357 | 0.1900 ± 0.0365 | 0.2687 ± 0.0368 | 0.6670 ± 0.0446 | 2.3843 ± 0.3655 |
| sdv_tvae | synthetic | catboost | 0.6850 ± 0.0203 | 0.4392 ± 0.0837 | 0.1767 ± 0.0494 | 0.2500 ± 0.0589 | 0.6770 ± 0.0424 | 1.1246 ± 0.1634 |
| sdv_tvae | synthetic | mlp | 0.7050 ± 0.0215 | 0.5572 ± 0.1638 | 0.1267 ± 0.0418 | 0.2030 ± 0.0600 | 0.5917 ± 0.0451 | 7.4588 ± 0.3639 |
| synthcity_ddpm | synthetic | logistic | 0.7033 ± 0.0058 | 0.3438 ± 0.2981 | 0.1611 ± 0.1456 | 0.2184 ± 0.1938 | 0.5453 ± 0.1655 | 1.8768 ± 0.4960 |
| synthcity_ddpm | synthetic | xgboost | 0.6967 ± 0.0104 | 0.3123 ± 0.2733 | 0.2222 ± 0.2658 | 0.2449 ± 0.2569 | 0.5855 ± 0.1780 | 1.3160 ± 0.5190 |
| synthcity_ddpm | synthetic | lightgbm | 0.6933 ± 0.0076 | 0.2851 ± 0.2550 | 0.1833 ± 0.2619 | 0.2000 ± 0.2552 | 0.5657 ± 0.1607 | 1.6257 ± 0.5854 |
| synthcity_ddpm | synthetic | catboost | 0.7067 ± 0.0161 | 0.3431 ± 0.2993 | 0.2111 ± 0.2263 | 0.2535 ± 0.2479 | 0.6095 ± 0.1300 | 0.9195 ± 0.2634 |
| synthcity_ddpm | synthetic | mlp | 0.6800 ± 0.0304 | 0.2710 ± 0.2410 | 0.1056 ± 0.1084 | 0.1441 ± 0.1352 | 0.4974 ± 0.1977 | 4.4415 ± 1.7518 |

## polish_companies_bankruptcy
- path: `/home/zhu11/TabPFN/sync_data_proj/datasets/Fin_data/polish_companies_bankruptcy`
- data: `/home/zhu11/TabPFN/sync_data_proj/datasets/Fin_data/polish_companies_bankruptcy/data.csv`
- samples: 43405
- CTGAN: {'epochs': 50, 'batch_size': 64, 'pac': 1}
- TVAE: {'epochs': 50, 'batch_size': 64}
- SynthCity DDPM: {'n_iter': 120, 'batch_size': 128}
- TabPFN conditional: {'sample_count': None, 'num_gibbs_rounds': 3, 'batch_size': 256, 'clip_quantile_low': 0.01, 'clip_quantile_high': 0.99, 'use_gpu': True}
- seed: 42

| generator | scheme | model | accuracy | precision | recall | f1 | roc_auc | log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gaussian_copula | synthetic | logistic | 0.9501 ± 0.0007 | 0.1435 ± 0.1148 | 0.0081 ± 0.0077 | 0.0154 ± 0.0144 | 0.6920 ± 0.0117 | 0.2114 ± 0.0132 |
| gaussian_copula | synthetic | xgboost | 0.9518 ± 0.0001 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.5590 ± 0.0488 | 0.1980 ± 0.0044 |
| gaussian_copula | synthetic | lightgbm | 0.9518 ± 0.0001 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.5685 ± 0.0127 | 0.1932 ± 0.0006 |
| gaussian_copula | synthetic | catboost | 0.9518 ± 0.0001 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.6571 ± 0.0214 | 0.1906 ± 0.0009 |
| gaussian_copula | synthetic | mlp | 0.8961 ± 0.0710 | 0.0448 ± 0.0135 | 0.0593 ± 0.0693 | 0.0383 ± 0.0309 | 0.4813 ± 0.0341 | 0.4999 ± 0.1357 |
| sdv_ctgan | synthetic | logistic | 0.7150 ± 0.1902 | 0.0873 ± 0.0193 | 0.4482 ± 0.2521 | 0.1369 ± 0.0123 | 0.6649 ± 0.0177 | 0.8882 ± 0.2171 |
| sdv_ctgan | synthetic | xgboost | 0.7115 ± 0.2039 | 0.1149 ± 0.0557 | 0.4779 ± 0.3038 | 0.1474 ± 0.0233 | 0.6898 ± 0.0284 | 0.5869 ± 0.3422 |
| sdv_ctgan | synthetic | lightgbm | 0.7081 ± 0.2088 | 0.1139 ± 0.0570 | 0.4703 ± 0.3008 | 0.1446 ± 0.0216 | 0.6872 ± 0.0365 | 0.5868 ± 0.3406 |
| sdv_ctgan | synthetic | catboost | 0.6959 ± 0.2234 | 0.1094 ± 0.0521 | 0.4823 ± 0.3315 | 0.1390 ± 0.0276 | 0.6949 ± 0.0424 | 0.5758 ± 0.3383 |
| sdv_ctgan | synthetic | mlp | 0.6798 ± 0.1723 | 0.0855 ± 0.0303 | 0.4688 ± 0.1854 | 0.1347 ± 0.0276 | 0.6299 ± 0.0451 | 3.8423 ± 2.0149 |
| sdv_tvae | synthetic | logistic | 0.7010 ± 0.1522 | 0.0813 ± 0.0113 | 0.4782 ± 0.2225 | 0.1452 ± 0.0234 | 0.6232 ± 0.0127 | 0.8782 ± 0.2171 |
| sdv_tvae | synthetic | xgboost | 0.7045 ± 0.2139 | 0.1142 ± 0.0547 | 0.4570 ± 0.3135 | 0.1524 ± 0.0245 | 0.6428 ± 0.0214 | 0.5569 ± 0.3522 |
| sdv_tvae | synthetic | lightgbm | 0.7191 ± 0.2178 | 0.1319 ± 0.0540 | 0.4643 ± 0.3123 | 0.1458 ± 0.0229 | 0.6912 ± 0.0321 | 0.5812 ± 0.3426 |
| sdv_tvae | synthetic | catboost | 0.6819 ± 0.2334 | 0.1195 ± 0.0223 | 0.4325 ± 0.3222 | 0.1410 ± 0.0226 | 0.6915 ± 0.0414 | 0.5769 ± 0.3283 |
| sdv_tvae | synthetic | mlp | 0.6591 ± 0.1523 | 0.0859 ± 0.0313 | 0.4691 ± 0.1891 | 0.1297 ± 0.0291 | 0.6322 ± 0.0552 | 3.6421 ± 2.0161 |
| synthcity_ddpm | synthetic | logistic | 0.2289 ± 0.4040 | 0.0384 ± 0.0215 | 0.7981 ± 0.4461 | 0.0733 ± 0.0410 | 0.5082 ± 0.0749 | 1.7705 ± 0.9058 |
| synthcity_ddpm | synthetic | xgboost | 0.4094 ± 0.4947 | 0.0750 ± 0.0895 | 0.6014 ± 0.5458 | 0.0579 ± 0.0468 | 0.5476 ± 0.0911 | 3.2816 ± 2.7719 |
| synthcity_ddpm | synthetic | lightgbm | 0.2398 ± 0.3987 | 0.0389 ± 0.0218 | 0.7967 ± 0.4454 | 0.0742 ± 0.0415 | 0.6223 ± 0.0690 | 2.6571 ± 1.9064 |
| synthcity_ddpm | synthetic | catboost | 0.2759 ± 0.3913 | 0.0404 ± 0.0230 | 0.7809 ± 0.4385 | 0.0768 ± 0.0435 | 0.4950 ± 0.1205 | 1.3868 ± 0.8589 |
| synthcity_ddpm | synthetic | mlp | 0.5809 ± 0.4866 | 0.1034 ± 0.1262 | 0.4278 ± 0.5234 | 0.0660 ± 0.0462 | 0.4896 ± 0.0582 | 5.2728 ± 7.4110 |
