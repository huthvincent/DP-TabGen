# Indistinguishability analysis (TVAE vs TabPFGen)

## What the metrics mean
- MMD: kernel-based distance between real and synthetic feature distributions (lower is better; 0 means identical).
- Marginals: Wasserstein distance for numeric features; TVD (total variation distance) for integer/categorical features (lower is better).
- Correlation gap: Frobenius norm of correlation matrix difference (lower means similar dependency structure).
- C2ST: accuracy/AUC of a classifier distinguishing real vs synthetic (closer to 0.5 for both indicates harder to distinguish).

## Inputs
- Real data: `/home/zhu11/TabPFN/sync_data_proj/datasets/Fin_data/polish_companies_bankruptcy/data.csv` (samples=43405)
- TVAE synthetic: `/home/zhu11/TabPFN/sync_data_proj/datasets/Results/polish_Indistinguishability_tvae.csv`
- TVAE params: {'tvae_epochs': 50, 'tvae_batch_size': 256, 'use_cuda': True}
- TabPFGen synthetic: `/home/zhu11/TabPFN/sync_data_proj/datasets/Results/polish_Indistinguishability.csv` (re-used for comparison)

## Summary
| generator | MMD | corr_gap | C2ST_acc | C2ST_auc |
| --- | --- | --- | --- | --- |
| TVAE | 0.030975 | 13.481735 | 0.5856 | 0.6567 |
| TabPFGen | 0.000564 | 3.829020 | 0.5212 | 0.5159 |

## Details: TVAE
- MMD (RBF): `0.030975`
- Correlation gap (Frobenius): `13.481735`
- C2ST accuracy: `0.5856`, ROC-AUC: `0.6567`

### Marginal distances
#### Numeric (Wasserstein)
- `Attr1`: 0.056832
- `Attr2`: 0.125532
- `Attr3`: 0.113096
- `Attr4`: 3.706635
- `Attr5`: 979.533188
- `Attr6`: 0.208198
- `Attr7`: 0.118745
- `Attr8`: 9.706388
- `Attr9`: 1.154364
- `Attr10`: 0.314787
- `Attr11`: 0.109538
- `Attr12`: 1.239373
- `Attr13`: 1.255792
- `Attr14`: 0.119050
- `Attr15`: 4262.557769
- `Attr16`: 1.167919
- `Attr17`: 10.302226
- `Attr18`: 0.124766
- `Attr19`: 0.674015
- `Attr20`: 203.159813
- `Attr21`: 2.633343
- `Attr22`: 0.103596
- `Attr23`: 0.658310
- `Attr24`: 0.239066
- `Attr25`: 0.267458
- `Attr26`: 1.064478
- `Attr27`: 1194.131636
- `Attr28`: 5.583862
- `Attr29`: 0.079089
- `Attr30`: 8.002218
- `Attr31`: 0.688117
- `Attr32`: 806.392772
- `Attr33`: 2.551805
- `Attr34`: 2.299486
- `Attr35`: 0.092373
- `Attr36`: 1.207382
- `Attr37`: 57.166764
- `Attr38`: 0.313043
- `Attr39`: 0.316964
- `Attr40`: 1.458732
- `Attr41`: 9.222224
- `Attr42`: 0.342965
- `Attr43`: 976.396153
- `Attr44`: 785.134525
- `Attr45`: 33.458475
- `Attr46`: 3.684168
- `Attr47`: 310.756534
- `Attr48`: 0.115127
- `Attr49`: 0.525932
- `Attr50`: 3.772772
- `Attr51`: 0.105312
- `Attr52`: 6.191373
- `Attr53`: 21.214197
- `Attr54`: 21.043048
- `Attr55`: 8705.552253
- `Attr56`: 26.755007
- `Attr57`: 0.416599
- `Attr58`: 31.131644
- `Attr59`: 1.568002
- `Attr60`: 406.298823
- `Attr61`: 9.134616
- `Attr62`: 1477.671337
- `Attr63`: 2.630857
- `Attr64`: 59.648741

#### Integer/Categorical (TVD)
- (none)

## Details: TabPFGen
- MMD (RBF): `0.000564`
- Correlation gap (Frobenius): `3.829020`
- C2ST accuracy: `0.5212`, ROC-AUC: `0.5159`

### Marginal distances
#### Numeric (Wasserstein)
- `Attr1`: 0.041640
- `Attr2`: 0.085947
- `Attr3`: 0.077458
- `Attr4`: 2.181809
- `Attr5`: 343.993689
- `Attr6`: 0.122066
- `Attr7`: 0.049548
- `Attr8`: 4.406622
- `Attr9`: 0.371801
- `Attr10`: 0.164881
- `Attr11`: 0.054428
- `Attr12`: 0.964016
- `Attr13`: 0.483268
- `Attr14`: 0.049532
- `Attr15`: 1530.604672
- `Attr16`: 0.854750
- `Attr17`: 4.474526
- `Attr18`: 0.051151
- `Attr19`: 0.412923
- `Attr20`: 18.136335
- `Attr21`: 1.037484
- `Attr22`: 0.050097
- `Attr23`: 0.402067
- `Attr24`: 0.097074
- `Attr25`: 0.173807
- `Attr26`: 0.822026
- `Attr27`: 278.129349
- `Attr28`: 2.954489
- `Attr29`: 0.020719
- `Attr30`: 2.055767
- `Attr31`: 0.403929
- `Attr32`: 475.111720
- `Attr33`: 1.537678
- `Attr34`: 1.016923
- `Attr35`: 0.049818
- `Attr36`: 0.347390
- `Attr37`: 24.650438
- `Attr38`: 0.166655
- `Attr39`: 0.246192
- `Attr40`: 0.764095
- `Attr41`: 1.888785
- `Attr42`: 0.130680
- `Attr43`: 161.329816
- `Attr44`: 152.942898
- `Attr45`: 13.562042
- `Attr46`: 2.083546
- `Attr47`: 201.871817
- `Attr48`: 0.059782
- `Attr49`: 0.120894
- `Attr50`: 1.934090
- `Attr51`: 0.074075
- `Attr52`: 1.839723
- `Attr53`: 7.904137
- `Attr54`: 8.024156
- `Attr55`: 1403.718487
- `Attr56`: 4.669521
- `Attr57`: 0.232511
- `Attr58`: 6.247497
- `Attr59`: 1.200083
- `Attr60`: 160.525330
- `Attr61`: 3.146696
- `Attr62`: 518.659193
- `Attr63`: 1.588767
- `Attr64`: 21.199177

#### Integer/Categorical (TVD)
- (none)
