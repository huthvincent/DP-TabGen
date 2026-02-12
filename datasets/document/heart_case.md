# Heart Dataset Cases

## Baseline (synthetic_100/200/1000)

| variant        | model    |   test_accuracy |   test_balanced_accuracy |   test_auroc |
|:---------------|:---------|----------------:|-------------------------:|-------------:|
| synthetic_100  | logistic |          0.5545 |                   0.5456 |       0.5544 |
| synthetic_100  | xgboost  |          0.5314 |                   0.5364 |       0.5223 |
| synthetic_100  | lightgbm |          0.4851 |                   0.4926 |       0.5192 |
| synthetic_100  | catboost |          0.4488 |                   0.4541 |       0.4584 |
| synthetic_100  | mlp      |          0.3927 |                   0.3946 |       0.3669 |
| synthetic_200  | logistic |          0.4587 |                   0.4485 |       0.4776 |
| synthetic_200  | xgboost  |          0.5083 |                   0.503  |       0.4889 |
| synthetic_200  | lightgbm |          0.4851 |                   0.4756 |       0.478  |
| synthetic_200  | catboost |          0.5149 |                   0.5091 |       0.4911 |
| synthetic_200  | mlp      |          0.4323 |                   0.4274 |       0.44   |
| synthetic_1000 | logistic |          0.538  |                   0.5057 |       0.5447 |
| synthetic_1000 | xgboost  |          0.5017 |                   0.4947 |       0.4749 |
| synthetic_1000 | lightgbm |          0.5182 |                   0.5105 |       0.4777 |
| synthetic_1000 | catboost |          0.4818 |                   0.4747 |       0.4481 |
| synthetic_1000 | mlp      |          0.5281 |                   0.5245 |       0.5091 |

## v2 config (8 rounds, clip 0.005/0.995)

| variant           | model    |   test_accuracy |   test_balanced_accuracy |   test_auroc |
|:------------------|:---------|----------------:|-------------------------:|-------------:|
| synthetic_100_v2  | logistic |          0.5677 |                   0.5529 |       0.5772 |
| synthetic_100_v2  | xgboost  |          0.4917 |                   0.4828 |       0.4934 |
| synthetic_100_v2  | lightgbm |          0.5215 |                   0.5163 |       0.5089 |
| synthetic_100_v2  | catboost |          0.5281 |                   0.5191 |       0.5095 |
| synthetic_100_v2  | mlp      |          0.4785 |                   0.4722 |       0.5007 |
| synthetic_200_v2  | logistic |          0.4521 |                   0.4369 |       0.3866 |
| synthetic_200_v2  | xgboost  |          0.4356 |                   0.4277 |       0.4391 |
| synthetic_200_v2  | lightgbm |          0.4422 |                   0.4343 |       0.4002 |
| synthetic_200_v2  | catboost |          0.4356 |                   0.4244 |       0.4159 |
| synthetic_200_v2  | mlp      |          0.4752 |                   0.4736 |       0.446  |
| synthetic_1000_v2 | logistic |          0.4587 |                   0.4276 |       0.3043 |
| synthetic_1000_v2 | xgboost  |          0.5017 |                   0.4947 |       0.4595 |
| synthetic_1000_v2 | lightgbm |          0.5017 |                   0.498  |       0.5004 |
| synthetic_1000_v2 | catboost |          0.5116 |                   0.4994 |       0.4609 |
| synthetic_1000_v2 | mlp      |          0.4851 |                   0.4833 |       0.4694 |

## v3a config (6 rounds, clip 0.01/0.99)

| variant            | model    |   test_accuracy |   test_balanced_accuracy |   test_auroc |
|:-------------------|:---------|----------------:|-------------------------:|-------------:|
| synthetic_100_v3a  | logistic |          0.835  |                   0.8289 |       0.9105 |
| synthetic_100_v3a  | xgboost  |          0.8284 |                   0.8234 |       0.893  |
| synthetic_100_v3a  | lightgbm |          0.8152 |                   0.8095 |       0.8838 |
| synthetic_100_v3a  | catboost |          0.8317 |                   0.8253 |       0.9015 |
| synthetic_100_v3a  | mlp      |          0.8284 |                   0.8245 |       0.8534 |
| synthetic_200_v3a  | logistic |          0.8482 |                   0.8444 |       0.9103 |
| synthetic_200_v3a  | xgboost  |          0.8449 |                   0.8419 |       0.9125 |
| synthetic_200_v3a  | lightgbm |          0.8416 |                   0.8389 |       0.9115 |
| synthetic_200_v3a  | catboost |          0.8482 |                   0.8433 |       0.9139 |
| synthetic_200_v3a  | mlp      |          0.8119 |                   0.8087 |       0.8749 |
| synthetic_1000_v3a | logistic |          0.8482 |                   0.8433 |       0.9118 |
| synthetic_1000_v3a | xgboost  |          0.8317 |                   0.8264 |       0.8965 |
| synthetic_1000_v3a | lightgbm |          0.8416 |                   0.8361 |       0.8906 |
| synthetic_1000_v3a | catboost |          0.8482 |                   0.8428 |       0.9117 |
| synthetic_1000_v3a | mlp      |          0.7921 |                   0.7887 |       0.8484 |

## v3b config (6 rounds, clip 0.005/0.995)

| variant            | model    |   test_accuracy |   test_balanced_accuracy |   test_auroc |
|:-------------------|:---------|----------------:|-------------------------:|-------------:|
| synthetic_100_v3b  | logistic |          0.835  |                   0.8284 |       0.9112 |
| synthetic_100_v3b  | xgboost  |          0.8086 |                   0.8018 |       0.8902 |
| synthetic_100_v3b  | lightgbm |          0.8251 |                   0.8198 |       0.8862 |
| synthetic_100_v3b  | catboost |          0.8218 |                   0.8162 |       0.9029 |
| synthetic_100_v3b  | mlp      |          0.8284 |                   0.825  |       0.8525 |
| synthetic_200_v3b  | logistic |          0.8482 |                   0.8444 |       0.91   |
| synthetic_200_v3b  | xgboost  |          0.8482 |                   0.845  |       0.9192 |
| synthetic_200_v3b  | lightgbm |          0.8383 |                   0.8353 |       0.9114 |
| synthetic_200_v3b  | catboost |          0.8416 |                   0.8389 |       0.9195 |
| synthetic_200_v3b  | mlp      |          0.8251 |                   0.8231 |       0.8752 |
| synthetic_1000_v3b | logistic |          0.8482 |                   0.8433 |       0.912  |
| synthetic_1000_v3b | xgboost  |          0.8383 |                   0.8325 |       0.9008 |
| synthetic_1000_v3b | lightgbm |          0.8449 |                   0.8392 |       0.8945 |
| synthetic_1000_v3b | catboost |          0.8383 |                   0.8325 |       0.9089 |
| synthetic_1000_v3b | mlp      |          0.8119 |                   0.8087 |       0.8582 |

## v3c config (best: 8 rounds, clip 0.01/0.99)

| variant            | model    |   test_accuracy |   test_balanced_accuracy |   test_auroc |
|:-------------------|:---------|----------------:|-------------------------:|-------------:|
| synthetic_100_v3c  | logistic |          0.8449 |                   0.8403 |       0.9095 |
| synthetic_100_v3c  | xgboost  |          0.8449 |                   0.8392 |       0.9071 |
| synthetic_100_v3c  | lightgbm |          0.8317 |                   0.8237 |       0.9028 |
| synthetic_100_v3c  | catboost |          0.835  |                   0.8295 |       0.9009 |
| synthetic_100_v3c  | mlp      |          0.8152 |                   0.8117 |       0.8593 |
| synthetic_200_v3c  | logistic |          0.8482 |                   0.8428 |       0.909  |
| synthetic_200_v3c  | xgboost  |          0.8548 |                   0.8494 |       0.9232 |
| synthetic_200_v3c  | lightgbm |          0.8449 |                   0.8386 |       0.9259 |
| synthetic_200_v3c  | catboost |          0.8548 |                   0.8472 |       0.926  |
| synthetic_200_v3c  | mlp      |          0.8449 |                   0.8364 |       0.8998 |
| synthetic_1000_v3c | logistic |          0.8515 |                   0.8458 |       0.9071 |
| synthetic_1000_v3c | xgboost  |          0.8284 |                   0.8228 |       0.9024 |
| synthetic_1000_v3c | lightgbm |          0.835  |                   0.8289 |       0.9043 |
| synthetic_1000_v3c | catboost |          0.8317 |                   0.8259 |       0.9085 |
| synthetic_1000_v3c | mlp      |          0.8086 |                   0.8045 |       0.8653 |
