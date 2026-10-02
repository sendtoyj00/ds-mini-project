# DS-MINI independent implementation report



설계서 p5/6/13/16/18의 관측 제한, Feature 정의, M0–M4, Pipeline 및 평가 순서를 준수합니다.

Group CV 중심 선택은 사용자의 추가 요구를 적용했습니다. Batch2/3 수명 값은 선택과 최종 refit 뒤에 평가 전용으로 읽었습니다.



## Selection

Selected: **M1_log10_RandomForest**, parameters `{"max_depth": 2, "max_features": 0.7, "min_samples_leaf": 3, "n_estimators": 100}`.

Rule fixed before evaluation: Group CV mean MAPE within 0.5 percentage points of best: fewer features, lower fold std, complexity, mean MAPE, candidate ID; Valid once without reselection

Valid 확인 뒤 후보 변경 없음. External test 후 후보 변경 없음.



## Performance Reporting (assignment format)

| index | value | unit | note |
| --- | --- | --- | --- |
| Train (Batch 1 CV) | 8.100 | % | Group CV fold mean; Train36 |
| Valid (Batch 1 Hold-out) | 13.792 | % | Hold-out10; one evaluation |
| Test (Batch 2) | 31.309 | % | Final refit on Batch1 46; Test39 |
| Gap (Train-Valid) | 5.692 | %p | Valid - Train; positive: possible overfitting |
| Gap (Valid-Test) | 17.516 | %p | Batch2 - Valid; positive: generalization degradation |
| Gap (Target-Test), Batch2 | 22.209 | %p | Batch2 - 9.1%; paper reference |
| Test (Batch 3) | 12.900 | % | Same final model; additional Test44 |
| Gap (Batch2-Batch3) | -18.409 | %p | Batch3 - Batch2; positive: Batch3 has higher error |
| Gap (Target-Test), Batch3 | 3.800 | %p | Batch3 - 9.1%; paper reference |



## Supplementary metrics (pooled OOF for Train)

| partition | n | mape_pct | mae | mean_signed_error | overprediction_rate_pct | baseline_mape_pct |
| --- | --- | --- | --- | --- | --- | --- |
| Train | 36 | 8.172 | 68.670 | -9.571 | 44.444 | 19.630 |
| Valid | 10 | 13.792 | 118.557 | 62.888 | 50.000 | 13.683 |
| Batch2 | 39 | 31.309 | 153.757 | 147.241 | 94.872 | 72.656 |
| Batch3 | 44 | 12.900 | 155.504 | -68.363 | 47.727 | 20.201 |



제출 표의 Train은 선택된 고정 후보의 Group CV fold 평균입니다. 보조 performance.csv의 Train은 pooled OOF이며 서로 다른 집계입니다. Gap은 뒤 단계 − 앞 단계로 계산하여 양수가 오차 악화를 나타내게 했습니다.

선택 CV fold 평균=8.100%, 표준편차=3.215%p.

전체 선택 절차의 nested Group CV fold 평균=9.948%, 표준편차=3.035%p. 튜닝/선택 낙관 편향을 포함하지 않는 보조 개발 성능입니다.



## Gaps (%p)

- Valid - Train (CV fold mean): 5.692%p

- Test - Valid: 17.516%p

- Batch2 - project goal: 22.209%p

- Batch3 - Batch2: -18.409%p

- Batch3 - project goal: 3.800%p



## Candidate comparison

| candidate_id | feature_set | feature_count | model | target | params | cv_mape_mean | cv_mape_std | cv_mae_mean | cv_oof_mape | cell_cv_mape_mean | cell_cv_mape_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M1_log10_RandomForest | M1 | 1 | RandomForest | log10 | {"max_depth": 2, "max_features": 0.7, "min_samples_leaf": 3, "n_estimators": 100} | 8.100 | 3.215 | 68.117 | 8.172 | 8.483 | 3.202 |
| M3_raw_Ridge | M3 | 5 | Ridge | raw | {"alpha": 1.0} | 8.134 | 1.473 | 72.281 | 8.093 | 8.107 | 2.014 |
| M3_clean_raw_Ridge | M3_clean | 5 | Ridge | raw | {"alpha": 1.0} | 8.136 | 1.477 | 72.288 | 8.095 | 8.110 | 2.013 |
| M1_raw_RandomForest | M1 | 1 | RandomForest | raw | {"max_depth": 3, "max_features": 0.7, "min_samples_leaf": 3, "n_estimators": 100} | 8.256 | 3.512 | 69.058 | 8.340 | 8.750 | 3.493 |
| M3_raw_ElasticNet | M3 | 5 | ElasticNet | raw | {"alpha": 0.01, "l1_ratio": 0.2} | 8.280 | 1.616 | 74.156 | 8.231 | 8.106 | 1.664 |
| M3_clean_raw_ElasticNet | M3_clean | 5 | ElasticNet | raw | {"alpha": 0.01, "l1_ratio": 0.2} | 8.281 | 1.618 | 74.156 | 8.232 | 8.108 | 1.663 |
| M2_log10_RandomForest | M2 | 3 | RandomForest | log10 | {"max_depth": 2, "max_features": 1.0, "min_samples_leaf": 3, "n_estimators": 100} | 8.328 | 3.669 | 69.671 | 8.393 | 8.492 | 3.153 |
| M3_raw_Linear | M3 | 5 | Linear | raw | {} | 8.344 | 1.704 | 74.931 | 8.291 | 8.111 | 1.542 |
| M3_clean_raw_Linear | M3_clean | 5 | Linear | raw | {} | 8.345 | 1.706 | 74.928 | 8.292 | 8.113 | 1.541 |
| M3_clean_log10_Ridge | M3_clean | 5 | Ridge | log10 | {"alpha": 1.0} | 8.474 | 2.705 | 78.335 | 8.401 | 8.082 | 2.091 |
| M4_log10_RandomForest | M4 | 7 | RandomForest | log10 | {"max_depth": 3, "max_features": 1.0, "min_samples_leaf": 3, "n_estimators": 100} | 8.475 | 3.779 | 71.170 | 8.522 | 9.122 | 4.228 |
| M4_clean_log10_RandomForest | M4_clean | 7 | RandomForest | log10 | {"max_depth": 3, "max_features": 1.0, "min_samples_leaf": 3, "n_estimators": 100} | 8.475 | 3.779 | 71.170 | 8.522 | 9.122 | 4.228 |
| M3_log10_Ridge | M3 | 5 | Ridge | log10 | {"alpha": 1.0} | 8.476 | 2.710 | 78.347 | 8.403 | 8.079 | 2.090 |
| M3_log10_RandomForest | M3 | 5 | RandomForest | log10 | {"max_depth": 3, "max_features": 1.0, "min_samples_leaf": 3, "n_estimators": 100} | 8.533 | 3.684 | 71.816 | 8.585 | 9.200 | 4.585 |
| M3_clean_log10_RandomForest | M3_clean | 5 | RandomForest | log10 | {"max_depth": 3, "max_features": 1.0, "min_samples_leaf": 3, "n_estimators": 100} | 8.533 | 3.684 | 71.816 | 8.585 | 9.200 | 4.585 |
| M3_clean_log10_ElasticNet | M3_clean | 5 | ElasticNet | log10 | {"alpha": 0.0001, "l1_ratio": 0.8} | 8.676 | 3.382 | 81.124 | 8.587 | 8.076 | 2.017 |
| M3_log10_ElasticNet | M3 | 5 | ElasticNet | log10 | {"alpha": 0.0001, "l1_ratio": 0.8} | 8.679 | 3.389 | 81.145 | 8.590 | 8.074 | 2.016 |
| M3_clean_log10_Linear | M3_clean | 5 | Linear | log10 | {} | 8.702 | 3.422 | 81.308 | 8.611 | 8.068 | 2.033 |
| M2_raw_RandomForest | M2 | 3 | RandomForest | raw | {"max_depth": 3, "max_features": 1.0, "min_samples_leaf": 3, "n_estimators": 100} | 8.703 | 4.114 | 72.161 | 8.776 | 8.785 | 3.454 |
| M3_log10_Linear | M3 | 5 | Linear | log10 | {} | 8.704 | 3.428 | 81.329 | 8.614 | 8.066 | 2.032 |
| M4_clean_raw_RandomForest | M4_clean | 7 | RandomForest | raw | {"max_depth": 3, "max_features": 1.0, "min_samples_leaf": 3, "n_estimators": 100} | 8.736 | 3.739 | 72.146 | 8.785 | 9.254 | 4.622 |
| M4_raw_RandomForest | M4 | 7 | RandomForest | raw | {"max_depth": 3, "max_features": 1.0, "min_samples_leaf": 3, "n_estimators": 100} | 8.736 | 3.739 | 72.146 | 8.785 | 9.254 | 4.622 |
| M4_clean_log10_ElasticNet | M4_clean | 7 | ElasticNet | log10 | {"alpha": 0.01, "l1_ratio": 0.2} | 8.849 | 2.983 | 82.423 | 8.782 | 9.119 | 2.434 |
| M4_log10_ElasticNet | M4 | 7 | ElasticNet | log10 | {"alpha": 0.01, "l1_ratio": 0.2} | 8.852 | 2.984 | 82.439 | 8.785 | 9.120 | 2.435 |
| M3_clean_raw_RandomForest | M3_clean | 5 | RandomForest | raw | {"max_depth": 3, "max_features": 1.0, "min_samples_leaf": 3, "n_estimators": 100} | 8.922 | 3.713 | 74.141 | 8.973 | 9.279 | 5.039 |
| M3_raw_RandomForest | M3 | 5 | RandomForest | raw | {"max_depth": 3, "max_features": 1.0, "min_samples_leaf": 3, "n_estimators": 100} | 8.922 | 3.713 | 74.141 | 8.973 | 9.279 | 5.039 |
| M4_raw_ElasticNet | M4 | 7 | ElasticNet | raw | {"alpha": 1.0, "l1_ratio": 0.8} | 9.124 | 1.997 | 75.574 | 9.058 | 8.722 | 4.034 |
| M4_clean_raw_ElasticNet | M4_clean | 7 | ElasticNet | raw | {"alpha": 1.0, "l1_ratio": 0.8} | 9.128 | 2.004 | 75.605 | 9.062 | 8.724 | 4.033 |
| M4_log10_Ridge | M4 | 7 | Ridge | log10 | {"alpha": 10.0} | 9.169 | 1.308 | 78.819 | 9.121 | 8.994 | 3.930 |
| M4_clean_log10_Ridge | M4_clean | 7 | Ridge | log10 | {"alpha": 10.0} | 9.175 | 1.305 | 78.881 | 9.127 | 8.999 | 3.926 |
| M2_raw_ElasticNet | M2 | 3 | ElasticNet | raw | {"alpha": 1.0, "l1_ratio": 0.8} | 9.285 | 0.985 | 80.179 | 9.275 | 9.478 | 2.185 |
| M1_raw_Linear | M1 | 1 | Linear | raw | {} | 9.288 | 1.763 | 82.835 | 9.308 | 9.301 | 1.461 |
| M1_raw_ElasticNet | M1 | 1 | ElasticNet | raw | {"alpha": 0.0001, "l1_ratio": 0.8} | 9.289 | 1.762 | 82.835 | 9.308 | 9.301 | 1.461 |
| M1_raw_Ridge | M1 | 1 | Ridge | raw | {"alpha": 0.01} | 9.289 | 1.759 | 82.835 | 9.309 | 9.303 | 1.462 |
| M2_raw_Ridge | M2 | 3 | Ridge | raw | {"alpha": 1.0} | 9.296 | 1.348 | 82.139 | 9.289 | 8.544 | 0.936 |
| M4_raw_Ridge | M4 | 7 | Ridge | raw | {"alpha": 10.0} | 9.611 | 1.503 | 78.461 | 9.546 | 9.053 | 4.495 |
| M4_clean_raw_Ridge | M4_clean | 7 | Ridge | raw | {"alpha": 10.0} | 9.617 | 1.505 | 78.512 | 9.552 | 9.053 | 4.496 |
| M2_raw_Linear | M2 | 3 | Linear | raw | {} | 9.704 | 2.043 | 85.267 | 9.682 | 8.460 | 0.913 |
| M2_log10_Ridge | M2 | 3 | Ridge | log10 | {"alpha": 1.0} | 9.894 | 2.954 | 90.934 | 9.858 | 9.224 | 1.733 |
| M4_clean_log10_Linear | M4_clean | 7 | Linear | log10 | {} | 9.930 | 4.713 | 88.878 | 9.798 | 9.552 | 3.312 |
| M4_log10_Linear | M4 | 7 | Linear | log10 | {} | 9.931 | 4.714 | 88.879 | 9.799 | 9.551 | 3.317 |
| M2_log10_Linear | M2 | 3 | Linear | log10 | {} | 10.004 | 3.044 | 92.346 | 9.950 | 8.924 | 1.972 |
| M2_log10_ElasticNet | M2 | 3 | ElasticNet | log10 | {"alpha": 0.0001, "l1_ratio": 0.2} | 10.005 | 3.044 | 92.357 | 9.952 | 8.932 | 1.963 |
| M1_log10_XGBoost | M1 | 1 | XGBoost | log10 | {"learning_rate": 0.03, "max_depth": 2, "min_child_weight": 5, "n_estimators": 120, "reg_lambda": 10.0} | 10.370 | 4.551 | 86.249 | 10.329 | 11.204 | 5.798 |
| M1_raw_XGBoost | M1 | 1 | XGBoost | raw | {"learning_rate": 0.03, "max_depth": 2, "min_child_weight": 5, "n_estimators": 120, "reg_lambda": 10.0} | 10.481 | 4.492 | 86.113 | 10.441 | 11.378 | 6.312 |
| M2_log10_XGBoost | M2 | 3 | XGBoost | log10 | {"learning_rate": 0.03, "max_depth": 2, "min_child_weight": 5, "n_estimators": 120, "reg_lambda": 10.0} | 10.719 | 5.199 | 88.364 | 10.670 | 11.208 | 5.810 |
| M2_raw_XGBoost | M2 | 3 | XGBoost | raw | {"learning_rate": 0.03, "max_depth": 2, "min_child_weight": 5, "n_estimators": 120, "reg_lambda": 10.0} | 10.727 | 5.070 | 87.270 | 10.675 | 11.451 | 6.467 |
| M3_raw_XGBoost | M3 | 5 | XGBoost | raw | {"learning_rate": 0.03, "max_depth": 1, "min_child_weight": 5, "n_estimators": 120, "reg_lambda": 10.0} | 10.847 | 4.802 | 88.909 | 10.799 | 11.924 | 6.726 |
| M3_clean_raw_XGBoost | M3_clean | 5 | XGBoost | raw | {"learning_rate": 0.03, "max_depth": 1, "min_child_weight": 5, "n_estimators": 120, "reg_lambda": 10.0} | 10.847 | 4.802 | 88.909 | 10.799 | 11.924 | 6.726 |
| M1_log10_Linear | M1 | 1 | Linear | log10 | {} | 10.956 | 4.212 | 100.842 | 10.913 | 10.382 | 2.429 |
| M1_log10_ElasticNet | M1 | 1 | ElasticNet | log10 | {"alpha": 0.0001, "l1_ratio": 0.2} | 10.956 | 4.208 | 100.833 | 10.914 | 10.383 | 2.428 |
| M1_log10_Ridge | M1 | 1 | Ridge | log10 | {"alpha": 0.01} | 10.956 | 4.208 | 100.835 | 10.914 | 10.383 | 2.428 |
| M3_clean_log10_XGBoost | M3_clean | 5 | XGBoost | log10 | {"learning_rate": 0.03, "max_depth": 1, "min_child_weight": 5, "n_estimators": 120, "reg_lambda": 10.0} | 11.027 | 4.741 | 92.084 | 10.984 | 11.939 | 6.248 |
| M3_log10_XGBoost | M3 | 5 | XGBoost | log10 | {"learning_rate": 0.03, "max_depth": 1, "min_child_weight": 5, "n_estimators": 120, "reg_lambda": 10.0} | 11.027 | 4.741 | 92.084 | 10.984 | 11.939 | 6.248 |
| M1_log10_LightGBM | M1 | 1 | LightGBM | log10 | {"learning_rate": 0.03, "max_depth": 2, "min_child_samples": 5, "n_estimators": 120, "num_leaves": 3, "reg_lambda": 10.0} | 11.030 | 3.754 | 91.837 | 11.023 | 10.536 | 3.153 |
| M2_log10_LightGBM | M2 | 3 | LightGBM | log10 | {"learning_rate": 0.03, "max_depth": 2, "min_child_samples": 5, "n_estimators": 120, "num_leaves": 3, "reg_lambda": 10.0} | 11.034 | 3.847 | 91.425 | 11.028 | 10.610 | 3.246 |
| M1_raw_LightGBM | M1 | 1 | LightGBM | raw | {"learning_rate": 0.03, "max_depth": 2, "min_child_samples": 5, "n_estimators": 120, "num_leaves": 3, "reg_lambda": 10.0} | 11.065 | 3.853 | 90.664 | 11.060 | 10.619 | 3.512 |
| M3_clean_log10_LightGBM | M3_clean | 5 | LightGBM | log10 | {"learning_rate": 0.03, "max_depth": 2, "min_child_samples": 5, "n_estimators": 120, "num_leaves": 3, "reg_lambda": 10.0} | 11.078 | 4.135 | 93.378 | 11.057 | 10.932 | 3.827 |
| M3_log10_LightGBM | M3 | 5 | LightGBM | log10 | {"learning_rate": 0.03, "max_depth": 2, "min_child_samples": 5, "n_estimators": 120, "num_leaves": 3, "reg_lambda": 10.0} | 11.078 | 4.135 | 93.378 | 11.057 | 10.932 | 3.827 |
| M2_raw_LightGBM | M2 | 3 | LightGBM | raw | {"learning_rate": 0.03, "max_depth": 2, "min_child_samples": 5, "n_estimators": 120, "num_leaves": 3, "reg_lambda": 10.0} | 11.084 | 4.038 | 90.223 | 11.078 | 10.596 | 3.569 |
| M4_clean_log10_XGBoost | M4_clean | 7 | XGBoost | log10 | {"learning_rate": 0.03, "max_depth": 1, "min_child_weight": 5, "n_estimators": 120, "reg_lambda": 10.0} | 11.101 | 4.700 | 92.896 | 11.058 | 11.913 | 6.179 |
| M4_log10_XGBoost | M4 | 7 | XGBoost | log10 | {"learning_rate": 0.03, "max_depth": 1, "min_child_weight": 5, "n_estimators": 120, "reg_lambda": 10.0} | 11.101 | 4.700 | 92.896 | 11.058 | 11.913 | 6.179 |
| M4_clean_log10_LightGBM | M4_clean | 7 | LightGBM | log10 | {"learning_rate": 0.03, "max_depth": 2, "min_child_samples": 5, "n_estimators": 120, "num_leaves": 3, "reg_lambda": 10.0} | 11.119 | 3.912 | 93.936 | 11.076 | 10.996 | 3.572 |
| M4_log10_LightGBM | M4 | 7 | LightGBM | log10 | {"learning_rate": 0.03, "max_depth": 2, "min_child_samples": 5, "n_estimators": 120, "num_leaves": 3, "reg_lambda": 10.0} | 11.119 | 3.912 | 93.936 | 11.076 | 10.996 | 3.572 |
| M4_clean_raw_XGBoost | M4_clean | 7 | XGBoost | raw | {"learning_rate": 0.03, "max_depth": 1, "min_child_weight": 5, "n_estimators": 120, "reg_lambda": 10.0} | 11.142 | 4.636 | 91.388 | 11.091 | 11.922 | 6.525 |
| M4_raw_XGBoost | M4 | 7 | XGBoost | raw | {"learning_rate": 0.03, "max_depth": 1, "min_child_weight": 5, "n_estimators": 120, "reg_lambda": 10.0} | 11.142 | 4.636 | 91.388 | 11.091 | 11.922 | 6.525 |
| M4_clean_raw_Linear | M4_clean | 7 | Linear | raw | {} | 11.267 | 5.775 | 92.909 | 11.112 | 10.592 | 4.949 |
| M4_raw_Linear | M4 | 7 | Linear | raw | {} | 11.271 | 5.782 | 92.938 | 11.115 | 10.596 | 4.961 |
| M3_clean_raw_LightGBM | M3_clean | 5 | LightGBM | raw | {"learning_rate": 0.03, "max_depth": 2, "min_child_samples": 5, "n_estimators": 120, "num_leaves": 3, "reg_lambda": 10.0} | 11.276 | 4.489 | 93.840 | 11.254 | 10.825 | 4.111 |
| M3_raw_LightGBM | M3 | 5 | LightGBM | raw | {"learning_rate": 0.03, "max_depth": 2, "min_child_samples": 5, "n_estimators": 120, "num_leaves": 3, "reg_lambda": 10.0} | 11.276 | 4.489 | 93.840 | 11.254 | 10.825 | 4.111 |
| M4_clean_raw_LightGBM | M4_clean | 7 | LightGBM | raw | {"learning_rate": 0.03, "max_depth": 2, "min_child_samples": 5, "n_estimators": 120, "num_leaves": 3, "reg_lambda": 10.0} | 11.613 | 3.936 | 95.844 | 11.556 | 10.943 | 3.599 |
| M4_raw_LightGBM | M4 | 7 | LightGBM | raw | {"learning_rate": 0.03, "max_depth": 2, "min_child_samples": 5, "n_estimators": 120, "num_leaves": 3, "reg_lambda": 10.0} | 11.613 | 3.936 | 95.844 | 11.556 | 10.943 | 3.599 |
| M0_raw_Dummy | M0 | 0 | Dummy | raw | {} | 19.756 | 6.142 | 155.314 | 19.630 | 21.045 | 5.555 |



## Generalization diagnostics

Batch2: MAPE=31.309%, mean signed error=147.2 cycles, overprediction=94.9%. Median baseline improvement=41.348%p.

| feature | missing_rate_pct | outside_batch1_pct | below_batch1_pct | above_batch1_pct |
| --- | --- | --- | --- | --- |
| log10_dq_var | 0.00 | 28.21 | 0.00 | 28.21 |

Batch3: MAPE=12.900%, mean signed error=-68.4 cycles, overprediction=47.7%. Median baseline improvement=7.301%p.

| feature | missing_rate_pct | outside_batch1_pct | below_batch1_pct | above_batch1_pct |
| --- | --- | --- | --- | --- |
| log10_dq_var | 0.00 | 2.27 | 2.27 | 0.00 |



범위 이탈 비율은 유효 관측값을 분모로 계산하며 결측 비율은 별도 표시합니다. Batch2는 단수명 방향, Batch3는 장수명 방향으로 Target 분포가 이동합니다. 이는 설계서에서도 이미 확인된 외부 데이터 특성이므로, 완전히 미관측인 test라는 주장은 하지 않습니다.

용량·온도·IR 배치 분포 및 Feature 추가의 CV 기여는 feature_distribution.csv와 feature_addition_effects.csv에서 확인합니다. Linear 최종 모델이면 linear_coefficients.csv와 linear_contributions.csv에서 계수·영향 방향을 확인합니다.

작은 표본에서 높은 fold 편차, 정책 중복, 학습 범위 밖 외삽, 입력과 수명 관계의 배치 차이를 함께 해석해야 합니다. 이 진단은 후보 재선택에 사용하지 않습니다.



## Further improvements

새로운 Batch1 계열 셀과 단수명 셀을 추가 확보하고, 미래의 독립 배치를 사전 등록한 프로토콜로 평가합니다. 배치·장비 측정 교정을 확보하면 입력 분포 이동 원인을 더 명확히 분리할 수 있습니다. 현재 외부 test를 보고 튜닝하지 않습니다.



## Implementation decisions / limitations

기존 split 파일은 없어서 seed 42의 36/10 분할을 생성 후 고정했습니다. 원본 139셀 중 Target 없는 10셀만 제외했습니다. 별도 VarCharge 2셀 파일은 설계서 139셀 cohort 밖이며 입력으로 사용하지 않습니다.

설계서에 이동 중앙값 window/ddof가 없어 기존 설정 11 및 sample variance(ddof=1)를 명시적으로 유지했습니다. 2% 급변은 오류 확정이 아닌 대안 실험입니다. 용량 중앙값 최소 1점, Theil–Sen 최소 2점을 사용하며 임의로 더 강한 유효표본 제약을 추가하지 않았습니다.

M0는 raw training target 중앙값 하나만 사용합니다. 나머지 M1–M4 및 clean 대안은 모든 6개 모델×두 Target을 동일 fold로 비교합니다. raw/log의 ElasticNet alpha는 단위가 다르므로 넓은 간격의 작은 grid를 사용합니다.

예측 clipping, log bias correction, target 기반 셀 제외, 사후 feature 변경은 적용하지 않습니다. 모든 결측 column fallback은 fold-local sklearn keep_empty_features 동작(0)으로 정의합니다.

충전시간 등 16개 원본 후보와 clean 용량 2개를 보존하되 모델에는 지정된 7개 및 clean 대안만 입력합니다. 설계서에 세부 정의가 없는 보존용 qd_change/qd_ratio는 Cycle96–100 median 대비 초기값으로 정의했습니다.

코드, 입력, 설정, 분할, 모델 해시와 이벤트 순서는 experiment_frozen.json / event_log.json / verification.json에 기록합니다.
