# ESS 배터리 수명 예측



## 모델링

최종 모델: **M3 + Ridge**, Target: raw, 파라미터: `{"alpha": 1.0}`.

Linear·Ridge·ElasticNet을 최종 후보로 사용하고 얕은 Random Forest·XGBoost·LightGBM은 비선형 비교군으로 평가한다. 학습 수명 범위를 벗어나는 예측이 필요하므로 외삽 가능한 모델을 사용한다.

M0 중앙값 기준, M1 ΔQ, M2 충전 조건, M3 초기 용량, M4 온도·IR, 급변 후보를 제외한 대안 용량 피처를 비교한다. 원 Target과 log10 Target의 점수는 원 사이클 단위로 계산한다.

선택 기준: Extrapolating Linear/Ridge/ElasticNet candidates only; Group CV mean MAPE within 0.5 percentage points of best, then fewer features, lower fold std, complexity, mean MAPE, candidate ID; Valid once without reselection

Batch 1 Train 36셀에서 충전 정책 Group CV로 후보를 선택하고, Valid 10셀을 한 번 평가한다. 최종 모델은 Batch 1 전체 46셀로 학습한다. 대치·표준화는 각 학습 fold 내부에서 학습하며 Test 셀을 사후 제외하지 않는다.



## 성능 결과

| index | value | unit | note |
| --- | --- | --- | --- |
| Train (Batch 1 CV) | 8.134 | % | Group CV fold mean; Train36 |
| Valid (Batch 1 Hold-out) | 6.271 | % | Hold-out10; one evaluation |
| Test (Batch 2) | 59.537 | % | Final refit on Batch1 46; Test39 |
| Gap (Train-Valid) | -1.864 | %p | Valid - Train; positive: possible overfitting |
| Gap (Valid-Test) | 53.267 | %p | Batch2 - Valid; positive: generalization degradation |
| Gap (Target-Test), Batch2 | 50.437 | %p | Batch2 - 9.1%; paper reference |
| Test (Batch 3) | 14.941 | % | Same final model; additional Test44 |
| Gap (Batch2-Batch3) | -44.596 | %p | Batch3 - Batch2; positive: Batch3 has higher error |
| Gap (Target-Test), Batch3 | 5.841 | %p | Batch3 - 9.1%; paper reference |

MAPE는 백분율(%), Gap은 퍼센트포인트(%p)다. Gap(Train-Valid)=Valid−Train, Gap(Valid-Test)=Test−Valid, Gap(Target-Test)=Test−9.1로 정의한다. 양수는 오차 악화를 의미한다.

Train은 고정 후보의 Group CV fold 평균이며 표준편차는 1.473%p다. 전체 후보 선택 절차의 nested Group CV 평균은 9.948%, 표준편차는 3.035%p다. pooled OOF 점수는 보조 지표로 구분한다.

과제 자료의 원논문 기준 9.1%와의 차이를 표시하되 데이터 구성과 검증 절차가 동일한 재현 실험으로 해석하지 않는다. 17%는 실행 전 가설이며 실제 점수를 대신할 수 없다.



## 보조 지표

| partition | n | mape_pct | mae | mean_signed_error | overprediction_rate_pct | baseline_mape_pct |
| --- | --- | --- | --- | --- | --- | --- |
| Train | 36 | 8.093 | 71.884 | 6.143 | 55.556 | 19.630 |
| Valid | 10 | 6.271 | 57.735 | 15.493 | 70.000 | 13.683 |
| Batch2 | 39 | 59.537 | 288.917 | 270.869 | 94.872 | 72.656 |
| Batch3 | 44 | 14.941 | 192.855 | -169.009 | 25.000 | 20.201 |



## 오류 분석

| cell_id | policy | actual | predicted | ape_pct | outside_features |
| --- | --- | --- | --- | --- | --- |
| b2c6 | 3.6C(9%)-5C | 393.000 | 904.800 | 130.229 | switch_soc;qd_slope_10_100 |
| b2c15 | 3.6C(9%)-5C | 396.000 | 797.912 | 101.493 | switch_soc;qd_slope_10_100 |
| b2c29 | 5.2C(58%)-4C | 452.000 | 877.043 | 94.036 | qd_initial_median;qd_slope_10_100 |
| b2c31 | 5.6C(26%)-4.5C | 425.000 | 821.916 | 93.392 | qd_initial_median;qd_slope_10_100 |
| b2c11 | 5.2C(50%)-4.25C | 449.000 | 850.145 | 89.342 | qd_initial_median;qd_slope_10_100 |
| b2c42 | 5.2C(50%)-4.25C | 474.000 | 878.446 | 85.326 | qd_initial_median;qd_slope_10_100 |
| b2c18 | 5.2C(50%)-4.25C | 449.000 | 823.836 | 83.482 | qd_slope_10_100 |
| b2c17 | 5.6C(26%)-4.5C | 471.000 | 838.748 | 78.078 | qd_slope_10_100 |
| b2c21 | 6C(60%)-3C | 408.000 | 723.818 | 77.406 | log10_dq_var;qd_initial_median;qd_slope_10_100 |
| b2c41 | 5.6C(26%)-4.5C | 442.000 | 781.614 | 76.836 | log10_dq_var;qd_initial_median;qd_slope_10_100 |

| partition | outside_input_range | n | mape_pct | mean_signed_error |
| --- | --- | --- | --- | --- |
| Batch2 | False | 6 | 29.037 | 65.668 |
| Batch2 | True | 33 | 65.083 | 308.178 |
| Batch3 | False | 36 | 15.551 | -184.422 |
| Batch3 | True | 8 | 12.199 | -99.651 |
| Train | False | 36 | 8.093 | 6.143 |
| Valid | False | 10 | 6.271 | 15.493 |

Batch 2의 단수명 셀을 과대 예측하는 편향을 확인한다. 초기 용량·기울기는 Batch 1에서 설명력이 있어도 배치가 달라지면 관계가 달라질 수 있다. 입력 범위 이탈과 선형 계수의 영향을 함께 살펴야 하며, 상관관계와 오차 동반만으로 원인을 확정하지 않는다.



## 입력 분포 진단

| batch | feature | missing_rate_pct | outside_batch1_pct |
| --- | --- | --- | --- |
| 2 | log10_dq_var | 0.00 | 28.21 |
| 3 | log10_dq_var | 0.00 | 2.27 |
| 2 | peak_c_rate | 0.00 | 0.00 |
| 3 | peak_c_rate | 0.00 | 0.00 |
| 2 | switch_soc | 0.00 | 5.13 |
| 3 | switch_soc | 0.00 | 0.00 |
| 2 | qd_initial_median | 0.00 | 51.28 |
| 3 | qd_initial_median | 0.00 | 13.64 |
| 2 | qd_slope_10_100 | 0.00 | 58.97 |
| 3 | qd_slope_10_100 | 0.00 | 6.82 |



## 모델 비교

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



## ESS 도메인 해석

초기 100사이클 관측으로 셀별 수명 위험을 비교하고 검사·교체 후보의 우선순위를 정하는 보조 도구로 사용할 수 있다. 과대 예측은 교체 지연 위험을 만들므로 단일 수명 값만으로 유지보수 시점을 확정하지 않는다.

실제 BESS 적용에는 사용 온도·SOC·부하·휴지시간·셀 화학계 차이를 반영한 운영 데이터, 장비 간 측정 교정, 예측 불확실성, 셀·모듈·팩 수준 검증이 필요하다.

Batch 2·3는 설계 EDA 및 분석에 이미 사용된 배치다. 본 점수는 재사용 배치 평가이며, 새로운 미사용 배치를 확보하여 일반화 성능을 검증해야 한다.



## 참고문헌

Severson et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. Nature Energy, 4, 383–391.

DS-MINI 설계서, 울산캠퍼스 4반 정예지. Mini Project 과제 자료.



## 팀 구성

정예지 · 울산캠퍼스 4반
