# 최종 설계 대조 및 구현 검토

설계서: `DS-MINI-Design-울산캠퍼스_4반-정예지.pdf`, 특히 p5/6/13/16/18.

| 요구 | 구현 및 확인 |
| --- | --- |
| cycle_life 회귀, 1 cell = 1 sample | 129 unique cell ID; Target은 features와 분리 |
| Cycle1–100 한정 | HDF5 summary를 디스크에서 100행까지만 slice; curve10/100만 dereference; max_feature_cycle 검증 |
| target/cell/batch/knee 입력 제외 | Feature allowlist; ID는 index, batch는 audit에만 존재 |
| 분석 cohort | B1 46 / B2 39 / B3 44; Target 결측 10셀 제외; 별도 VarCharge2 파일은 cohort 밖 |
| ΔQ | Vdlin/Qdlin의 2.0–3.5 V 300점 선형 보간; 95% 유효, 길이, 중복, 범위, gap0.05, variance>0 확인 |
| 초기 용량 | Cycle2–6 유효값 median |
| 기울기 | Cycle10–100 유효값 Theil–Sen slope |
| 온도·IR | Cycle1–100 유효 median; IR≤0 결측; 초기화 모두0 행 결측 |
| 급변 | 2% flag 원본 유지; clean M3/M4 대안 비교; 10행 확인 |
| 정책 | C1/SOC/C2 parsing; max C-rate; 동일 numeric protocol group, newstructure 보수적으로 통합 |
| M0–M4 | 지정 Feature 그대로; clean 대안 및 raw/log10 Target, 모든 6개 알고리즘 비교 |
| M0 | raw Train cycle_life median 하나; 같은 baseline을 model/target별로 불필요하게 중복하지 않음 |
| split | 기존 manifest 없음 확인; seed42 Train36/Valid10 생성 및 고정; 이후 임의 분할 없음 |
| 학습 Pipeline | Median Imputer; 선형만 StandardScaler; 각 CV fold 안에서 fit |
| 로그 Target | TransformedTargetRegressor와 정확한 10**x inverse; raw cycle 단위 지표 |
| 탐색 | compact grid; no early stopping using Valid/Test, depth/leaf/regularization 제한 |
| CV | 모든 후보 동일 GroupKFold; fold group 교집합0; 보조 cell CV; 전체 선택 nested Group CV |
| 후보 선택 | 평가 전에 최저 평균+0.5%p → Feature 수 → fold std → 모델복잡도 순 규칙 고정 |
| Valid | 후보 확정 후 1회, 재선택 없음 |
| Final | 동일 선택·Hyperparameter로 Batch1 46 refit; model hash 동결 |
| External | model/selection hash 동결 후 Target 값 읽기; 동일 final model로 B2/B3 평가 |
| 지표 | MAPE%, MAE, prediction−actual mean signed error, 과대예측%; median baseline; Gap %p |
| 진단 | 전체 Feature 분포, selected 범위 이탈, actual/predicted 및 signed-error plots, feature-addition matched CV, nested 선택 변동성 |
| Linear 계수 | 최종 모델이 Linear/Ridge/ElasticNet일 때 coefficient/contribution 저장하는 코드 구현; 이번 final은 RF이므로 해당 없음 |
| 자동 검증 | ID, X/y, cohort, split, 관측범위, allowlist, group fold 분리, finite, MAPE, log inverse, reload, 평가순서 검증 |
| 산출물 | 후보/Feature 비교, trial/fold/nested CV, 선택/동결, performance/gap, cell predictions, Feature/audit/range, plot, models, 실행 Notebook |

## 설계에 없는 제약 검토

설계서에 정의하지 않은 초기 용량 최소3점/기울기 최소30점의 기존 설정은 적용하지 않았습니다. 중앙값 최소1점, slope 최소2점이라는 수학적 계산 가능 조건만 적용합니다. 물리적 유효조건 외에 추가 outlier 삭제, Target clipping, 사후 셀 제외, 단수명 threshold loss, 부호 강제, 단조 강제, feature 추가/제외를 도입하지 않았습니다.

11-cycle centered 이동 median과 sample variance(ddof=1)는 설계서의 구체 수치가 없으므로 기존 configuration 값을 유지하고 명시했습니다. centered window도 100사이클로 먼저 제한한 데이터 안에서만 계산합니다. 모든 결측 column은 sklearn keep_empty_features=True의 fold-local 0 fallback으로 처리합니다. 방전 곡선 QC 실패는 NaN과 사유를 남기며 강제로 외삽하거나 중복을 평균내지 않습니다.

보존 후보 중 qd_change/qd_ratio는 설계서에서 이름별 상세 수식을 지정하지 않아 Cycle96–100 median 대비 Cycle2–6 median으로 정의했습니다. 이 보존용 후보들은 모델에 입력하지 않습니다. t80는 SOC percentage/100으로 계산하며, switch_soc Feature 자체는 설계서 표시와 동일한 % 단위입니다.

## 개선 수단 검토

기존 Ridge alpha3개를5개, ElasticNet을 raw/log Target 단위를 고려한 작은 log-spaced alpha4개×l1 ratio2개로 확장했습니다. RF depth/leaf/max_features, boosting depth/leaves/iterations/child/regularization을 제한했습니다. n_estimators와 learning_rate는 동시 대규모 탐색 대신 작고 명확한 조합을 사용합니다. 선형·tree의 preprocessing을 분리하고, target 역변환을 sklearn estimator 안에서 처리합니다. Group CV를 주 선택 기준으로 승격하고 nested CV로 선택 편향과 불안정을 보여줍니다.

M1 RF는 Feature1개, depth2, min leaf3, 100trees, log10 Target입니다. 선택 CV mean8.100%, std3.215%p; 비교 Ridge M3 raw8.134%, std1.473%p. 적은 Feature를 우선하는 사전 고정 규칙에 따라 RF를 선택했으며, 외부 성능을 이유로 Ridge로 바꾸지 않았습니다. Nested fold에서 선택 모델이 달라지는 현상과 nested mean9.948%, std3.035%p를 명시하여 모델 선정의 불안정성을 감추지 않습니다.

## 실제 일반화 저하 해석

Batch2: Core 범위 이탈28.21%; 과대예측37/39(94.87%); mean signed error+147.24cycles. 실제 수명 최소392cycles인데 최종 모델 예측 최저603.92cycles입니다. RF는 학습 Target 범위 밖으로 외삽할 수 없고, 이번 예측은 603.92–1143.94cycles에 제한됩니다. 이는 단수명 셀에서 큰 상대오차가 발생하는 구체적인 패턴입니다.

Batch3: Core 범위 이탈2.27%여도 실제 수명 최대1935cycles를 예측1143.94cycles 이하로 압축합니다. 입력 범위 이탈 비율만으로 Target shift나 입력−수명 관계의 변화를 모두 설명할 수 없습니다. mean signed error−68.36cycles이며 특히 장수명 꼬리를 과소예측합니다. 관찰 기반 해석이며 배치 이동의 인과 원인을 입증하는 것은 아닙니다.

Valid MAPE13.792%는 median baseline13.683%보다0.110%p 나쁩니다. CV→Valid Gap5.620%p와 nested 결과는 소표본/정책 구성 차이에 따른 불안정성을 보여줍니다. Batch1 내부 일반화 향상이 Holdout에서 입증되었다고 주장하지 않습니다. 프로젝트9.1% 목표와 Batch2 차이는22.209%p입니다.

Batch2/3의 기존 EDA가 설계서에 있으므로 완전히 미관측 test라고 주장하지 않습니다. 이번 학습·선택에는 외부 Target 값이 들어가지 않으며, Target 존재 여부만 cohort 정의에 사용했습니다. 평가 이후 model/grid/Feature/선택 규칙을 변경하지 않았습니다.

## 검증 실행

- `python -m unittest discover -s tests -v`: 8 tests passed.
- `python -m src.run_experiment`: 전체 실행 성공.
- `python -m src.verify_results outputs/modeling_20261002_rebuilt`: 저장 결과 검증 통과.
- 입력/소스/config/split SHA256를 final frozen 기록과 대조해 일치 확인.
- Notebook13 code cells 실제 실행, 오류 없는 output 저장; Jupyter 로컬 커널 포트는 실행 환경 승인을 받아 사용.
- Prediction 진단 plot을 직접 확인; 잘림/겹침 없음.

첫 실행은 보고서의 선택적 tabulate 의존성 때문에 보고서 저장 단계에서 실패했습니다. 의존성을 제거한 뒤 동일 split/config/grid/선택 규칙으로 전체 재실행했고 모든 점수와 선택이 동일했습니다. `outputs/modeling_20261002_attempt1`은 실패 실행 기록이며, 공식 결과는 `outputs/modeling_20261002_rebuilt`입니다. 기존 사용자 파일 삭제와 requirements 변경은 작업 시작 전 상태로 유지했습니다.
