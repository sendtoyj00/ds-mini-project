# MIT–Stanford Battery Dataset

Severson et al. (2019)의 MATLAB v7.3(HDF5) 원본 파일을 이 폴더에 둡니다. 대용량 원본은 Git 추적에서 제외합니다.

| 파일 | 용도 | 분석 셀 수 |
| --- | --- | --- |
| `2017-05-12_batchdata_updated_struct_errorcorrect.mat` | Batch 1 학습·검증 | 46 |
| `2018-02-20_batchdata_updated_struct_errorcorrect.mat` | Batch 2 최종 평가 | 39 |
| `2018-04-12_batchdata_updated_struct_errorcorrect.mat` | Batch 3 추가 평가 | 44 |

세 파일의 원본 139셀 중 유효한 cycle_life가 없는 10셀을 제외하여 총 129셀을 분석합니다. 별도 `2018-04-03_varcharge_batchdata_updated_struct_errorcorrect.mat`는 설계서의 분석 cohort에 포함하지 않습니다.

모델 입력은 초기 Cycle 1–100 관측으로 제한합니다. 전체 열화 곡선과 knee 분석은 `src/eda.py`에서 설명용으로만 읽습니다. 초기화 기록과 IR≤0은 결측 처리합니다. ΔQ는 2.0–3.5V 공통 격자로 정렬하여 계산하며 범위 밖 외삽은 하지 않습니다.

입력 파일 SHA-256은 실험별 `experiment_frozen.json`에 기록합니다. 셀 ID는 각 원본 파일의 행 번호를 보존한 `b1c0`, `b2c0` 형태입니다.

참고문헌: Severson et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. Nature Energy, 4, 383–391.
