"""Build and execute the three reproducible project notebooks."""
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
SETUP='''from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
from IPython.display import display, Image, Markdown
ROOT = Path.cwd()
if not (ROOT / 'src').exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
'''

def build(name,cells):
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_code_cell(SETUP)]+[
        nbf.v4.new_markdown_cell(source) if kind=='md' else nbf.v4.new_code_cell(source)
        for kind,source in cells])
    nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
    nb.metadata.language_info={'name':'python','version':'3.11'}
    path=ROOT/'notebooks'/name
    nbf.write(nb,path)
    NotebookClient(nb,timeout=600,resources={'metadata':{'path':str(ROOT)}}).execute()
    nbf.write(nb,path)
    print(f'Executed {path.name}',flush=True)

build('01_EDA.ipynb',[
('md','# ESS 배터리 수명 예측 · EDA\n\nMIT–Stanford 데이터의 배치별 수명, 열화 곡선, ΔQ(V), 충전 정책을 분석한다. 수명과 전체 열화 곡선은 설명용이며 모델 입력은 초기 100사이클로 제한한다. Batch 2·3 분석은 외부 배치 진단이며 독립적인 미관측 Test를 의미하지 않는다.'),
('code',"from src.eda import run_eda\nOUT=ROOT/'results/eda'\ntable=run_eda(ROOT,OUT)\ndisplay(table.groupby('batch').size().rename('cells').to_frame())"),
('md','## Cycle Life 분포\n\n550사이클 미만을 단수명으로 정의하여 분류 가능성도 점검한다. Batch 1의 단수명 셀은 1개라 회귀를 선택한다.'),
('code',"display(pd.read_csv(OUT/'lifetime_summary.csv').round(3))\ndisplay(Image(filename=str(OUT/'lifetime_distribution.png')))"),
('md','Batch 2 중앙값은 472사이클이며 76.9%가 Batch 1 수명 범위 밖이다. 내부 검증만으로 외부 성능을 판단하기 어렵다.'),
('md','## 열화 곡선 분석\n\n상단은 전체 수명, 하단은 초기 100사이클의 상대 용량이다. 곡선 색은 수명을 나타낸다. 전체 수명 데이터는 EDA 함수에서만 읽는다. 비교를 위해 전체 곡선의 표시 범위는 상대 용량 0.70–1.10, 초기 곡선은 0.85–1.06으로 고정한다. 범위 밖 관측은 삭제하지 않고 capacity_plot_outliers.csv에 보존한다.'),
('code',"display(Image(filename=str(OUT/'capacity_degradation.png')))\nknee=pd.read_csv(OUT/'knee_diagnostics.csv')\ndisplay(knee.groupby('batch').agg(detected_n=('knee_cycle','count'),median_knee=('knee_cycle','median'),median_life_ratio=('knee_life_ratio','median')).round(3))"),
('md','초기 총용량 차이는 작고 후반에 용량 저하가 가속된다. knee는 이동 중앙값으로 평활화한 곡선의 연속 두 직선 회귀로 탐색한다. 단일 직선 대비 SSE가 50% 이상 감소하고 후기 기울기가 더 작을 때 후보로 표시한다. 탐색 구간·평활화에 의존하는 설명용 추정이며 물리적 knee의 확정값이나 모델 피처로 사용하지 않는다.'),
('md','## ΔQ(V) 곡선 분석\n\nCycle 100−Cycle 10을 2.0–3.5V 공통 300점 격자에서 계산한다. 각 배치 수명 하위·상위 1/3의 중앙값과 25–75% 범위를 비교한다.'),
('code',"display(Image(filename=str(OUT/'delta_q_curves.png')))\ncorr=pd.read_csv(OUT/'correlations.csv')\ndisplay(corr[corr.feature.eq('log10_dq_var')].round(3))\ndisplay(Image(filename=str(OUT/'feature_lifetime.png')))"),
('md','log10_dq_var의 수명 상관은 모든 배치에서 음수다. Batch 2의 높은 상관은 Batch 1에서 학습한 절편·기울기가 동일함을 보장하지 않는다. 원 수명과 로그 수명은 모델링에서 별도로 비교한다.'),
('md','## 충전 속도와 수명의 관계'),
('code',"policy=pd.read_csv(OUT/'policy_lifetime.csv')\ndisplay(policy.round(2))\ndisplay(corr[corr.feature.isin(['peak_c_rate','switch_soc','chargetime_median_100'])].round(3))"),
('md','최대 C-rate와 수명은 배치 내부에서도 음의 상관을 보인다. 같은 충전 정책에서도 수명 차이가 나타나므로 정책만으로 예측하지 않고 ΔQ와 결합한다.'),
('md','## 추가 확인 · 입력 분포와 결측'),
('code',"display(Image(filename=str(OUT/'feature_distributions.png')))\ndisplay(table.groupby('batch')[['log10_dq_var','qd_initial_median','qd_slope_10_100','ir_median_100']].agg(['median','count']).round(6))"),
('md','Batch 2 IR은 6셀에서 결측이다. 용량·온도·IR은 보조 피처로 단계적으로 평가한다. 전체 배치 통계로 대치하거나 표준화하지 않는다. 모델 선택은 Batch 1 Train CV로 수행한다.\n\n참고: Severson et al. (2019), Nature Energy 4, 383–391; DS-MINI 설계서; Mini Project 과제 자료.')])

build('02_feature_engineering.ipynb',[
('md','# ESS 배터리 수명 예측 · Feature Engineering\n\n초기 100사이클에서 셀 단위 피처를 생성하고 품질 기준 및 단계적 피처 세트를 확인한다.'),
('code',"from src.preprocess import load_data, load_targets\nfrom src.features import prepare_features, build_feature_sets\ncfg=json.loads((ROOT/'config/experiment.json').read_text())\ncells,excluded=load_data(ROOT/'data')\nx,meta=prepare_features(cells,cfg)\ndisplay(pd.DataFrame(excluded))\ndisplay(meta.groupby('batch').agg(cells=('policy','size'),max_cycle=('max_feature_cycle','max'),initialization_rows=('initialization_rows','sum'),capacity_jump_rows=('capacity_jump_count','sum')))"),
('md','## 전처리 원칙\n\n초기화 기록은 결측 처리하고 IR≤0은 무효 측정으로 처리한다. 2% 용량 급변 후보는 원본을 유지하며 급변을 제외한 대안 피처도 생성한다. 보간은 외삽 없이 수행한다. 결측 대치·선형 모델 표준화는 모델 Pipeline의 각 학습 fold에서 수행한다.'),
('code',"display(meta[['curve10_status','curve100_status','dq_status']].value_counts().to_frame('cells'))\ndisplay(x.groupby(meta.batch).count())\nassert meta.max_feature_cycle.le(100).all()\nassert meta.dq_status.eq('ok').all()\nassert 'cycle_life' not in x.columns"),
('md','## 피처 정의\n\n- Core: log10(Var[Q100(V)−Q10(V)]), 표본분산 ddof=1\n- 충전 조건: 최대 C-rate, 전환 SOC\n- 초기 용량: Cycle 2–6 중앙값, Cycle 10–100 Theil–Sen 기울기\n- 상태: 초기 100사이클 유효 온도·IR 중앙값\n\n곡선 기준: 길이 일치·3점 이상, 유한값 95% 이상, 중복 전압 없음, 전압 범위 확보, 공백 0.05V 이하, ΔQ 분산 양수.'),
('code',"sets=build_feature_sets()\ndisplay(pd.DataFrame([{'Feature Set':k,'count':len(v),'features':', '.join(v) or 'Train target median'} for k,v in sets.items()]))\ndisplay(x.head().round(6))"),
('md','## Batch 1 Train의 중복 정보\n\nTrain 36셀만 사용하여 후보 간 상관을 계산한다. 셀 ID·배치·수명·knee는 입력에서 제외한다.'),
('code',"split=json.loads((ROOT/'config/split_manifest.json').read_text())\ntrain=x.loc[split['train']]\npairs=[('log10_dq_var','dq_min'),('c_rate_1','peak_c_rate'),('peak_c_rate','c_rate_gap'),('qd_change','qd_ratio'),('t80_min','chargetime_median_100')]\ndisplay(pd.DataFrame([{'feature_1':a,'feature_2':b,'Pearson r':train[a].corr(train[b])} for a,b in pairs]).round(4))\ndisplay(train[sets['M4']].corr().round(3))"),
('md','ΔQ 최소값과 분산, 충전시간 계열, 용량 변화량은 중복 정보를 포함하므로 대표 변수를 사용한다. M0–M4 및 clean 대안은 동일 fold에서 비교하며 Test 정답으로 피처를 선택하지 않는다.')])

build('03_modeling.ipynb',[
('md','# ESS 배터리 수명 예측 · Modeling\n\nBatch 1 학습·내부 검증, Batch 2 최종 평가, Batch 3 추가 평가. 점수는 실제 예측값으로 계산한다.'),
('code',"from src.train import main\nOUT=ROOT/'results/experiment'\nif not (OUT/'verification.json').exists():\n    main(ROOT,OUT,ROOT/'config/experiment.json',ROOT/'config/split_manifest.json')\nfrom src.verify_results import verify_pipeline\ndisplay(verify_pipeline(OUT))\nchosen=json.loads((OUT/'selected_model.json').read_text())\ndisplay(chosen)"),
('md','## 모델 선택 및 근거\n\n최종 후보는 외삽 가능한 Linear·Ridge·ElasticNet이다. 얕은 Random Forest·제한된 XGBoost·LightGBM은 비교군이다. 후보별 원 수명·로그 수명, M0–M4 및 clean 대안을 공통 충전 정책 Group CV로 비교한다. 최저 MAPE+0.5%p 안에서 피처 수·fold 편차·복잡도 순으로 선택한다. Valid는 한 번 확인하고 최종적으로 Batch 1 전체를 재학습한다.'),
('code',"comparison=pd.read_csv(OUT/'candidate_comparison.csv')\ndisplay(comparison.sort_values('cv_mape_mean').drop(columns=['features']).round(4))\ndisplay(pd.read_csv(OUT/'feature_addition_effects.csv').round(3))\ndisplay(pd.read_csv(OUT/'nested_cv_results.csv').round(3))"),
('md','## 성능 결과 · 과제 Format\n\nTrain은 Group CV fold 평균, Valid는 Hold-out 10셀, Test는 Batch 2 39셀이다. Gap은 뒤 단계 오차−앞 단계 오차이며 원논문 기준은 과제 자료의 9.1%다. MAPE는 %, Gap은 %p로 표시한다.'),
('code',"display(pd.read_csv(OUT/'performance_reporting.csv').round(3))\ndisplay(pd.read_csv(OUT/'performance.csv').round(3))\ndisplay(Image(filename=str(OUT/'prediction_diagnostics.png')))"),
('md','## 오류 분석\n\n과대 예측과 입력 범위 이탈을 셀별로 확인한다. MAPE가 큰 셀도 평가에서 제외하지 않는다.'),
('code',"errors=pd.read_csv(OUT/'error_analysis_top_cells.csv')\ndisplay(errors[['cell_id','policy','actual','predicted','ape_pct','outside_features']].round(3))\ndisplay(pd.read_csv(OUT/'error_by_input_range.csv').round(3))\ndisplay(pd.read_csv(OUT/'feature_distribution.csv').query('selected == True').round(3))\nif (OUT/'linear_coefficients.csv').exists():\n    display(pd.read_csv(OUT/'linear_coefficients.csv').round(6))"),
('md','내부 CV의 좋은 성능이 Batch 2로 이어지지 않는다. 초기 용량·기울기의 배치별 관계 변화와 입력 범위 이탈은 오차 원인 가설이다. 높은 ΔQ 상관만으로 15–20% MAPE를 보장할 수 없다. 이 배치들은 설계 EDA와 분석에 이미 사용되었으므로 새로운 미사용 배치의 검증이 필요하다.'),
('md','## ESS 도메인 해석\n\n초기 100사이클 관측으로 검사·교체 후보를 우선순위화하는 보조 도구로 사용할 수 있다. 수명을 과대 예측하면 교체가 지연될 수 있으므로 실제 운영에는 온도·SOC·부하·휴지시간·화학계 차이, 측정 교정, 불확실성 및 팩 단위 검증이 필요하다.\n\n참고문헌: Severson et al. (2019), Nature Energy 4, 383–391.\n\n팀: 울산캠퍼스 4반 정예지.')])
