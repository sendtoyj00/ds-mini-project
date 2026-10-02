"""Post-freeze reports and diagnostics; no selection or fitting here."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'ds-mini-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from src.modeling import LINEAR, matrix


def analyze_distribution_shift(x,meta,features,out):
    rows=[]; reference=x.loc[meta.batch==1]
    for feature in x.columns:
        ref=reference[feature].dropna(); lo,hi=ref.min(),ref.max()
        for batch in [1,2,3]:
            vals=x.loc[meta.batch==batch,feature];valid=vals.dropna()
            below=valid<lo;above=valid>hi
            rows.append(dict(batch=batch,feature=feature,selected=feature in features,n=len(vals),valid_n=len(valid),
                             missing_rate_pct=vals.isna().mean()*100,minimum=valid.min(),maximum=valid.max(),
                             median=valid.median(),q25=valid.quantile(.25),q75=valid.quantile(.75),
                             batch1_min=lo,batch1_max=hi,
                             below_batch1_pct=below.mean()*100 if len(valid) else np.nan,
                             above_batch1_pct=above.mean()*100 if len(valid) else np.nan,
                             outside_batch1_pct=(below|above).mean()*100 if len(valid) else np.nan,
                             wasserstein_distance=wasserstein_distance(ref,valid) if len(ref) and len(valid) else np.nan))
    table=pd.DataFrame(rows);table.to_csv(out/'feature_distribution.csv',index=False)
    flags=[]
    for cid,row in x[features].iterrows():
        outside=[f for f in features if pd.notna(row[f]) and (row[f]<reference[f].min() or row[f]>reference[f].max())]
        flags.append(dict(cell_id=cid,batch=int(meta.loc[cid,'batch']),outside_features=';'.join(outside),
                          any_outside=bool(outside),outside_feature_count=len(outside),missing_feature_count=int(row.isna().sum())))
    pd.DataFrame(flags).to_csv(out/'input_range_flags.csv',index=False)
    fs=[f for f in x.columns if f in features] or ['log10_dq_var']
    fig,axes=plt.subplots(len(fs),1,figsize=(9,3*len(fs)),squeeze=False)
    for ax,f in zip(axes[:,0],fs):
        vals=[x.loc[meta.batch==b,f].dropna() for b in [1,2,3]]
        ax.boxplot(vals,tick_labels=['Batch1','Batch2','Batch3']);ax.set_ylabel(f)
    fig.tight_layout();fig.savefig(out/'feature_distributions.png',dpi=160);plt.close(fig)
    # All feature groups are also shown, including those not finally selected.
    fig,axes=plt.subplots(5,4,figsize=(16,16));
    for ax,f in zip(axes.ravel(),x.columns):
        ax.boxplot([x.loc[meta.batch==b,f].dropna() for b in [1,2,3]],tick_labels=['B1','B2','B3']);ax.set_title(f,fontsize=9)
    for ax in axes.ravel()[len(x.columns):]:ax.set_visible(False)
    fig.tight_layout();fig.savefig(out/'all_feature_distributions.png',dpi=130);plt.close(fig)
    return table


def prediction_plots(predictions,out):
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for col,part in enumerate(['Valid','Batch2','Batch3']):
        s=predictions.loc[predictions.partition==part]
        ax=axes[0,col];ax.scatter(s.actual,s.predicted);bounds=[min(s.actual.min(),s.predicted.min()),max(s.actual.max(),s.predicted.max())]
        ax.plot(bounds,bounds,'k--');ax.set(title=part,xlabel='Actual cycle life',ylabel='Predicted cycle life')
        axes[1,col].hist(s.signed_error,bins=12);axes[1,col].axvline(0,color='black',linestyle='--')
        axes[1,col].set(xlabel='Prediction - actual (cycles)',ylabel='Cells')
    fig.tight_layout();fig.savefig(out/'prediction_diagnostics.png',dpi=160);plt.close(fig)


def linear_coefficients(pipe,chosen,x,meta,out):
    if chosen['model'] not in LINEAR:return
    fitted=pipe.regressor_ if chosen['target']=='log10' else pipe
    coef=fitted['model'].coef_;scaler=fitted['scaler']
    raw_coef=coef/scaler.scale_;intercept=float(fitted['model'].intercept_-np.dot(raw_coef,scaler.mean_))
    pd.DataFrame({'feature':chosen['features'],'standardized_coefficient':coef,'input_unit_coefficient':raw_coef,
                  'direction':np.where(coef>0,'increasing','decreasing')}).to_csv(out/'linear_coefficients.csv',index=False)
    imputed=fitted['imputer'].transform(matrix(x,chosen['features']))
    contributions=pd.DataFrame(imputed*raw_coef,columns=chosen['features'],index=x.index)
    contributions['intercept']=intercept;contributions['batch']=meta.batch
    contributions.to_csv(out/'linear_contributions.csv')
    (out/'linear_interpretation.json').write_text(json.dumps({'target_scale':chosen['target'],'intercept_input_units':intercept,
        'interpretation':'Additive contributions are on log10(cycle_life) scale for log target; they are not causal effects.'},indent=2))


def create_performance_report(performance,comparison,nested,chosen,distribution,cfg,out):
    """Write the assignment format and post-evaluation error analysis."""
    out=Path(out)
    standard=build_performance_reporting(performance,chosen,cfg)
    standard.to_csv(out/'performance_reporting.csv',index=False)
    standard.loc[standard['unit'].eq('%p')].to_csv(out/'performance_gaps.csv',index=False)
    base=comparison.loc[comparison.feature_set.isin(['M1','M2','M3','M4'])].copy()
    base['previous_set']=base.feature_set.map({'M2':'M1','M3':'M2','M4':'M3'})
    prev=base[['feature_set','model','target','cv_mape_mean']].rename(columns={'feature_set':'previous_set','cv_mape_mean':'previous_cv_mape'})
    effects=base.merge(prev,on=['previous_set','model','target'],how='left')
    effects['improvement_pp']=effects.previous_cv_mape-effects.cv_mape_mean
    effects.to_csv(out/'feature_addition_effects.csv',index=False)
    comparison.sort_values('cv_mape_mean').groupby('feature_set',sort=False).head(1).to_csv(out/'feature_set_comparison.csv',index=False)
    predictions=pd.read_csv(out/'predictions.csv')
    meta=pd.read_csv(out/'feature_audit.csv')
    features=pd.read_csv(out/'features.csv')
    flags=pd.read_csv(out/'input_range_flags.csv')
    errors=predictions.merge(meta,on='cell_id').merge(features,on='cell_id').merge(
        flags[['cell_id','any_outside','outside_features']],on='cell_id')
    top=errors.loc[errors.partition.eq('Batch2')].sort_values('ape_pct',ascending=False).head(10)
    top.to_csv(out/'error_analysis_top_cells.csv',index=False)
    strata=[]
    for part,g in errors.groupby('partition'):
        for flag,h in g.groupby('any_outside'):
            strata.append(dict(partition=part,outside_input_range=flag,n=len(h),mape_pct=h.ape_pct.mean(),
                               mean_signed_error=h.signed_error.mean()))
    pd.DataFrame(strata).to_csv(out/'error_by_input_range.csv',index=False)
    lines=['# ESS 배터리 수명 예측', '', '## 모델링',
           f"최종 모델: **{chosen['feature_set']} + {chosen['model']}**, Target: {chosen['target']}, 파라미터: `{chosen['params']}`.",
           'Linear·Ridge·ElasticNet을 최종 후보로 사용하고 얕은 Random Forest·XGBoost·LightGBM은 비선형 비교군으로 평가한다. 학습 수명 범위를 벗어나는 예측이 필요하므로 외삽 가능한 모델을 사용한다.',
           'M0 중앙값 기준, M1 ΔQ, M2 충전 조건, M3 초기 용량, M4 온도·IR, 급변 후보를 제외한 대안 용량 피처를 비교한다. 원 Target과 log10 Target의 점수는 원 사이클 단위로 계산한다.',
           f"선택 기준: {cfg['selection']}",
           'Batch 1 Train 36셀에서 충전 정책 Group CV로 후보를 선택하고, Valid 10셀을 한 번 평가한다. 최종 모델은 Batch 1 전체 46셀로 학습한다. 대치·표준화는 각 학습 fold 내부에서 학습하며 Test 셀을 사후 제외하지 않는다.',
           '', '## 성능 결과',markdown_table(standard,3),
           'MAPE는 백분율(%), Gap은 퍼센트포인트(%p)다. Gap(Train-Valid)=Valid−Train, Gap(Valid-Test)=Test−Valid, Gap(Target-Test)=Test−9.1로 정의한다. 양수는 오차 악화를 의미한다.',
           f"Train은 고정 후보의 Group CV fold 평균이며 표준편차는 {chosen['cv_mape_std']:.3f}%p다. 전체 후보 선택 절차의 nested Group CV 평균은 {nested.mape_pct.mean():.3f}%, 표준편차는 {nested.mape_pct.std():.3f}%p다. pooled OOF 점수는 보조 지표로 구분한다.",
           '과제 자료의 원논문 기준 9.1%와의 차이를 표시하되 데이터 구성과 검증 절차가 동일한 재현 실험으로 해석하지 않는다. 17%는 실행 전 가설이며 실제 점수를 대신할 수 없다.',
           '', '## 보조 지표',markdown_table(performance,3),
           '', '## 오류 분석',markdown_table(top[['cell_id','policy','actual','predicted','ape_pct','outside_features']],3),
           markdown_table(pd.DataFrame(strata),3),
           'Batch 2의 단수명 셀을 과대 예측하는 편향을 확인한다. 초기 용량·기울기는 Batch 1에서 설명력이 있어도 배치가 달라지면 관계가 달라질 수 있다. 입력 범위 이탈과 선형 계수의 영향을 함께 살펴야 하며, 상관관계와 오차 동반만으로 원인을 확정하지 않는다.',
           '', '## 입력 분포 진단',
           markdown_table(distribution.loc[distribution.selected & distribution.batch.isin([2,3]),['batch','feature','missing_rate_pct','outside_batch1_pct']],2),
           '', '## 모델 비교',markdown_table(comparison.sort_values('cv_mape_mean').drop(columns=['features']),3),
           '', '## ESS 도메인 해석',
           '초기 100사이클 관측으로 셀별 수명 위험을 비교하고 검사·교체 후보의 우선순위를 정하는 보조 도구로 사용할 수 있다. 과대 예측은 교체 지연 위험을 만들므로 단일 수명 값만으로 유지보수 시점을 확정하지 않는다.',
           '실제 BESS 적용에는 사용 온도·SOC·부하·휴지시간·셀 화학계 차이를 반영한 운영 데이터, 장비 간 측정 교정, 예측 불확실성, 셀·모듈·팩 수준 검증이 필요하다.',
           'Batch 2·3는 설계 EDA 및 분석에 이미 사용된 배치다. 본 점수는 재사용 배치 평가이며, 새로운 미사용 배치를 확보하여 일반화 성능을 검증해야 한다.',
           '', '## 참고문헌',
           'Severson et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. Nature Energy, 4, 383–391.',
           'DS-MINI 설계서, 울산캠퍼스 4반 정예지. Mini Project 과제 자료.',
           '', '## 팀 구성', '정예지 · 울산캠퍼스 4반']
    (out/'performance_report.md').write_text('\n\n'.join(lines)+'\n')


def markdown_table(frame, digits=3):
    """No optional tabulate dependency needed for reproducible reports."""
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return f'{v:.{digits}f}' if np.isfinite(v) else 'NA'
        return str(v).replace('|', r'\|').replace('\n',' ')
    lines=['| ' + ' | '.join(map(str,frame.columns)) + ' |',
           '| ' + ' | '.join(['---']*len(frame.columns)) + ' |']
    lines.extend('| ' + ' | '.join(fmt(v) for v in row) + ' |' for row in frame.itertuples(index=False,name=None))
    return '\n'.join(lines)


def build_performance_reporting(performance, chosen, cfg):
    """Assignment reporting: macro fold mean for Train, explicit % vs percentage points.

    Gap labels follow the assignment; formulas define the sign so positive means
    the later evaluation has a larger error. Original pooled metrics stay intact.
    """
    p = performance.set_index('partition')
    train = float(chosen['cv_mape_mean'])
    valid = float(p.loc['Valid', 'mape_pct'])
    test = float(p.loc['Batch2', 'mape_pct'])
    extra = float(p.loc['Batch3', 'mape_pct'])
    goal = float(cfg['project_goal_mape_pct'])
    rows = [
        ('Train (Batch 1 CV)', train, '%', 'Group CV fold mean; Train36'),
        ('Valid (Batch 1 Hold-out)', valid, '%', 'Hold-out10; one evaluation'),
        ('Test (Batch 2)', test, '%', 'Final refit on Batch1 46; Test39'),
        ('Gap (Train-Valid)', valid-train, '%p', 'Valid - Train; positive: possible overfitting'),
        ('Gap (Valid-Test)', test-valid, '%p', 'Batch2 - Valid; positive: generalization degradation'),
        ('Gap (Target-Test), Batch2', test-goal, '%p', f'Batch2 - {goal:g}%; paper reference'),
        ('Test (Batch 3)', extra, '%', 'Same final model; additional Test44'),
        ('Gap (Batch2-Batch3)', extra-test, '%p', 'Batch3 - Batch2; positive: Batch3 has higher error'),
        ('Gap (Target-Test), Batch3', extra-goal, '%p', f'Batch3 - {goal:g}%; paper reference'),
    ]
    result = pd.DataFrame(rows, columns=['index', 'value', 'unit', 'note'])
    assert np.isfinite(result.value).all()
    return result
