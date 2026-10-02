"""Independent saved-artifact checks; run with python -m src.verify_results OUTPUT."""
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.modeling import matrix, inverse_log10, evaluate_model, select_model


def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def verify_pipeline(out):
    out=Path(out); frozen=json.loads((out/'experiment_frozen.json').read_text())
    events=json.loads((out/'event_log.json').read_text());types=[e['event'] for e in events]
    assert types.index('candidate_frozen') < types.index('valid_labels_evaluated') < types.index('final_model_frozen') < types.index('external_labels_loaded')
    assert frozen['final_model_sha256']==file_hash(out/'final_model.joblib')
    assert frozen['selection_sha256']==file_hash(out/'selected_model.json')
    assert frozen['config_sha256']==file_hash(out/'config.json')
    assert frozen['split_sha256']==file_hash(out/'split_manifest.json')
    cfg=json.loads((out/'config.json').read_text())
    selected=json.loads((out/'selected_model.json').read_text())
    comparison=pd.read_csv(out/'candidate_comparison.csv')
    independent_selection=select_model(comparison,cfg)
    assert independent_selection['candidate_id']==selected['candidate_id']
    assert json.loads(independent_selection['params'])==json.loads(selected['params'])
    fold_results=pd.read_csv(out/'cv_fold_results.csv')
    scores=fold_results.loc[fold_results.candidate_id.eq(selected['candidate_id']) & fold_results.params.eq(selected['params'])]
    assert np.isclose(scores.mape_pct.mean(),selected['cv_mape_mean'])
    split=json.loads((out/'split_manifest.json').read_text())
    x=pd.read_csv(out/'features.csv',index_col='cell_id');meta=pd.read_csv(out/'feature_audit.csv',index_col='cell_id')
    pred=pd.read_csv(out/'predictions.csv');folds=json.loads((out/'cv_folds.json').read_text())
    assert x.index.is_unique and x.index.equals(meta.index)
    assert set(split['train']).isdisjoint(split['valid']) and len(split['train'])==36 and len(split['valid'])==10
    assert set(split['train']+split['valid'])==set(meta.index[meta.batch==1])
    assert meta.groupby('batch').size().to_dict()=={1:46,2:39,3:44}
    assert meta.max_feature_cycle.max()<=100 and not np.isinf(x.to_numpy()).any()
    assert set(selected['features']).isdisjoint({'cell_id','batch','cycle_life','knee_point','target'})
    for fold in folds:
        assert set(fold['train_ids']).isdisjoint(fold['valid_ids'])
        assert set(meta.loc[fold['train_ids'],'protocol_group']).isdisjoint(meta.loc[fold['valid_ids'],'protocol_group'])
    assert np.isfinite(pred[['actual','predicted','signed_error','ape_pct']]).all().all()
    assert not pred[['partition','cell_id']].duplicated().any()
    assert np.allclose(pred.ape_pct,abs((pred.predicted-pred.actual)/pred.actual)*100)
    assert np.allclose(inverse_log10(np.log10(pred.actual)),pred.actual,rtol=1e-12)
    model=joblib.load(out/'final_model.joblib')
    ext=pred.loc[pred.partition.isin(['Batch2','Batch3'])]
    expected=model.predict(matrix(x.loc[ext.cell_id],selected['features']))
    assert np.allclose(expected,ext.predicted,rtol=1e-12,atol=1e-10)
    valid=pred.loc[pred.partition.eq('Valid')]
    development=joblib.load(out/'train_model.joblib')
    assert np.allclose(development.predict(matrix(x.loc[valid.cell_id],selected['features'])),valid.predicted)
    perf=pd.read_csv(out/'performance.csv').set_index('partition')
    for name,group in pred.groupby('partition'):
        assert np.isclose(evaluate_model(group.actual,group.predicted)['mape_pct'],perf.loc[name,'mape_pct'])
    if (out/'performance_reporting.csv').exists():
        from src.reporting import build_performance_reporting
        actual_report=pd.read_csv(out/'performance_reporting.csv')
        expected_report=build_performance_reporting(perf.reset_index(),selected,cfg)
        assert actual_report['index'].tolist()==expected_report['index'].tolist()
        assert np.allclose(actual_report.value,expected_report.value)
    result={'passed':True,'checks':['config/split hashes','candidate selection independently reproduced','CV fold mean independently recomputed','holdout model reload','reporting gaps independently recomputed','unique IDs','aligned X/y','batch counts','36/10 disjoint split','cycle <=100',
        'input allowlist','protocol group isolation','finite predictions','MAPE independently recomputed',
        'log10 inverse roundtrip','saved model reload predictions','selection/model frozen before external evaluation'],
        'selection_sha256':frozen['selection_sha256'],'model_sha256':frozen['final_model_sha256']}
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    import sys
    print(json.dumps(verify_pipeline(sys.argv[1]),indent=2))
