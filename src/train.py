"""Executable experiment: B1 development -> freeze -> holdout -> refit -> external tests."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocess import load_data,load_targets,FILES
from src.features import prepare_features,build_feature_sets
from src.modeling import (build_candidates,compare_candidates,select_model,build_pipeline,matrix,
                          run_cv,make_folds,nested_selection_cv,evaluate_model)
from src.reporting import analyze_distribution_shift,prediction_plots,linear_coefficients,create_performance_report
from src.verify_results import verify_pipeline,file_hash


def write_json(path,obj):
    path=Path(path);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def get_split(ids,path,seed):
    if path.exists():
        split=json.loads(path.read_text())
    else:
        train,valid=train_test_split(sorted(ids),test_size=10,random_state=seed)
        split={'train':sorted(train),'valid':sorted(valid),'seed':seed,'source':'new deterministic split; no pre-existing manifest found'}
        write_json(path,split)
    assert len(split['train'])==36 and len(split['valid'])==10
    assert set(split['train']).isdisjoint(split['valid'])
    assert set(split['train']+split['valid'])==set(ids)
    return split


def prediction_rows(ids,y,pred,partition,baseline):
    assert list(y.index)==list(ids)
    return pd.DataFrame({'cell_id':ids,'partition':partition,'actual':y.values,'predicted':pred,
                         'signed_error':pred-y.values,'absolute_error':abs(pred-y.values),
                         'ape_pct':abs((pred-y.values)/y.values)*100,'baseline_prediction':baseline})


def main(root,output,config,split_path):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise FileExistsError('Use a new output directory; frozen results are never overwritten')
    cfg=json.loads(Path(config).read_text());write_json(out/'config.json',cfg)
    warning_log=open(out/'warnings.log','w')
    def log_warning(message,category,filename,lineno,file=None,line=None):
        warning_log.write(f'{category.__name__}: {message} ({filename}:{lineno})\n');warning_log.flush()
    warnings.showwarning=log_warning;warnings.simplefilter('always')
    events=[]
    def event(kind,**info):
        events.append({'event':kind,'utc':datetime.now(timezone.utc).isoformat(),**info});write_json(out/'event_log.json',events)
    event('rules_frozen',config_sha256=file_hash(out/'config.json'))
    print('Loading first-100-cycle observations',flush=True)
    cells,excluded=load_data(root/'data');write_json(out/'exclusions.json',{'excluded':excluded,
        'out_of_cohort_files':['2018-04-03_varcharge_batchdata_updated_struct_errorcorrect.mat']})
    x,meta=prepare_features(cells,cfg);x.to_csv(out/'features.csv');meta.to_csv(out/'feature_audit.csv')
    event('features_prepared', core_missing_n=int(x.log10_dq_var.isna().sum()))
    split=get_split(meta.index[meta.batch==1].tolist(),Path(split_path),cfg['seed']);write_json(out/'split_manifest.json',split)
    sets=build_feature_sets();write_json(out/'feature_sets.json',sets)
    ids=split['train'];X=x.loc[ids]; y=load_targets(root/'data',ids);groups=meta.loc[ids,'protocol_group']
    assert X.index.equals(y.index)
    folds=make_folds(X,groups,cfg)
    write_json(out/'cv_folds.json',[{'fold':i,'train_ids':X.index[t].tolist(),'valid_ids':X.index[v].tolist()} for i,(t,v) in enumerate(folds)])
    candidates=list(build_candidates(sets,cfg));print(f'Comparing {len(candidates)} candidates using common Group CV',flush=True)
    comparison,trials,details=compare_candidates(candidates,X,y,groups,cfg)
    comparison.to_csv(out/'candidate_comparison.csv',index=False);trials.to_csv(out/'hyperparameter_trials.csv',index=False)
    details.to_csv(out/'cv_fold_results.csv',index=False)
    chosen=select_model(comparison,cfg);write_json(out/'selected_model.json',chosen)
    event('candidate_frozen',selection_sha256=file_hash(out/'selected_model.json'),candidate_id=chosen['candidate_id'])
    print('Selected',chosen['candidate_id'],chosen['params'],flush=True)
    nested,nested_pred=nested_selection_cv(candidates,X,y,groups,cfg)
    nested.to_csv(out/'nested_cv_results.csv',index=False)
    prediction_rows(ids,y,nested_pred,'NestedTrain',np.nan).drop(columns='baseline_prediction').to_csv(out/'nested_cv_predictions.csv',index=False)
    params=json.loads(chosen['params']);stats,_,train_pred=run_cv(chosen,params,X,y,folds,cfg)
    base={'features':[],'model':'Dummy','target':'raw'};_,_,base_pred=run_cv(base,{},X,y,folds,cfg)
    performances=[];predictions=[]
    def record(name,idx,target,pred,baseline):
        performances.append(dict(partition=name,n=len(target),**evaluate_model(target,pred),
                                 baseline_mape_pct=evaluate_model(target,baseline)['mape_pct']))
        predictions.append(prediction_rows(idx,target,pred,name,baseline))
    record('Train',ids,y,train_pred,base_pred)
    pipe=build_pipeline(chosen['model'],chosen['target'],params,cfg).fit(matrix(X,chosen['features']),y)
    valid_ids=split['valid'];yv=load_targets(root/'data',valid_ids)
    vp=pipe.predict(matrix(x.loc[valid_ids],chosen['features']));joblib.dump(pipe,out/'train_model.joblib')
    reloaded=joblib.load(out/'train_model.joblib');assert np.allclose(vp,reloaded.predict(matrix(x.loc[valid_ids],chosen['features'])),rtol=1e-12)
    record('Valid',valid_ids,yv,vp,np.repeat(y.median(),len(yv)));event('valid_labels_evaluated',candidate_unchanged=True)
    print('Valid:',evaluate_model(yv,vp),flush=True)
    b1=meta.index[meta.batch==1].tolist();y1=load_targets(root/'data',b1)
    final=build_pipeline(chosen['model'],chosen['target'],params,cfg).fit(matrix(x.loc[b1],chosen['features']),y1)
    joblib.dump(final,out/'final_model.joblib')
    source_hashes={str(p.relative_to(root)):file_hash(p) for p in sorted((root/'src').glob('*.py'))}
    frozen={'selected':chosen,'selection_sha256':file_hash(out/'selected_model.json'),
            'final_model_sha256':file_hash(out/'final_model.joblib'),'config_sha256':file_hash(out/'config.json'),
            'split_sha256':file_hash(out/'split_manifest.json'),'source_sha256':source_hashes,
            'data_sha256':{filename:file_hash(root/'data'/filename) for filename in FILES.values()},
            'versions':{m:importlib.metadata.version(m) for m in ['numpy','pandas','scikit-learn','xgboost','lightgbm','h5py']},
            'external_target_used_for_selection':False,'final_training_n':46}
    write_json(out/'experiment_frozen.json',frozen);event('final_model_frozen',model_sha256=frozen['final_model_sha256'])
    for batch in [2,3]:
        assert file_hash(out/'selected_model.json')==frozen['selection_sha256']
        assert file_hash(out/'final_model.joblib')==frozen['final_model_sha256']
        idx=meta.index[meta.batch==batch].tolist();yp=load_targets(root/'data',idx);event('external_labels_loaded',batch=batch)
        pred=final.predict(matrix(x.loc[idx],chosen['features']));record(f'Batch{batch}',idx,yp,pred,np.repeat(y1.median(),len(idx)))
        print(f'Batch{batch}:',evaluate_model(yp,pred),flush=True)
    performance=pd.DataFrame(performances);prediction=pd.concat(predictions,ignore_index=True)
    performance.to_csv(out/'performance.csv',index=False);prediction.to_csv(out/'predictions.csv',index=False)
    distribution=analyze_distribution_shift(x,meta,chosen['features'],out)
    prediction_plots(prediction,out);linear_coefficients(final,chosen,x,meta,out)
    create_performance_report(performance,comparison,nested,chosen,distribution,cfg,out)
    from src.reporting import build_performance_reporting
    (root/'results').mkdir(exist_ok=True)
    build_performance_reporting(performance,chosen,cfg).to_csv(root/'results/model_performance.csv',index=False)
    event('reporting_complete');result=verify_pipeline(out);warning_log.close();print('Verification:',result['passed'],flush=True)
    return out

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='results/experiment')
    parser.add_argument('--config',default='config/experiment.json');parser.add_argument('--split',default='config/split_manifest.json')
    args=parser.parse_args();main(Path.cwd(),args.output,args.config,args.split)
