"""Small candidate spaces and fold-local preprocessing, with nested selection audit."""
import json
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.model_selection import GroupKFold, KFold, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

ORDER = {'Dummy':0, 'Ridge':1, 'ElasticNet':2, 'Linear':3, 'RandomForest':4, 'XGBoost':5, 'LightGBM':6}
LINEAR = {'Linear', 'Ridge', 'ElasticNet'}


def inverse_log10(y):
    return np.power(10.0, y)


def evaluate_model(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    assert y.shape == pred.shape and np.isfinite(y).all() and (y > 0).all()
    assert np.isfinite(pred).all()
    mape = float(np.mean(np.abs((pred-y)/y))*100)
    assert np.isclose(mape, mean_absolute_percentage_error(y, pred)*100)
    return dict(mape_pct=mape, mae=float(np.mean(np.abs(pred-y))),
                mean_signed_error=float(np.mean(pred-y)), overprediction_rate_pct=float(np.mean(pred>y)*100))


def build_pipeline(model, target, params, cfg):
    factories = {'Dummy': lambda: DummyRegressor(strategy='median'), 'Linear': LinearRegression,
                 'Ridge': Ridge, 'ElasticNet': lambda: ElasticNet(max_iter=50000, tol=1e-6),
                 'RandomForest': lambda: RandomForestRegressor(random_state=cfg['seed'], n_jobs=1),
                 'XGBoost': lambda: XGBRegressor(random_state=cfg['seed'], n_jobs=1, objective='reg:squarederror', verbosity=0),
                 'LightGBM': lambda: LGBMRegressor(random_state=cfg['seed'], n_jobs=1, verbosity=-1,
                                                   deterministic=True, force_col_wise=True)}
    estimator = factories[model]().set_params(**params)
    steps = [('imputer', SimpleImputer(strategy='median', keep_empty_features=True))]
    if model in LINEAR:
        steps.append(('scaler', StandardScaler()))
    steps.append(('model', estimator)); pipe = Pipeline(steps)
    return TransformedTargetRegressor(regressor=pipe, func=np.log10, inverse_func=inverse_log10,
                                      check_inverse=True) if target=='log10' else pipe


def matrix(x, features):
    # M0 has no input information. A constant lets sklearn accept a zero-feature baseline.
    return x[features] if features else pd.DataFrame({'constant': np.zeros(len(x))}, index=x.index)


def build_candidates(sets, cfg):
    for name, features in sets.items():
        models = ['Dummy'] if name == 'M0' else list(cfg['grids'])
        targets = ['raw'] if name == 'M0' else ['raw','log10']
        # Median of log target back-transformed differs for even sample counts; M0 is raw median only.
        for target in targets:
            for model in models:
                yield dict(candidate_id=f'{name}_{target}_{model}', feature_set=name, features=features,
                           feature_count=len(features), model=model, target=target,
                           grid={} if model=='Dummy' else cfg['grids'][model])


def make_folds(x, groups, cfg, cell=False, count=None):
    n = count or (cfg['cell_cv_folds'] if cell else cfg['group_cv_folds'])
    if cell:
        folds = list(KFold(n_splits=min(n,len(x)), shuffle=True, random_state=cfg['seed']).split(x))
    else:
        n = min(n, len(set(groups)))
        if n < 2:
            raise ValueError('At least two protocol groups required')
        folds = list(GroupKFold(n_splits=n).split(x, groups=groups))
        for train, valid in folds:
            assert not set(groups.iloc[train]) & set(groups.iloc[valid])
    return folds


def run_cv(candidate, params, x, y, folds, cfg):
    X = matrix(x, candidate['features']); results = []; oof = np.full(len(y), np.nan)
    for fold, (train, valid) in enumerate(folds):
        pipe = build_pipeline(candidate['model'], candidate['target'], params, cfg)
        pipe.fit(X.iloc[train], y.iloc[train])
        pred = pipe.predict(X.iloc[valid]); oof[valid] = pred
        results.append(dict(fold=fold, train_n=len(train), valid_n=len(valid), **evaluate_model(y.iloc[valid],pred)))
    assert np.isfinite(oof).all()
    scores = pd.DataFrame(results)
    return dict(cv_mape_mean=float(scores.mape_pct.mean()), cv_mape_std=float(scores.mape_pct.std(ddof=1)),
                cv_mae_mean=float(scores.mae.mean()), cv_oof_mape=evaluate_model(y,oof)['mape_pct']), results, oof


def tune_model(candidate, x, y, folds, cfg):
    rows, details = [], []
    for params in ParameterGrid(candidate['grid']):
        stats, fold_results, _ = run_cv(candidate,params,x,y,folds,cfg)
        key = json.dumps(params,sort_keys=True)
        rows.append({**{k:v for k,v in candidate.items() if k!='grid'},'params':key, **stats})
        details.extend(dict(candidate_id=candidate['candidate_id'],params=key,**r) for r in fold_results)
    table = pd.DataFrame(rows).sort_values(['cv_mape_mean','cv_mape_std','params'],kind='stable')
    return table.iloc[0].to_dict(), rows, details


def select_model(table, cfg):
    eligible = table.loc[table.model.isin(cfg.get('selection_models', list(ORDER)))].copy()
    if eligible.empty:
        raise ValueError('No eligible final-model candidates')
    table = eligible
    floor = table.cv_mape_mean.min()
    pool = table.loc[table.cv_mape_mean <= floor + cfg['tie_tolerance_percentage_points']].copy()
    pool['model_complexity'] = pool.model.map(ORDER)
    return pool.sort_values(['feature_count','cv_mape_std','model_complexity','cv_mape_mean','candidate_id'],
                            kind='stable').iloc[0].drop('model_complexity').to_dict()


def compare_candidates(candidates,x,y,groups,cfg,cell_diagnostic=True):
    folds = make_folds(x,groups,cfg); best, trials, details = [], [], []
    for candidate in candidates:
        winner, rows, result = tune_model(candidate,x,y,folds,cfg)
        if cell_diagnostic:
            stats, _, _ = run_cv(candidate,json.loads(winner['params']),x,y,make_folds(x,groups,cfg,cell=True),cfg)
            winner['cell_cv_mape_mean'] = stats['cv_mape_mean']
            winner['cell_cv_mape_std'] = stats['cv_mape_std']
        best.append(winner);trials.extend(rows);details.extend(result)
    return pd.DataFrame(best), pd.DataFrame(trials), pd.DataFrame(details)


def nested_selection_cv(candidates,x,y,groups,cfg):
    """Outer groups see neither inner tuning nor feature/model selection."""
    folds = make_folds(x,groups,cfg); oof = np.full(len(y),np.nan); rows=[]
    for fold,(train,valid) in enumerate(folds):
        print(f'Nested Group CV outer fold {fold+1}/{len(folds)}',flush=True)
        table,_,_ = compare_candidates(candidates,x.iloc[train],y.iloc[train],groups.iloc[train],cfg,False)
        chosen = select_model(table,cfg)
        pipe = build_pipeline(chosen['model'],chosen['target'],json.loads(chosen['params']),cfg)
        X = matrix(x,chosen['features']);pipe.fit(X.iloc[train],y.iloc[train])
        pred=pipe.predict(X.iloc[valid]);oof[valid]=pred
        rows.append(dict(fold=fold,candidate_id=chosen['candidate_id'],params=chosen['params'],
                         **evaluate_model(y.iloc[valid],pred)))
    assert np.isfinite(oof).all()
    return pd.DataFrame(rows),oof
