"""Edge cases, fold isolation and label-free feature contracts."""
import copy
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from src.preprocess import Cell,load_data
from src.features import align_curve,parse_policy,cell_features,prepare_features,build_feature_sets
from src.modeling import build_pipeline,make_folds,inverse_log10,evaluate_model,select_model

CFG=json.loads(Path('config/experiment.json').read_text())

class FeatureTests(unittest.TestCase):
    def test_curve_qc(self):
        v=np.linspace(2,3.5,1000);q=v**2
        aligned,status=align_curve((v[::-1],q[::-1]),CFG)
        self.assertEqual(status,'ok');self.assertEqual(len(aligned),300)
        for pair,reason in [((v,q[:-1]),'length'),((np.r_[v[:10],v[9:]],np.r_[q[:10],q[9:]]),'duplicate_voltage'),
                            ((v[1:],q[1:]),'range'),((np.array([2,2.4,3.5]),np.ones(3)),'voltage_gap')]:
            self.assertEqual(align_curve(pair,CFG)[1],reason)
        q[:60]=np.nan;self.assertEqual(align_curve((v,q),CFG)[1],'nonfinite')
        q[:60]=v[:60]**2;q[100:120]=np.nan;self.assertEqual(align_curve((v,q),CFG)[1],'ok')

    def test_policy(self):
        self.assertEqual(parse_policy('6C(40%)-3C')[0],(6.,40.,3.))
        self.assertEqual(parse_policy('4.8C(80%)-4.8C')[1],parse_policy('4.8C(80%)-4.8C-newstructure')[1])
        self.assertTrue(np.isnan(parse_policy('SLOWCYCLE')[0][0]))

    def test_feature_formulas(self):
        cycle=np.arange(1,101);qd=1+cycle*.001
        s=pd.DataFrame({'cycle':cycle,'QDischarge':qd,'IR':np.full(100,.02),
                        'Tavg':np.full(100,30.),'chargetime':np.full(100,10.)})
        v=np.linspace(2,3.5,1000);q=v**2
        cell=Cell('synthetic',1,'6C(40%)-3C',s,{10:(v,q),100:(v,q+v*.01)})
        f,a=cell_features(cell,CFG)
        self.assertAlmostEqual(f['qd_initial_median'],1.004)
        self.assertAlmostEqual(f['qd_slope_10_100'],.001)
        self.assertAlmostEqual(f['log10_dq_var'],np.log10(np.var(np.linspace(2,3.5,300)*.01,ddof=1)),places=9)
        self.assertEqual(f['tavg_median_100'],30);self.assertEqual(f['ir_median_100'],.02)
        cell.summary['IR']=0;self.assertTrue(np.isnan(cell_features(cell,CFG)[0]['ir_median_100']))
        cell.summary.loc[99,'cycle']=101
        with self.assertRaises(AssertionError):cell_features(cell,CFG)

    def test_real_cohort(self):
        cells,excluded=load_data('data');x,m=prepare_features(cells,CFG)
        self.assertEqual(len(excluded),10);self.assertEqual(len(cells),129)
        self.assertTrue(m.dq_status.eq('ok').all())
        self.assertEqual(x.loc[m.batch==2,'ir_median_100'].isna().sum(),6)
        self.assertEqual(m.capacity_jump_count.sum(),10)
        self.assertEqual(m.loc[m.batch==1,'initialization_rows'].sum(),46)
        self.assertNotIn('cycle_life',x.columns)
        self.assertEqual(build_feature_sets()['M4'],['log10_dq_var','peak_c_rate','switch_soc',
                            'qd_initial_median','qd_slope_10_100','tavg_median_100','ir_median_100'])

class PipelineTests(unittest.TestCase):
    def test_fold_local_imputation_and_scaling(self):
        x=pd.DataFrame({'a':[1.,3.,np.nan],'b':[np.nan]*3});y=np.array([100.,200.,300.])
        pipe=build_pipeline('Ridge','raw',{'alpha':1.},CFG).fit(x,y)
        self.assertTrue(np.allclose(pipe['imputer'].statistics_,[2.,0.]))
        self.assertTrue(np.allclose(pipe['scaler'].mean_,[2.,0.]))
        pipe.predict(pd.DataFrame({'a':[1e9,np.nan],'b':[1e9,np.nan]}))
        self.assertTrue(np.allclose(pipe['imputer'].statistics_,[2.,0.]))

    def test_log_prediction_and_reload(self):
        import joblib,tempfile
        x=pd.DataFrame({'a':np.arange(10.)});y=10**(2+x.a*.05)
        pipe=build_pipeline('Linear','log10',{},CFG).fit(x,y)
        np.testing.assert_allclose(pipe.predict(x),y,rtol=1e-10)
        np.testing.assert_allclose(inverse_log10(np.log10(y)),y)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'model.joblib';joblib.dump(pipe,p)
            np.testing.assert_array_equal(joblib.load(p).predict(x),pipe.predict(x))

    def test_groups_and_selection(self):
        x=pd.DataFrame({'x':np.arange(12.)});g=pd.Series(['a']*3+['b']*3+['c']*3+['d']*3)
        for t,v in make_folds(x,g,CFG):self.assertFalse(set(g.iloc[t])&set(g.iloc[v]))
        table=pd.DataFrame([dict(candidate_id='complex',features=['a','b'],feature_count=2,model='Linear',cv_mape_mean=5.,cv_mape_std=.2),
                            dict(candidate_id='simple',features=['a'],feature_count=1,model='Ridge',cv_mape_mean=5.3,cv_mape_std=.4),
                            dict(candidate_id='too_bad',features=[],feature_count=0,model='Dummy',cv_mape_mean=6.,cv_mape_std=0)])
        self.assertEqual(select_model(table,CFG)['candidate_id'],'simple')

    def test_extrapolation_candidate_pool(self):
        table=pd.DataFrame([
            dict(candidate_id='tree',feature_count=1,model='RandomForest',cv_mape_mean=1.,cv_mape_std=.1),
            dict(candidate_id='ridge',feature_count=3,model='Ridge',cv_mape_mean=8.,cv_mape_std=1.),
            dict(candidate_id='linear',feature_count=1,model='Linear',cv_mape_mean=8.3,cv_mape_std=2.)])
        self.assertEqual(select_model(table,CFG)['candidate_id'],'linear')
        with self.assertRaises(ValueError):
            select_model(table.loc[table.model.eq('RandomForest')],CFG)

    def test_metrics(self):
        result=evaluate_model([100,200],[110,180])
        self.assertAlmostEqual(result['mape_pct'],10.);self.assertEqual(result['mean_signed_error'],-5.)
        self.assertEqual(result['overprediction_rate_pct'],50.)
        with self.assertRaises(AssertionError):evaluate_model([0],[10])
        with self.assertRaises(AssertionError):evaluate_model([100],[np.inf])

if __name__=='__main__':unittest.main()
