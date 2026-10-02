"""Verify assignment aggregation/sign conventions independently of pooled OOF."""
import unittest
import pandas as pd
from src.reporting import build_performance_reporting

class ReportingTests(unittest.TestCase):
    def test_cv_fold_mean_and_gap_direction(self):
        performance=pd.DataFrame({'partition':['Train','Valid','Batch2','Batch3'],
                                  'mape_pct':[8.17,13.79,31.31,12.90]})
        result=build_performance_reporting(performance,{'cv_mape_mean':8.10},{'project_goal_mape_pct':9.1}).set_index('index')
        self.assertAlmostEqual(result.loc['Train (Batch 1 CV)','value'],8.10)
        self.assertAlmostEqual(result.loc['Gap (Train-Valid)','value'],5.69)
        self.assertAlmostEqual(result.loc['Gap (Valid-Test)','value'],17.52)
        self.assertAlmostEqual(result.loc['Gap (Target-Test), Batch2','value'],22.21)
        self.assertAlmostEqual(result.loc['Gap (Batch2-Batch3)','value'],-18.41)
        self.assertAlmostEqual(result.loc['Gap (Target-Test), Batch3','value'],3.80)
        self.assertEqual(result.loc['Test (Batch 2)','unit'],'%')
        self.assertEqual(result.loc['Gap (Train-Valid)','unit'],'%p')
        self.assertEqual(performance.loc[0,'mape_pct'],8.17)
