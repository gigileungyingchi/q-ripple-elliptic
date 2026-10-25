"""Validation-only selection, compulsory counterparts and a complete short flow."""
import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from elliptic_models import ExperimentFlow,PairedSelection,Evaluation,RiskGraph,PropagationStudy


class SelectionTests(unittest.TestCase):
    def frame(self):
        rows=[]
        for seed in (42,43):
            for depth in (1,2):
                for family,val,test in [('Classical',.8 if depth==1 else .7,.1 if depth==1 else .9),
                                         ('Quantum',.6 if depth==1 else .9,.99 if depth==1 else .2)]:
                    rows.append({'Stage':'Test','Features':'Local','Seed':seed,'Axis':'Main',
                        'Pair':'Kernel','Hops':depth,'Depth':1,'Depth kind':'passes',
                        'Model':family,'Family':family,'Reference':'Classical' if family=='Quantum' else '',
                        'Validation PR-AUC':val,**{m:test for m in Evaluation.CORE_METRICS}})
        return pd.DataFrame(rows)

    def test_quantum_best_uses_validation_and_keeps_exact_classical_context(self):
        selected=PairedSelection.best(self.frame())
        self.assertEqual(len(selected),3)
        q=selected[selected.Family=='Quantum'].iloc[0]
        self.assertEqual(q.Hops,2)  # Test AP would incorrectly choose depth 1.
        classical=selected[selected.Selection.str.contains('Matched')].iloc[0]
        self.assertEqual(classical.Hops,q.Hops)
        self.assertAlmostEqual(classical['PR-AUC'],.9)
        self.assertEqual(q.Seeds,2)

    def test_missing_per_seed_counterpart_fails_loudly(self):
        frame=self.frame()
        frame=frame[~((frame.Family=='Classical')&(frame.Hops==2)&(frame.Seed==42))]
        with self.assertRaises(ValueError):PairedSelection.pairs(frame)


class SavedResultTests(unittest.TestCase):
    def test_reuse_requires_matching_stage_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            first=ExperimentFlow(directory,hops=(1,4))
            frame=SelectionTests().frame()
            first.store.save('01_attributes_all',frame)
            current=ExperimentFlow(directory,hops=(1,4))
            with patch.object(current,'classification') as fit:
                pd.testing.assert_frame_equal(current.run('Attributes',reuse_saved=True),frame)
                fit.assert_not_called()
            changed=ExperimentFlow(directory,hops=(1,2,4))
            with patch.object(changed,'classification',return_value=frame) as fit:
                changed.run('Attributes',reuse_saved=True)
                fit.assert_called_once()


class ExperimentFlowTests(unittest.TestCase):
    def test_all_stages_save_full_and_selected_results_without_cloud_access(self):
        rng=np.random.default_rng(18)
        frame=pd.DataFrame(rng.normal(size=(12,165)),columns=[f'x{i}' for i in range(165)])
        frame['txId'],frame['time_step'],frame['label']=np.arange(12),1,[0,1]*6
        graph=RiskGraph(frame,pd.DataFrame({'txId1':np.arange(11),'txId2':np.arange(1,12)}),seed_fraction=0)
        with tempfile.TemporaryDirectory() as directory:
            flow=ExperimentFlow(directory,seeds=(7,),sizes=dict(train=12,val=12,test=12),
                hops=(1,4),circuit_depths=(1,2),message_layers=(1,2),nodes=3,landmarks=4)
            flow.graph=graph
            flow.cohorts={7:PropagationStudy(graph,{s:[1] for s in ('train','val','test')},
                flow.sizes,max_hops=4,n_landmarks=4,seed=7)}
            with patch('elliptic_models.experiment_flow.display'),patch.object(flow.store,'plot',side_effect=lambda _,fig:plt.close(fig)):
                for stage,method in [('Attributes',flow.classification),('Propagation',flow.propagation),
                                     ('Topology features',flow.topology_features),('Quantum topology',flow.quantum_topology)]:
                    results=method()
                    selected=flow.best(stage)
                    flow.heatmaps(stage)
                    flow.interpretation(stage)
                    self.assertTrue(np.isfinite(results[list(Evaluation.CORE_METRICS)].to_numpy()).all())
                    self.assertEqual(set(selected.Features),{'Local','All'})
                    for _,q in selected[selected.Family=='Quantum'].iterrows():
                        matches=selected[(selected.Family=='Classical')&(selected.Model==q.Reference)]
                        for key in PairedSelection.context:matches=matches[matches[key]==q[key]]
                        self.assertEqual(len(matches),1)
                    stem=flow.stages[stage]
                    for suffix in ('_all','_best','_pairs'):
                        self.assertTrue((flow.store.directory/f'{stem}{suffix}.csv').exists())
                flow.kernel_heatmaps(hops=4,count=4)
                flow.summary()
            self.assertTrue((flow.store.directory/'06_summary_best_pairs.csv').exists())
            self.assertTrue((flow.store.directory/'04_kernel_spectral_summaries_quantum.csv').exists())
            self.assertTrue((flow.store.directory/'06_classical_progression.csv').exists())
            self.assertTrue((flow.store.directory/'result_index.csv').exists())


if __name__=='__main__':unittest.main()
