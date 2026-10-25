"""Cumulative inputs, compulsory matched controls and validation-only selection."""
import tempfile,unittest
from pathlib import Path
import numpy as np
import pandas as pd
from elliptic_models import ExperimentFlow,FourCaseStudy,RiskGraph,PropagationStudy


class FourCaseTests(unittest.TestCase):
    def test_cumulative_inputs_controls_and_selection(self):
        rng=np.random.default_rng(18)
        frame=pd.DataFrame(rng.normal(size=(12,165)),columns=[f'x{i}' for i in range(165)])
        frame['txId'],frame['time_step'],frame['label']=np.arange(12),1,[0,1]*6
        graph=RiskGraph(frame,pd.DataFrame({'txId1':np.arange(11),'txId2':np.arange(1,12)}),seed_fraction=0)
        with tempfile.TemporaryDirectory() as directory:
            flow=ExperimentFlow(directory,seeds=(7,),sizes=dict(train=12,val=12,test=12),
                hops=(1,4),nodes=3,landmarks=4)
            flow.graph=graph
            flow.cohorts={7:PropagationStudy(graph,{s:[1] for s in ('train','val','test')},flow.sizes,
                max_hops=4,n_landmarks=4,seed=7)}
            study=FourCaseStudy(flow,passes=(1,2),steps=(1,2)).run(reuse_saved=False).reports()
            self.assertEqual(len(study.selected),24)
            for features in ('Local','All'):
                chosen=study.selected[study.selected.Features==features]
                for case in study.cases:
                    rows=chosen[chosen.Case==case]
                    self.assertEqual(set(rows.Arm),set(study.arms))
                    q=rows[rows.Arm=='Quantum circuit'].iloc[0]
                    c=rows[rows.Arm=='Classical RBF'].iloc[0]
                    self.assertEqual((q.Passes,q.Steps,q.Hops),(c.Passes,c.Steps,c.Hops))
                base=chosen[chosen.Arm=='RF only'].set_index('Case')
                np.testing.assert_allclose(base.loc[study.cases[2],study.metrics].astype(float),
                                           base.loc[study.cases[3],study.metrics].astype(float))
                self.assertGreater(base.loc[study.cases[2],'Raw columns retained'],base.loc[study.cases[1],'Raw columns retained'])
            samples,_=flow.topology_inputs(flow.cohorts[7],4)
            attrs,prop,spectra,retained,_=study._materialize(flow.cohorts[7].full_inputs,4,samples)
            for features in attrs:
                for split in attrs[features]:
                    np.testing.assert_array_equal(retained[features][split][:,:attrs[features][split].shape[1]],attrs[features][split])
                    np.testing.assert_array_equal(retained[features][split][:,-spectra[split].shape[1]:],spectra[split])
            altered=study.results.copy()
            qmask=altered.Arm=='Quantum circuit'
            winner=(altered.Passes==1)|(altered.Steps==1)
            altered.loc[qmask,'Validation PR-AUC']=np.where(winner[qmask],.9,.1)
            altered.loc[qmask,'Illicit F1']=np.where(winner[qmask],.1,.99)
            selected=study.select(altered)
            q=selected[selected.Arm=='Quantum circuit']
            self.assertTrue(((q.Passes==1)|(q.Steps==1)).all())
            self.assertTrue((Path(directory)/'results/experiment_flow/07_four_case_paired_deltas.csv').exists())


if __name__=='__main__':unittest.main()
