"""Cumulative four-case ablations with paired kernels and unchanged RF readouts."""
from dataclasses import replace
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

from .additive import AdditiveKernelClassifier, FullGraphInputs
from .data import AnglePreprocessor, EllipticData, StratifiedSampler
from .experiments import RBFKernel
from .metrics import Evaluation
from .quantum import QuantumFeatureMap, StatevectorKernel, NystromProjector
from .reporting import ResultPlots
from .topology import RootedSubgraphSampler, TopologyPreprocessor
from .topology_encoders import GraphHamiltonianFeatureMap
from .topology_research import GraphDescriptors


class FourCaseStudy:
    """Use a common radius and preserve all numerical inputs in both kernel arms.

    Case 1: attributes. Case 2: attributes + full-period propagation. Cases 3/4:
    the entire case-2 vector + unencoded spectral descriptors. The optional
    kernel branch sees attributes / propagation / spectra / packed adjacency.
    Every case includes RF-only, RBF+RF and quantum-kernel+RF arms.
    """
    cases=('1 Raw attributes','2 + Propagation','3 + Spectral topology','4 + Direct topology')
    arms=('RF only','Classical RBF','Quantum circuit')
    metrics=list(Evaluation.CORE_METRICS)
    keys=['Case','Features','Arm','Model','Passes','Steps','Hops','Raw columns retained']

    def __init__(self, flow, hops=4, passes=(1,2), steps=(1,2,4,8)):
        self.flow,self.store=flow,flow.store
        self.hops,self.passes,self.steps=hops,tuple(passes),tuple(steps)
        self.spec={'version':1,'hops':hops,'passes':self.passes,'steps':self.steps,
                   'nodes':flow.nodes,'landmarks':flow.landmarks,'seeds':flow.seeds,
                   'spectra_bypass_in_cases_3_and_4':True,'selection':'mean validation AP',
                   'flow':flow.configuration}

    @staticmethod
    def _angles(values):
        prep=AnglePreprocessor()
        return {'train':prep.fit_transform(values['train']),
                **{s:prep.transform(values[s]) for s in ('val','test')}}

    @staticmethod
    def _blocks(X,y,kernel,landmarks,seed,indices=None):
        indices=StratifiedSampler.landmarks(y['train'],landmarks,seed) if indices is None else indices
        refs=X['train'][indices]
        projector=NystromProjector().fit(kernel.evaluate(refs))
        return {s:projector.transform(kernel.evaluate(X[s],refs)) for s in X}

    @staticmethod
    def _readout(raw,phi,y,seed):
        model=AdditiveKernelClassifier(None,seed=seed).fit_readout(raw['train'],phi['train'],y['train'])
        joined={s:model.join(raw[s],phi[s]) for s in raw}
        score=model.model_.predict_proba(joined['test'])[:,1]
        val=model.model_.predict_proba(joined['val'])[:,1]
        return {**Evaluation.classification(y['test'],model.model_.predict(joined['test']),score),
            'Validation PR-AUC':Evaluation.classification(y['val'],model.model_.predict(joined['val']),val)['PR-AUC'],
            'Raw columns retained':raw['train'].shape[1]}

    @staticmethod
    def _materialize(full,hops,samples):
        attrs={'Local':full.local,'All':full.original}
        prop={'Local':full.local_raw(hops),'All':full.raw(hops)}
        spectral={s:np.array([GraphDescriptors.spectral(a) for a in items]) for s,items in samples.items()}
        retained={f:{s:np.column_stack([prop[f][s],spectral[s]]) for s in spectral} for f in attrs}
        signals={s:prop['All'][s][:,attrs['All'][s].shape[1]:] for s in spectral}
        # These assertions guard against inadvertently replacing attributes or spectra.
        for features in attrs:
            for split in spectral:
                np.testing.assert_array_equal(retained[features][split][:,:prop[features][split].shape[1]],prop[features][split])
                np.testing.assert_array_equal(prop[features][split][:,:attrs[features][split].shape[1]],attrs[features][split])
        return attrs,prop,spectral,retained,signals

    def _evaluate(self,seed,y,full,samples,packed,landmarks,indices=None,measured=None,steps=None):
        attrs,prop,spectral,retained,signals=self._materialize(full,self.hops,samples)
        rows=[]
        bases=[attrs,prop,retained,retained]
        rf_cache={}
        for number,case in enumerate(self.cases):
            # Cases 2–4 share their kernel inputs across Local / All; only the bypass changes.
            inputs={f:(attrs[f] if number==0 else signals if number==1 else spectral if number==2 else packed)
                    for f in attrs}
            block_cache={}
            configs=[(p,0) for p in self.passes] if number<3 else [(0,s) for s in (steps or self.steps)]
            for features in ('Local','All'):
                raw=bases[number][features]
                meta={'Case':case,'Features':features,'Seed':seed,'Hops':0 if number==0 else self.hops}
                rf_key=(features,min(number,2))
                if rf_key not in rf_cache:
                    empty={s:np.empty((len(y[s]),0)) for s in y}
                    rf_cache[rf_key]=self._readout(raw,empty,y,seed)
                rows.append({**rf_cache[rf_key],**meta,'Arm':'RF only','Model':
                    ['Raw RF','RF + propagation','RF + propagation + spectra','RF + propagation + spectra'][number],
                    'Passes':0,'Steps':0})
                shared=features if number==0 else 'shared'
                if shared not in block_cache:
                    X=self._angles(inputs[features]) if number<3 else inputs[features]
                    gamma=1/(X['train'].shape[1]*max(X['train'].var(),1e-12))
                    classical=self._blocks(X,y,RBFKernel(gamma),landmarks,seed,indices)
                    quantum={}
                    for passes,depth in configs:
                        fm=(QuantumFeatureMap.uploading(X['train'].shape[1],4,passes) if number<3 else
                            GraphHamiltonianFeatureMap(self.flow.nodes,steps=depth,topology_only=True,field_axis='Y'))
                        quantum[passes,depth]=self._blocks(X,y,StatevectorKernel(fm),landmarks,seed,indices)
                    block_cache[shared]=(classical,quantum)
                classical,quantum=block_cache[shared]
                classical_score=self._readout(raw,classical,y,seed)
                for passes,depth in configs:
                    for arm,score in [('Classical RBF',classical_score),
                        ('Quantum circuit',self._readout(raw,quantum[passes,depth],y,seed))]:
                        rows.append({**score,**meta,'Arm':arm,'Model':
                            ('RF + adjacency RBF' if number==3 else 'RF + RBF '+['attributes','propagation','spectra'][number])
                            if arm=='Classical RBF' else ('RF + Hamiltonian circuit' if number==3 else 'RF + quantum '+['attributes','propagation','spectra'][number]),
                            'Passes':passes,'Steps':depth})
                if number==3 and measured is not None:
                    rows.append({**self._readout(raw,measured,y,seed),**meta,
                        'Arm':'IBM circuit','Model':'RF + IBM Hamiltonian circuit','Passes':0,'Steps':4})
        return pd.DataFrame(rows)

    def run(self,reuse_saved=True):
        path=self.store.directory/'07_four_case_all.csv'
        config=self.store.directory/'07_four_case_configuration.json'
        spec=json.loads(json.dumps(self.spec))
        roots=hashlib.sha256()
        for seed,study in self.flow.cohorts.items():
            for split in ('train','val','test'):
                roots.update(np.asarray(study.roots[split]).tobytes());roots.update(np.asarray(study.y[split]).tobytes())
        spec['cohort_sha256']=roots.hexdigest()
        spec['implementation_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if reuse_saved and path.exists() and config.exists() and json.loads(config.read_text())==spec:
            self.results=pd.read_csv(path,keep_default_na=False)
            print('Restored cumulative four-case simulator fits.');return self
        frames=[]
        for seed,study in self.flow.cohorts.items():
            print('Cumulative four-case simulator study:',seed,flush=True)
            samples,packed=self.flow.topology_inputs(study,self.hops)
            frames.append(self._evaluate(seed,study.y,study.full_inputs,samples,packed,self.flow.landmarks))
        self.results=pd.concat(frames,ignore_index=True)
        self.store.save('07_four_case_all',self.results)
        config.write_text(json.dumps(spec,indent=2))
        return self

    @classmethod
    def aggregate(cls,frame):
        group=frame.groupby(cls.keys,sort=False,dropna=False)
        result=group[cls.metrics+['Validation PR-AUC']].mean()
        result['Seeds']=group.Seed.nunique()
        for metric in cls.metrics:result[metric+' SD']=group[metric].std()
        result['AP SD']=result['PR-AUC SD']
        return result.reset_index()

    @classmethod
    def select(cls,frame):
        means=cls.aggregate(frame);selected=[]
        for (case,features),items in means.groupby(['Case','Features'],sort=False):
            q=items[items.Arm=='Quantum circuit'].sort_values(['Passes','Steps'],kind='stable')
            q=q.loc[q['Validation PR-AUC'].idxmax()]
            match=items[(items.Arm=='Classical RBF')&(items.Passes==q.Passes)&(items.Steps==q.Steps)]
            rf=items[items.Arm=='RF only']
            if len(match)!=1 or len(rf)!=1:raise ValueError('Every quantum winner requires its exact RBF and RF controls')
            selected.extend([rf.iloc[0].to_dict(),match.iloc[0].to_dict(),q.to_dict()])
            ibm=items[items.Arm=='IBM circuit']
            if len(ibm):selected.extend(ibm.to_dict('records'))
        result=pd.DataFrame(selected)
        result['Setting']=result.apply(lambda r:('raw attributes' if not r.Hops else f'{int(r.Hops)} hops')+
            (f'; passes={int(r.Passes)}' if r.Passes else '')+(f'; steps={int(r.Steps)}' if r.Steps else ''),axis=1)
        return result

    def reports(self):
        self.selected=self.select(self.results)
        self.store.save('07_four_case_aggregate',self.aggregate(self.results))
        self.store.save('07_four_case_selected_full',self.selected)
        pairs=[]
        for _,q in self.selected[self.selected.Arm=='Quantum circuit'].iterrows():
            subset=self.results[(self.results.Case==q.Case)&(self.results.Features==q.Features)&
                (self.results.Passes==q.Passes)&(self.results.Steps==q.Steps)]
            for seed,items in subset.groupby('Seed'):
                c=items[items.Arm=='Classical RBF'].iloc[0];v=items[items.Arm=='Quantum circuit'].iloc[0]
                pairs.append({'Case':q.Case,'Features':q.Features,'Seed':seed,'Setting':q.Setting,
                    **{'Δ '+m:v[m]-c[m] for m in self.metrics}})
        self.store.save('07_four_case_paired_deltas',pd.DataFrame(pairs))
        return self

    def case(self,number):
        selected=self.selected[self.selected.Case==self.cases[number-1]]
        self.store.show(f'07_case_{number}_best',selected,['Case','Features','Arm','Model','Setting'])
        heat=selected.copy();heat['Label']=heat.Features+' / '+heat.Arm
        self.store.save(f'07_case_{number}_heatmap',heat[['Label']+self.metrics])
        self.store.plot(f'07_case_{number}_heatmap',ResultPlots.metric_heatmap(heat,by=('Label',),
            title=self.cases[number-1]+': all five test metrics'))
        return selected

    def summary(self):
        self.store.show('07_four_case_summary',self.selected,['Case','Features','Arm','Model','Setting'])
        self.plot_progression(self.selected,'07_four_case_progression',self.metrics)
        self.plot_progression(self.selected,'07_four_case_key_metrics',['Illicit F1','PR-AUC'])
        # The whole simulator depth sweep is retained without long printed tables.
        graph=self.aggregate(self.results[self.results.Case==self.cases[3]])
        fig,axes=plt.subplots(1,2,figsize=(11,3.5))
        for ax,features in zip(axes,('Local','All')):
            points=graph[graph.Features==features];values=[]
            for arm in ('Classical RBF','Quantum circuit'):
                p=points[points.Arm==arm].sort_values('Steps')
                ax.plot(p.Steps,p['Validation PR-AUC'],marker='o',label=arm);values.extend(p['Validation PR-AUC'])
            ax.set(title=features+': Hamiltonian product-formula steps',xlabel='Steps',ylabel='Mean validation AP')
            ResultPlots.relevant_ylim(ax,values);ax.grid(alpha=.2)
        axes[0].legend(fontsize=8);fig.tight_layout();self.store.plot('07_cumulative_depth_sweep',fig)
        display(Markdown('**Interpretation:** compare RF-only across cases for added numerical information, '
            'then compare the matched RBF and circuit arms for representation value. Case 4 retains the '
            'entire case-3 numerical vector in both arms. Main results use four hops throughout; older '
            'independently selected radii belong to the exploratory results, not these cumulative comparisons.'))
        self.index()

    def plot_progression(self,selected,name,metrics):
        fig,axes=plt.subplots(2,len(metrics),figsize=(max(10,4*len(metrics)),7),squeeze=False)
        for row,features in enumerate(('Local','All')):
            for col,metric in enumerate(metrics):
                ax=axes[row,col];values=[];errors=[]
                for arm in self.arms:
                    points=selected[(selected.Features==features)&(selected.Arm==arm)].set_index('Case').loc[list(self.cases)]
                    sd=points[metric+' SD'].fillna(0)
                    ax.errorbar(range(4),points[metric],yerr=sd,marker='o',capsize=3,label=arm)
                    values.extend(points[metric]);errors.extend(sd)
                ax.set_xticks(range(4),['Raw','+ propagation','+ spectra','+ direct graph'],rotation=20,ha='right',fontsize=8)
                ax.set(title=features+': '+metric,ylabel='Mean test score');ax.grid(alpha=.2)
                ResultPlots.relevant_ylim(ax,values,errors)
        axes[0,0].legend(fontsize=8);fig.suptitle('Four cumulative cases at a common four-hop radius; error bars = sampling SD')
        fig.tight_layout();self.store.plot(name,fig)

    def hardware(self):
        """Reanalyse existing measured graph kernels with all four matched input cases.

        No IBM account is loaded and no jobs are submitted. The graph circuits
        excluded numerical attributes, so the same measurements can be reused
        with new Local/All classical bypasses without new QPU execution.
        """
        output=self.flow.root/'experiment_results'
        manifest=json.loads((output/'ibm_final_manifest.json').read_text())
        if manifest['phase']!='complete':raise ValueError('Complete the recorded topology hardware run first')
        z=np.load(output/'ibm_final_inputs.npz');kernels=np.load(output/'ibm_final_kernels.npz')
        roots={s:z['roots_'+s] for s in ('train','val','test')};y={s:z['y_'+s] for s in roots}
        full=FullGraphInputs(self.flow.graph,roots,y,max_hops=self.hops,feature_set='All')
        sampler=RootedSubgraphSampler(self.flow.graph,self.flow.nodes,EllipticData.feature_columns('Local'))
        samples={s:[replace(a,attributes=np.zeros((self.flow.nodes,2))) for a in sampler.collect(roots[s],self.hops)] for s in roots}
        prep=TopologyPreprocessor();packed={'train':prep.fit_transform(samples['train'])}
        packed.update({s:prep.transform(samples[s]) for s in ('val','test')})
        for s in roots:np.testing.assert_allclose(packed[s],z['X_'+s],atol=1e-12)
        projector=NystromProjector().fit(kernels['W_measured'])
        measured={s:projector.transform(kernels['C_'+s]) for s in roots}
        results=self._evaluate(manifest['seed'],y,full,samples,packed,len(z['landmarks']),z['landmarks'],measured,steps=(4,))
        selected=self.select(results)
        selected['Execution']=selected.Arm.map({'RF only':'Classical','Classical RBF':'Classical',
            'Quantum circuit':'Noiseless simulator','IBM circuit':'IBM measured'})
        self.store.save('08_hardware_four_case_all',results)
        self.store.save('08_hardware_four_case_selected_full',selected)
        self.store.show('08_hardware_four_case_summary',selected,['Case','Features','Arm','Model','Execution','Setting'])
        display(Markdown('**Hardware pilot:** all four cases use the same recorded balanced 12/8/12 roots; '
            'Local and All are both included. Compare each circuit arm with the RF and RBF controls first. '
            'Noiseless and IBM execution are two implementations of the final graph circuit. Earlier '
            'quantum feature-map arms remain simulator-only; only case 4 has IBM measurements. '
            'The numerical spectral bypass is now retained, so these readout scores can differ from '
            'the earlier four-row hardware report. No circuits were resubmitted.'))
        self.index()
        return selected

    def index(self):
        self.store.save('result_index',pd.DataFrame({'CSV':[p.name for p in sorted(self.store.directory.glob('*.csv')) if p.name!='result_index.csv']}))
