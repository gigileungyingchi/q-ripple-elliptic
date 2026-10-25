"""Research notebook orchestration: fixed cohorts, paired selection, complete CSVs."""
from dataclasses import replace
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

from .data import EllipticData, AnglePreprocessor
from .graph import RiskGraph
from .graph_studies import PropagationStudy, KernelComparison
from .experiments import RBFKernel
from .quantum import QuantumFeatureMap, StatevectorKernel
from .topology import RootedSubgraphSampler, TopologyPreprocessor
from .topology_encoders import GraphHamiltonianFeatureMap, GraphMessageFeatureMap, HamiltonianMomentKernel, HermitianWalkFeatureMap
from .topology_research import GraphDescriptors, StructuralProbeStudy, GraphRetention
from .hardware_experiments import HamiltonianHardwareExperiment
from .reporting import ResultReport, ResultPlots
from .metrics import Evaluation


class ResultStore:
    """Every displayed table and every full experiment gets its own CSV."""
    def __init__(self, directory):
        self.directory=Path(directory)
        self.figures=self.directory/'figures'
        self.figures.mkdir(parents=True,exist_ok=True)

    def save(self, name, frame):
        frame.to_csv(self.directory/f'{name}.csv',index=False)
        return frame

    def show(self, name, frame, identifiers):
        table=ResultReport.table(frame,identifiers,extras=('Validation PR-AUC','Seeds','AP SD'))
        self.save(name,table)
        display(table)

    def plot(self, name, figure):
        ResultPlots.save(figure,self.figures/name)
        display(figure)
        plt.close(figure)


class PairedSelection:
    """Pick by mean validation AP; always retain a quantum winner's counterpart."""
    keys=['Stage','Features','Axis','Pair','Hops','Depth','Depth kind','Model','Family','Reference']
    context=['Stage','Features','Axis','Pair','Hops','Depth','Depth kind']
    scores=list(ResultReport.metrics)+['Validation PR-AUC']

    @classmethod
    def aggregate(cls, frame):
        grouped=frame.groupby(cls.keys,dropna=False,sort=False)
        means=grouped[cls.scores].mean()
        means['Seeds']=grouped.Seed.nunique()
        means['AP SD']=grouped['PR-AUC'].std()
        return means.reset_index()

    @classmethod
    def best(cls, frame):
        means=cls.aggregate(frame)
        selected=[]
        for features,rows in means.groupby('Features',sort=False):
            quantum=rows[rows.Family=='Quantum']
            classical=rows[rows.Family=='Classical']
            if quantum.empty or classical.empty:raise ValueError('Both families are required')
            q=quantum.loc[quantum['Validation PR-AUC'].idxmax()]
            c=classical.loc[classical['Validation PR-AUC'].idxmax()]
            match=classical[classical.Model==q.Reference]
            for key in cls.context:match=match[match[key]==q[key]]
            if len(match)!=1:raise ValueError(f'Missing/ambiguous counterpart for {q.Model}')
            if np.isclose(match.iloc[0]['Validation PR-AUC'],c['Validation PR-AUC'],atol=1e-12,rtol=0):
                c=match.iloc[0]
            for row,selection in [(c,'Best classical'),(match.iloc[0],'Matched classical'),(q,'Best quantum')]:
                record=row.to_dict();record['Selection']=selection
                signature=tuple(record[key] for key in cls.keys)
                previous=next((r for r in selected if tuple(r[key] for key in cls.keys)==signature),None)
                if previous:previous['Selection']+=' + '+selection.lower()
                else:selected.append(record)
        result=pd.DataFrame(selected)
        result['Setting']=result.apply(cls.setting,axis=1)
        return result

    @staticmethod
    def setting(row):
        if row.Model=='Raw RF':return 'raw attributes'
        if 'legacy' in row.Model:return '9 legacy features'
        parts=[]
        if row.Hops:parts.append(f'hops={int(row.Hops)}')
        if row.Depth:parts.append(f'{row["Depth kind"]}={int(row.Depth)}')
        if row.Pair and row.Pair not in row.Model:parts.append(row.Pair)
        return '; '.join(parts)

    @classmethod
    def pairs(cls, frame):
        rows=[]
        for _,q in frame[frame.Family=='Quantum'].iterrows():
            match=frame[(frame.Model==q.Reference)&(frame.Family=='Classical')&(frame.Seed==q.Seed)]
            for key in cls.context:match=match[match[key]==q[key]]
            if len(match)!=1:raise ValueError(f'Unmatched per-seed quantum run: {q.Model}')
            c=match.iloc[0]
            record={key:q[key] for key in cls.context}
            record.update(Seed=q.Seed,Classical=c.Model,Quantum=q.Model)
            for metric in cls.scores:
                record[metric+' classical']=c[metric]
                record[metric+' quantum']=q[metric]
                record['Δ '+metric]=q[metric]-c[metric]
            rows.append(record)
        return pd.DataFrame(rows)


class ExperimentFlow:
    """Four scientific questions; heavy experiment code stays outside the notebook.

    All simulator stages use the same chronological non-seed roots per sampling
    seed. All hybrid readouts retain raw 93/165 attributes. The quantum branch in
    the final stage receives actual adjacency/root/seed markers, not summaries.
    """
    stages={'Attributes':'01_attributes','Propagation':'02_propagation',
            'Topology features':'03_topology_features','Quantum topology':'04_quantum_topology'}

    def __init__(self, root, seeds=(42,43,44), sizes=None, hops=(1,2,3,4,6,8),
                 circuit_depths=(1,2,4,8), message_layers=(1,2,3,4), nodes=8, landmarks=12):
        self.root=Path(root)
        self.store=ResultStore(self.root/'results'/'experiment_flow')
        self.seeds=tuple(seeds)
        self.sizes=sizes or {'train':80,'val':60,'test':100}
        self.hops=tuple(hops)
        self.circuit_depths=tuple(circuit_depths)
        self.message_layers=tuple(message_layers)
        self.nodes,self.landmarks=nodes,landmarks
        self.tables,self.samples,self.inputs,self.references,self.rf_cache={},{},{},{},{}
        self.configuration={'seeds':self.seeds,'sizes':self.sizes,'hops':self.hops,
            'hamiltonian_steps':self.circuit_depths,'message_layers':self.message_layers,
            'nodes':nodes,'landmarks':landmarks,'feature_sets':['Local','All'],
            'time':.8,'field_axis':'Y','raw_inputs_retained':True,
            'hardware_policy':{'attributes':False,'propagation':False,'topology_features':False,'quantum_topology':True}}
        config_path=self.store.directory/'configuration.json'
        self.saved_configuration=json.loads(config_path.read_text()) if config_path.exists() else None
        config_path.write_text(json.dumps(self.configuration,indent=2))

    def prepare(self):
        self.data=EllipticData.load(self.root/'data')
        self.graph=RiskGraph(self.data.transactions,self.data.edges,seed_fraction=.2,random_state=42)
        periods={'train':range(1,31),'val':range(31,35),'test':range(35,50)}
        self.cohorts={seed:PropagationStudy(self.graph,periods,self.sizes,max_hops=max(self.hops),
            n_landmarks=self.landmarks,seed=seed) for seed in self.seeds}
        counts,targets=[],[]
        for seed,study in self.cohorts.items():
            np.savez_compressed(self.store.directory/f'cohort_{seed}.npz',
                **{f'roots_{s}':study.roots[s] for s in study.roots},
                **{f'y_{s}':study.y[s] for s in study.y})
            for split,y in study.y.items():
                counts.append({'Seed':seed,'Split':split,'Targets':len(y),'Illicit':int(y.sum()),'Illicit fraction':y.mean()})
                targets.extend({'Seed':seed,'Split':split,'Time step':int(t),'Period row':int(i),'Label':int(label)}
                    for (t,i),label in zip(study.roots[split],y))
        self.store.save('cohorts',pd.DataFrame(targets))
        self.store.save('cohort_summary',pd.DataFrame(counts))
        print(f'Simulator cohorts ready: {self.sizes}; {len(self.seeds)} sampling seeds. All four stages share roots.')
        print('Full and selected tables:',self.store.directory)
        return self

    @staticmethod
    def meta(stage,seed,pair='',hops=0,depth=0,kind='',axis='Main'):
        return {'Stage':stage,'Seed':seed,'Pair':pair,'Hops':hops,'Depth':depth,'Depth kind':kind,'Axis':axis}

    def raw(self,study,features,hops=0):
        if hops:return study.full_inputs.local_raw(hops) if features=='Local' else study.full_inputs.raw(hops)
        return study.full_inputs.local if features=='Local' else study.full_inputs.original

    def baseline(self,study,features,stage,hops=0):
        key=(study.seed,features,hops)
        if key not in self.rf_cache:
            compare=KernelComparison(self.raw(study,features,hops),study.y,self.landmarks,study.seed)
            self.rf_cache[key]=compare.random_forest('Raw RF' if hops==0 else 'RF + propagation')
        return {**self.rf_cache[key],**self.meta(stage,study.seed,hops=hops),
            'Features':features,'Family':'Classical','Reference':''}

    def finish(self,stage,rows,diagnostics=None):
        frame=pd.DataFrame(rows).fillna({'Reference':'','Pair':'','Depth kind':''})
        self.tables[stage]=frame
        stem=self.stages[stage]
        self.store.save(stem+'_all',frame)
        self.store.save(stem+'_aggregate',PairedSelection.aggregate(frame))
        self.store.save(stem+'_pairs',PairedSelection.pairs(frame))
        if diagnostics is not None:self.store.save(stem+'_diagnostics',pd.DataFrame(diagnostics))
        return frame

    def classification(self):
        rows,diagnostics=[],[]
        for seed,study in self.cohorts.items():
            for features in ('Local','All'):
                raw=self.raw(study,features)
                prep=AnglePreprocessor()
                X={'train':prep.fit_transform(raw['train'])}
                X.update({s:prep.transform(raw[s]) for s in ('val','test')})
                compare=KernelComparison(X,study.y,self.landmarks,seed)
                gamma=1/(X['train'].shape[1]*max(X['train'].var(),1e-12))
                compare.nystrom('Classical kernel',RBFKernel(gamma))
                classical=compare.additive('RF + RBF attributes','Classical kernel',raw)
                rows.append(self.baseline(study,features,'Attributes'))
                for passes in (1,2):
                    meta={**self.meta('Attributes',seed,'Attributes',depth=passes,kind='passes'),'Features':features}
                    rows.append({**classical,**meta,'Family':'Classical','Reference':''})
                    fm=QuantumFeatureMap.uploading(X['train'].shape[1],4,passes)
                    compare.nystrom(f'Quantum p={passes}',StatevectorKernel(fm))
                    quantum=compare.additive('RF + quantum attributes',f'Quantum p={passes}',raw)
                    rows.append({**quantum,**meta,'Family':'Quantum','Reference':'RF + RBF attributes'})
                diagnostics.extend({**r,'Features':features} for r in compare.rows if 'Raw columns retained' not in r)
        return self.finish('Attributes',rows,diagnostics)

    def propagation(self):
        rows,diagnostics=[],[]
        for seed,study in self.cohorts.items():
            print('Simulator propagation sweep:',seed,flush=True)
            source=study.run_depths(self.hops)
            diagnostics.extend(source[source.Diagnostic].to_dict('records'))
            for _,record in source[~source.Diagnostic].iterrows():
                features=record['Original features'];prefix='Local' if features=='Local' else 'Original'
                name=record.Model
                base={**record.to_dict(),**self.meta('Propagation',seed,hops=int(record.Hops)),
                      'Features':features,'Family':'Classical','Reference':''}
                if name in ('Local-only RF (93)','All-original RF (165)'):
                    rows.append({**base,'Model':'Raw RF'})
                elif name==prefix+' + legacy propagation RF':rows.append({**base,'Model':'RF + legacy propagation'})
                elif name==prefix+' + propagation RF':rows.append({**base,'Model':'RF + propagation'})
                elif name==prefix+' + propagation + RBF':
                    for pair in ('Hop-aware','Generic'):
                        rows.append({**base,'Model':'RF + RBF propagation','Pair':pair,'Depth':1,'Depth kind':'passes'})
                else:
                    pair='Hop-aware' if 'hop quantum' in name else 'Generic'
                    rows.append({**base,'Model':f'RF + {pair.lower()} quantum propagation',
                        'Family':'Quantum','Reference':'RF + RBF propagation','Pair':pair,'Depth':1,'Depth kind':'passes'})
        return self.finish('Propagation',rows,diagnostics)

    def topology_inputs(self,study,hops):
        key=(study.seed,hops)
        if key not in self.samples:
            sampler=RootedSubgraphSampler(self.graph,self.nodes,EllipticData.feature_columns('Local'))
            samples={s:sampler.collect(study.roots[s],hops) for s in study.roots}
            # Topology branch deliberately omits numerical node attributes. Raw
            # 93/165 attributes bypass it unchanged in every primary readout.
            samples={s:[replace(a,attributes=np.zeros((self.nodes,2))) for a in items] for s,items in samples.items()}
            prep=TopologyPreprocessor()
            X={'train':prep.fit_transform(samples['train'])}
            X.update({s:prep.transform(samples[s]) for s in ('val','test')})
            self.samples[key],self.inputs[key]=samples,X
        return self.samples[key],self.inputs[key]

    def topology_features(self):
        rows,diagnostics=[],[]
        for seed,study in self.cohorts.items():
            print('Simulator topology-feature sweep:',seed,flush=True)
            for hops in self.hops:
                samples,X=self.topology_inputs(study,hops)
                for features in ('Local','All'):
                    rows.append(self.baseline(study,features,'Topology features'))
                    rows.append(self.baseline(study,features,'Topology features',hops))
                for name,descriptor in [('Motifs',GraphDescriptors.summary),('Spectra',GraphDescriptors.spectral)]:
                    values={s:np.array([descriptor(a) for a in items]) for s,items in samples.items()}
                    prep=AnglePreprocessor();Z={'train':prep.fit_transform(values['train'])}
                    Z.update({s:prep.transform(values[s]) for s in ('val','test')})
                    compare=KernelComparison(Z,study.y,self.landmarks,seed)
                    gamma=1/(Z['train'].shape[1]*max(Z['train'].var(),1e-12))
                    compare.nystrom('Classical kernel',RBFKernel(gamma))
                    compare.nystrom('Quantum kernel',StatevectorKernel(QuantumFeatureMap.uploading(Z['train'].shape[1],4,2)))
                    for features in ('Local','All'):
                        raw=self.raw(study,features,hops)
                        meta={**self.meta('Topology features',seed,name,hops,2,'passes'),'Features':features}
                        for family,reference in [('Classical','Classical kernel'),('Quantum','Quantum kernel')]:
                            model='RF + '+('RBF' if family=='Classical' else 'quantum')+' topology features'
                            row=compare.additive(model,reference,raw)
                            rows.append({**row,**meta,'Family':family,'Reference':'RF + RBF topology features' if family=='Quantum' else ''})
                        augmented={s:np.column_stack([raw[s],values[s]]) for s in raw}
                        rf=KernelComparison(augmented,study.y,self.landmarks,seed)
                        rows.append({**rf.random_forest(f'RF + {name.lower()}'),**meta,'Family':'Classical','Reference':''})
                    diagnostics.extend({**r,'Hops':hops,'Descriptor':name} for r in compare.rows if 'Raw columns retained' not in r)
                compare=KernelComparison(X,study.y,self.landmarks,seed)
                compare.wl(samples,name='WL kernel')
                for features in ('Local','All'):
                    rows.append({**compare.additive('RF + rooted WL','WL kernel',self.raw(study,features,hops)),
                        **self.meta('Topology features',seed,'WL',hops),
                        'Features':features,'Family':'Classical','Reference':''})
        # Raw baselines were added once per radius but contain no radius input.
        frame=pd.DataFrame(rows).drop_duplicates(PairedSelection.keys+['Seed'])
        return self.finish('Topology features',frame.to_dict('records'),diagnostics)

    def topology_reference(self,study,hops):
        key=(study.seed,hops)
        if key not in self.references:
            _,X=self.topology_inputs(study,hops)
            compare=KernelComparison(X,study.y,self.landmarks,study.seed)
            gamma=1/(X['train'].shape[1]*max(X['train'].var(),1e-12))
            compare.nystrom('Classical kernel',RBFKernel(gamma))
            classical={features:compare.additive('RF + adjacency RBF','Classical kernel',self.raw(study,features,hops))
                       for features in ('Local','All')}
            self.references[key]=(compare,classical)
        return self.references[key]

    def encoding_configuration(self,study,hops,depth,pair,feature_map,axis):
        compare,classical=self.topology_reference(study,hops)
        fm=feature_map.feature_map if isinstance(feature_map,HamiltonianMomentKernel) else feature_map
        kernel=feature_map if isinstance(feature_map,HamiltonianMomentKernel) else StatevectorKernel(fm)
        kind='steps' if 'Hamiltonian' in pair else 'layers' if 'Message' in pair else 'time setting'
        reference=f'{pair} {axis} h={hops} d={depth}'
        compare.nystrom(reference,kernel)
        logical_depth=np.mean([fm.bind(row).depth() for row in compare.X['train'][:8]])
        rows=[]
        for features in ('Local','All'):
            meta={**self.meta('Quantum topology',study.seed,pair,hops,depth,kind,axis),'Features':features}
            rows.append({**classical[features],**meta,'Family':'Classical','Reference':''})
            quantum=compare.additive('RF + '+pair,reference,self.raw(study,features,hops))
            rows.append({**quantum,**meta,'Family':'Quantum','Reference':'RF + adjacency RBF',
                         'Mean logical depth':logical_depth,'Qubits':fm.circuit.num_qubits})
        return rows

    def quantum_topology(self):
        rows=[]
        fixed_hops=4
        for seed,study in self.cohorts.items():
            print('Simulator direct topology radius/circuit sweep:',seed,flush=True)
            for hops in self.hops:
                for features in ('Local','All'):
                    rows.append(self.baseline(study,features,'Quantum topology'))
                    rows.append(self.baseline(study,features,'Quantum topology',hops))
                rows.extend(self.encoding_configuration(study,hops,4,'Hamiltonian fidelity quantum',
                    GraphHamiltonianFeatureMap(self.nodes,steps=4,topology_only=True,field_axis='Y'),'Radius'))
            for steps in self.circuit_depths:
                rows.extend(self.encoding_configuration(study,fixed_hops,steps,'Hamiltonian fidelity quantum',
                    GraphHamiltonianFeatureMap(self.nodes,steps=steps,topology_only=True,field_axis='Y'),'Circuit depth'))
                rows.extend(self.encoding_configuration(study,fixed_hops,steps,'Hamiltonian moments quantum',
                    HamiltonianMomentKernel(GraphHamiltonianFeatureMap(self.nodes,steps=steps,topology_only=True,field_axis='Y')),'Circuit depth'))
            for layers in self.message_layers:
                rows.extend(self.encoding_configuration(study,fixed_hops,layers,'Message quantum',
                    GraphMessageFeatureMap(self.nodes,repetitions=layers,topology_only=True),'Circuit depth'))
            rows.extend(self.encoding_configuration(study,fixed_hops,1,'Hermitian walk quantum',
                HermitianWalkFeatureMap(self.nodes),'Architecture'))
        frame=pd.DataFrame(rows).drop_duplicates(PairedSelection.keys+['Seed'])
        result=self.finish('Quantum topology',frame.to_dict('records'))
        retention=[]
        for seed,study in self.cohorts.items():
            for hops in self.hops:
                for split,samples in self.samples[seed,hops].items():
                    retention.append(GraphRetention.describe(self.graph,study.roots[split],samples,split).assign(Seed=seed))
        self.store.save('04_retention',pd.concat(retention,ignore_index=True))
        self.depth_plot(result)
        return result

    def run(self, stage, reuse_saved=False):
        """Execute a stage, or explicitly reuse completed fits with matching settings."""
        path=self.store.directory/f'{self.stages[stage]}_all.csv'
        current=json.loads(json.dumps(self.configuration))
        stage_config=self.store.directory/f'{self.stages[stage]}_configuration.json'
        saved=json.loads(stage_config.read_text()) if stage_config.exists() else self.saved_configuration
        if reuse_saved and saved==current and path.exists():
            stage_config.write_text(json.dumps(current,indent=2))
            frame=self.results(stage)
            print(f'{stage}: restored completed simulator fits from {path.name}.')
            if stage=='Quantum topology':self.depth_plot(frame)
            return frame
        methods={'Attributes':self.classification,'Propagation':self.propagation,
                 'Topology features':self.topology_features,'Quantum topology':self.quantum_topology}
        result=methods[stage]()
        stage_config.write_text(json.dumps(current,indent=2))
        return result

    def results(self, stage):
        """Load a completed experiment without repeating model fits or QPU jobs."""
        if stage not in self.tables:
            self.tables[stage]=pd.read_csv(self.store.directory/f'{self.stages[stage]}_all.csv',keep_default_na=False)
        return self.tables[stage]

    def heatmaps(self, stage):
        """Show selected scores and validation response surfaces, exporting both matrices.

        Every surface cell represents a single prespecified configuration averaged
        across seeds. No test score participates in selecting or ordering cells.
        """
        frame=self.results(stage)
        selected=PairedSelection.best(frame)
        selected['Label']=selected.apply(lambda r: f'{r.Features}: {r.Model} [{r.Setting}]',axis=1)
        stem=self.stages[stage]
        self.store.save(stem+'_metric_heatmap',selected[['Label']+list(Evaluation.CORE_METRICS)])
        self.store.plot(stem+'_metric_heatmap',ResultPlots.metric_heatmap(
            selected,by=('Label',),title=stage+': validation-selected models, all five test metrics'))
        means=PairedSelection.aggregate(frame)
        means['Variant']=means.apply(lambda r: r.Model+(' / '+r.Pair if r.Pair else '')+
            (' / '+r.Axis if stage=='Quantum topology' else ''),axis=1)
        if stage=='Attributes':
            means['Sweep']=means.Depth.map(lambda n:f'passes={int(n)}' if n else 'raw')
        elif stage=='Quantum topology':
            means['Sweep']=means.apply(lambda r:f'h={int(r.Hops)}, {r["Depth kind"]}={int(r.Depth)}',axis=1)
        else: means['Sweep']=means.Hops.map(lambda n:f'h={int(n)}' if n else 'raw')
        # Identical RBF arms can occur in multiple deliberately matched pairs.
        # Keep the pair in the row label instead of averaging different settings.
        for features in ('Local','All'):
            subset=means[means.Features==features]
            matrix=subset.pivot(index='Variant',columns='Sweep',values='Validation PR-AUC')
            self.store.save(stem+'_'+features.lower()+'_validation_heatmap',matrix.reset_index())
            self.matrix_plot(matrix,stem+'_'+features.lower()+'_validation_heatmap',
                f'{features}: {stage} configuration response (mean validation AP)',focused=True)

    def matrix_plot(self, matrix, name, title, focused=False):
        """Annotated similarities or scores; blank cells mean not evaluated."""
        values=matrix.to_numpy(dtype=float)
        finite=values[np.isfinite(values)]
        lo,hi=(max(0,float(finite.min())-.005),min(1,float(finite.max())+.005)) if focused else (0,1)
        fig,ax=plt.subplots(figsize=(max(7,.7*len(matrix.columns)+3),max(3,.36*len(matrix)+1.7)))
        cmap=plt.get_cmap('viridis').with_extremes(bad='#eeeeee')
        artist=ax.imshow(np.ma.masked_invalid(values),vmin=lo,vmax=hi,cmap=cmap,aspect='auto')
        ax.set_xticks(range(len(matrix.columns)),matrix.columns,rotation=40,ha='right',fontsize=8)
        ax.set_yticks(range(len(matrix)),matrix.index,fontsize=8)
        for i,j in np.ndindex(values.shape):
            v=values[i,j]
            if np.isfinite(v):ax.text(j,i,f'{v:.3f}',ha='center',va='center',fontsize=7,color='black' if v>(lo+hi)/2 else 'white')
        ax.set_title(title,fontsize=10)
        fig.colorbar(artist,ax=ax,label='Validation AP (focused scale)' if focused else 'Similarity',shrink=.8)
        fig.tight_layout();self.store.plot(name,fig)

    def interpretation(self, stage):
        """Report what happened, separating information gains from kernel changes."""
        frame=self.results(stage);best=PairedSelection.best(frame)
        observations=[]
        records=[]
        for features,rows in best.groupby('Features',sort=False):
            c=rows[rows.Selection.str.contains('Best classical')].iloc[0]
            q=rows[rows.Family=='Quantum'].iloc[0]
            match=rows[(rows.Family=='Classical')&rows.Selection.str.contains('Matched',case=False)].iloc[0]
            raw=PairedSelection.aggregate(self.results('Attributes'))
            baseline=raw[(raw.Features==features)&(raw.Model=='Raw RF')].iloc[0]
            change=c['PR-AUC']-baseline['PR-AUC']
            delta=q['PR-AUC']-match['PR-AUC']
            f1_change=c['Illicit F1']-baseline['Illicit F1']
            f1_delta=q['Illicit F1']-match['Illicit F1']
            observations.append(f'**{features}, classical leader by validation AP:** '
                f'test AP {c["PR-AUC"]:.4f} (raw RF {baseline["PR-AUC"]:.4f}; Δ {change:+.4f}); '
                f'illicit F1 **{c["Illicit F1"]:.4f}** (raw RF {baseline["Illicit F1"]:.4f}; Δ {f1_change:+.4f}). '
                f'**Matched classical → quantum:** AP {match["PR-AUC"]:.4f} → {q["PR-AUC"]:.4f} (Δ {delta:+.4f}); '
                f'illicit F1 **{match["Illicit F1"]:.4f} → {q["Illicit F1"]:.4f}** (Δ {f1_delta:+.4f}).')
            for metric in Evaluation.CORE_METRICS:
                records.append({'Stage':stage,'Features':features,'Metric':metric,
                    'Classical minus raw RF':c[metric]-baseline[metric],
                    'Quantum minus matched classical':q[metric]-match[metric]})
        self.store.save(self.stages[stage]+'_interpretation',pd.DataFrame(records))
        display(Markdown('\n\n'.join(observations)+'\n\nThese are held-out scores after validation selection; '
            'read gains by metric and feature set. These configurations were selected by AP, not F1.'))

    def kernel_heatmaps(self, hops=4, count=24):
        """Recover the old kernel view with matched numerical and direct-topology inputs.

        Displays training examples only; computes kernels but does not fit classifiers.
        Label order is identical in each pair. Quantum topology uses no node attributes.
        """
        study=self.cohorts[self.seeds[0]]
        _,X=self.topology_inputs(study,hops)
        inputs=X['train'][:count]
        gamma=1/(X['train'].shape[1]*max(X['train'].var(),1e-12))
        kernels={'Adjacency RBF':RBFKernel(gamma),
            'Direct Hamiltonian quantum':StatevectorKernel(GraphHamiltonianFeatureMap(
                self.nodes,steps=4,topology_only=True,field_axis='Y'))}
        # Preserve the older summary-feature quantum-kernel view as well.
        samples=self.samples[study.seed,hops]['train']
        values=np.array([GraphDescriptors.spectral(a) for a in samples])
        angles=AnglePreprocessor().fit_transform(values)
        summary_gamma=1/(angles.shape[1]*max(angles.var(),1e-12))
        cases=[(name,kernel,inputs) for name,kernel in kernels.items()]
        cases.extend([('Spectral summaries RBF',RBFKernel(summary_gamma),angles[:count]),
            ('Spectral summaries quantum',StatevectorKernel(QuantumFeatureMap.uploading(angles.shape[1],4,2)),angles[:count])])
        for name,kernel,Z in cases:
            matrix=pd.DataFrame(kernel.evaluate(Z))
            slug=name.lower().replace(' ','_')
            self.store.save('04_kernel_'+slug,matrix.rename_axis('Training row').reset_index())
            self.matrix_plot(matrix,'04_kernel_'+slug,name+': training similarities (same roots)')
        roots=study.roots['train'][:len(inputs)]
        self.store.save('04_kernel_example_roots',pd.DataFrame({'Training row':range(len(inputs)),
            'Time step':[int(t) for t,i in roots],'Period row':[int(i) for t,i in roots],
            'Label':study.y['train'][:len(inputs)]}))

    def best(self,stage):
        frame=self.tables.get(stage)
        if frame is None:frame=pd.read_csv(self.store.directory/f'{self.stages[stage]}_all.csv',keep_default_na=False)
        selected=PairedSelection.best(frame)
        self.store.save(self.stages[stage]+'_best_full',selected)
        self.store.show(self.stages[stage]+'_best',selected,['Features','Selection','Model','Setting'])
        return selected

    def depth_plot(self,frame):
        """Full simulator sweeps are plotted, never printed as long tables."""
        means=PairedSelection.aggregate(frame[frame.Axis=='Circuit depth'])
        fig,axes=plt.subplots(2,3,figsize=(14,7),squeeze=False)
        for row,features in enumerate(('Local','All')):
            for col,pair in enumerate(('Hamiltonian fidelity quantum','Hamiltonian moments quantum','Message quantum')):
                ax=axes[row,col];values=[];errors=[]
                subset=means[(means.Features==features)&(means.Pair==pair)]
                for family,points in subset.groupby('Family',sort=False):
                    points=points.sort_values('Depth')
                    ax.errorbar(points.Depth,points['Validation PR-AUC'],marker='o',label=family)
                    values.extend(points['Validation PR-AUC']);errors.extend(np.zeros(len(points)))
                quantum=subset[subset.Family=='Quantum']
                best=quantum.loc[quantum['Validation PR-AUC'].idxmax()]
                ax.scatter([best.Depth],[best['Validation PR-AUC']],marker='*',s=110,color='black',zorder=5)
                ax.set(title=features+': '+pair.replace(' quantum',''),xlabel=best['Depth kind'],ylabel='Mean validation AP')
                ResultPlots.relevant_ylim(ax,values,errors);ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=8)
        fig.suptitle(f'Simulator depth sweep: fixed 4-hop radius / {self.nodes}-node budget; ★ = selected quantum depth')
        fig.tight_layout()
        self.store.plot('04_simulator_depth_sweep',fig)
        self.store.save('04_simulator_depth_sweep_selected',PairedSelection.best(frame[frame.Axis=='Circuit depth']))

    def structural_check(self):
        study=StructuralProbeStudy();results=study.run(trials=6,seed=self.seeds[0],field_axis='Y')
        self.store.save('04_structural_probes_all',results)
        self.store.save('04_evolution_convergence',study.convergence)
        case=results[results.Case=='Degree + triangle collision (8 nodes)']
        names=['Classical summary RBF','Quantum summary rotations','Classical spectral RBF',
               'Classical rooted WL','Quantum Hamiltonian invariant moments (product)']
        selected=case[case.Encoder.isin(names)][['Encoder','Different graph similarity','Isomorphic mean similarity','Mean relabeling error']]
        self.store.save('04_structural_check',selected)
        display(selected.round(4))

    def hardware(self,enabled=True):
        experiment=HamiltonianHardwareExperiment(self.root)
        manifest=json.loads(experiment.manifest_path.read_text()) if experiment.manifest_path.exists() else None
        if enabled and (manifest is None or manifest['phase']!='complete'):
            from qiskit_ibm_runtime import QiskitRuntimeService
            service=QiskitRuntimeService()
            manifest=experiment.prepare(pilot_path=self.store.directory/f'cohort_{self.seeds[0]}.npz')
            manifest=experiment.submit(service,backend_name='ibm_fez',max_execution_time=180)
            manifest=experiment.collect(service)
        if manifest is None:
            print('Final topology hardware is disabled and no recorded run exists.');return
        jobs=pd.DataFrame(manifest['jobs'])
        self.store.save('05_hardware_jobs',jobs)
        if manifest['phase']!='complete':
            print('Physical jobs submitted/running; call this method to recover the same IDs.');return
        experiment.analyze(manifest)
        models=pd.read_csv(experiment.output/'ibm_final_classification.csv')
        self.store.show('05_hardware_best',models,['Model','Execution'])
        self.store.save('05_hardware_classification_all',models)
        self.store.save('05_hardware_observables',pd.read_csv(experiment.output/'ibm_final_observables.csv'))
        resources=experiment.tables();self.store.save('05_hardware_resources',resources)
        kernels=np.load(experiment.output/'ibm_final_kernels.npz')
        self.store.save('05_hardware_structural_kernel',pd.DataFrame([
            {'Row':i,'Column':j,'Ideal':kernels['probe_ideal'][i,j],'Measured':kernels['probe_measured'][i,j]}
            for i in range(4) for j in range(4)]))
        for key,label in [('probe_ideal','Ideal'),('probe_measured','IBM measured')]:
            self.matrix_plot(pd.DataFrame(kernels[key]),'05_hardware_'+key,
                label+': structural graph-pair fidelity kernel')
        seconds=resources['Charged QPU seconds'].sum()
        mae=np.abs(kernels['probe_ideal']-kernels['probe_measured']).mean()
        display(Markdown(f'**Final topology hardware enabled:** recovered two completed jobs on **{manifest["backend"]}** '
            f'({seconds:.0f} charged QPU seconds); no duplicate submissions. The 12-case balanced test set is separate '
            f'from the simulator cohorts. Structural kernel MAE **{mae:.3f}** indicates strong noise; AP alone is not evidence of a quantum gain.'))
        _,large=experiment.figures()
        plt.close(_)
        self.store.plot('05_hardware_40_node_observables',large)
        (self.store.directory/'hardware_manifest.json').write_text(json.dumps(manifest,indent=2))

    def ordering_heatmap(self):
        """Keep the archived ordering-robustness heatmap, explicitly separate from this pilot."""
        names=('permutation_mixture.csv','permuted_test_scores.csv')
        frames=[]
        for name in names:
            path=self.root/'experiment_results'/name
            if path.exists():
                frame=pd.read_csv(path);frame['Source']=name
                frames.append(frame)
        if not frames:
            print('No archived ordering results found; rerun the archive to produce this diagnostic.');return
        frame=pd.concat(frames,ignore_index=True)
        self.store.save('04_archived_ordering_results',frame)
        scores=frame.groupby(['Source','Model'],sort=False)[list(Evaluation.CORE_METRICS)].mean().reset_index()
        self.store.save('04_archived_ordering_heatmap',scores)
        self.store.plot('04_archived_ordering_heatmap',ResultPlots.metric_heatmap(scores,by=('Source','Model'),
            title='Archived ordering robustness: separate cohorts, five test metrics'))

    def classical_progression(self):
        """Track the information question separately from matched quantum kernel deltas."""
        records=[]
        for stage in self.stages:
            selected=PairedSelection.best(self.results(stage))
            records.extend(selected[selected.Selection.str.contains('Best classical')].to_dict('records'))
        table=pd.DataFrame(records)
        self.store.save('06_classical_progression',table)
        raw=PairedSelection.aggregate(self.results('Attributes'))
        fig,axes=plt.subplots(2,3,figsize=(14,7))
        stages=list(self.stages)
        for ax,metric in zip(axes.flat,Evaluation.CORE_METRICS):
            values=[]
            for features,color in [('Local','C0'),('All','C1')]:
                points=table[table.Features==features].set_index('Stage').loc[stages]
                baseline=raw[(raw.Features==features)&(raw.Model=='Raw RF')].iloc[0][metric]
                ax.plot(range(4),points[metric],marker='o',color=color,label=features+' selected classical')
                ax.axhline(baseline,color=color,linestyle=':',alpha=.7,label=features+' raw RF')
                values.extend(points[metric]);values.append(baseline)
            ax.set_xticks(range(4),['Attributes','Propagation','Topology stats','Topology circuit inputs'],rotation=20,ha='right',fontsize=8)
            ax.set(title=metric,ylabel='Mean test score');ax.grid(alpha=.2)
            ResultPlots.relevant_ylim(ax,values)
        axes.flat[-1].set_visible(False);axes.flat[0].legend(fontsize=7)
        fig.suptitle('Information added: validation-selected classical models versus raw RF references')
        fig.tight_layout();self.store.plot('06_classical_progression',fig)

    def summary(self):
        self.classical_progression()
        selected=[];deltas=[]
        for stage,stem in self.stages.items():
            best=pd.read_csv(self.store.directory/f'{stem}_best_full.csv',keep_default_na=False)
            for features,rows in best.groupby('Features',sort=False):
                q=rows[rows.Family=='Quantum'].iloc[0]
                c=rows[(rows.Family=='Classical')&rows.Selection.str.contains('Matched',case=False)].iloc[0]
                selected.extend([c.to_dict(),q.to_dict()])
                pairs=pd.read_csv(self.store.directory/f'{stem}_pairs.csv',keep_default_na=False)
                subset=pairs[(pairs.Quantum==q.Model)&(pairs.Features==features)]
                for key in PairedSelection.context:subset=subset[subset[key]==q[key]]
                deltas.extend(subset.to_dict('records'))
        table=pd.DataFrame(selected);paired=pd.DataFrame(deltas)
        self.store.show('06_summary_best_pairs',table,['Stage','Features','Family','Model','Setting'])
        self.store.save('06_summary_paired_deltas',paired)
        heat=table.copy()
        heat['Label']=heat.apply(lambda r:f'{r.Stage} / {r.Features} / {r.Family}',axis=1)
        self.store.save('06_summary_metric_heatmap',heat[['Label']+list(Evaluation.CORE_METRICS)])
        self.store.plot('06_summary_metric_heatmap',ResultPlots.metric_heatmap(heat,by=('Label',),
            title='Final selected quantum and matched classical results: five test metrics'))
        metrics=list(Evaluation.CORE_METRICS)
        fig,axes=plt.subplots(2,3,figsize=(14,7))
        stage_names=list(self.stages)
        for ax,metric in zip(axes.flat,metrics):
            values=[];errors=[]
            for features in ('Local','All'):
                means=paired[paired.Features==features].groupby('Stage')['Δ '+metric].agg(['mean','std']).loc[stage_names]
                sd=means['std'].fillna(0)
                ax.errorbar(range(4),means['mean'],yerr=sd,marker='o',capsize=3,label=features)
                values.extend(means['mean']);errors.extend(sd)
            ax.axhline(0,color='gray',linestyle='--',linewidth=1)
            ax.set_xticks(range(4),['Attributes','Propagation','Topology stats','Topology circuit'],rotation=20,ha='right',fontsize=8)
            ax.set(title='Δ '+metric,ylabel='Quantum − matched classical')
            ResultPlots.relevant_ylim(ax,values,errors,include_zero=True,bounded=False);ax.grid(alpha=.2)
        axes.flat[0].legend(fontsize=8);axes.flat[-1].set_visible(False)
        fig.suptitle('Validation-selected pairs on the same simulator roots; bars = sampling SD')
        fig.tight_layout();self.store.plot('06_experiment_summary',fig)
        index=pd.DataFrame({'CSV':[p.name for p in sorted(self.store.directory.glob('*.csv')) if p.name!='result_index.csv']})
        self.store.save('result_index',index)
        display(Markdown('**Interpretation:** graph information, kernel choice and direct quantum topology are separate changes. '
            'The classical rows above are matched counterparts, not necessarily the classical validation-AP leaders. '
            'Read F1 alongside AP: their changes can differ. Use paired differences to assess representation value. '
            'The final QPU experiment establishes execution and noise/cost evidence; '
            'its small balanced classification scores do not establish quantum advantage. All full tables are in `results/experiment_flow/`.'))

