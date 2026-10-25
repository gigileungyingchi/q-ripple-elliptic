"""Recoverable small-kernel and larger observable IBM Hamiltonian experiments."""
from dataclasses import replace
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from qiskit.quantum_info import Statevector
from qiskit.transpiler import generate_preset_pass_manager
from sklearn.metrics.pairwise import rbf_kernel

from .data import EllipticData, StratifiedSampler
from .graph import RiskGraph
from .topology import RootedSubgraphSampler, TopologyPreprocessor, TopologyDiagnostics
from .topology_research import StructuralProbeStudy
from .topology_encoders import GraphHamiltonianFeatureMap
from .additive import FullGraphInputs, AdditiveKernelClassifier
from .quantum import StatevectorKernel, NystromProjector
from .metrics import Evaluation


class HamiltonianHardwareExperiment:
    """Submit once, save IDs immediately, collect without resubmitting.

    Small balanced classifiers share roots, topology-only quantum inputs, raw
    Local 93+propagation bypass, landmark indices and RF settings. Larger runs
    measure Z observables, not exponentially small all-zero overlaps. They test
    hardware execution/cost, not classification or classical intractability.
    """
    def __init__(self, root):
        self.root=Path(root)
        self.output=self.root/'experiment_results'
        self.manifest_path=self.output/'ibm_final_manifest.json'
        self.input_path=self.output/'ibm_final_inputs.npz'

    @staticmethod
    def feature_map(nodes, edges='directed'):
        return GraphHamiltonianFeatureMap(nodes,time=.8,steps=4,field=.37,coupling=.7,
                                           edges=edges,topology_only=True,field_axis='Y')

    def prepare(self, small_nodes=8, large_nodes=40, small_shots=1024, large_shots=2048, seed=17, pilot_path=None):
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text())
        data=EllipticData.load(self.root/'data')
        graph=RiskGraph(data.transactions,data.edges,seed_fraction=.2,random_state=42)
        pilot=np.load(Path(pilot_path) if pilot_path is not None else self.output/'encoding_roots_local.npz')
        roots,y,selected={},{},{}
        for split,size in [('train',12),('val',8),('test',12)]:
            selected[split]=StratifiedSampler.balanced_indices(pilot[f'y_{split}'],size//2,seed)
            roots[split]=pilot[f'roots_{split}'][selected[split]]
            y[split]=pilot[f'y_{split}'][selected[split]]
            if len(np.unique(y[split])) != 2:raise ValueError('Each hardware split requires both labels')
        full=FullGraphInputs(graph,roots,y,feature_set='Local')
        raw=full.raw(4)
        sampler=RootedSubgraphSampler(graph,small_nodes,EllipticData.feature_columns('Local'))
        samples={s:[replace(a,attributes=np.zeros((small_nodes,2)))
                    for a in sampler.collect(roots[s],4)] for s in roots}
        prep=TopologyPreprocessor()
        X={'train':prep.fit_transform(samples['train'])}
        X.update({s:prep.transform(samples[s]) for s in ('val','test')})
        landmarks=StratifiedSampler.landmarks(y['train'],4,seed)
        small_map=self.feature_map(small_nodes)
        indices={}; cursor=0
        pair_entries=[]
        for i in range(len(landmarks)):
            for j in range(i,len(landmarks)):
                pair_entries.append({'kind':'W','i':i,'j':j})
        for split in ('train','val','test'):
            for i in range(len(X[split])):
                for j in range(len(landmarks)):
                    pair_entries.append({'kind':split,'i':i,'j':j})
        if small_nodes != 8:raise ValueError('The structural collision probe requires eight nodes')
        probe_samples=StructuralProbeStudy.cases()['Degree + triangle collision (8 nodes)']
        probe=TopologyPreprocessor().fit_transform(probe_samples)
        probe=np.vstack([probe,TopologyDiagnostics.permute_inputs(probe,8,seed)])
        for i in range(4):
            for j in range(i,4):pair_entries.append({'kind':'probe','i':i,'j':j})
        large_sampler=RootedSubgraphSampler(graph,large_nodes,EllipticData.feature_columns('Local'))
        large_samples,large_roots=[],[]
        for t,root in pilot['roots_train']:
            candidate=large_sampler.sample(int(t),int(root),hops=6,policy='hubs')
            if candidate.mask.sum()==large_nodes:
                large_samples.append(replace(candidate,attributes=np.zeros((large_nodes,2))))
                large_roots.append([int(t),int(root)])
            if len(large_samples)==2:break
        if len(large_samples)<2:raise ValueError('Cannot retain requested large graph size from training roots')
        large_X=TopologyPreprocessor().fit_transform(large_samples)
        np.savez_compressed(self.input_path,
            **{f'X_{s}':X[s] for s in X}, **{f'raw_{s}':raw[s] for s in raw},
            **{f'y_{s}':y[s] for s in y}, **{f'roots_{s}':roots[s] for s in roots},
            **{f'indices_{s}':selected[s] for s in selected}, landmarks=landmarks,
            probe=probe,large_X=large_X,large_roots=np.array(large_roots))
        manifest={'phase':'prepared','small_nodes':small_nodes,'large_nodes':large_nodes,
            'small_shots':small_shots,'large_shots':large_shots,'seed':seed,'landmarks':4,
            'field_axis':'Y','time':.8,'steps':4,'hops':4,'large_hops':6,
            'quantum_inputs':'adjacency, root and known seed flags; numerical attributes excluded',
            'raw_bypass':'unchanged Local 93 + full-period legacy/structural/directional propagation',
            'target_sizes':{s:len(y[s]) for s in y},'balanced_subsets':True,
            'kernel_entries':pair_entries,'large_entries':[
                {'graph':i,'edge_mode':mode} for i in range(2) for mode in ('directed','reverse','none')],
            'large_edges':[int(np.count_nonzero(s.adjacency)) for s in large_samples],
            'large_reachable_nodes':[s.reachable_nodes for s in large_samples],
            'generic_statevector_GiB':16*(2**large_nodes)/(2**30),
            'small_circuits':len(pair_entries),'large_circuits':6,
            'requested_shots':len(pair_entries)*small_shots+6*large_shots,'jobs':[]}
        self._save(manifest)
        return manifest

    def _save(self,manifest):
        self.manifest_path.write_text(json.dumps(manifest,indent=2,default=str))

    def circuits(self,manifest):
        z=np.load(self.input_path)
        fm=self.feature_map(manifest['small_nodes'])
        references=z['X_train'][z['landmarks']]
        small=[]
        for entry in manifest['kernel_entries']:
            kind,i,j=entry['kind'],entry['i'],entry['j']
            x,y=(references[i],references[j]) if kind=='W' else (
                (z['probe'][i],z['probe'][j]) if kind=='probe' else (z['X_'+kind][i],references[j]))
            circuit=fm.bind(x).compose(fm.bind(y).inverse());circuit.measure_all();small.append(circuit)
        large=[]
        for entry in manifest['large_entries']:
            fm=self.feature_map(manifest['large_nodes'],entry['edge_mode'])
            circuit=fm.bind(z['large_X'][entry['graph']]);circuit.measure_all();large.append(circuit)
        return {'small':small,'large':large}

    def submit(self,service,backend_name=None,max_execution_time=180):
        """Only missing groups are submitted; an existing job ID is never replaced."""
        from qiskit_ibm_runtime import SamplerV2
        manifest=json.loads(self.manifest_path.read_text())
        if len(manifest['jobs'])==2:return manifest
        backend=service.backend(backend_name) if backend_name else service.least_busy(
            operational=True,simulator=False,min_num_qubits=manifest['large_nodes'])
        if backend.configuration().simulator:raise ValueError('A physical IBM backend is required')
        manifest['backend']=backend.name
        manifest['backend_qubits']=backend.num_qubits
        self._save(manifest)
        groups=self.circuits(manifest)
        manager=generate_preset_pass_manager(backend=backend,optimization_level=2,seed_transpiler=17)
        sampler=SamplerV2(mode=backend,options={'max_execution_time':max_execution_time,
            'environment':{'job_tags':['q-ripple','hamiltonian-topology']}})
        done={job['group'] for job in manifest['jobs']}
        for group,circuits in groups.items():
            if group in done:continue
            print(f'Transpiling {group}: {len(circuits)} circuits on {backend.name}',flush=True)
            started=time.perf_counter();isa=manager.run(circuits)
            resources={'transpile_seconds':time.perf_counter()-started,
                'transpiled_depth_max':max(c.depth() for c in isa),
                'two_qubit_gates_max':max(sum(i.operation.num_qubits==2 for i in c.data) for c in isa),
                'two_qubit_gates_mean':float(np.mean([sum(i.operation.num_qubits==2 for i in c.data) for c in isa]))}
            shots=manifest[group+'_shots']
            job=sampler.run(isa,shots=shots)
            record={'group':group,'job_id':job.job_id(),'status':'SUBMITTED','circuits':len(isa),
                'shots':shots,'max_execution_time':max_execution_time,**resources}
            manifest['jobs'].append(record);manifest['phase']='submitted';self._save(manifest)
            print(json.dumps(record),flush=True)
        return manifest

    @staticmethod
    def observables(counts,nodes):
        shots=sum(counts.values())
        values=[];weights=[]
        for bitstring,count in counts.items():
            bits=np.array([int(b) for b in bitstring.replace(' ','')[::-1]])[:nodes]
            spins=1-2*bits
            m=spins.mean()
            values.append([m,m*m,spins[0]])
            weights.append(count)
        a=np.array(values);weights=np.array(weights)
        means=np.average(a,axis=0,weights=weights)
        se=np.sqrt(np.maximum(np.average((a-means)**2,axis=0,weights=weights),0)/shots)
        return dict(zip(['Magnetization','Magnetization squared','Root Z'],means)) | dict(zip(
            ['SE magnetization','SE magnetization squared','SE root Z'],se))

    def collect(self,service):
        manifest=json.loads(self.manifest_path.read_text())
        for record in manifest['jobs']:
            job=service.job(record['job_id']);record['status']=job.status()
            if record['status']=='DONE':
                counts_path=self.output/f'ibm_final_{record["group"]}_counts.json'
                if not counts_path.exists():
                    result=job.result()
                    counts=[pub.data.meas.get_counts() for pub in result]
                    counts_path.write_text(json.dumps(counts))
                record['metrics']=job.metrics()
            elif record['status'] in ('ERROR','CANCELLED'):
                record['error']=job.error_message()
        self._save(manifest)
        if len(manifest['jobs'])==2 and all(r['status']=='DONE' for r in manifest['jobs']):
            self.analyze(manifest);manifest['phase']='complete';self._save(manifest)
        return manifest

    def analyze(self,manifest=None):
        manifest=manifest or json.loads(self.manifest_path.read_text())
        z=np.load(self.input_path);fm=self.feature_map(manifest['small_nodes'])
        refs=z['X_train'][z['landmarks']]
        ideal_kernel=StatevectorKernel(fm)
        W=np.empty((len(refs),len(refs)));probe=np.empty((4,4))
        blocks={s:np.empty((len(z['X_'+s]),len(refs))) for s in ('train','val','test')}
        counts=json.loads((self.output/'ibm_final_small_counts.json').read_text())
        for entry,histogram in zip(manifest['kernel_entries'],counts):
            value=histogram.get('0'*manifest['small_nodes'],0)/sum(histogram.values())
            i,j=entry['i'],entry['j']
            if entry['kind']=='W':W[i,j]=W[j,i]=value
            elif entry['kind']=='probe':probe[i,j]=probe[j,i]=value
            else:blocks[entry['kind']][i,j]=value
        ideal_probe=ideal_kernel.evaluate(z['probe'])
        np.savez_compressed(self.output/'ibm_final_kernels.npz',W_measured=W,
            probe_ideal=ideal_probe,probe_measured=probe,
            **{f'C_{s}':a for s,a in blocks.items()})
        raw={s:z['raw_'+s] for s in blocks};y={s:z['y_'+s] for s in blocks}
        gamma=1/(z['X_train'].shape[1]*max(z['X_train'].var(),1e-12))
        projected={}
        for name,kernel in [('Classical topology RBF',None),('Ideal Hamiltonian',ideal_kernel)]:
            matrix=rbf_kernel(refs,gamma=gamma) if kernel is None else kernel.evaluate(refs)
            projector=NystromProjector().fit(matrix)
            projected[name]={s:projector.transform(rbf_kernel(z['X_'+s],refs,gamma=gamma)
                if kernel is None else kernel.evaluate(z['X_'+s],refs)) for s in blocks}
        projector=NystromProjector().fit(W)
        projected['IBM Hamiltonian']={s:projector.transform(blocks[s]) for s in blocks}
        # A true raw+propagation reference is always included.
        projected['Raw + propagation RF']={s:np.empty((len(y[s]),0)) for s in y}
        rows=[]
        for name,phi in projected.items():
            model=AdditiveKernelClassifier(None,seed=manifest['seed']).fit_readout(raw['train'],phi['train'],y['train'])
            joined={s:model.join(raw[s],phi[s]) for s in blocks}
            val=model.model_.predict_proba(joined['val'])[:,1]
            score=model.model_.predict_proba(joined['test'])[:,1]
            rows.append({'Model':name,'Backend':manifest['backend'],'Execution':'Physical' if name.startswith('IBM') else 'Local',
                'Train targets':len(y['train']),'Validation targets':len(y['val']),'Test targets':len(y['test']),
                'Balanced subsets':True,'Raw columns retained':raw['train'].shape[1],
                'Validation PR-AUC':Evaluation.classification(y['val'],model.model_.predict(joined['val']),val)['PR-AUC'],
                **Evaluation.classification(y['test'],model.model_.predict(joined['test']),score)})
        pd.DataFrame(rows).to_csv(self.output/'ibm_final_classification.csv',index=False)
        large_counts=json.loads((self.output/'ibm_final_large_counts.json').read_text())
        observables=[]
        for entry,histogram in zip(manifest['large_entries'],large_counts):
            row={'Graph':entry['graph'],'Edge mode':entry['edge_mode'],
                'Active nodes':manifest['large_nodes'],'Backend':manifest['backend'],
                'Shots':sum(histogram.values()),**self.observables(histogram,manifest['large_nodes'])}
            if entry['edge_mode']=='none':
                # Independent one-qubit evolution is a feasible analytic check
                # at 40 nodes; no 2**40 statevector is constructed.
                from qiskit import QuantumCircuit
                from .topology import TopologyFeatureMap
                _,root,seeds,_,_=TopologyFeatureMap.unpack(z['large_X'][entry['graph']],manifest['large_nodes'])
                expectations=[]
                for node in range(manifest['large_nodes']):
                    circuit=QuantumCircuit(1);circuit.h(0)
                    circuit.ry(float(.2+.7*root[node]+1.1*seeds[node]),0)
                    circuit.rz(float(.4*root[node]),0)
                    circuit.ry(2*.37*manifest['time'],0)
                    state=Statevector.from_instruction(circuit).data
                    expectations.append(float(abs(state[0])**2-abs(state[1])**2))
                expectations=np.array(expectations);n=manifest['large_nodes']
                row['Analytic Magnetization']=expectations.mean()
                row['Analytic Magnetization squared']=(expectations.sum()**2+np.sum(1-expectations**2))/(n*n)
                row['Analytic Root Z']=expectations[0]
            observables.append(row)
        pd.DataFrame(observables).to_csv(self.output/'ibm_final_observables.csv',index=False)
        return {'structural_kernel_mae':float(np.abs(probe-ideal_probe).mean())}

    def tables(self):
        manifest=json.loads(self.manifest_path.read_text())
        resources=[]
        for record in manifest['jobs']:
            usage=record.get('metrics',{}).get('usage',{})
            resources.append({'Group':record['group'],'Job ID':record['job_id'],
                'Status':record['status'],'Backend':manifest['backend'],
                'Logical qubits':manifest['small_nodes'] if record['group']=='small' else manifest['large_nodes'],
                'Circuits':record['circuits'],'Shots per circuit':record['shots'],
                'Maximum ISA depth':record['transpiled_depth_max'],
                'Maximum two-qubit gates':record['two_qubit_gates_max'],
                'Transpile seconds':record['transpile_seconds'],
                'Charged QPU seconds':usage.get('qpu_charge_time_seconds',usage.get('quantum_seconds',np.nan))})
        return pd.DataFrame(resources)

    def figures(self):
        import matplotlib.pyplot as plt
        from .reporting import ResultPlots
        rows=pd.read_csv(self.output/'ibm_final_classification.csv')
        fig,axes=plt.subplots(2,3,figsize=(14,8))
        for ax,metric in zip(axes.flat,Evaluation.CORE_METRICS):
            ax.plot(range(len(rows)),rows[metric],'o')
            ax.set_xticks(range(len(rows)),rows.Model.str.replace('Hamiltonian','Ham.',regex=False),rotation=25,ha='right',fontsize=8)
            ax.set(title=metric,ylabel='Balanced hardware-subset score')
            ResultPlots.relevant_ylim(ax,rows[metric]);ax.grid(alpha=.2)
        kernels=np.load(self.output/'ibm_final_kernels.npz')
        a,b=kernels['probe_ideal'],kernels['probe_measured']
        ax=axes.flat[-1];ax.scatter(a.ravel(),b.ravel(),s=22)
        ax.plot([0,1],[0,1],'--',color='gray',linewidth=1)
        ax.set(xlabel='Ideal structural fidelity',ylabel='IBM structural fidelity',
               title=f'Structural kernel MAE {np.abs(a-b).mean():.3f}')
        ResultPlots.relevant_ylim(ax,b.ravel())
        ax.grid(alpha=.2)
        fig.suptitle('Small matched pilot: raw Local 93 + propagation retained')
        fig.tight_layout()
        observable=pd.read_csv(self.output/'ibm_final_observables.csv')
        large_fig,axes=plt.subplots(1,3,figsize=(13,4))
        for ax,metric,se in zip(axes,['Magnetization','Magnetization squared','Root Z'],
                                ['SE magnetization','SE magnetization squared','SE root Z']):
            values,errors=[],[]
            for graph,frame in observable.groupby('Graph'):
                frame=frame.set_index('Edge mode').loc[['directed','reverse','none']]
                ax.errorbar(range(3),frame[metric],yerr=frame[se],marker='o',capsize=3,label=f'Graph {graph}')
                values.extend(frame[metric]);errors.extend(frame[se])
                analytic='Analytic '+metric
                if analytic in frame and np.isfinite(frame.loc['none',analytic]):
                    ax.scatter([2],[frame.loc['none',analytic]],marker='x',s=70,color='black')
                    values.append(frame.loc['none',analytic]);errors.append(0.)
            ax.set_xticks(range(3),['Directed','Reversed','Edgeless'])
            ax.set(title=metric,ylabel='Measured observable ± shot SE')
            ResultPlots.relevant_ylim(ax,values,errors,bounded=False);ax.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        large_fig.suptitle('40 active graph qubits: observables; × = analytic edgeless reference')
        large_fig.tight_layout()
        return fig,large_fig
