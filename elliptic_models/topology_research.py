"""Structural probes and matched Elliptic topology-encoding experiments."""
from collections import deque

import networkx as nx
import numpy as np
import pandas as pd
from qiskit.quantum_info import Statevector

from .additive import FullGraphInputs
from .data import AnglePreprocessor, StratifiedSampler
from .experiments import RBFKernel
from .graph import WeisfeilerLehmanKernel, RiskGraph
from .graph_studies import KernelComparison
from .quantum import QuantumFeatureMap, StatevectorKernel
from .topology import (TopologySample, TopologyPreprocessor, TopologyFeatureMap,
                       TopologyDiagnostics, RootedSubgraphSampler, PermutationAveragedKernel)
from .topology_encoders import (GraphHamiltonianFeatureMap, GraphMessageFeatureMap,
                                HermitianWalkFeatureMap, HamiltonianMomentKernel)


class GraphDescriptors:
    """Explicitly limited summary and a stronger permutation-invariant control."""
    @staticmethod
    def summary(sample):
        n = int(sample.mask.sum())
        A = sample.adjacency[:n, :n]
        U = ((A + A.T) > 0).astype(float)
        indegree, outdegree = A.sum(0), A.sum(1)
        seed = sample.seeds[:n]
        return np.array([n, A.sum(), indegree.mean(), indegree.std(),
            outdegree.mean(), outdegree.std(), U.sum() / max(n*(n-1), 1),
            np.trace(U @ U @ U) / 6, (U @ U).sum(), indegree[0], outdegree[0],
            (A.T @ seed)[0], (A @ seed)[0], (A.T @ A.T @ seed)[0]], dtype=float)

    @staticmethod
    def spectral(sample):
        """Sorted spectra plus rooted walk returns; not a complete graph invariant."""
        A = sample.adjacency
        U = ((A + A.T) > 0).astype(float)
        power, returns = np.eye(len(A)), []
        for _ in range(6):
            power = power @ A
            returns.append(power[0, 0])
        return np.r_[GraphDescriptors.summary(sample), np.linalg.eigvalsh(U),
                     np.linalg.svd(A, compute_uv=False), returns]

    @staticmethod
    def topology_row(row, q):
        mask, root, seeds, attrs, A = TopologyFeatureMap.unpack(row, q)
        return np.r_[mask, root, seeds, np.zeros_like(attrs).ravel(), A.ravel()]


class StructuralProbeStudy:
    """Probe collisions in a chosen summary, not all possible classical methods.

    No illicit labels: fidelity, invariant summary collisions, relabeling error
    and product-formula error are the appropriate outcomes, not classification
    accuracy. Every permutation carries adjacency, attributes, root and seeds.
    """
    @staticmethod
    def cases():
        graphs = [
            ('Degree collision (6 nodes)', nx.circular_ladder_graph(3), nx.complete_bipartite_graph(3, 3)),
            ('Degree + triangle collision (8 nodes)', nx.cubical_graph(),
             nx.Graph(list(nx.cycle_graph(8).edges()) + [(i, i+4) for i in range(4)])),
        ]
        result = {}
        for label, a, b in graphs:
            samples = []
            for graph in (a, b):
                A = nx.to_numpy_array(graph, nodelist=sorted(graph))
                n = len(A)
                samples.append(TopologySample(A, np.zeros((n, 2)), np.zeros(n),
                    np.ones(n), np.arange(n), n, 4))
            result[label] = samples
        return result

    def run(self, trials=12, seed=42, field_axis='X'):
        rows, convergence = [], []
        self.samples = self.cases()
        for case, samples in self.samples.items():
            q = len(samples[0].mask)
            X = TopologyPreprocessor().fit_transform(samples)
            summary = np.array([GraphDescriptors.summary(s) for s in samples])
            spectral = np.array([GraphDescriptors.spectral(s) for s in samples])
            # Normalizing vectors here is deterministic, not fitted to illicit labels.
            summary_angles = 2*np.arctan(summary)
            spectral_angles = 2*np.arctan(spectral)
            graphs = [s.to_graph() for s in samples]
            wl = WeisfeilerLehmanKernel(n_iter=3, directed=True)
            wl_original = wl.fit_transform(graphs)[0, 1]
            maps = {
                'Quantum summary rotations': QuantumFeatureMap.uploading(summary.shape[1], 4),
                'Quantum graph message': GraphMessageFeatureMap(q),
                'Quantum generic ring': GraphMessageFeatureMap(q, edges='generic'),
                'Quantum Hamiltonian': GraphHamiltonianFeatureMap(q, topology_only=True, field_axis=field_axis),
                'Quantum Hermitian walk': HermitianWalkFeatureMap(q),
            }
            kernels = {
                'Classical summary RBF': (RBFKernel(1/summary.shape[1]), summary_angles),
                'Classical spectral RBF': (RBFKernel(1/spectral.shape[1]), spectral_angles),
                **{name: (StatevectorKernel(fm), summary_angles if name == 'Quantum summary rotations' else X)
                   for name, fm in maps.items()},
                'Quantum Hamiltonian invariant moments (exact)': (HamiltonianMomentKernel(maps['Quantum Hamiltonian'], exact=True), X),
                'Quantum Hamiltonian invariant moments (product)': (HamiltonianMomentKernel(maps['Quantum Hamiltonian']), X),
                'Quantum Hamiltonian mixture (3)': (
                    PermutationAveragedKernel(maps['Quantum Hamiltonian'], trials=3, seed=seed), X),
            }
            for name, (kernel, values) in kernels.items():
                different = float(kernel.evaluate(values[:1], values[1:])[0, 0])
                isomorphic = []
                for trial in range(trials):
                    permuted = TopologyDiagnostics.permute_inputs(X, q, seed+trial)
                    inputs = values if 'summary' in name.lower() or 'spectral' in name.lower() else permuted
                    # Both graphs contribute relabeling fidelities, paired to themselves.
                    overlaps = kernel.evaluate(values, inputs)
                    isomorphic.extend(np.diag(overlaps).tolist())
                rows.append({'Case': case, 'Encoder': name, 'Node budget': q,
                    'Different graph similarity': different,
                    'Isomorphic mean similarity': np.mean(isomorphic),
                    'Isomorphic minimum similarity': np.min(isomorphic),
                    'Mean relabeling error': 1-np.mean(isomorphic),
                    'Separation gap': np.mean(isomorphic)-different,
                    'Relabeling comparisons': len(isomorphic),
                    'Summary vector distance': np.linalg.norm(summary[0]-summary[1])})
            wl_isomorphic = []
            for trial in range(trials):
                permuted = TopologyDiagnostics.permute_inputs(X, q, seed+trial)
                relabeled_graphs = []
                for row in permuted:
                    mask, root, seeds, attrs, A = TopologyFeatureMap.unpack(row, q)
                    relabeled_graphs.append(TopologySample(A, attrs, seeds, mask,
                        np.arange(q), q, 4).to_graph())
                wl_isomorphic.extend(np.diag(wl.transform(relabeled_graphs)).tolist())
            rows.append({'Case': case, 'Encoder': 'Classical rooted WL', 'Node budget': q,
                'Different graph similarity': wl_original, 'Isomorphic mean similarity': np.mean(wl_isomorphic),
                'Isomorphic minimum similarity': np.min(wl_isomorphic), 'Mean relabeling error': 1-np.mean(wl_isomorphic),
                'Separation gap': np.mean(wl_isomorphic)-wl_original, 'Relabeling comparisons': len(wl_isomorphic),
                'Summary vector distance': np.linalg.norm(summary[0]-summary[1])})
            for steps in (1, 2, 4, 8):
                fm = GraphHamiltonianFeatureMap(q, steps=steps, topology_only=True, field_axis=field_axis)
                for graph_id, row in enumerate(X):
                    approximate = Statevector.from_instruction(fm.simulation_bind(row)).data
                    exact = fm.exact_state(row)
                    convergence.append({'Case': case, 'Graph': graph_id, 'Steps': steps,
                        'Evolution infidelity': max(0., 1-abs(np.vdot(exact, approximate))**2),
                        'Logical depth': fm.bind(row).depth()})
        self.results, self.convergence = pd.DataFrame(rows), pd.DataFrame(convergence)
        return self.results


class GraphRetention:
    """Reachable/retained edges and distances, beyond the node-retention fraction."""
    @staticmethod
    def describe(graph, roots, samples, split):
        rows, cache = [], {}
        for (t, root), sample in zip(roots, samples):
            t, root = int(t), int(root)
            if t not in cache:
                p = graph.prepare_period(t)
                cache[t] = (p, RiskGraph.connectivity(p.adjacency),
                            {identifier: i for i, identifier in enumerate(p.frame.txId)})
            p, U, mapping = cache[t]
            distances, queue = {root: 0}, deque([root])
            while queue:
                node = queue.popleft()
                if distances[node] >= sample.hops:
                    continue
                for other in U.indices[U.indptr[node]:U.indptr[node+1]]:
                    if int(other) not in distances:
                        distances[int(other)] = distances[node]+1
                        queue.append(int(other))
            retained = [mapping[x] for x in sample.node_ids[sample.mask.astype(bool)]]
            reachable = list(distances)
            edges = p.adjacency[reachable, :][:, reachable].nnz
            retained_edges = int(np.count_nonzero(sample.adjacency))
            rows.append({'Split': split, 'Hops': sample.hops, 'Node budget': len(sample.mask),
                'Retained nodes': len(retained), 'Reachable nodes': len(reachable),
                'Retention fraction': len(retained)/len(reachable),
                'Retained edges': retained_edges, 'Reachable edges': edges,
                'Edge retention fraction': retained_edges/edges if edges else 1.,
                'Maximum retained distance': max(distances[x] for x in retained),
                'Maximum reachable distance': max(distances.values()),
                'Retained seeds': int(sample.seeds.sum())})
        return pd.DataFrame(rows)


class TopologyEncodingStudy:
    """Same pilot roots/raw bypass/readout across direct-encoding ablations.

    Classical RBF receives the same packed adjacency/attributes as graph circuits;
    rooted WL and spectral RBF are additional invariant topology controls.
    Kernel-only rows isolate representation quality; additive rows retain raw
    inputs. All validation selection is separate from the descriptive test scores.
    """
    def __init__(self, graph, roots, y, feature_set='Local', sizes=None, n_landmarks=12, seed=42):
        self.graph, self.feature_set, self.seed = graph, feature_set, seed
        self.roots, self.y = {}, {}
        for split in roots:
            idx = StratifiedSampler.indices(y[split], None if sizes is None else sizes[split], seed)
            self.roots[split], self.y[split] = roots[split][idx], y[split][idx]
        self.full_inputs = FullGraphInputs(graph, self.roots, self.y, feature_set=feature_set)
        self.n_landmarks, self.prepared, self.runs = n_landmarks, {}, {}
        self.retention, self.rows = [], []

    def prepare(self, nodes=6, hops=4):
        key = (nodes, hops)
        if key not in self.prepared:
            sampler = RootedSubgraphSampler(self.graph, nodes, self.full_inputs.columns)
            samples = {s: sampler.collect(self.roots[s], hops) for s in self.roots}
            prep = TopologyPreprocessor()
            X = {'train': prep.fit_transform(samples['train'])}
            X.update({s: prep.transform(samples[s]) for s in ('val', 'test')})
            for split in samples:
                self.retention.append(GraphRetention.describe(self.graph, self.roots[split], samples[split], split))
            self.prepared[key] = (samples, prep, X)
        return self.prepared[key]

    @staticmethod
    def _undirected_row(row, nodes):
        mask, root, seeds, attrs, A = TopologyFeatureMap.unpack(row, nodes)
        return np.r_[mask, root, seeds, attrs.ravel(), ((A+A.T)>0).astype(float).ravel()]

    @staticmethod
    def encoders(nodes=6, layers=2):
        return {
            'Graph message quantum': GraphMessageFeatureMap(nodes, repetitions=layers),
            'Generic ring quantum': GraphMessageFeatureMap(nodes, repetitions=layers, edges='generic'),
            'Topology message quantum': GraphMessageFeatureMap(nodes, repetitions=layers, topology_only=True),
            'Hamiltonian quantum': GraphHamiltonianFeatureMap(nodes),
            'Topology Hamiltonian quantum': GraphHamiltonianFeatureMap(nodes, topology_only=True),
            'Reversed Hamiltonian quantum': GraphHamiltonianFeatureMap(nodes, edges='reverse'),
            'Undirected Hamiltonian quantum': GraphHamiltonianFeatureMap(nodes, edges='undirected'),
            'Hermitian walk quantum': HermitianWalkFeatureMap(nodes),
        }

    def configuration(self, nodes=6, hops=4, layers=2, axis='Architecture', names=None):
        key = (nodes, hops, layers)
        samples, prep, X = self.prepare(nodes, hops)
        raw = self.full_inputs.raw(hops)
        metadata = {'Hops': hops, 'Node budget': nodes, 'Message layers': layers,
            'Original features': self.feature_set, 'Seed': self.seed, 'Axis': axis,
            'Train targets': len(self.y['train']), 'Validation targets': len(self.y['val']),
            'Test targets': len(self.y['test']), 'Sampling': 'nearest', 'Diagnostic': True}
        compare = KernelComparison(X, self.y, self.n_landmarks, self.seed)
        for baseline_name, inputs in [('Raw RF', self.full_inputs.original), ('Raw + propagation RF', raw)]:
            rf = KernelComparison(inputs, self.y, self.n_landmarks, self.seed)
            compare.rows.append(rf.random_forest(baseline_name,
                {**metadata, 'Diagnostic': False, 'Raw columns retained': inputs['train'].shape[1]}))
        gamma = 1/(X['train'].shape[1]*max(X['train'].var(), 1e-12))
        compare.nystrom('Graph RBF kernel', RBFKernel(gamma), metadata)
        compare.additive('Raw + graph RBF', 'Graph RBF kernel', raw, {**metadata, 'Diagnostic': False})
        # Match the information withheld in topology-only/generic/undirected arms.
        for control, adapt in [
            ('topology', lambda row: GraphDescriptors.topology_row(row, nodes)),
            ('generic ring', GraphMessageFeatureMap(nodes, edges='generic').adapted),
            ('undirected graph', lambda row: self._undirected_row(row, nodes)),
        ]:
            Z = {split: np.array([adapt(row) for row in values]) for split, values in X.items()}
            auxiliary = KernelComparison(Z, self.y, self.n_landmarks, self.seed)
            g = 1/(Z['train'].shape[1]*max(Z['train'].var(), 1e-12))
            reference = control+' RBF kernel'
            auxiliary.nystrom(reference, RBFKernel(g), metadata)
            compare.models[reference] = auxiliary.models[reference]
            compare.rows.extend(auxiliary.rows)
            compare.additive('Raw + '+control+' RBF', reference, raw, {**metadata, 'Diagnostic': False})
        compare.wl(samples, name='Rooted WL kernel', metadata=metadata)
        compare.additive('Raw + rooted WL', 'Rooted WL kernel', raw, {**metadata, 'Diagnostic': False})
        for descriptor, producer in [('Summary', GraphDescriptors.summary), ('Spectral', GraphDescriptors.spectral)]:
            values = {s: np.array([producer(sample) for sample in samples[s]]) for s in samples}
            angles = AnglePreprocessor()
            Z = {'train': angles.fit_transform(values['train'])}
            Z.update({s: angles.transform(values[s]) for s in ('val', 'test')})
            auxiliary = KernelComparison(Z, self.y, self.n_landmarks, self.seed)
            g = 1/(Z['train'].shape[1]*max(Z['train'].var(), 1e-12))
            auxiliary.nystrom(f'{descriptor} RBF kernel', RBFKernel(g), metadata)
            if descriptor == 'Summary':
                fm = QuantumFeatureMap.uploading(Z['train'].shape[1], 4)
                auxiliary.nystrom('Numerical summary quantum kernel', StatevectorKernel(fm), metadata)
            for reference in list(auxiliary.models):
                compare.models[reference] = auxiliary.models[reference]
                compare.additive('Raw + '+reference.replace(' kernel', ''), reference, raw,
                                 {**metadata, 'Diagnostic': False})
            compare.rows.extend(auxiliary.rows)
        maps = self.encoders(nodes, layers)
        chosen = list(maps) if names is None else list(names)
        for name in chosen:
            fm = maps[name]
            resource_rows = [fm.bind(row) for row in X['train'][:min(8, len(X['train']))]]
            resources = {'Qubits': fm.circuit.num_qubits,
                'Mean logical depth': np.mean([c.depth() for c in resource_rows]),
                'Mean logical two-qubit gates': np.nan if isinstance(fm, HermitianWalkFeatureMap) else np.mean([
                    sum(i.operation.num_qubits == 2 for i in c.data) for c in resource_rows]),
                'Topology only': name in ('Topology message quantum', 'Topology Hamiltonian quantum', 'Hermitian walk quantum'),
                'Evolution time': getattr(fm, 'time', np.nan), 'Trotter steps': getattr(fm, 'steps', np.nan)}
            reference = name+' kernel'
            compare.nystrom(reference, StatevectorKernel(fm), {**metadata, **resources})
            compare.additive('Raw + '+name, reference, raw, {**metadata, **resources, 'Diagnostic': False})
        moment_kernel = HamiltonianMomentKernel(GraphHamiltonianFeatureMap(nodes))
        if names is None:
            reference = 'Hamiltonian moment quantum kernel'
            compare.nystrom(reference, moment_kernel, {**metadata, 'Qubits': nodes})
            compare.additive('Raw + Hamiltonian moment quantum', reference, raw,
                {**metadata, 'Diagnostic': False, 'Qubits': nodes})
        self.runs[key] = compare
        self.rows.extend(compare.rows)
        return pd.DataFrame(compare.rows)

    def run(self, node_budgets=(4, 6, 8, 10), radii=(2, 4, 6), layers=(1, 2, 3)):
        self.rows = []
        self.configuration()
        for q in node_budgets:
            if q != 6:
                self.configuration(nodes=q, axis='Node budget', names=['Hamiltonian quantum'])
        for h in radii:
            if h != 4:
                self.configuration(hops=h, axis='Radius', names=['Hamiltonian quantum'])
        for depth in layers:
            if depth != 2:
                self.configuration(layers=depth, axis='Message layers', names=['Graph message quantum'])
        self.results = pd.DataFrame(self.rows)
        self.retention_results = pd.concat(self.retention, ignore_index=True)
        return self.results

    def permutation_control(self, trials=3):
        """Prespecified Hamiltonian at six nodes/four hops; no test-driven choice."""
        _, _, X = self.prepare(6, 4)
        fm = GraphHamiltonianFeatureMap(6)
        compare = KernelComparison(X, self.y, self.n_landmarks, self.seed)
        raw = self.full_inputs.raw(4)
        rows = []
        for name, kernel in [('Ordered Hamiltonian quantum', StatevectorKernel(fm)),
                             ('Mixture Hamiltonian quantum', PermutationAveragedKernel(fm, trials, self.seed))]:
            compare.nystrom(name+' kernel', kernel)
            rows.append(compare.additive('Raw + '+name, name+' kernel', raw,
                {'Hops': 4, 'Node budget': 6, 'Original features': self.feature_set, 'Trials': trials}))
        sensitivity = []
        for trial in range(trials):
            permuted = TopologyDiagnostics.permute_inputs(X['test'], 6, self.seed+trial)
            for reference in ('Ordered Hamiltonian quantum kernel', 'Mixture Hamiltonian quantum kernel'):
                model = compare.models[reference]
                before = model['kernel'].evaluate(X['test'][:16])
                after = model['kernel'].evaluate(permuted[:16])
                sensitivity.append({'Encoder': reference, 'Trial': trial,
                    'Original features': self.feature_set,
                    'Mean absolute kernel change': np.abs(before-after).mean()})
        return pd.DataFrame(rows), pd.DataFrame(sensitivity)
