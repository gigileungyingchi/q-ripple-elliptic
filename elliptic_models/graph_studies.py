"""Research experiments for depth-aware and direct-topology quantum kernels."""
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .data import AnglePreprocessor, StratifiedSampler, EllipticData
from .experiments import RBFKernel
from .graph import WeisfeilerLehmanKernel
from .metrics import Evaluation
from .additive import FullGraphInputs, AdditiveKernelClassifier
from .multihop import DirectionalPropagation, HopFeatureMap
from .quantum import NystromProjector, QuantumFeatureMap, StatevectorKernel
from .topology import (RootedSubgraphSampler, TopologyPreprocessor,
                       TopologyFeatureMap, TopologyDiagnostics, PermutationAveragedKernel)


class KernelComparison:
    """Fixed splits and landmark indices; identical logistic readout/C selection.

    Kernel blocks are evaluated once, independent of the C search. Validation
    controls C selection. Test predictions are descriptive for prespecified
    configurations; downstream configuration selection must use validation only.
    """
    def __init__(self, X, y, n_landmarks=16, seed=42, C_values=(0.1, 1., 10.)):
        if any(len(np.unique(y[s])) != 2 for s in ('train', 'val', 'test')):
            raise ValueError('Every split needs both labels; increase sample sizes')
        self.X, self.y, self.seed = X, y, seed
        self.C_values = tuple(C_values)
        self.landmark_indices = StratifiedSampler.landmarks(y['train'], n_landmarks, seed)
        self.rows, self.scores, self.models = [], {}, {}

    def _record(self, name, model, blocks, metadata, start):
        val_scores = model.predict_proba(blocks['val'])[:, 1]
        scores = model.predict_proba(blocks['test'])[:, 1]
        row = {'Model': name, 'Seed': self.seed, **metadata,
               'Validation PR-AUC': Evaluation.classification(self.y['val'],
                   model.predict(blocks['val']), val_scores)['PR-AUC'],
               **Evaluation.classification(self.y['test'], model.predict(blocks['test']), scores),
               'Seconds': perf_counter() - start}
        for budget in (10, 25, 50):
            discovery = Evaluation.discovery(self.y['test'], scores, [budget]).iloc[0]
            row[f'Recall@{budget}'] = discovery['recall']
        self.rows.append(row)
        self.scores[name] = scores
        self.models[name] = model
        return row

    def random_forest(self, name='Random Forest', metadata=None):
        start = perf_counter()
        model = RandomForestClassifier(n_estimators=150, min_samples_leaf=5,
            class_weight='balanced_subsample', n_jobs=-1, random_state=self.seed)
        model.fit(self.X['train'], self.y['train'])
        return self._record(name, model, self.X, metadata or {}, start)

    def nystrom(self, name, kernel, metadata=None):
        start = perf_counter()
        references = self.X['train'][self.landmark_indices]
        projector = NystromProjector().fit(kernel.evaluate(references))
        blocks = {s: projector.transform(kernel.evaluate(self.X[s], references))
                  for s in ('train', 'val', 'test')}
        row = self._select_readout(name, blocks, metadata or {}, start, projector, references)
        self.models[name]['kernel'] = kernel
        return row

    def _select_readout(self, name, blocks, metadata, start, projector=None, references=None):
        best, best_ap, best_c = None, -np.inf, None
        for C in self.C_values:
            model = LogisticRegression(C=C, class_weight='balanced', max_iter=2000,
                                       random_state=self.seed).fit(blocks['train'], self.y['train'])
            ap = Evaluation.classification(self.y['val'], model.predict(blocks['val']),
                                            model.predict_proba(blocks['val'])[:, 1])['PR-AUC']
            if ap > best_ap:
                best, best_ap, best_c = model, ap, C
        row = self._record(name, best, blocks,
                           {**metadata, 'C': best_c, 'Landmarks': len(self.landmark_indices)}, start)
        self.models[name] = {'readout': best, 'projector': projector, 'references': references, 'blocks': blocks, 'Seconds': row['Seconds']}
        return row

    def additive(self, name, reference_name, raw, metadata=None):
        """Reuse graph blocks while retaining every raw column in the readout."""
        start = perf_counter()
        reference = self.models[reference_name]
        hybrid = AdditiveKernelClassifier(reference.get('kernel'), len(self.landmark_indices), self.seed)
        hybrid.projector_ = reference['projector']
        hybrid.references_ = reference['references']
        hybrid.fit_readout(raw['train'], reference['blocks']['train'], self.y['train'])
        blocks = {s: hybrid.join(raw[s], reference['blocks'][s]) for s in raw}
        row = self._record(name, hybrid.model_, blocks,
                           {**(metadata or {}), 'Raw columns retained': hybrid.n_raw_features_,
                            'Landmarks': len(self.landmark_indices), 'Readout': 'Random Forest'}, start)
        row['Seconds'] += reference.get('Seconds', 0.)
        self.models[name] = {'readout': hybrid.model_, 'projector': hybrid.projector_,
                            'references': hybrid.references_, 'blocks': blocks,
                            'raw': raw, 'estimator': hybrid}
        return row

    def wl(self, samples, name='WL on retained subgraphs', directed=True, metadata=None):
        start = perf_counter()
        graphs = {s: [sample.to_graph() for sample in samples[s]] for s in samples}
        # Explicit WL coordinates from the training-only vocabulary. A linear
        # logistic readout gives a classical topology reference without SVM
        # objective changes. Nyström uses the identical root landmark indices.
        kernel = WeisfeilerLehmanKernel(n_iter=2, directed=directed)
        train_kernel = kernel.fit_transform(graphs['train'])
        idx = self.landmark_indices
        projector = NystromProjector().fit(train_kernel[idx, :][:, idx])
        blocks = {'train': projector.transform(train_kernel[:, idx])}
        blocks.update({s: projector.transform(kernel.transform(graphs[s])[:, idx])
                       for s in ('val', 'test')})
        return self._select_readout(name, blocks, metadata or {}, start)


class PropagationStudy:
    """Matched roots: original attributes plus graph additions; graph-only diagnostics."""
    def __init__(self, risk_graph, periods, sizes, max_hops=8, n_landmarks=16, seed=42, feature_set='All'):
        self.graph, self.periods = risk_graph, periods
        self.seed, self.n_landmarks = seed, n_landmarks
        self.propagation = DirectionalPropagation(max_hops)
        self.X, self.y, self.roots, coverage = {}, {}, {}, []
        for split, times in periods.items():
            X, y, roots = self.propagation.dataset(risk_graph, times)
            coverage.append(self.propagation.coverage(X).assign(Split=split, Targets=len(y)))
            idx = StratifiedSampler.indices(y, sizes[split], seed)
            self.X[split], self.y[split], self.roots[split] = X.iloc[idx], y[idx], roots[idx]
        self.full_inputs = FullGraphInputs(risk_graph, self.roots, self.y, max_hops, feature_set)
        self.feature_set = feature_set
        self.coverage = pd.concat(coverage, ignore_index=True)
        self.runs, self.preprocessors, self.angles = {}, {}, {}

    def run_depths(self, depths=(1, 2, 3, 4, 6, 8)):
        rows = []
        self.raw_inputs = {}
        for depth in depths:
            if depth > self.propagation.max_hops or depth < 1:
                raise ValueError('Depth outside prepared propagation range')
            columns = self.propagation.columns(depth)
            raw = {s: frame[columns].to_numpy() for s, frame in self.X.items()}
            prep = AnglePreprocessor()
            X = {'train': prep.fit_transform(raw['train'])}
            X.update({s: prep.transform(raw[s]) for s in ('val', 'test')})
            comparison = KernelComparison(X, self.y, self.n_landmarks, self.seed)
            # RF on raw aggregated signals (tree splits do not need scaling).
            rf = KernelComparison(raw, self.y, self.n_landmarks, self.seed)
            rows.append(rf.random_forest('Propagation Random Forest', {'Hops': depth}))
            gamma = 1 / (X['train'].shape[1] * max(X['train'].var(), 1e-12))
            rows.append(comparison.nystrom('Propagation RBF Nyström', RBFKernel(gamma), {'Hops': depth}))
            for name, fm in [
                ('Hop-aware quantum Nyström', HopFeatureMap.build(depth)),
                ('Generic quantum Nyström', QuantumFeatureMap.uploading(4 * depth, 4, passes=1)),
            ]:
                rows.append(comparison.nystrom(name, StatevectorKernel(fm),
                            {'Hops': depth, 'Repetitions': 1, 'Circuit depth': fm.circuit.depth(), 'Qubits': 4}))
            # All existing graph-only rows remain explicitly diagnostic.
            for row in rows:
                row.setdefault('Diagnostic', True)
            retained = self.full_inputs.raw(depth)
            self.raw_inputs[depth] = retained
            metadata = {'Hops': depth, 'Diagnostic': False, 'Original features': self.feature_set,
                        'Original feature count': len(self.full_inputs.columns)}
            for name, inputs in [
                ('Local-only RF (93)', self.full_inputs.local),
                ('All-original RF (165)', self.full_inputs.original),
                ('Original + legacy propagation RF', {s: np.column_stack([self.full_inputs.original[s], legacy])
                     for s, legacy in self.full_inputs.legacy().items()}),
                ('Original + propagation RF', retained),
                ('Local + legacy propagation RF', {s: np.column_stack([self.full_inputs.local[s], legacy])
                     for s, legacy in self.full_inputs.legacy().items()}),
                ('Local + propagation RF', self.full_inputs.local_raw(depth)),
            ]:
                raw_comparison = KernelComparison(inputs, self.y, self.n_landmarks, self.seed)
                rows.append(raw_comparison.random_forest(name, {**metadata,
                    'Original features': 'Local' if name.startswith('Local') else self.feature_set,
                    'Original feature count': 93 if name.startswith('Local') else len(self.full_inputs.columns),
                    'Raw columns retained': inputs['train'].shape[1]}))
                comparison.scores.update(raw_comparison.scores)
            for name, reference in [
                ('Original + propagation + RBF', 'Propagation RBF Nyström'),
                ('Original + propagation + hop quantum', 'Hop-aware quantum Nyström'),
                ('Original + propagation + generic quantum', 'Generic quantum Nyström'),
            ]:
                rows.append(comparison.additive(name, reference, retained, metadata))
                rows.append(comparison.additive(name.replace('Original', 'Local', 1), reference,
                    self.full_inputs.local_raw(depth), {**metadata, 'Original features': 'Local',
                                                       'Original feature count': 93}))
            comparison.scores.update(rf.scores)
            self.runs[depth], self.preprocessors[depth], self.angles[depth] = comparison, prep, X
        self.results = pd.DataFrame(rows)
        return self.results

    def run_repetitions(self, fixed_hops=4, repetitions=(1, 2, 3), reverse_control=True):
        if fixed_hops not in self.angles:
            raise ValueError('Run this hop depth before the repetition control')
        comparison = KernelComparison(self.angles[fixed_hops], self.y, self.n_landmarks, self.seed)
        rows = []
        configurations = [(rep, 'forward') for rep in repetitions]
        if reverse_control:
            configurations.append((1, 'reverse'))
        for rep, order in configurations:
            fm = HopFeatureMap.build(fixed_hops, rep, order)
            reference = f'Hop-only reps={rep}, order={order}'
            comparison.nystrom(reference, StatevectorKernel(fm), {'Diagnostic': True})
            for prefix, retained in [('Original', self.raw_inputs[fixed_hops]),
                                     ('Local', self.full_inputs.local_raw(fixed_hops))]:
                rows.append(comparison.additive(f'{prefix} + hop quantum reps={rep}, order={order}',
                            reference, retained,
                            {'Hops': fixed_hops, 'Repetitions': rep, 'Hop order': order,
                             'Original features': 'Local' if prefix == 'Local' else self.feature_set,
                             'Circuit depth': fm.circuit.depth(), 'Qubits': 4, 'Diagnostic': False}))
        self.repetition_run = comparison
        return pd.DataFrame(rows)

    @staticmethod
    def marginal_depth(results):
        """Differences to the previous tested depth, not single-hop attribution."""
        rows = results.sort_values(['Seed', 'Model', 'Hops']).copy()
        groups = rows.groupby(['Seed', 'Model'])
        rows['Previous tested depth'] = groups.Hops.shift()
        rows['Δ validation PR-AUC'] = groups['Validation PR-AUC'].diff()
        rows['Δ test PR-AUC'] = groups['PR-AUC'].diff()
        return rows


class TopologyStudy:
    """Direct topology versus baselines on exactly the same sampled subgraphs."""
    def __init__(self, risk_graph, roots, y, n_qubits=6, n_landmarks=16, seed=42, full_inputs=None):
        self.full_inputs = full_inputs or FullGraphInputs(risk_graph, roots, y)
        self.prefix = 'Local' if self.full_inputs.feature_set == 'Local' else 'Original'
        self.sampler = RootedSubgraphSampler(risk_graph, n_qubits, feature_columns=self.full_inputs.columns)
        self.roots, self.y, self.q, self.n_landmarks, self.seed = roots, y, n_qubits, n_landmarks, seed
        self.runs, self.samples, self.inputs, self.preprocessors = {}, {}, {}, {}

    def run(self, depths=(1, 2, 3, 4, 6, 8), policies=('nearest',),
            edge_modes=('directed', 'reverse', 'undirected', 'none'), repetitions=2,
            search_direction='both'):
        rows, retention = [], []
        self.repetitions_ = repetitions
        for policy in policies:
            for h in depths:
                key = (h, policy)
                samples = {s: self.sampler.collect(roots, h, policy, search_direction)
                           for s, roots in self.roots.items()}
                prep = TopologyPreprocessor()
                X = {'train': prep.fit_transform(samples['train'])}
                X.update({s: prep.transform(samples[s]) for s in ('val', 'test')})
                metadata = {'Hops': h, 'Sampling': policy, 'Search direction': search_direction,
                            'Qubits': self.q, 'Repetitions': repetitions,
                            'Original features': self.full_inputs.feature_set,
                            'Node attribute count': len(self.full_inputs.columns)}
                compare = KernelComparison(X, self.y, self.n_landmarks, self.seed)
                rows.append(compare.random_forest('Retained-subgraph Random Forest', metadata))
                gamma = 1 / (X['train'].shape[1] * max(X['train'].var(), 1e-12))
                rows.append(compare.nystrom('Retained-subgraph RBF Nyström', RBFKernel(gamma), metadata))
                rows.append(compare.wl(samples, metadata=metadata))
                for mode in edge_modes:
                    fm = TopologyFeatureMap(self.q, repetitions=repetitions, edges=mode)
                    depths_used = [fm.logical_depth(row) for row in X['train']]
                    rows.append(compare.nystrom(f'Topology quantum ({mode})', StatevectorKernel(fm),
                        {**metadata, 'Edge mode': mode, 'Mean circuit depth': np.mean(depths_used)}))
                # Primary models preserve all root attributes AND full-period context.
                for row in rows:
                    row.setdefault('Diagnostic', True)
                retained = self.full_inputs.raw(h)
                additive_metadata = {**metadata, 'Diagnostic': False,
                    'Original features': self.full_inputs.feature_set,
                    'Original feature count': len(self.full_inputs.columns)}
                for name, data in [('Local-only RF (93)', self.full_inputs.local),
                                   *([('All-original RF (165)', self.full_inputs.original)] if self.prefix == 'Original' else []),
                                   (f'{self.prefix} + propagation RF', retained)]:
                    baseline = KernelComparison(data, self.y, self.n_landmarks, self.seed)
                    rows.append(baseline.random_forest(name, {**additive_metadata,
                        'Original feature count': 93 if name.startswith('Local-only') else len(self.full_inputs.columns),
                        'Raw columns retained': data['train'].shape[1]}))
                    compare.scores.update(baseline.scores)
                for label, reference in [
                    ('Original + graph + RBF', 'Retained-subgraph RBF Nyström'),
                    ('Original + graph + WL', 'WL on retained subgraphs'),
                    *[(f'Original + graph + quantum ({mode})', f'Topology quantum ({mode})')
                      for mode in edge_modes],
                ]:
                    kernel_metadata = dict(additive_metadata)
                    if reference.startswith('Topology quantum ('):
                        kernel_metadata['Edge mode'] = reference[len('Topology quantum ('):-1]
                    rows.append(compare.additive(label.replace('Original', self.prefix, 1), reference, retained, kernel_metadata))
                for split in samples:
                    retention.append(TopologyDiagnostics.retention(samples[split]).assign(
                        Split=split, Sampling=policy, Search_direction=search_direction))
                self.runs[key], self.samples[key], self.inputs[key], self.preprocessors[key] = compare, samples, X, prep
        self.retention = pd.concat(retention, ignore_index=True)
        self.results = pd.DataFrame(rows)
        return self.results

    def permutation_checks(self, hops, policy='nearest', trials=3, sample_size=24):
        X = self.inputs[(hops, policy)]
        fm = TopologyFeatureMap(self.q, repetitions=getattr(self, 'repetitions_', 2))
        diagnostic = TopologyDiagnostics.kernel_sensitivity(X['test'][:sample_size], fm, trials, self.seed)
        compare = KernelComparison(X, self.y, self.n_landmarks, self.seed)
        compare.nystrom('Original graph-only', StatevectorKernel(fm))
        raw = self.full_inputs.raw(hops)
        rows = [compare.additive('Original + graph: original ordering', 'Original graph-only', raw)]
        averaged = PermutationAveragedKernel(fm, trials=trials, seed=self.seed)
        compare.nystrom('Mixture graph-only', averaged)
        rows.append(compare.additive('Original + graph: permutation mixture', 'Mixture graph-only', raw))
        # Hold training/model fixed; change only the test slot ordering.
        fitted = compare.models['Original + graph: original ordering']
        changed_X = TopologyDiagnostics.permute_inputs(X['test'], self.q, self.seed)
        features = fitted['projector'].transform(StatevectorKernel(fm).evaluate(changed_X, fitted['references']))
        features = AdditiveKernelClassifier.join(raw['test'], features)
        model = fitted['readout']
        scores = model.predict_proba(features)[:, 1]
        robustness = {'Model': 'Original + graph, permuted test slots',
                      **Evaluation.classification(self.y['test'], model.predict(features), scores),
                      'Mean absolute score change': np.abs(scores - compare.scores[rows[0]['Model']]).mean()}
        return diagnostic, pd.DataFrame(rows), pd.DataFrame([robustness])
