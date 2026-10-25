"""Matched classical/quantum experiments with validation-based selection."""
from time import perf_counter
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel

from .data import AnglePreprocessor, StratifiedSampler
from .metrics import Evaluation
from .quantum import QuantumFeatureMap, StatevectorKernel, KernelClassifier
from .graph import WeisfeilerLehmanKernel


class RBFKernel:
    def __init__(self, gamma):
        self.gamma = gamma

    def evaluate(self, X, Y=None):
        return rbf_kernel(X, Y, gamma=self.gamma)


class ClassificationExperiment:
    """Fit all preprocessing on train; tune C on val; evaluate test once per fit.

    Input split entries are (X, y). No cross-split sampling or preprocessing.
    For the 2x2 ablation, use reduced inputs of dimension n_qubits; full_feature
    encoding is a separate experiment because chunking changes the hypothesis.
    """
    def __init__(self, splits, n_qubits=4, n_landmarks=16, seed=42, full_features=False):
        self.y = {name: np.asarray(y) for name, (_, y) in splits.items()}
        self.preprocessor = AnglePreprocessor(None if full_features else n_qubits)
        self.X = {'train': self.preprocessor.fit_transform(splits['train'][0])}
        self.X.update({s: self.preprocessor.transform(splits[s][0]) for s in ('val', 'test')})
        self.raw = {s: np.asarray(X) for s, (X, _) in splits.items()}
        self.n_qubits, self.n_landmarks, self.seed = n_qubits, n_landmarks, seed
        self.models, self.scores, self.rows = {}, {}, []

    def _record(self, name, model, train_input, val_input, test_input, metadata, start):
        predictions = model.predict(test_input)
        scores = (model.predict_proba(test_input)[:, 1] if hasattr(model, 'predict_proba')
                  else model.decision_function(test_input))
        val_scores = (model.predict_proba(val_input)[:, 1] if hasattr(model, 'predict_proba')
                      else model.decision_function(val_input))
        val_prediction = model.predict(val_input)
        self.rows.append({'Model': name, **metadata, 'Seed': self.seed,
                          'Validation PR-AUC': Evaluation.classification(self.y['val'], val_prediction, val_scores)['PR-AUC'],
                          **Evaluation.classification(self.y['test'], predictions, scores),
                          'Seconds': perf_counter() - start})
        self.models[name] = model
        self.scores[name] = scores

    def run(self, passes=(1, 2), C_values=(0.1, 1., 10.), include_zz=True, include_exact=True):
        self.rows, self.models, self.scores = [], {}, {}
        for name, data in [('Random Forest (raw)', self.raw), ('Random Forest (PCA/angles)', self.X)]:
            start = perf_counter()
            model = RandomForestClassifier(n_estimators=150, min_samples_leaf=5,
                    class_weight='balanced_subsample', n_jobs=-1, random_state=self.seed)
            model.fit(data['train'], self.y['train'])
            self._record(name, model, data['train'], data['val'], data['test'], {}, start)
        gamma = 1 / (self.X['train'].shape[1] * max(self.X['train'].var(), 1e-12))
        configs = [('RBF Nyström', RBFKernel(gamma), self.n_landmarks, {})]
        if include_exact:
            configs.insert(0, ('RBF-SVM', RBFKernel(gamma), None, {}))
        for p in passes:
            fm = QuantumFeatureMap.uploading(self.X['train'].shape[1], self.n_qubits, p)
            for approx in ((False, True) if include_exact else (True,)):
                configs.append((f'Quantum passes={p}, Nyström={approx}', StatevectorKernel(fm),
                                self.n_landmarks if approx else None,
                                {'Passes': p, 'Nyström': approx, 'Depth': fm.circuit.depth(),
                                 'Qubits': self.n_qubits}))
        if include_zz and include_exact and self.X['train'].shape[1] == self.n_qubits:
            fm = QuantumFeatureMap.zz(self.X['train'].shape[1], reps=2)
            configs.append(('Quantum ZZ (original baseline)', StatevectorKernel(fm), None,
                            {'Depth': fm.circuit.depth(), 'Qubits': fm.circuit.num_qubits}))
        for name, kernel, landmarks, metadata in configs:
            start = perf_counter()
            # Kernel blocks are independent of C: evaluate each only once.
            if landmarks is None:
                references = self.X['train']
                Ktrain = kernel.evaluate(references)
                projector = None
            else:
                from .quantum import NystromProjector
                idx = StratifiedSampler.landmarks(self.y['train'], landmarks, self.seed)
                references = self.X['train'][idx]
                projector = NystromProjector().fit(kernel.evaluate(references))
                Ktrain = projector.transform(kernel.evaluate(self.X['train'], references))
            blocks = {s: kernel.evaluate(self.X[s], references) for s in ('val', 'test')}
            if projector:
                blocks = {s: projector.transform(K) for s, K in blocks.items()}
            best, best_ap, best_c = None, -np.inf, None
            for C in C_values:
                model = SVC(kernel='precomputed' if landmarks is None else 'linear',
                            C=C, class_weight='balanced')
                model.fit(Ktrain, self.y['train'])
                ap = Evaluation.classification(self.y['val'], model.predict(blocks['val']),
                                               model.decision_function(blocks['val']))['PR-AUC']
                if ap > best_ap:
                    best, best_ap, best_c = model, ap, C
            self._record(name, best, Ktrain, blocks['val'], blocks['test'],
                         {**metadata, 'C': best_c, 'Landmarks': landmarks or 0}, start)
        return pd.DataFrame(self.rows)

    @staticmethod
    def ablation_summary(results):
        """Paired deltas to one-pass exact kernel within each seed/feature group."""
        quantum = results[results['Model'].str.startswith('Quantum passes=')].copy()
        groups = [c for c in ('Features', 'Seed', 'Qubits') if c in quantum]
        baseline = quantum[(quantum.Passes == 1) & (quantum['Nyström'] == False)]
        baseline = baseline[groups + ['PR-AUC', 'Validation PR-AUC']].rename(
            columns={'PR-AUC': 'Base PR-AUC', 'Validation PR-AUC': 'Base validation PR-AUC'})
        quantum = quantum.merge(baseline, on=groups, validate='many_to_one')
        quantum['Δ PR-AUC'] = quantum['PR-AUC'] - quantum['Base PR-AUC']
        quantum['Δ validation PR-AUC'] = quantum['Validation PR-AUC'] - quantum['Base validation PR-AUC']
        return quantum


class GraphExperiment:
    def __init__(self, risk_graph, periods=None, sizes=None, seed=42):
        self.graph = risk_graph
        self.periods = periods or {'train': range(1, 31), 'val': range(31, 35), 'test': range(35, 50)}
        self.sizes = sizes or {'train': 160, 'val': 100, 'test': 160}
        self.seed = seed

    def feature_splits(self, hops, feature_set='Local'):
        result = {}
        for name, periods in self.periods.items():
            X, y, _ = self.graph.make_dataset(periods, hops, feature_set)
            idx = StratifiedSampler.indices(y, self.sizes[name], self.seed)
            result[name] = (X.iloc[idx], y[idx])
        return result

    def run_features(self, hops=(1, 2, 3), n_qubits=4, landmarks=16):
        rows, self.feature_runs = [], {}
        for h in hops:
            splits = self.feature_splits(h)
            experiment = ClassificationExperiment(splits, n_qubits, landmarks, self.seed)
            table = experiment.run(C_values=(0.1, 1., 10.), include_zz=True)
            # Transaction-only baseline uses exactly the same non-seed rows.
            from .data import EllipticData
            columns = EllipticData.feature_columns('Local')
            base = RandomForestClassifier(n_estimators=150, min_samples_leaf=5,
                    class_weight='balanced_subsample', n_jobs=-1, random_state=self.seed)
            start = perf_counter()
            base.fit(splits['train'][0][columns], splits['train'][1])
            scores = base.predict_proba(splits['test'][0][columns])[:, 1]
            val_scores = base.predict_proba(splits['val'][0][columns])[:, 1]
            base_row = {'Model': 'Random Forest (transaction only)', 'Seed': self.seed,
                        'Validation PR-AUC': Evaluation.classification(splits['val'][1],
                            base.predict(splits['val'][0][columns]), val_scores)['PR-AUC'],
                        **Evaluation.classification(splits['test'][1],
                            base.predict(splits['test'][0][columns]), scores),
                        'Seconds': perf_counter() - start}
            experiment.scores[base_row['Model']] = scores
            table = pd.concat([table, pd.DataFrame([base_row])], ignore_index=True)
            table['Hops'] = h
            rows.append(table)
            self.feature_runs[h] = experiment
        return pd.concat(rows, ignore_index=True)

    def run_legacy_reference(self, full_data=False):
        """Restore original 165-column RF and nine propagation features, no PCA."""
        from .additive import FullGraphInputs
        from .graph_studies import KernelComparison
        roots, labels = {}, {}
        for split, periods in self.periods.items():
            references, y = [], []
            for t in periods:
                p = self.graph.prepare_period(t)
                references.extend((t, int(i)) for i in p.candidates)
                y.extend(p.labels[p.candidates])
            y = np.asarray(y)
            idx = StratifiedSampler.indices(y, None if full_data else self.sizes[split], self.seed)
            roots[split], labels[split] = np.asarray(references)[idx], y[idx]
        full = FullGraphInputs(self.graph, roots, labels)
        legacy = full.legacy()
        rows = []
        self.legacy_scores, self.legacy_y = {}, labels
        for name, X in [
            ('Local-only RF (93)', full.local),
            ('All-original RF (165)', full.original),
            ('Local + legacy propagation RF',
             {s: np.column_stack([full.local[s], legacy[s]]) for s in labels}),
            ('Original + legacy propagation RF',
             {s: np.column_stack([full.original[s], legacy[s]]) for s in labels}),
        ]:
            comparison = KernelComparison(X, labels, n_landmarks=min(16, len(labels['train'])), seed=self.seed)
            rows.append(comparison.random_forest(name,
                {'Hops': 2, 'Diagnostic': False, 'Protocol': 'Original notebook reference',
                 'Raw columns retained': X['train'].shape[1]}))
            self.legacy_scores.update(comparison.scores)
        return pd.DataFrame(rows)

    def run_wl(self, hops=(1, 2, 3), max_nodes=30, direction='both', n_iter=2,
               C_values=(0.1, 1., 10.)):
        self.wl_scores = {}
        rows = []
        for h in hops:
            start = perf_counter()
            splits = {s: self.graph.sample_subgraphs(ts, self.sizes[s], h, max_nodes,
                                                      direction, self.seed)
                      for s, ts in self.periods.items()}
            kernel = WeisfeilerLehmanKernel(n_iter=n_iter, directed=direction != 'both')
            Ktrain = kernel.fit_transform(splits['train'][0])
            Kval = kernel.transform(splits['val'][0])
            Ktest = kernel.transform(splits['test'][0])
            best, best_ap, best_c = None, -np.inf, None
            for C in C_values:
                model = SVC(kernel='precomputed', C=C, class_weight='balanced').fit(Ktrain, splits['train'][1])
                ap = Evaluation.classification(splits['val'][1], model.predict(Kval),
                                               model.decision_function(Kval))['PR-AUC']
                if ap > best_ap:
                    best, best_ap, best_c = model, ap, C
            scores = best.decision_function(Ktest)
            rows.append({'Model': 'WL subgraph SVM', 'Hops': h, 'C': best_c,
                         'Validation PR-AUC': best_ap, 'Seconds': perf_counter() - start,
                         **Evaluation.classification(splits['test'][1], best.predict(Ktest), scores)})
            self.wl_scores[h] = (splits['test'][1], scores)
        return pd.DataFrame(rows)
