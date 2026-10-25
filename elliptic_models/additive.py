"""Preserve original attributes while adding graph and quantum information."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .data import EllipticData, StratifiedSampler
from .graph import RiskGraph
from .multihop import DirectionalPropagation
from .quantum import NystromProjector


class FullGraphInputs:
    """Aligned, uncompressed root attributes plus full-period graph descriptors.

    Node caps affect the optional quantum topology branch, never these inputs.
    Labels validate row alignment; they are not used to construct features.
    """
    def __init__(self, graph, roots, y, max_hops=8, feature_set='All'):
        self.graph, self.roots, self.y = graph, roots, y
        self.columns = EllipticData.feature_columns(feature_set)
        self.feature_set, self.max_hops = feature_set, max_hops
        self.original, self.local = {}, {}
        self._legacy, self._directional, self._structural, self._raw = {}, {}, {}, {}
        for split, references in roots.items():
            values = np.empty((len(references), len(self.columns)))
            for t in np.unique(references[:, 0]):
                positions = np.flatnonzero(references[:, 0] == t)
                indices = references[positions, 1].astype(int)
                period = graph.prepare_period(int(t))
                if np.intersect1d(indices, period.seeds).size:
                    raise ValueError('Known seeds cannot be evaluation roots')
                if not np.array_equal(period.labels[indices], np.asarray(y[split])[positions]):
                    raise ValueError('Root/label alignment mismatch')
                values[positions] = period.frame.iloc[indices][self.columns].to_numpy()
            self.original[split] = values
            # Local features are x0..x92, first in both supported feature sets.
            self.local[split] = values[:, :93].copy()

    @staticmethod
    def legacy_features(A, seeds):
        """The nine descriptors from the original graph notebook.

        U @ degree - degree counts two-step undirected walks excluding returns
        to the root, exactly matching the old zero-diagonal U**2 definition.
        """
        U = RiskGraph.connectivity(A)
        degree = np.asarray(U.sum(axis=1)).ravel()
        seed = np.zeros(A.shape[0]); seed[np.asarray(seeds, dtype=int)] = 1.
        return pd.DataFrame({
            'log_in_degree': np.log1p(np.asarray(A.sum(axis=0)).ravel()),
            'log_out_degree': np.log1p(np.asarray(A.sum(axis=1)).ravel()),
            'log_neighbours': np.log1p(degree),
            'log_two_step_walks': np.log1p(np.maximum(U @ degree - degree, 0)),
            'seed_predecessors': np.log1p(A.T @ seed),
            'seed_successors': np.log1p(A @ seed),
            'seed_two_hop_paths': np.log1p(A.T @ (A.T @ seed)),
            'forward_risk': RiskGraph.diffusion(A, seeds),
            'reverse_risk': RiskGraph.diffusion(A.T.tocsr(), seeds),
        })

    def _gather(self, split, producer):
        references = self.roots[split]
        rows = []
        positions = []
        for t in np.unique(references[:, 0]):
            pos = np.flatnonzero(references[:, 0] == t)
            frame = producer(int(t))
            rows.append(frame.iloc[references[pos, 1].astype(int)].to_numpy())
            positions.extend(pos)
        values = np.concatenate(rows)
        return values[np.argsort(positions)]

    def legacy(self):
        def period_features(t):
            if t not in self._legacy:
                p = self.graph.prepare_period(t)
                self._legacy[t] = self.legacy_features(p.adjacency, p.seeds)
            return self._legacy[t]
        return {s: self._gather(s, period_features) for s in self.roots}

    def raw(self, hops):
        """Original columns are the exact prefix; no PCA, clipping or selection."""
        if not 1 <= hops <= self.max_hops:
            raise ValueError('Requested hops outside prepared range')
        if hops not in self._raw:
            def graph_features(t):
                p = self.graph.prepare_period(t)
                if t not in self._legacy:
                    self._legacy[t] = self.legacy_features(p.adjacency, p.seeds)
                if t not in self._directional:
                    self._directional[t] = DirectionalPropagation(self.max_hops).transform(p.adjacency, p.seeds)
                if (t, hops) not in self._structural:
                    self._structural[t, hops] = RiskGraph.structural_features(p.adjacency, p.seeds, hops)
                legacy = self._legacy[t]
                structural = self._structural[t, hops]
                extra = structural[[c for c in structural if c not in legacy]]
                return pd.concat([legacy, extra, self._directional[t][DirectionalPropagation.columns(hops)]], axis=1)
            self._raw[hops] = {s: np.column_stack([self.original[s], self._gather(s, graph_features)])
                               for s in self.roots}
        return self._raw[hops]


    def local_raw(self, hops):
        """Same graph context and roots, with exactly the 93 local root attributes."""
        return {s: np.column_stack([self.local[s], values[:, len(self.columns):]])
                for s, values in self.raw(hops).items()}


class AdditiveKernelClassifier:
    """RF sees every original input plus Nyström features; no replacement/PCA.

    The encoding and Nyström blocks are approximations. The original numerical
    inputs are retained exactly in the classical readout, including on hardware.
    """
    def __init__(self, kernel, n_landmarks=16, seed=42):
        self.kernel, self.n_landmarks, self.seed = kernel, n_landmarks, seed

    @staticmethod
    def join(raw, graph):
        raw, graph = np.asarray(raw), np.asarray(graph)
        if raw.ndim != 2 or graph.ndim != 2 or len(raw) != len(graph):
            raise ValueError('Raw and graph feature rows must be aligned matrices')
        return np.column_stack([raw, graph])

    def fit_readout(self, raw, graph_features, y):
        self.n_raw_features_ = np.asarray(raw).shape[1]
        self.training_features_ = self.join(raw, graph_features)
        self.model_ = RandomForestClassifier(n_estimators=150, min_samples_leaf=5,
            class_weight='balanced_subsample', n_jobs=-1, random_state=self.seed)
        self.model_.fit(self.training_features_, y)
        return self

    def fit(self, raw, encoded, y):
        indices = StratifiedSampler.landmarks(y, self.n_landmarks, self.seed)
        self.references_ = np.asarray(encoded)[indices]
        self.projector_ = NystromProjector().fit(self.kernel.evaluate(self.references_))
        graph = self.projector_.transform(self.kernel.evaluate(encoded, self.references_))
        return self.fit_readout(raw, graph, y)

    def features(self, raw, encoded):
        if np.asarray(raw).shape[1] != self.n_raw_features_:
            raise ValueError('Raw feature count changed')
        graph = self.projector_.transform(self.kernel.evaluate(encoded, self.references_))
        return self.join(raw, graph)
