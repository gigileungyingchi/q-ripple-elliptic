"""Directed propagation, configurable rooted neighborhoods and a native WL kernel."""
from dataclasses import dataclass
import hashlib

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize

from .data import StratifiedSampler


@dataclass
class GraphPeriod:
    frame: pd.DataFrame
    adjacency: sparse.csr_matrix
    labels: np.ndarray
    seeds: np.ndarray
    candidates: np.ndarray


class RiskGraph:
    def __init__(self, transactions, edges, seed_fraction=0.2, random_state=42):
        if not 0 <= seed_fraction < 1:
            raise ValueError('seed_fraction must be in [0, 1)')
        self.transactions = transactions
        self.edges = edges
        self.seed_fraction = seed_fraction
        self.random_state = random_state
        self._periods = {}

    def prepare_period(self, time_step):
        """Oracle seed simulation: reveal some period-local illicit labels.

        Candidates exclude revealed seeds, including in validation/test. This
        assumes investigators already know these seed labels at prediction time.
        Unknown-label nodes remain in the topology but never become targets.
        """
        if time_step in self._periods:
            return self._periods[time_step]
        part = self.transactions[self.transactions.time_step == time_step].reset_index(drop=True)
        mapping = {tx: i for i, tx in enumerate(part.txId)}
        edges = self.edges[self.edges.txId1.isin(mapping) & self.edges.txId2.isin(mapping)]
        A = sparse.csr_matrix((np.ones(len(edges)),
                               (edges.txId1.map(mapping).to_numpy(dtype=int),
                                edges.txId2.map(mapping).to_numpy(dtype=int))),
                              shape=(len(part), len(part)))
        A.data[:] = 1
        A.setdiag(0)
        A.eliminate_zeros()
        y = part.label.to_numpy(dtype=int)
        illicit = np.flatnonzero(y == 1)
        count = min(len(illicit), max(1, round(len(illicit) * self.seed_fraction))) \
            if self.seed_fraction and len(illicit) else 0
        seeds = np.random.default_rng(self.random_state + time_step).choice(
            illicit, count, replace=False)
        result = GraphPeriod(part, A, y, seeds, np.setdiff1d(np.flatnonzero(y >= 0), seeds))
        self._periods[time_step] = result
        return result

    @staticmethod
    def diffusion(A, seeds, alpha=0.8, n_iter=20):
        if not 0 <= alpha < 1 or n_iter < 0:
            raise ValueError('Require 0 <= alpha < 1 and n_iter >= 0')
        start = np.zeros(A.shape[0])
        if not len(seeds):
            return start
        start[seeds] = 1 / len(seeds)
        degree = np.asarray(A.sum(axis=1)).ravel()
        inverse = np.divide(1., degree, out=np.zeros_like(degree, dtype=float), where=degree > 0)
        P = sparse.diags(inverse) @ A
        scores = start.copy()
        for _ in range(n_iter):
            scores = (1 - alpha) * start + alpha * (P.T @ scores + scores[degree == 0].sum() * start)
        return scores

    @staticmethod
    def connectivity(A, direction='both'):
        if direction not in ('both', 'out', 'in'):
            raise ValueError('direction must be both, out or in')
        U = (A + A.T if direction == 'both' else A if direction == 'out' else A.T).tocsr(copy=True)
        U.data[:] = 1
        U.setdiag(0)
        U.eliminate_zeros()
        return U

    @classmethod
    def structural_features(cls, A, seeds, hops=2):
        """Walk/exposure counts at EVERY length 1..hops plus global diffusion.

        Walk counts include repeated vertices; they are not unique hop-shell
        sizes. Sparse matrix-vector recurrences avoid densifying A**hops.
        Diffusion is global and is held fixed across hop-count comparisons.
        """
        if not isinstance(hops, int) or hops < 1:
            raise ValueError('hops must be a positive integer')
        U = cls.connectivity(A)
        features = {
            'log_in_degree': np.log1p(np.asarray(A.sum(axis=0)).ravel()),
            'log_out_degree': np.log1p(np.asarray(A.sum(axis=1)).ravel()),
        }
        walks = np.ones(A.shape[0])
        forward = np.zeros(A.shape[0])
        forward[seeds] = 1
        reverse = forward.copy()
        for h in range(1, hops + 1):
            walks = U @ walks
            forward = A.T @ forward
            reverse = A @ reverse
            features[f'log_walks_{h}'] = np.log1p(walks)
            features[f'seed_from_{h}'] = np.log1p(forward)
            features[f'seed_to_{h}'] = np.log1p(reverse)
        features['forward_risk'] = cls.diffusion(A, seeds)
        features['reverse_risk'] = cls.diffusion(A.T.tocsr(), seeds)
        return pd.DataFrame(features)

    @classmethod
    def extract_subgraph(cls, A, center, seeds, hops=2, max_nodes=30, direction='both'):
        """BFS ego graph, root retained, nearest nodes kept before farther nodes.

        max_nodes=None gives the complete radius-hops neighborhood. Capping
        during BFS bounds exploration; a capped graph may omit valid neighbors.
        Search direction controls reachability; directed edges are retained.
        """
        if not isinstance(hops, int) or hops < 0:
            raise ValueError('hops must be a nonnegative integer')
        if max_nodes is not None and max_nodes < 1:
            raise ValueError('max_nodes must be positive or None')
        if not 0 <= center < A.shape[0]:
            raise ValueError('Root outside graph')
        U = cls.connectivity(A, direction)
        selected, frontier = [int(center)], [int(center)]
        seen = {int(center)}
        for _ in range(hops):
            neighbors = set()
            for node in frontier:
                neighbors.update(U.indices[U.indptr[node]:U.indptr[node + 1]].tolist())
            next_nodes = sorted(neighbors - seen)
            if max_nodes is not None:
                next_nodes = next_nodes[:max_nodes - len(selected)]
            selected.extend(next_nodes)
            seen.update(next_nodes)
            frontier = next_nodes
            if not frontier or (max_nodes is not None and len(selected) >= max_nodes):
                break
        B = A[selected, :][:, selected]
        graph = nx.from_scipy_sparse_array(B, create_using=nx.DiGraph)
        seed_set = set(seeds)
        for i, node in enumerate(selected):
            graph.nodes[i].update(transaction_index=node,
                                  label='ROOT' if node == center else 'SEED' if node in seed_set else 'OTHER')
        graph.graph.update(hops=hops, direction=direction, max_nodes=max_nodes)
        return graph

    def make_dataset(self, time_steps, hops=2, feature_set='Local'):
        from .data import EllipticData
        features, labels, times = [], [], []
        for t in time_steps:
            p = self.prepare_period(t)
            if not len(p.candidates):
                continue
            original = p.frame[EllipticData.feature_columns(feature_set)]
            combined = pd.concat([original, self.structural_features(p.adjacency, p.seeds, hops)], axis=1)
            features.append(combined.iloc[p.candidates])
            labels.append(p.labels[p.candidates])
            times.append(np.full(len(p.candidates), t))
        if not features:
            raise ValueError('No labelled non-seed candidates in requested periods')
        return pd.concat(features, ignore_index=True), np.concatenate(labels), np.concatenate(times)

    def sample_subgraphs(self, time_steps, size=120, hops=2, max_nodes=30, direction='both', seed=42):
        """Sample roots once per split; identical roots for all hop settings."""
        roots, labels = [], []
        for t in time_steps:
            p = self.prepare_period(t)
            roots.extend((t, int(i)) for i in p.candidates)
            labels.extend(p.labels[p.candidates])
        labels = np.asarray(labels)
        idx = StratifiedSampler.indices(labels, size, seed)
        graphs = []
        for j in idx:
            t, root = roots[j]
            p = self.prepare_period(t)
            graphs.append(self.extract_subgraph(p.adjacency, root, p.seeds, hops, max_nodes, direction))
        return graphs, labels[idx]


class WeisfeilerLehmanKernel:
    """Normalized WL subtree counts, train-only vocabulary, optional directed WL.

    No GraKeL dependency. n_iter is refinement depth, independent of ego hops.
    Stable hashes align unseen/test graph colors without fitting on test graphs.
    """
    def __init__(self, n_iter=2, directed=False):
        if n_iter < 0:
            raise ValueError('n_iter must be nonnegative')
        self.n_iter = n_iter
        self.directed = directed

    def _counts(self, graph):
        from collections import Counter
        colors = {n: str(d.get('label', 'OTHER')) for n, d in graph.nodes(data=True)}
        counts = Counter(f'0:{c}' for c in colors.values())
        for iteration in range(1, self.n_iter + 1):
            updated = {}
            for n in graph:
                if self.directed:
                    neighbors = (sorted(colors[v] for v in graph.predecessors(n)),
                                 sorted(colors[v] for v in graph.successors(n)))
                else:
                    neighbors = sorted(colors[v] for v in set(graph.predecessors(n)) | set(graph.successors(n)))
                updated[n] = hashlib.sha256(repr((colors[n], neighbors)).encode()).hexdigest()
            colors = updated
            counts.update(f'{iteration}:{c}' for c in colors.values())
        return counts

    def fit_transform(self, graphs):
        self.vectorizer_ = DictVectorizer()
        self.train_ = normalize(self.vectorizer_.fit_transform([self._counts(g) for g in graphs]))
        return (self.train_ @ self.train_.T).toarray()

    def transform(self, graphs):
        # Include unseen colors in the norm, as a true normalized kernel must.
        counts = [self._counts(g) for g in graphs]
        norms = np.asarray([np.sqrt(sum(v * v for v in c.values())) for c in counts])
        X = self.vectorizer_.transform(counts)
        X = sparse.diags(np.divide(1., norms, out=np.zeros_like(norms), where=norms > 0)) @ X
        return (X @ self.train_.T).toarray()
