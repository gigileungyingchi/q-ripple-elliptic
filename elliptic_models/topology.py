"""Small rooted directed graphs as fixed-register quantum inputs.

Ordering is a structural heuristic, NOT an isomorphism-canonical labeling.
Permutation sensitivity is measured rather than claiming invariance.
"""
from dataclasses import dataclass
from collections import deque
import hashlib

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from sklearn.preprocessing import StandardScaler

from .graph import RiskGraph
from .data import EllipticData
from .quantum import StatevectorKernel


@dataclass
class TopologySample:
    adjacency: np.ndarray
    attributes: np.ndarray
    seeds: np.ndarray
    mask: np.ndarray
    node_ids: np.ndarray
    reachable_nodes: int
    hops: int

    def to_graph(self):
        count = int(self.mask.sum())
        graph = nx.from_numpy_array(self.adjacency[:count, :count], create_using=nx.DiGraph)
        for node in graph:
            graph.nodes[node]['label'] = 'ROOT' if node == 0 else 'SEED' if self.seeds[node] else 'OTHER'
        return graph


class RootedSubgraphSampler:
    """Explore radius first, then select up to n_qubits with root-path closure.

    Policies: nearest, hubs, seeds, paths (seed-path coverage). All retain root
    and choose whole BFS-parent paths when they fit; sampled graphs stay weakly
    connected. Search direction changes reachability, original edge orientation
    is always retained. Hidden labels are never read by sample().
    """
    policies = ('nearest', 'hubs', 'seeds', 'paths')

    def __init__(self, risk_graph, n_qubits=6, feature_columns=None):
        feature_columns = EllipticData.feature_columns('All') if feature_columns is None else feature_columns
        if n_qubits < 2 or len(feature_columns) < 2:
            raise ValueError('Use at least two qubits and two or more node attributes')
        if any(c in ('label', 'class', 'txId') for c in feature_columns):
            raise ValueError('Only observed numerical transaction attributes may be encoded')
        self.risk_graph, self.n_qubits = risk_graph, n_qubits
        self.feature_columns = tuple(feature_columns)
        self._connectivity = {}

    def sample(self, time_step, root, hops=3, policy='nearest', direction='both'):
        if policy not in self.policies or not isinstance(hops, int) or hops < 0:
            raise ValueError('Invalid sampling policy or radius')
        period = self.risk_graph.prepare_period(int(time_step))
        A = period.adjacency
        if not 0 <= root < A.shape[0]:
            raise ValueError('Root outside period graph')
        key = (int(time_step), direction)
        if key not in self._connectivity:
            self._connectivity[key] = RiskGraph.connectivity(A, direction)
        U = self._connectivity[key]
        degree = np.asarray((A + A.T).sum(axis=1)).ravel()
        seed_set = set(period.seeds.tolist())
        # Two values break ordering ties; every requested attribute is encoded below.
        attrs = period.frame[list(self.feature_columns[:2])].to_numpy(dtype=float)
        txids = period.frame.txId.to_numpy()

        def tie_key(n):
            # ID is only the final tie-breaker; this still is not canonical.
            return (n not in seed_set, -float(degree[n]), tuple(attrs[n]), str(txids[n]))

        root = int(root)
        distances, parents = {root: 0}, {}
        queue = deque([root])
        while queue:
            node = queue.popleft()
            if distances[node] >= hops:
                continue
            neighbors = U.indices[U.indptr[node]:U.indptr[node + 1]]
            for other in sorted(map(int, neighbors), key=tie_key):
                if other not in distances:
                    distances[other] = distances[node] + 1
                    parents[other] = node
                    queue.append(other)

        def path(node):
            result = []
            while node != root:
                result.append(node)
                node = parents[node]
            return result[::-1]

        seed_paths = set()
        for node in distances.keys() & seed_set:
            seed_paths.update(path(node))
        def priority(n):
            if policy == 'hubs':
                return (-degree[n], distances[n], tie_key(n))
            if policy == 'seeds':
                return (n not in seed_set, distances[n], tie_key(n))
            if policy == 'paths':
                return (n not in seed_paths, distances[n], tie_key(n))
            return (distances[n], tie_key(n))

        selected, seen = [root], {root}
        for node in sorted(set(distances) - {root}, key=priority):
            addition = [v for v in path(node) if v not in seen]
            if len(selected) + len(addition) <= self.n_qubits:
                selected.extend(addition)
                seen.update(addition)
            if len(selected) == self.n_qubits:
                break
        # Ordering is independent of candidate visitation order after selection.
        nodes = [root] + sorted(seen - {root}, key=lambda n: (distances[n], tie_key(n)))
        q, n = self.n_qubits, len(nodes)
        B = np.zeros((q, q))
        B[:n, :n] = A[nodes, :][:, nodes].toarray()
        features = np.zeros((q, len(self.feature_columns)))
        features[:n] = period.frame.iloc[nodes][list(self.feature_columns)].to_numpy(dtype=float)
        mask, seeds = np.zeros(q), np.zeros(q)
        mask[:n] = 1
        seeds[:n] = [node in seed_set for node in nodes]
        ids = np.full(q, -1, dtype=txids.dtype)
        ids[:n] = txids[nodes]
        return TopologySample(B, features, seeds, mask, ids, len(distances), hops)

    def collect(self, roots, hops, policy='nearest', direction='both'):
        return [self.sample(t, root, hops, policy, direction) for t, root in roots]


class TopologyPreprocessor:
    """Train-only standardization of real node attributes; padded nodes stay zero.

    Repeated occurrences weight a transaction by neighborhood frequency. Neither
    transaction IDs nor hidden labels are part of the packed input.
    """
    def fit(self, samples):
        self.scaler_ = StandardScaler().fit(np.concatenate([
            sample.attributes[sample.mask.astype(bool)] for sample in samples]))
        self.n_qubits_ = len(samples[0].mask)
        return self

    def transform(self, samples):
        rows = []
        for sample in samples:
            if len(sample.mask) != self.n_qubits_:
                raise ValueError('All samples must use the same padded register')
            real = sample.mask.astype(bool)
            angles = np.zeros_like(sample.attributes)
            angles[real] = 2 * np.arctan(self.scaler_.transform(sample.attributes[real]))
            root = np.zeros(self.n_qubits_)
            root[0] = 1
            rows.append(np.concatenate([sample.mask, root, sample.seeds,
                                        angles.ravel(), sample.adjacency.ravel()]))
        return np.asarray(rows)

    def fit_transform(self, samples):
        return self.fit(samples).transform(samples)


class TopologyFeatureMap:
    """One qubit per retained node, padded to fixed q, observed edges as gates.

    rzx(gamma, source, destination) attaches the first gate wire to source and
    the second to destination (Qiskit's tensor matrix order is little-endian).
    Undirected control uses symmetric RZZ once per unordered pair. Edgeless
    control holds attributes and node slots fixed but drops all interactions.
    These gate families also differ, so directed-vs-undirected is not a pure
    causal test of edge direction. Reverse-edge RZX is the matched-gate control.
    """
    def __init__(self, n_qubits=6, repetitions=2, gamma=0.7, edges='directed'):
        if n_qubits < 2 or repetitions < 1:
            raise ValueError('Require >=2 qubits and positive repetitions')
        if edges not in ('directed', 'reverse', 'undirected', 'none'):
            raise ValueError('Invalid edge encoding')
        self.n_qubits, self.repetitions, self.gamma, self.edges = n_qubits, repetitions, gamma, edges
        # The common kernel/hardware interface needs the fixed logical width.
        self.circuit = QuantumCircuit(n_qubits)
        self._depths = {}

    @staticmethod
    def unpack(row, q):
        row = np.asarray(row, dtype=float)
        count, remainder = divmod(row.size - 3 * q - q * q, q)
        if remainder or count < 2:
            raise ValueError('Packed topology input has incorrect width')
        end = 3 * q + count * q
        return row[:q], row[q:2*q], row[2*q:3*q], row[3*q:end].reshape(q, count), row[end:].reshape(q, q)

    def bind(self, row):
        q = self.n_qubits
        mask, root, seeds, attributes, A = self.unpack(row, q)
        active = np.flatnonzero(mask)
        circuit = QuantumCircuit(q)
        for _ in range(self.repetitions):
            for node in active:
                circuit.h(int(node))
                circuit.ry(float(attributes[node, 0] + 0.2 + 0.7 * root[node] + 1.1 * seeds[node]), int(node))
                circuit.rz(float(attributes[node, 1] + 0.4 * root[node]), int(node))
                for feature in range(2, attributes.shape[1]):
                    gate = circuit.ry if feature % 2 == 0 else circuit.rz
                    gate(float(attributes[node, feature]), int(node))
            if self.edges in ('directed', 'reverse'):
                for source, dest in np.argwhere(A > 0):
                    if mask[source] and mask[dest] and source != dest:
                        wires = (int(source), int(dest)) if self.edges == 'directed' else (int(dest), int(source))
                        circuit.rzx(self.gamma, *wires)
            elif self.edges == 'undirected':
                for source, dest in np.argwhere(np.triu((A + A.T) > 0, k=1)):
                    circuit.rzz(self.gamma, int(source), int(dest))
        return circuit

    def logical_depth(self, row):
        mask, _, _, attributes, A = self.unpack(row, self.n_qubits)
        key = (mask.tobytes(), A.tobytes(), attributes.shape[1])
        if key not in self._depths:
            self._depths[key] = self.bind(row).depth()
        return self._depths[key]

    def simulation_bind(self, row):
        """Exactly fuse local rotations into one unitary per node and stage.

        No feature truncation: this equals bind(row), which retains every gate
        for hardware depth reporting. Fusion accelerates ideal simulation only.
        """
        mask, root, seeds, attributes, A = self.unpack(row, self.n_qubits)
        circuit = QuantumCircuit(self.n_qubits)
        unitaries = {}
        H = np.array([[1., 1.], [1., -1.]]) / np.sqrt(2)
        for node in np.flatnonzero(mask):
            angles = attributes[node].copy()
            angles[0] += 0.2 + 0.7 * root[node] + 1.1 * seeds[node]
            angles[1] += 0.4 * root[node]
            unitary = H.astype(complex)
            for feature, angle in enumerate(angles):
                if feature % 2 == 0:
                    c, s = np.cos(angle / 2), np.sin(angle / 2)
                    unitary = np.array([[c, -s], [s, c]]) @ unitary
                else:
                    unitary = np.diag(np.exp([-0.5j * angle, 0.5j * angle])) @ unitary
            unitaries[int(node)] = unitary
        for _ in range(self.repetitions):
            for node, unitary in unitaries.items():
                circuit.unitary(unitary, [node])
            if self.edges in ('directed', 'reverse'):
                for source, dest in np.argwhere(A > 0):
                    if mask[source] and mask[dest] and source != dest:
                        wires = (int(source), int(dest)) if self.edges == 'directed' else (int(dest), int(source))
                        circuit.rzx(self.gamma, *wires)
            elif self.edges == 'undirected':
                for source, dest in np.argwhere(np.triu((A + A.T) > 0, k=1)):
                    circuit.rzz(self.gamma, int(source), int(dest))
        return circuit


class TopologyDiagnostics:
    @staticmethod
    def retention(samples):
        import pandas as pd
        return pd.DataFrame([{'Retained nodes': int(s.mask.sum()), 'Reachable nodes': s.reachable_nodes,
                              'Retention fraction': s.mask.sum() / s.reachable_nodes,
                              'Truncated': s.mask.sum() < s.reachable_nodes,
                              'Retained seeds': int(s.seeds.sum()), 'Hops': s.hops}
                             for s in samples])

    @staticmethod
    def permute_inputs(X, q, seed=42):
        """Permute real non-root slots, carrying all markers/attributes/edges."""
        rng = np.random.default_rng(seed)
        rows = []
        for row in X:
            mask, root, seeds, attrs, A = TopologyFeatureMap.unpack(row, q)
            order = np.arange(q)
            nonroot = np.flatnonzero((mask > 0) & (root == 0))
            order[nonroot] = rng.permutation(nonroot)
            rows.append(np.concatenate([mask[order], root[order], seeds[order],
                                        attrs[order].ravel(), A[order, :][:, order].ravel()]))
        return np.asarray(rows)

    @classmethod
    def kernel_sensitivity(cls, X, feature_map, trials=3, seed=42):
        """Measure K(X,X) changes after independent per-graph slot permutations."""
        kernel = StatevectorKernel(feature_map)
        baseline = kernel.evaluate(X)
        rows = []
        for trial in range(trials):
            permuted = cls.permute_inputs(X, feature_map.n_qubits, seed + trial)
            changed = kernel.evaluate(permuted)
            rows.append({'Trial': trial, 'Mean absolute kernel change': np.abs(changed - baseline).mean(),
                         'Max absolute kernel change': np.abs(changed - baseline).max()})
        import pandas as pd
        return pd.DataFrame(rows)


class PermutationAveragedKernel:
    """Finite-mixture density kernel; a robustness option, not exact invariance.

    Each graph uses the same deterministic set of random slot permutations,
    independent of its features/labels. Average every cross-permutation fidelity,
    giving a PSD kernel Tr(rho_x rho_y). Normalize by the density self-overlaps.
    Hardware needs trials**2 more pairs and is not implemented for this wrapper.
    """
    def __init__(self, feature_map, trials=4, seed=42):
        if trials < 1:
            raise ValueError('trials must be positive')
        self.feature_map, self.trials, self.seed = feature_map, trials, seed
        self._cache = {}

    def _states(self, X):
        key = hashlib.sha256(np.asarray(X).tobytes()).hexdigest()
        if key not in self._cache:
            states = []
            for trial in range(self.trials):
                # Reset the same RNG per graph to make the mapping a function of x.
                rows = np.vstack([TopologyDiagnostics.permute_inputs(
                    [row], self.feature_map.n_qubits, self.seed + trial) for row in X])
                states.append(StatevectorKernel(self.feature_map).encode(rows))
            self._cache[key] = np.stack(states, axis=1)
        return self._cache[key]

    def evaluate(self, X, Y=None):
        sx = self._states(X)
        sy = sx if Y is None else self._states(Y)
        values = np.mean(np.abs(np.einsum('iad,jbd->ijab', sx, sy.conj())) ** 2, axis=(2, 3))
        nx = np.mean(np.abs(np.einsum('iad,ibd->iab', sx, sx.conj())) ** 2, axis=(1, 2))
        ny = nx if Y is None else np.mean(np.abs(np.einsum('iad,ibd->iab', sy, sy.conj())) ** 2, axis=(1, 2))
        return values / np.sqrt(nx[:, None] * ny[None, :])
