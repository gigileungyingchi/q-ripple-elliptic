"""Behavioral checks: leakage boundaries, graph radius and measured fidelities."""
import tempfile
import unittest
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from qiskit.primitives import StatevectorSampler
from qiskit.providers.fake_provider import GenericBackendV2

from elliptic_models import (AnglePreprocessor, Evaluation, IBMFidelityKernel,
    NystromProjector, QuantumFeatureMap, RiskGraph, StatevectorKernel,
    StratifiedSampler, WeisfeilerLehmanKernel)


class ModelTests(unittest.TestCase):
    def test_radius_direction_and_nearest_truncation(self):
        # Root 4 -> 3 -> 0 -> 1; lower node IDs must not displace nearer nodes.
        A = sparse.csr_matrix(([1., 1., 1.], ([4, 3, 0], [3, 0, 1])), shape=(5, 5))
        for hops, expected in [(0, {4}), (1, {4, 3}), (2, {4, 3, 0}), (3, {4, 3, 0, 1})]:
            g = RiskGraph.extract_subgraph(A, 4, np.array([0]), hops, None, 'out')
            self.assertEqual({d['transaction_index'] for _, d in g.nodes(data=True)}, expected)
        capped = RiskGraph.extract_subgraph(A, 4, np.array([0]), 3, 2, 'out')
        self.assertEqual({d['transaction_index'] for _, d in capped.nodes(data=True)}, {4, 3})
        incoming = RiskGraph.extract_subgraph(A, 4, [], 3, None, 'in')
        self.assertEqual(len(incoming), 1)

    def test_diffusion_sinks_and_hop_exposure(self):
        A = sparse.csr_matrix(([1., 1.], ([0, 1], [1, 2])), shape=(3, 3))
        scores = RiskGraph.diffusion(A, [0], n_iter=100)
        self.assertAlmostEqual(scores.sum(), 1.)
        features = RiskGraph.structural_features(A, [0], hops=3)
        self.assertEqual(features.loc[2, 'seed_from_1'], 0)
        self.assertAlmostEqual(features.loc[2, 'seed_from_2'], np.log(2))
        self.assertIn('seed_from_3', features)
        np.testing.assert_array_equal(RiskGraph.diffusion(A, []), np.zeros(3))

    def test_seeds_are_excluded_and_unknown_topology_retained(self):
        frame = pd.DataFrame({'txId': range(6), 'time_step': [1] * 6,
                              'label': [1, 1, 0, 0, -1, 1]})
        graph = RiskGraph(frame, pd.DataFrame({'txId1': [0, 4], 'txId2': [4, 2]}))
        p = graph.prepare_period(1)
        self.assertFalse(set(p.seeds) & set(p.candidates))
        self.assertNotIn(4, p.candidates)
        self.assertEqual(p.adjacency[4, 2], 1)
        ego = graph.extract_subgraph(p.adjacency, 2, p.seeds, 3, None)
        for _, attrs in ego.nodes(data=True):
            if attrs['transaction_index'] not in p.seeds and attrs['transaction_index'] != 2:
                self.assertEqual(attrs['label'], 'OTHER')

    def test_nystrom_full_landmarks_recovers_kernel(self):
        X = np.array([[0., 0.], [0.2, 1.], [1.5, 0.3]])
        kernel = StatevectorKernel(QuantumFeatureMap.uploading(2, 2, 2))
        W = kernel.evaluate(X)
        features = NystromProjector().fit(W).transform(W)
        np.testing.assert_allclose(features @ features.T, W, atol=1e-9)
        np.testing.assert_allclose(np.diag(W), 1.)
        np.testing.assert_allclose(kernel.evaluate(X, X), W)

    def test_train_only_preprocessing(self):
        X = np.array([[0., 1.], [1., 0.], [2., 3.], [3., 1.]])
        prep = AnglePreprocessor(2).fit(X)
        before = prep.pipeline.named_steps['standard'].mean_.copy()
        test = prep.transform(np.array([[1e6, -1e6]]))
        np.testing.assert_array_equal(before, prep.pipeline.named_steps['standard'].mean_)
        self.assertTrue(np.all((test >= 0) & (test <= np.pi)))
        idx = StratifiedSampler.landmarks([0, 0, 0, 1], 4)
        self.assertEqual(len(set(idx)), 4)

    def test_wl_kernel_alignment(self):
        g = nx.DiGraph([(0, 1), (1, 2)])
        nx.set_node_attributes(g, {0: 'ROOT', 1: 'OTHER', 2: 'SEED'}, 'label')
        renamed = nx.relabel_nodes(g, {0: 7, 1: 9, 2: 4})
        kernel = WeisfeilerLehmanKernel()
        K = kernel.fit_transform([g, renamed])
        np.testing.assert_allclose(K, np.ones((2, 2)))
        np.testing.assert_allclose(kernel.transform([renamed]), np.ones((1, 2)))

    def test_runtime_path_with_local_finite_shot_sampler(self):
        fm = QuantumFeatureMap.uploading(2, 2, 2)
        X = np.array([[0.2, 0.3], [1.4, 0.8]])
        backend = GenericBackendV2(num_qubits=2, basis_gates=['id', 'rz', 'sx', 'x', 'cx'])
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'jobs.json'
            kernel = IBMFidelityKernel(fm, backend, shots=16000, batch_size=2,
                max_circuits=3, log_path=log, sampler=StatevectorSampler(seed=8))
            measured = kernel.evaluate(X)
            np.testing.assert_allclose(measured, StatevectorKernel(fm).evaluate(X), atol=0.025)
            self.assertEqual(kernel.submitted_circuits, 3)
            self.assertTrue(log.exists())
            with self.assertRaises(ValueError):
                kernel.evaluate(X)

    def test_discovery_metrics(self):
        y = [0, 1, 1]
        scores = [0.8, 0.9, 0.2]
        self.assertEqual(Evaluation.inspections_to_find(y, scores, 2), 3)
        result = Evaluation.discovery(y, scores, [1, 10])
        self.assertEqual(result.iloc[0]['found'], 1)
        self.assertEqual(result.iloc[1]['budget'], 3)


if __name__ == '__main__':
    unittest.main()
