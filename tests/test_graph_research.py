"""Graph semantics, encoding safety, permutation diagnostics and hardware adapter."""
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy import sparse
from qiskit.quantum_info import Statevector
from qiskit.primitives import StatevectorSampler
from qiskit.providers.fake_provider import GenericBackendV2

from elliptic_models import (DirectionalPropagation, HopFeatureMap, RiskGraph,
    RootedSubgraphSampler, TopologyPreprocessor, TopologySample, TopologyFeatureMap,
    TopologyDiagnostics, PermutationAveragedKernel, StatevectorKernel, IBMFidelityKernel,
    KernelComparison)


def example_sample():
    return TopologySample(np.array([[0., 1., 0., 0.], [0., 0., 1., 0.],
                                    [0., 0., 0., 0.], [0., 0., 0., 0.]]),
                          np.array([[0.2, 0.5], [1.2, -0.4], [-0.6, 0.9], [0., 0.]]),
                          np.array([0., 0., 1., 0.]), np.array([1., 1., 1., 0.]),
                          np.array([10, 20, 30, -1]), 5, 3)


class GraphResearchTests(unittest.TestCase):
    def test_normalized_directional_signals(self):
        # 0->2 and 1->2, 2->3. At node 2 half its predecessors are seeds.
        A = sparse.csr_matrix(([1., 1., 1.], ([0, 1, 2], [2, 2, 3])), shape=(5, 5))
        values = DirectionalPropagation(8).transform(A, [0])
        self.assertEqual(values.shape, (5, 32))
        self.assertEqual(values.loc[2, 'in_seed_h1'], 0.5)
        self.assertEqual(values.loc[3, 'in_seed_h2'], 0.5)
        self.assertEqual(values.loc[3, 'out_seed_h2'], 0.)
        np.testing.assert_array_equal(values.loc[4], np.zeros(32))
        outgoing_seed = DirectionalPropagation(2).transform(A, [3])
        self.assertEqual(outgoing_seed.loc[0, 'out_seed_h2'], 1.)

    def test_hop_order_and_repetition_are_independent(self):
        X = np.linspace(0.1, 2., 12)
        forward = HopFeatureMap.build(3)
        reverse = HopFeatureMap.build(3, order='reverse')
        twice = HopFeatureMap.build(3, repetitions=2)
        self.assertEqual(len(twice.parameters), 12)
        self.assertGreater(twice.circuit.depth(), forward.circuit.depth())
        a = Statevector.from_instruction(forward.bind(X)).data
        b = Statevector.from_instruction(reverse.bind(X)).data
        self.assertLess(abs(np.vdot(a, b)) ** 2, 0.999)

    def test_sampler_paths_padding_and_no_target_labels(self):
        frame = pd.DataFrame({'txId': [10, 11, 12, 13, 14], 'time_step': [1]*5,
                              'label': [1, 0, 0, -1, 0],
                              'x0': [0., 1., 2., 3., 4.], 'x1': [4., 3., 2., 1., 0.]})
        edges = pd.DataFrame({'txId1': [10, 11, 12, 11], 'txId2': [11, 12, 13, 14]})
        graph = RiskGraph(frame, edges)
        sampler = RootedSubgraphSampler(graph, n_qubits=4, feature_columns=('x0', 'x1'))
        for policy in sampler.policies:
            sample = sampler.sample(1, 4, hops=8, policy=policy)
            self.assertEqual(sample.node_ids[0], 14)
            self.assertEqual(sample.reachable_nodes, 5)
            self.assertLessEqual(sample.mask.sum(), 4)
            import networkx as nx
            self.assertTrue(nx.is_weakly_connected(sample.to_graph()))
        # Freeze revealed seeds; changing hidden labels must change no circuit inputs.
        first = sampler.sample(1, 4)
        period = graph.prepare_period(1)
        period.frame.loc[period.candidates, 'label'] = 1
        second = sampler.sample(1, 4)
        prep = TopologyPreprocessor().fit([first])
        np.testing.assert_array_equal(prep.transform([first]), prep.transform([second]))
        isolated = sampler.sample(1, 4, hops=0)
        self.assertEqual(isolated.mask.sum(), 1)
        row = prep.transform([isolated])[0]
        mask, _, _, attrs, A = TopologyFeatureMap.unpack(row, 4)
        np.testing.assert_array_equal(attrs[mask == 0], 0.)
        self.assertEqual(A.sum(), 0)

    def test_train_only_node_scaling_and_ids_not_encoded(self):
        sample = example_sample()
        prep = TopologyPreprocessor().fit([sample])
        before = prep.scaler_.mean_.copy()
        hidden = replace(sample, attributes=sample.attributes * 1e6)
        prep.transform([hidden])
        np.testing.assert_array_equal(prep.scaler_.mean_, before)
        renamed = replace(sample, node_ids=np.array([1000, 2000, 3000, -1]))
        np.testing.assert_array_equal(prep.transform([sample]), prep.transform([renamed]))

    def test_directed_edges_root_and_padding_affect_state(self):
        sample = example_sample()
        row = TopologyPreprocessor().fit_transform([sample])[0]
        forward = TopologyFeatureMap(4, edges='directed')
        reverse = TopologyFeatureMap(4, edges='reverse')
        a = Statevector.from_instruction(forward.bind(row)).data
        b = Statevector.from_instruction(reverse.bind(row)).data
        self.assertLess(abs(np.vdot(a, b))**2, 0.999)
        # Fourth qubit is padding: it remains |0> in little-endian indexing.
        self.assertAlmostEqual(np.sum(np.abs(a[8:])**2), 0.)
        changed = row.copy()
        changed[4:8] = 0.  # remove root marker only
        c = Statevector.from_instruction(forward.bind(changed)).data
        self.assertLess(abs(np.vdot(a, c))**2, 0.999)

    def test_permutation_mixture_is_consistent_psd_kernel(self):
        sample = example_sample()
        other = replace(sample, attributes=sample.attributes + np.array([0.3, -0.2]))
        X = TopologyPreprocessor().fit_transform([sample, other])
        X = np.vstack([X, TopologyDiagnostics.permute_inputs(X[:1], 4, seed=7)])
        kernel = PermutationAveragedKernel(TopologyFeatureMap(4), trials=3)
        K = kernel.evaluate(X)
        np.testing.assert_allclose(K, K.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(K), 1., atol=1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(K).min(), -1e-10)
        # Evaluating rows separately must give the same function of each graph.
        np.testing.assert_allclose(kernel.evaluate(X[:1], X[1:]), K[:1, 1:])

    def test_topology_runtime_compute_uncompute(self):
        sample = example_sample()
        X = TopologyPreprocessor().fit_transform([sample, replace(sample, adjacency=sample.adjacency.T)])
        fm = TopologyFeatureMap(4)
        backend = GenericBackendV2(num_qubits=4, basis_gates=['id', 'rz', 'sx', 'x', 'cx'])
        with tempfile.TemporaryDirectory() as directory:
            hardware = IBMFidelityKernel(fm, backend, shots=12000, max_circuits=3,
                sampler=StatevectorSampler(seed=11), log_path=Path(directory)/'jobs.json')
            np.testing.assert_allclose(hardware.evaluate(X), StatevectorKernel(fm).evaluate(X), atol=0.025)


class FullInputTests(unittest.TestCase):
    def test_full_local_columns_survive_quantum_readout_unchanged(self):
        from elliptic_models import AdditiveKernelClassifier
        rng = np.random.default_rng(51)
        raw = rng.normal(size=(12, 165))
        X = rng.normal(size=(12, 4))
        y = np.array([0, 1] * 6)
        model = AdditiveKernelClassifier(StatevectorKernel(HopFeatureMap.build(1)), n_landmarks=4)
        model.fit(raw, X, y)
        np.testing.assert_array_equal(model.training_features_[:, :165], raw)
        transformed = model.features(raw, X)
        np.testing.assert_array_equal(transformed[:, :165], raw)
        self.assertGreater(transformed.shape[1], 165)

    def test_legacy_two_hop_counts_match_original_definition(self):
        from elliptic_models import FullGraphInputs
        A = sparse.csr_matrix(([1., 1., 1.], ([0, 1, 2], [1, 2, 0])), shape=(4, 4))
        U = RiskGraph.connectivity(A)
        original = (U @ U).toarray()
        np.fill_diagonal(original, 0)
        features = FullGraphInputs.legacy_features(A, [0])
        np.testing.assert_allclose(features.log_two_step_walks, np.log1p(original.sum(axis=1)))
        self.assertEqual(features.shape[1], 9)

    def test_full_node_attributes_and_exact_simulator_fusion(self):
        rng = np.random.default_rng(52)
        sample = example_sample()
        full = replace(sample, attributes=rng.normal(size=(4, 165)))
        X = TopologyPreprocessor().fit_transform([full])
        _, _, _, attrs, _ = TopologyFeatureMap.unpack(X[0], 4)
        self.assertEqual(attrs.shape, (4, 165))
        fm = TopologyFeatureMap(4)
        exact = Statevector.from_instruction(fm.bind(X[0])).data
        fused = Statevector.from_instruction(fm.simulation_bind(X[0])).data
        np.testing.assert_allclose(exact, fused, atol=1e-11)
        np.testing.assert_allclose(StatevectorKernel(fm).encode(X)[0], exact, atol=1e-11)
        changed = X.copy()
        # Change feature 164 of the root: late attributes must affect the state.
        changed[0, 3 * 4 + 164] += .8
        self.assertLess(abs(np.vdot(exact, StatevectorKernel(fm).encode(changed)[0]))**2, .999)

    def test_full_graph_inputs_align_roots_and_preserve_original_prefix(self):
        from elliptic_models import FullGraphInputs
        rng = np.random.default_rng(53)
        columns = [f'x{i}' for i in range(165)]
        frame = pd.DataFrame(rng.normal(size=(6, 165)), columns=columns)
        frame['txId'], frame['time_step'], frame['label'] = np.arange(6), 1, [0, 1, 0, 1, -1, 0]
        graph = RiskGraph(frame, pd.DataFrame({'txId1': [0, 1, 2], 'txId2': [1, 2, 3]}), seed_fraction=0)
        roots = {'train': np.array([[1, 3], [1, 0], [1, 2]])}
        labels = {'train': np.array([1, 0, 0])}
        inputs = FullGraphInputs(graph, roots, labels)
        expected = frame.iloc[[3, 0, 2]][columns].to_numpy()
        np.testing.assert_array_equal(inputs.original['train'], expected)
        for hops in [1, 2, 8]:
            np.testing.assert_array_equal(inputs.raw(hops)['train'][:, :165], expected)
        self.assertEqual(inputs.legacy()['train'].shape[1], 9)
        with self.assertRaises(ValueError):
            FullGraphInputs(graph, roots, {'train': np.array([0, 0, 0])})


if __name__ == '__main__':
    unittest.main()
