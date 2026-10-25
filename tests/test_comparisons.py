"""Prevent missing feature arms, leaked aggregate attributes, and unmatched pairs."""
import unittest
import numpy as np
import pandas as pd

from elliptic_models import (RiskGraph, PropagationStudy, FullGraphInputs,
    TopologyStudy, TopologyFeatureMap, MatchedComparisons, Evaluation)


class MatchedStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(71)
        frame = pd.DataFrame(rng.normal(size=(12, 165)), columns=[f'x{i}' for i in range(165)])
        frame['txId'], frame['time_step'], frame['label'] = np.arange(12), 1, [0, 1] * 6
        edges = pd.DataFrame({'txId1': np.arange(11), 'txId2': np.arange(1, 12)})
        cls.graph = RiskGraph(frame, edges, seed_fraction=0)
        # Shared periods here test plumbing only, not scientific performance.
        cls.study = PropagationStudy(cls.graph, {s: [1] for s in ('train', 'val', 'test')},
            {s: 12 for s in ('train', 'val', 'test')}, max_hops=1, n_landmarks=4)
        cls.results = cls.study.run_depths([1])

    def test_local_attributes_and_identical_graph_context(self):
        full = self.study.full_inputs
        for split in full.roots:
            local, all_inputs = full.local_raw(1)[split], full.raw(1)[split]
            np.testing.assert_array_equal(local[:, :93], full.local[split])
            np.testing.assert_array_equal(local[:, 93:], all_inputs[:, 165:])
            self.assertEqual(all_inputs.shape[1] - local.shape[1], 72)

    def test_every_propagation_arm_exists_with_five_metrics(self):
        for prefix in ('Local', 'Original'):
            for suffix in ('legacy propagation RF', 'propagation RF', 'propagation + RBF',
                           'propagation + hop quantum', 'propagation + generic quantum'):
                rows = self.results[self.results.Model == f'{prefix} + {suffix}']
                self.assertEqual(len(rows), 1)
                self.assertTrue(np.isfinite(rows[list(Evaluation.CORE_METRICS)].to_numpy()).all())
        for prefix in ('Local', 'Original'):
            pairs = MatchedComparisons.pair(self.results, f'{prefix} + propagation + RBF',
                f'{prefix} + propagation + hop quantum', ['Seed', 'Hops'], prefix)
            self.assertEqual(len(pairs), 1)
            for metric in Evaluation.CORE_METRICS:
                self.assertAlmostEqual(pairs[f'Δ {metric}'].iloc[0],
                    pairs[f'{metric} quantum'].iloc[0] - pairs[f'{metric} classical'].iloc[0])

    def test_local_topology_has_only_93_attributes_and_same_graphs(self):
        reference = self.study
        local = FullGraphInputs(self.graph, reference.roots, reference.y, max_hops=1, feature_set='Local')
        all_study = TopologyStudy(self.graph, reference.roots, reference.y, n_qubits=3,
                                 n_landmarks=4, full_inputs=reference.full_inputs)
        local_study = TopologyStudy(self.graph, reference.roots, reference.y, n_qubits=3,
                                   n_landmarks=4, full_inputs=local)
        results = local_study.run(depths=[1], edge_modes=['directed'])
        quantum = results[results.Model == 'Local + graph + quantum (directed)'].iloc[0]
        self.assertEqual(quantum['Edge mode'], 'directed')
        all_samples = all_study.sampler.collect(reference.roots['train'], 1, 'nearest', 'both')
        for a, b in zip(all_samples, local_study.samples[1, 'nearest']['train']):
            np.testing.assert_array_equal(a.node_ids, b.node_ids)
            np.testing.assert_array_equal(a.adjacency, b.adjacency)
            self.assertEqual(b.attributes.shape[1], 93)
        row = local_study.inputs[1, 'nearest']['train'][0]
        self.assertEqual(TopologyFeatureMap.unpack(row, 3)[3].shape[1], 93)
        self.assertTrue({'Local + propagation RF', 'Local + graph + RBF',
                         'Local + graph + quantum (directed)'}.issubset(set(results.Model)))

    def test_pairing_rejects_missing_or_duplicate_configurations(self):
        args = ('Propagation RBF Nyström', 'Hop-aware quantum Nyström', ['Seed', 'Hops'], 'kernel')
        with self.assertRaises(ValueError):
            MatchedComparisons.pair(self.results[self.results.Model != args[1]], *args)
        with self.assertRaises(pd.errors.MergeError):
            MatchedComparisons.pair(pd.concat([self.results, self.results]), *args)


if __name__ == '__main__':
    unittest.main()
