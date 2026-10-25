"""Hamiltonian convention, invariance, directed walks and matched experiments."""
import unittest
from dataclasses import replace
import numpy as np
import pandas as pd
import networkx as nx
from qiskit.quantum_info import Statevector
from qiskit.primitives import StatevectorSampler
from qiskit.providers.fake_provider import GenericBackendV2

from elliptic_models import (TopologySample, TopologyPreprocessor, TopologyDiagnostics,
    GraphHamiltonianFeatureMap, GraphMessageFeatureMap, HermitianWalkFeatureMap,
    HamiltonianMomentKernel, StructuralProbeStudy, GraphDescriptors, StatevectorKernel,
    TopologyEncodingStudy, RiskGraph, IBMFidelityKernel, Evaluation)


def sample():
    return TopologySample(np.array([[0., 1., 0.], [0., 0., 1.], [0., 0., 0.]]),
        np.array([[.1, .2], [.3, -.4], [-.5, .6]]), np.array([0., 0., 1.]),
        np.ones(3), np.arange(3), 3, 2)


class DynamicsTests(unittest.TestCase):
    def setUp(self):
        self.X=TopologyPreprocessor().fit_transform([sample()])

    def test_hamiltonian_hermitian_and_exact_fusion(self):
        fm=GraphHamiltonianFeatureMap(3)
        H=fm.hamiltonian(self.X[0]).to_matrix()
        np.testing.assert_allclose(H,H.conj().T)
        a=Statevector.from_instruction(fm.bind(self.X[0])).data
        b=Statevector.from_instruction(fm.simulation_bind(self.X[0])).data
        np.testing.assert_allclose(a,b,atol=1e-12)
        # Logical source=0 has Z, destination=1 has X in little-endian labels.
        self.assertIn('IXZ',fm.hamiltonian(self.X[0]).to_list()[3][0])

    def test_product_formula_converges_at_fixed_time(self):
        errors=[]
        for steps in (1, 4, 8):
            fm=GraphHamiltonianFeatureMap(3,steps=steps,time=1.2)
            exact=fm.exact_state(self.X[0])
            state=Statevector.from_instruction(fm.simulation_bind(self.X[0])).data
            errors.append(1-abs(np.vdot(exact,state))**2)
        self.assertLess(errors[-1],errors[1])
        self.assertLess(errors[1],errors[0])

    def test_exact_observable_kernel_is_relabeling_invariant_and_psd(self):
        X=np.vstack([self.X,TopologyDiagnostics.permute_inputs(self.X,3,seed=1)])
        kernel=HamiltonianMomentKernel(GraphHamiltonianFeatureMap(3),exact=True)
        np.testing.assert_allclose(kernel.features(X)[0],kernel.features(X)[1],atol=1e-12)
        K=kernel.evaluate(X)
        np.testing.assert_allclose(K,1.,atol=1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(K).min(),-1e-10)

    def test_walk_uses_valid_hermitian_lift_and_ignores_numeric_attributes(self):
        walk=HermitianWalkFeatureMap(3)
        H=walk.hamiltonian(self.X[0])
        np.testing.assert_allclose(H,H.T.conj())
        self.assertEqual(walk.circuit.num_qubits,3)
        changed=replace(sample(),attributes=sample().attributes+100)
        prep=TopologyPreprocessor().fit([sample()])
        Z=prep.transform([changed])
        np.testing.assert_allclose(walk.hamiltonian(self.X[0]),walk.hamiltonian(Z[0]))
        state=Statevector.from_instruction(walk.bind(self.X[0])).data
        self.assertAlmostEqual(np.linalg.norm(state),1.)
        reversed_sample=replace(sample(),adjacency=sample().adjacency.T)
        B=prep.transform([reversed_sample])
        self.assertFalse(np.allclose(H,walk.hamiltonian(B[0])))

    def test_generic_ring_removes_input_adjacency_but_graph_map_uses_it(self):
        prep=TopologyPreprocessor().fit([sample()])
        X=prep.transform([sample(),replace(sample(),adjacency=np.zeros((3,3)))])
        generic=StatevectorKernel(GraphMessageFeatureMap(3,edges='generic')).evaluate(X)
        graph=StatevectorKernel(GraphMessageFeatureMap(3)).evaluate(X)
        self.assertAlmostEqual(generic[0,1],1.)
        self.assertLess(graph[0,1],.999)

    def test_summary_collision_is_not_a_collision_for_all_classical_methods(self):
        a,b=StructuralProbeStudy.cases()['Degree + triangle collision (8 nodes)']
        np.testing.assert_array_equal(GraphDescriptors.summary(a),GraphDescriptors.summary(b))
        self.assertFalse(nx.is_isomorphic(a.to_graph(),b.to_graph()))
        self.assertFalse(np.allclose(GraphDescriptors.spectral(a),GraphDescriptors.spectral(b)))

    def test_hamiltonian_hardware_adapter_with_local_finite_shots(self):
        X=np.vstack([self.X,TopologyDiagnostics.permute_inputs(self.X,3,seed=1)])
        fm=GraphHamiltonianFeatureMap(3,steps=2)
        backend=GenericBackendV2(num_qubits=3,seed=8)
        kernel=IBMFidelityKernel(fm,backend,shots=512,max_circuits=3,
                                sampler=StatevectorSampler(seed=4))
        observed=kernel.evaluate(X)
        np.testing.assert_allclose(observed,StatevectorKernel(fm).evaluate(X),atol=.1)


class EncodingStudyTests(unittest.TestCase):
    def test_matched_classification_and_retention(self):
        rng=np.random.default_rng(82)
        frame=pd.DataFrame(rng.normal(size=(12,165)),columns=[f'x{i}' for i in range(165)])
        frame['txId'],frame['time_step'],frame['label']=np.arange(12),1,[0,1]*6
        graph=RiskGraph(frame,pd.DataFrame({'txId1':np.arange(11),'txId2':np.arange(1,12)}),seed_fraction=0)
        roots={s:np.c_[np.ones(12,dtype=int),np.arange(12)] for s in ('train','val','test')}
        y={s:np.array([0,1]*6) for s in roots}
        study=TopologyEncodingStudy(graph,roots,y,feature_set='Local',n_landmarks=4)
        results=study.configuration(nodes=3,hops=2,names=['Hamiltonian quantum','Hermitian walk quantum'])
        primary=results[~results.Diagnostic]
        self.assertTrue(np.isfinite(primary[list(Evaluation.CORE_METRICS)].to_numpy()).all())
        run=study.runs[(3,2,2)]
        for name in ['Raw + Hamiltonian quantum','Raw + Hermitian walk quantum']:
            estimator=run.models[name]['estimator']
            raw=study.full_inputs.raw(2)['train']
            np.testing.assert_array_equal(estimator.training_features_[:,:raw.shape[1]],raw)
            self.assertEqual(len(study.full_inputs.columns),93)
        retention=pd.concat(study.retention)
        self.assertTrue(retention['Maximum retained distance'].le(2).all())
        self.assertTrue(retention['Edge retention fraction'].between(0,1).all())
        self.assertTrue(retention['Retained nodes'].le(3).all())


if __name__=='__main__':unittest.main()
