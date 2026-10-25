"""Hardware bit order, observable uncertainty and recovery without resubmission."""
import json,tempfile,unittest
from pathlib import Path
import numpy as np
from qiskit.quantum_info import Statevector
from elliptic_models import HamiltonianHardwareExperiment,GraphHamiltonianFeatureMap,TopologySample,TopologyPreprocessor


class HardwareExperimentTests(unittest.TestCase):
    def test_observables_use_logical_little_endian_root(self):
        values=HamiltonianHardwareExperiment.observables({'01':4},2)
        self.assertEqual(values['Root Z'],-1)
        self.assertEqual(values['Magnetization'],0)
        self.assertEqual(values['SE root Z'],0)
        values=HamiltonianHardwareExperiment.observables({'00':2,'11':2},2)
        self.assertEqual(values['Magnetization'],0)
        self.assertEqual(values['Magnetization squared'],1)
        self.assertAlmostEqual(values['SE magnetization'],.5)

    def test_existing_job_ids_are_recovered_without_a_service_or_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            e=HamiltonianHardwareExperiment(directory)
            e.output.mkdir()
            manifest={'jobs':[{'group':'small','job_id':'existing-small'},
                              {'group':'large','job_id':'existing-large'}]}
            e.manifest_path.write_text(json.dumps(manifest))
            self.assertEqual(e.submit(None),manifest)

    def test_y_field_edgeless_dynamics_match_independent_qubit_reference(self):
        sample=TopologySample(np.zeros((3,3)),np.zeros((3,2)),np.array([0.,1.,0.]),
                              np.ones(3),np.arange(3),3,2)
        row=TopologyPreprocessor().fit_transform([sample])[0]
        fm=GraphHamiltonianFeatureMap(3,edges='none',topology_only=True,field_axis='Y')
        state=Statevector.from_instruction(fm.bind(row)).data
        exact=fm.exact_state(row)
        np.testing.assert_allclose(state,exact,atol=1e-12)
        self.assertTrue(any('Y' in term for term,_ in fm.hamiltonian(row).to_list()))
        from qiskit import QuantumCircuit
        means=[]
        for node in range(3):
            root=float(node==0)
            c=QuantumCircuit(1);c.h(0)
            c.ry(.2+.7*root+1.1*sample.seeds[node],0);c.rz(.4*root,0)
            c.ry(2*.37*.8,0)
            a=Statevector.from_instruction(c).data
            means.append(abs(a[0])**2-abs(a[1])**2)
        counts={format(i,'03b'):p for i,p in enumerate(abs(state)**2)}
        observed=HamiltonianHardwareExperiment.observables(counts,3)
        self.assertAlmostEqual(observed['Magnetization'],np.mean(means))
        self.assertAlmostEqual(observed['Root Z'],means[0])
        self.assertAlmostEqual(observed['Magnetization squared'],
            (np.sum(means)**2+np.sum(1-np.array(means)**2))/9)


if __name__=='__main__':unittest.main()
