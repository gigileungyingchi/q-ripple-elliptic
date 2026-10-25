"""Compute-uncompute fidelities from IBM Runtime SamplerV2 measurements.

Authentication and backend selection belong in the notebook; importing this
module never connects or submits a job. Classical SVM training stays local.
"""
import json
from pathlib import Path

import numpy as np
from qiskit.transpiler import generate_preset_pass_manager


class IBMFidelityKernel:
    def __init__(self, feature_map, backend, shots=2048, batch_size=64,
                 max_circuits=2000, log_path=None, sampler=None):
        if min(shots, batch_size, max_circuits) < 1:
            raise ValueError('shots, batch_size and max_circuits must be positive')
        self.feature_map = feature_map
        self.backend = backend
        self.shots = shots
        self.batch_size = batch_size
        self.max_circuits = max_circuits
        self.log_path = Path(log_path) if log_path else None
        self.records = []
        self.submitted_circuits = 0
        if sampler is None:
            from qiskit_ibm_runtime import SamplerV2
            sampler = SamplerV2(mode=backend)
        self.sampler = sampler
        self.pass_manager = generate_preset_pass_manager(
            backend=backend, optimization_level=1, seed_transpiler=42)

    @staticmethod
    def estimate_circuits(n_train, n_eval, n_landmarks=None):
        m = n_train if n_landmarks is None else n_landmarks
        return m * (m + 1) // 2 + (n_eval if n_landmarks is None else n_train + n_eval) * m

    def overlap_circuit(self, x, y):
        # Probability of measuring all zeros is |<phi(y)|phi(x)>|^2.
        circuit = self.feature_map.bind(x).compose(self.feature_map.bind(y).inverse())
        circuit.measure_all()
        return circuit

    def _save(self):
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text(json.dumps(self.records, indent=2))

    def evaluate(self, X, Y=None):
        X = np.asarray(X)
        symmetric = Y is None
        Y = X if symmetric else np.asarray(Y)
        count = len(X) * (len(X) + 1) // 2 if symmetric else len(X) * len(Y)
        if count + self.submitted_circuits > self.max_circuits:
            raise ValueError(f'Circuit budget exceeded: requested {count}, already submitted '
                             f'{self.submitted_circuits}, budget {self.max_circuits}')
        output = np.empty((len(X), len(Y)))
        pairs = ((i, j) for i in range(len(X)) for j in range(i if symmetric else 0, len(Y)))
        from itertools import islice
        while batch := list(islice(pairs, self.batch_size)):
            circuits = [self.overlap_circuit(X[i], Y[j]) for i, j in batch]
            isa = self.pass_manager.run(circuits)
            job = self.sampler.run(isa, shots=self.shots)
            self.submitted_circuits += len(batch)
            record = {'job_id': job.job_id(), 'backend': self.backend.name,
                      'shots': self.shots, 'pairs': batch,
                      'transpiled_depth_max': max(c.depth() for c in isa),
                      'two_qubit_gates_max': max(sum(1 for instruction in c.data
                                                   if instruction.operation.num_qubits == 2) for c in isa)}
            self.records.append(record)
            # Persist job IDs BEFORE waiting; service.job(id).result() can recover jobs.
            self._save()
            result = job.result()
            estimates = []
            for (i, j), pub in zip(batch, result):
                counts = pub.data.meas.get_counts()
                value = counts.get('0' * self.feature_map.circuit.num_qubits, 0) / sum(counts.values())
                output[i, j] = value
                if symmetric:
                    output[j, i] = value
                estimates.append(value)
            record['fidelities'] = estimates
            self._save()
        return output
