"""Feature maps, exact statevector fidelities and Nyström kernel classifiers."""
from dataclasses import dataclass
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from .data import StratifiedSampler


@dataclass
class QuantumFeatureMap:
    circuit: QuantumCircuit
    parameters: tuple

    @classmethod
    def zz(cls, n_qubits, reps=2):
        circuit = zz_feature_map(n_qubits, reps=reps, entanglement='linear')
        return cls(circuit, tuple(circuit.parameters))

    @classmethod
    def uploading(cls, n_features, n_qubits=4, passes=1):
        """Chunk features across qubits, repeat all chunks on each pass.

        For the controlled ablation n_features == n_qubits: one pass is a
        single encoding, two or more passes re-upload the SAME inputs.
        With more features, chunking itself already re-uploads data.
        """
        if min(n_features, n_qubits, passes) < 1:
            raise ValueError('Feature count, qubits and passes must be positive')
        parameters = ParameterVector('x', n_features)
        circuit = QuantumCircuit(n_qubits)
        circuit.h(range(n_qubits))
        for _ in range(passes):
            for start in range(0, n_features, n_qubits):
                for q in range(n_qubits):
                    if start + q < n_features:
                        circuit.ry(parameters[start + q], q)
                for q in range(n_qubits - 1):
                    circuit.cz(q, q + 1)
                if n_qubits > 2:
                    circuit.cz(n_qubits - 1, 0)
        return cls(circuit, tuple(parameters))

    def bind(self, row):
        if len(row) != len(self.parameters):
            raise ValueError('Input dimension does not match feature map')
        return self.circuit.assign_parameters(dict(zip(self.parameters, map(float, row))))


class StatevectorKernel:
    """Reference simulator. No IBM jobs or credentials are used."""
    def __init__(self, feature_map, batch_size=256):
        self.feature_map = feature_map
        self.batch_size = batch_size
        self._states = {}

    def encode(self, X):
        builder = getattr(self.feature_map, 'simulation_bind', self.feature_map.bind)
        rows = []
        for row in np.asarray(X):
            key = np.asarray(row, dtype=float).tobytes()
            if key not in self._states:
                self._states[key] = Statevector.from_instruction(builder(row)).data
            rows.append(self._states[key])
        return np.asarray(rows)

    @staticmethod
    def fidelity(states, references, batch_size=256):
        result = np.empty((len(states), len(references)))
        for start in range(0, len(states), batch_size):
            result[start:start + batch_size] = np.clip(
                np.abs(states[start:start + batch_size] @ references.conj().T) ** 2, 0, 1)
        return result

    def evaluate(self, X, Y=None):
        states = self.encode(X)
        references = states if Y is None else self.encode(Y)
        return self.fidelity(states, references, self.batch_size)


class NystromProjector:
    """Positive eigenspace projection, also robust to indefinite shot-noisy W."""
    def fit(self, W):
        values, vectors = np.linalg.eigh((W + W.T) / 2)
        keep = values > max(float(values.max()), 1.0) * 1e-10
        if not keep.any():
            raise ValueError('Landmark kernel has no positive eigenvalues')
        self.projection_ = vectors[:, keep] / np.sqrt(values[keep])
        return self

    def transform(self, C):
        return C @ self.projection_


class KernelClassifier:
    """Same SVC objective for exact and Nyström branches by default.

    Any kernel implementing evaluate(X, Y=None) works, including IBMFidelityKernel.
    Nyström reduces circuit pairs to O(n*m), not the cost of encoding a circuit.
    """
    def __init__(self, kernel, n_landmarks=None, C=1.0, seed=42, readout='svc'):
        self.kernel = kernel
        self.n_landmarks = n_landmarks
        self.C = C
        self.seed = seed
        self.readout = readout

    def fit(self, X, y):
        X = np.asarray(X)
        if self.n_landmarks is None:
            if self.readout != 'svc':
                raise ValueError('Exact kernels use the SVC readout')
            self.references_ = X
            self.model_ = SVC(kernel='precomputed', C=self.C, class_weight='balanced')
            features = self.kernel.evaluate(X)
            # A finite-shot square kernel need not be exactly symmetric.
            features = (features + features.T) / 2
        else:
            self.landmark_indices_ = StratifiedSampler.landmarks(y, self.n_landmarks, self.seed)
            self.references_ = X[self.landmark_indices_]
            W = self.kernel.evaluate(self.references_)
            self.projector_ = NystromProjector().fit(W)
            features = self.projector_.transform(self.kernel.evaluate(X, self.references_))
            if self.readout == 'svc':
                self.model_ = SVC(kernel='linear', C=self.C, class_weight='balanced')
            elif self.readout == 'logistic':
                self.model_ = LogisticRegression(C=self.C, class_weight='balanced', max_iter=2000)
            else:
                raise ValueError('Readout must be svc or logistic')
        self.model_.fit(features, y)
        return self

    def features(self, X):
        block = self.kernel.evaluate(X, self.references_)
        return block if self.n_landmarks is None else self.projector_.transform(block)

    def predict(self, X):
        return self.model_.predict(self.features(X))

    def decision_function(self, X):
        return self.model_.decision_function(self.features(X))
