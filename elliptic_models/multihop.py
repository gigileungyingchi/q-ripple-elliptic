"""Directional, degree-normalized walk aggregation and depth-aware circuits."""
import numpy as np
import pandas as pd
from scipy import sparse
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from .quantum import QuantumFeatureMap


class DirectionalPropagation:
    """Four signals per length-k walk, with A[i,j] meaning transaction i -> j.

    P_in @ s averages predecessor signals; P_out @ s averages successor
    signals. Sinks/isolates have zero aggregates. Repeated walks can revisit
    nodes: these are NOT shortest-path shells or probability-conserving flow.
    Only revealed seeds, never target labels, enter the initial signal.
    """
    channels = ('in_seed', 'out_seed', 'in_degree', 'out_degree')

    def __init__(self, max_hops=8):
        if not isinstance(max_hops, int) or max_hops < 1:
            raise ValueError('max_hops must be a positive integer')
        self.max_hops = max_hops

    @staticmethod
    def normalize_rows(A):
        A = sparse.csr_matrix(A, dtype=float)
        degree = np.asarray(A.sum(axis=1)).ravel()
        inverse = np.divide(1., degree, out=np.zeros_like(degree), where=degree > 0)
        return sparse.diags(inverse) @ A

    @classmethod
    def columns(cls, depth):
        return [f'{channel}_h{h}' for h in range(1, depth + 1) for channel in cls.channels]

    def transform(self, A, seeds):
        Pout, Pin = self.normalize_rows(A), self.normalize_rows(A.T)
        seed = np.zeros(A.shape[0])
        seed[np.asarray(seeds, dtype=int)] = 1.
        base = np.column_stack([seed, np.log1p(np.asarray(A.sum(axis=0)).ravel()),
                                np.log1p(np.asarray(A.sum(axis=1)).ravel())])
        incoming, outgoing = base.copy(), base.copy()
        values = {}
        for h in range(1, self.max_hops + 1):
            incoming, outgoing = Pin @ incoming, Pout @ outgoing
            for name, signal in zip(self.channels, [incoming[:, 0], outgoing[:, 0],
                                                    incoming[:, 1], outgoing[:, 2]]):
                values[f'{name}_h{h}'] = signal
        return pd.DataFrame(values)

    def dataset(self, risk_graph, time_steps):
        X, y, roots = [], [], []
        for t in time_steps:
            period = risk_graph.prepare_period(t)
            if not len(period.candidates):
                continue
            X.append(self.transform(period.adjacency, period.seeds).iloc[period.candidates])
            y.extend(period.labels[period.candidates])
            roots.extend((t, int(i)) for i in period.candidates)
        if not X:
            raise ValueError('No labeled non-seed candidates')
        return pd.concat(X, ignore_index=True), np.asarray(y), np.asarray(roots)

    @classmethod
    def coverage(cls, frame):
        """Nonzero seed exposure by exact walk length and cumulatively."""
        rows = []
        cumulative_in = np.zeros(len(frame), dtype=bool)
        cumulative_out = cumulative_in.copy()
        depths = sorted(int(c.rsplit('h', 1)[1]) for c in frame if c.startswith('in_seed_h'))
        for h in depths:
            incoming = frame[f'in_seed_h{h}'].to_numpy() > 0
            outgoing = frame[f'out_seed_h{h}'].to_numpy() > 0
            cumulative_in |= incoming
            cumulative_out |= outgoing
            rows.append({'Hops': h, 'Upstream nonzero': incoming.mean(),
                         'Downstream nonzero': outgoing.mean(),
                         'Any exposure through depth': (cumulative_in | cumulative_out).mean()})
        return pd.DataFrame(rows)


class HopFeatureMap:
    """Four qubits = four directional channels, one stage per graph depth.

    Repetitions and hop order are independent of input depth. A reversed-order
    control receives identical features, but applies the stages in reverse.
    """
    @staticmethod
    def build(max_hops=4, repetitions=1, order='forward'):
        if max_hops < 1 or repetitions < 1 or order not in ('forward', 'reverse'):
            raise ValueError('Require positive depth/repetitions and forward/reverse order')
        x = ParameterVector('hop', 4 * max_hops)
        circuit = QuantumCircuit(4)
        circuit.h(range(4))
        sequence = range(max_hops) if order == 'forward' else range(max_hops - 1, -1, -1)
        for _ in range(repetitions):
            for h in sequence:
                circuit.ry(x[4 * h], 0)
                circuit.ry(x[4 * h + 1], 1)
                circuit.rz(x[4 * h + 2], 2)
                circuit.rz(x[4 * h + 3], 3)
                circuit.cx(0, 1)
                circuit.cx(1, 2)
                circuit.cx(2, 3)
        return QuantumFeatureMap(circuit, tuple(x))
