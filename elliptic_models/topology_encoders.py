"""Direct graph dynamics, with explicit circuit and input-resource conventions."""
import numpy as np
from scipy.sparse.linalg import expm_multiply
from qiskit import QuantumCircuit
from qiskit.circuit.library import HamiltonianGate
from qiskit.quantum_info import SparsePauliOp, Statevector

from .topology import TopologyFeatureMap


class GraphMessageFeatureMap(TopologyFeatureMap):
    """Existing graph-edge message circuit plus fixed-ring/attribute ablations.

    topology_only removes numerical node attributes, retaining root/seed flags.
    Generic interactions use the same RZX family on a fixed active-node ring;
    gate counts can differ from the observed graph and are reported separately.
    """
    def __init__(self, n_qubits=6, repetitions=2, edges='directed', topology_only=False):
        self.graph_edges, self.topology_only = edges, topology_only
        super().__init__(n_qubits, repetitions, edges='directed' if edges == 'generic' else edges)

    def adapted(self, row):
        mask, root, seeds, attrs, A = self.unpack(row, self.n_qubits)
        attrs, A = attrs.copy(), A.copy()
        if self.topology_only:
            attrs[:] = 0
        if self.graph_edges == 'generic':
            A[:] = 0
            active = np.flatnonzero(mask)
            if len(active) > 1:
                for i, j in zip(active, np.roll(active, -1)):
                    A[i, j] = 1
        return np.concatenate([mask, root, seeds, attrs.ravel(), A.ravel()])

    def bind(self, row):
        return super().bind(self.adapted(row))

    def simulation_bind(self, row):
        return super().simulation_bind(self.adapted(row))


class GraphHamiltonianFeatureMap:
    r"""Node-register H = field Σ P_i + coupling Σ Z_source X_destination (P=X or Y).

    Prepare all scaled node attributes and root/seed flags once, then use a
    symmetric second-order product formula with fixed total time. Increasing
    steps improves the evolution approximation; it is not a hop-radius change.
    bind and simulation_bind implement the SAME product formula. exact_state
    is a small-system convergence reference, never the hardware target.
    """
    def __init__(self, n_qubits=6, time=0.8, steps=4, field=0.37,
                 coupling=0.7, edges='directed', topology_only=False, field_axis='X'):
        if n_qubits < 2 or steps < 1 or int(steps) != steps:
            raise ValueError('Use >=2 nodes and a positive integer step count')
        if edges not in ('directed', 'reverse', 'undirected', 'none'):
            raise ValueError('Unknown Hamiltonian edge mode')
        self.n_qubits, self.time, self.steps = n_qubits, float(time), int(steps)
        if field_axis not in ('X', 'Y'):
            raise ValueError('Field axis must be X or Y')
        self.field_axis = field_axis
        self.field, self.coupling, self.edges = field, coupling, edges
        self.preparation = GraphMessageFeatureMap(n_qubits, repetitions=1,
                                                  edges='none', topology_only=topology_only)
        self.circuit = QuantumCircuit(n_qubits)

    def terms(self, row):
        mask, _, _, _, A = TopologyFeatureMap.unpack(row, self.n_qubits)
        active = np.flatnonzero(mask)
        terms = [(self.field_axis, (int(i),), self.field) for i in active]
        if self.edges in ('directed', 'reverse'):
            for source, dest in np.argwhere(A > 0):
                if source != dest and mask[source] and mask[dest]:
                    wires = (int(source), int(dest))
                    if self.edges == 'reverse':
                        wires = wires[::-1]
                    terms.append(('ZX', wires, self.coupling * A[source, dest]))
        elif self.edges == 'undirected':
            for i, j in np.argwhere(np.triu((A + A.T) > 0, 1)):
                if mask[i] and mask[j]:
                    terms.append(('ZZ', (int(i), int(j)), self.coupling))
        return terms

    def hamiltonian(self, row):
        entries = []
        for paulis, wires, coefficient in self.terms(row):
            label = ['I'] * self.n_qubits
            for letter, wire in zip(paulis, wires):
                label[self.n_qubits - 1 - wire] = letter
            entries.append((''.join(label), coefficient))
        return SparsePauliOp.from_list(entries or [('I' * self.n_qubits, 0.)])

    def _evolve(self, circuit, row):
        terms = self.terms(row)
        dt = self.time / self.steps
        for _ in range(self.steps):
            # Each half step exp(-i dt c P/2) has Pauli-rotation angle dt*c.
            for paulis, wires, coefficient in terms + terms[::-1]:
                angle = dt * coefficient
                if paulis == 'X':
                    circuit.rx(angle, wires[0])
                elif paulis == 'Y':
                    circuit.ry(angle, wires[0])
                elif paulis == 'ZX':
                    circuit.rzx(angle, *wires)  # Z on source, X on destination.
                else:
                    circuit.rzz(angle, *wires)
        return circuit

    def bind(self, row):
        return self._evolve(self.preparation.bind(row), row)

    def simulation_bind(self, row):
        return self._evolve(self.preparation.simulation_bind(row), row)

    def exact_state(self, row):
        state = Statevector.from_instruction(self.preparation.simulation_bind(row)).data
        return expm_multiply(-1j * self.time * self.hamiltonian(row).to_matrix(sparse=True), state)


class HermitianWalkFeatureMap:
    r"""Small directed-adjacency lift H=[[0,A],[A.T,0]], root source state.

    This is a Hermitian bipartite lift, not directed P**k diffusion. Evolution
    alternates source/destination roles. Seed flags supply an on-site potential;
    numerical attributes are deliberately absent. Logical width is ceil(log2(2n))
    for n node slots, unlike the one-qubit-per-node encoders. Dense HamiltonianGate
    synthesis is a small-system control, not an efficient large-graph loading claim.
    """
    def __init__(self, n_nodes=6, time=0.8, seed_potential=0.4):
        if n_nodes < 2:
            raise ValueError('Use at least two node slots')
        self.n_nodes, self.time, self.seed_potential = n_nodes, float(time), seed_potential
        self.n_qubits = int(np.ceil(np.log2(2 * n_nodes)))
        self.circuit = QuantumCircuit(self.n_qubits)

    def hamiltonian(self, row):
        mask, _, seeds, _, adjacency = TopologyFeatureMap.unpack(row, self.n_nodes)
        A = adjacency * mask[:, None] * mask[None, :]
        # Fixed per-graph degree bound, independent of labels and data splits.
        scale = max(float(A.sum(0).max()), float(A.sum(1).max()), 1.)
        A = A / scale
        n = self.n_nodes
        H = np.zeros((2 ** self.n_qubits, 2 ** self.n_qubits))
        H[:n, n:2*n], H[n:2*n, :n] = A, A.T
        potential = np.tile(seeds * mask * self.seed_potential, 2)
        H[np.arange(2*n), np.arange(2*n)] = potential
        return H

    def bind(self, row):
        _, root, _, _, _ = TopologyFeatureMap.unpack(row, self.n_nodes)
        index = int(np.argmax(root))  # root in the source half of the lift
        circuit = QuantumCircuit(self.n_qubits)
        for bit in range(self.n_qubits):
            if index & (1 << bit):
                circuit.x(bit)
        circuit.append(HamiltonianGate(self.hamiltonian(row), self.time), range(self.n_qubits))
        return circuit


class HamiltonianMomentKernel:
    """RBF on permutation-invariant sums of Hamiltonian-state observables.

    With exact evolution the representation is invariant to simultaneous graph/
    root/attribute relabeling. A finite ordered product formula only approximates
    this property. Moments are compressed readouts; they are not full topology
    or a complete invariant. Exact evolution is a small-system reference.
    Z moments share computational-basis measurements; directed ZX sums need
    additional measurement bases. This class computes ideal observables locally.
    """
    def __init__(self, feature_map, exact=False, gamma=4.):
        self.feature_map, self.exact, self.gamma = feature_map, exact, gamma
        self.cache = {}

    def features(self, X):
        values = []
        q = self.feature_map.n_qubits
        bits = ((np.arange(2**q)[:, None] >> np.arange(q)) & 1)
        spins = 1-2*bits
        for row in X:
            key = np.asarray(row, dtype=float).tobytes()
            if key not in self.cache:
                mask, root, _, _, A = TopologyFeatureMap.unpack(row, q)
                state = (self.feature_map.exact_state(row) if self.exact else
                         Statevector.from_instruction(self.feature_map.simulation_bind(row)).data)
                probability = abs(state)**2
                magnetization = spins[:, mask.astype(bool)].mean(axis=1)
                root_spin = spins[:, int(np.argmax(root))]
                moments = [probability @ magnetization**power for power in (1, 2, 3, 4)]
                moments += [probability @ (root_spin * magnetization**power) for power in (0, 1, 2, 3)]
                # Invariant average directed ZX expectation, globally and at the root.
                terms = [[], [], []]
                r = int(np.argmax(root))
                for i, j in np.argwhere(A > 0):
                    if i == j or not mask[i] or not mask[j]:
                        continue
                    label=['I']*q; label[q-1-i]='Z'; label[q-1-j]='X'
                    item=(''.join(label), 1.)
                    terms[0].append(item)
                    if i == r: terms[1].append(item)
                    if j == r: terms[2].append(item)
                sv=Statevector(state)
                moments += [float(np.real(sv.expectation_value(SparsePauliOp.from_list(group)))/len(group))
                            if group else 0. for group in terms]
                self.cache[key] = np.array(moments)
            values.append(self.cache[key])
        return np.array(values)

    def evaluate(self, X, Y=None):
        from sklearn.metrics.pairwise import rbf_kernel
        a=self.features(X)
        b=a if Y is None else self.features(Y)
        return rbf_kernel(a,b,gamma=self.gamma)
