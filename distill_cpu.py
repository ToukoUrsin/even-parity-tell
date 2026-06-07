#!/usr/bin/env python3
"""CPU distillation peak-finder (Kremer pcsim method 1), corrected for this box.

Differences from distill.py (which targets CUDA + CircuitPermMPS and is BROKEN here):
  - CircuitMPS, NOT CircuitPermMPS. CircuitPermMPS returns samples in PHYSICAL
    (permuted) order in quimb 1.11.2 -> scrambled bitstring; it also trips the
    swap_back kwarg crash. CircuitMPS keeps logical qubit order. Validated: it
    recovers the known 8_1/8_11/8_27/48_8/56_9 secrets exactly.
  - numpy complex128 backend (no CUDA on this Apple box).
  - gpu_patch.apply() still applied: it drops the stray `swap_back` kwarg that
    quimb pipes into the backend SVD on long-range gates (else CircuitMPS also
    crashes on dense circuits).

Decode = OBFUSCATION_RE marginal insight: draw `shots` samples, majority-vote each
bit. A peaked circuit cannot hide its single-qubit marginals, so per-bit voting
recovers the secret even when the global MPS fidelity is ~0 from truncation.

min_margin near 1 = every bit decisive (trust). near 0 = some bit unresolved (raise max_bond).

Usage: distill_cpu.py <circuit.qasm> [max_bond=128] [shots=1000] [cutoff=1e-10] [dtype=128|64]
"""
import sys, time, json
import numpy as np
import gpu_patch; gpu_patch.apply()          # drop stray swap_back kwarg on long-range gates
from qiskit import QuantumCircuit
import quimb.tensor as qt
from qiskit_quimb import quimb_circuit

_DT = {"128": np.complex128, "64": np.complex64}


def distill(path, max_bond=128, shots=1000, cutoff=1e-10, seed=1234, dt="128"):
    dtype = _DT[str(dt)]
    def to_backend(x):
        return np.asarray(x, dtype=dtype)
    name = path.split("/")[-1]
    t0 = time.time()
    circuit = QuantumCircuit.from_qasm_file(path)
    n = circuit.num_qubits
    qc = quimb_circuit(
        circuit,
        quimb_circuit_class=qt.CircuitMPS,
        to_backend=to_backend,
        max_bond=max_bond,
        cutoff=cutoff,
        progbar=False,
    )
    t_build = time.time() - t0
    try:
        maxbond_reached = int(qc.psi.max_bond())
    except Exception:
        maxbond_reached = None
    samples = list(qc.sample(shots, seed=seed))           # logical order q[0..n-1]
    bit_probs = np.array([[int(s) for s in ss] for ss in samples]).mean(axis=0)
    voted_q0 = "".join(str(i) for i in (bit_probs > 0.5).astype(int).tolist())  # q[0..n-1]
    qiskit_str = voted_q0[::-1]                                                 # q[n-1..0]
    margins = np.abs(bit_probs - 0.5) * 2.0
    weak = [int(i) for i in np.where(margins < 0.2)[0]]
    rec = dict(name=name, n=n, max_bond=max_bond, maxbond_reached=maxbond_reached,
               shots=shots, dtype=str(dt), cutoff=cutoff,
               secret_qiskit=qiskit_str, secret_q0=voted_q0,
               confidence=round(float(margins.mean()), 3),
               min_margin=round(float(margins.min()), 3),
               n_weak_bits=len(weak), weak_qubits=weak,
               t_build=round(t_build, 1), t_total=round(time.time() - t0, 1))
    print("DISTILL:" + json.dumps(rec), flush=True)
    return rec


if __name__ == "__main__":
    path = sys.argv[1]
    mb = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    shots = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    cutoff = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-10
    dt = sys.argv[5] if len(sys.argv) > 5 else "128"
    distill(path, max_bond=mb, shots=shots, cutoff=cutoff, dt=dt)
