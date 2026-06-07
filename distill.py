#!/usr/bin/env python3
"""Kremer "distillation" peak-finder (pcsim method 1), as a script.

Why it beats plain CircuitMPS on these circuits:
  - CircuitPermMPS dynamically permutes qubits so long-range gates stay local
    -> no SWAP blow-up on dense/all-to-all circuits (the wall my gpu_mps hit).
  - complex64 on GPU tensor cores -> 2x throughput / memory vs complex128; we
    only need an argmax bitstring, not 15 digits.
  - The MPS fidelity may be ~0. We do NOT read the peak off the state. We draw a
    few hundred samples and MAJORITY-VOTE each bit. A peaked circuit cannot hide
    its large single-qubit marginals, so the per-bit vote recovers the secret
    even when the global state is badly truncated. (This is your OBFUSCATION_RE
    marginal insight, made into the decoder.)

Confidence = how bimodal the per-bit vote is. min_margin near 1 = every bit is
decisive (trust it); near 0 = some bit unresolved (raise max_bond).

Usage: distill.py <circuit.qasm> [max_bond=128] [shots=1000]
"""
import sys, time, json
import numpy as np
import gpu_patch; gpu_patch.apply()      # shim quimb 1.14 swap_back leak
from qiskit import QuantumCircuit
import quimb
from qiskit_quimb import quimb_circuit
import torch


# complex128 + cutoff 1e-10 is numerically safe; complex64 is ~2x faster/lighter
# but needs a looser cutoff (~1e-6) or SVD noise -> NaN. Validated: qc.sample()
# already returns bits in LOGICAL order q[0..n-1] -> no remap needed.
DTYPE = torch.complex128


def to_backend(x):
    return torch.tensor(x, dtype=DTYPE, device="cuda")


def distill(path, max_bond=128, shots=1000, cutoff=1e-10, seed=1234):
    name = path.split("/")[-1]
    t0 = time.time()
    circuit = QuantumCircuit.from_qasm_file(path)
    n = circuit.num_qubits
    qc = quimb_circuit(
        circuit,
        quimb_circuit_class=quimb.tensor.CircuitPermMPS,
        to_backend=to_backend,
        max_bond=max_bond,
        cutoff=cutoff,
        progbar=False,
    )
    t_build = time.time() - t0
    samples = list(qc.sample(shots, seed=seed))   # already q[0..n-1] order
    bit_probs = np.array([[int(s) for s in ss] for ss in samples]).mean(axis=0)
    voted_q0 = "".join(str(i) for i in (bit_probs > 0.5).astype(int).tolist())  # q[0..n-1]
    qiskit_str = voted_q0[::-1]                                                  # q[n-1..0]
    margins = np.abs(bit_probs - 0.5) * 2.0
    rec = dict(name=name, n=n, max_bond=max_bond, shots=shots,
               secret_qiskit=qiskit_str, secret_q0=voted_q0,
               confidence=round(float(margins.mean()), 3),
               min_margin=round(float(margins.min()), 3),
               n_weak_bits=int((margins < 0.2).sum()),
               t_build=round(t_build, 1), t_total=round(time.time() - t0, 1))
    print("DISTILL:" + json.dumps(rec), flush=True)
    return rec


if __name__ == "__main__":
    path = sys.argv[1]
    mb = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    shots = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    distill(path, max_bond=mb, shots=shots)
