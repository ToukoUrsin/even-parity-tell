#!/usr/bin/env python3
"""
Tensor-network (MPS) solver for QMill peaked-circuit challenges.

Strategy
--------
1. Simulate the circuit as a Matrix Product State (MPS). An MPS represents the
   n-qubit state as a chain of rank-3 tensors connected by "bond" indices whose
   size (bond dimension chi) measures entanglement. Cost ~ n * chi^3, NOT 2^n.
   This only works if the circuit keeps entanglement bounded -- peaked /
   AI-compressed circuits tend to.

2. Find the secret (peak bitstring) by GREEDY MAX-MARGINAL DECODING:
   sweep qubit 0..n-1; at each step compute P(q_i = 0) vs P(q_i = 1) conditioned
   on the bits already fixed (via <Z_i> on the current first site), keep the more
   likely value, collapse that qubit, and continue on the smaller MPS.
   For a state concentrated on one basis string this returns that string.

We validate the decoder on the 8-qubit circuits (known exact answers) before
trusting it on 40 / 48 qubits.
"""
import sys
import time
import numpy as np
import quimb as qu
import quimb.tensor as qtn


def load_circuit_mps(path, max_bond=None, cutoff=1e-10):
    """Build a CircuitMPS from an OpenQASM 2.0 file."""
    gate_opts = {"cutoff": cutoff}
    if max_bond is not None:
        gate_opts["max_bond"] = max_bond
    circ = qtn.CircuitMPS.from_openqasm2_file(path, gate_opts=gate_opts)
    return circ


def greedy_peak(psi, verbose=False):
    """
    Greedy max-marginal decode of the most-likely bitstring from an MPS `psi`.

    Sweeps site 0..n-1. At each site, form the local density matrix conditioned
    on previously-fixed bits, take argmax of P(0) vs P(1), fix it, project, and
    absorb into the next site. The product of the chosen conditional probs is the
    EXACT probability of the returned bitstring, |<x|psi>|^2.

    Returns (bits_q0_to_qn-1, prob). bits[k] is the value of qubit q[k].
    """
    psi = psi.copy()
    psi.canonicalize(0, inplace=True)  # right-canonical: site 0 is ortho center
    n = psi.L
    bits = []
    prob = 1.0
    for i in range(n):
        ki = psi[i]
        bi = ki.H
        ix = psi.site_ind(i)
        pi = (ki & bi).contract(output_inds=[ix]).data
        pi = np.asarray(pi).real
        pi = np.clip(pi, 0.0, None)
        pi /= pi.sum()
        xi = int(np.argmax(pi))           # GREEDY (vs random sampling)
        bits.append(xi)
        prob *= float(pi[xi])
        if verbose:
            print(f"   q[{i}] p0={pi[0]:.4f} p1={pi[1]:.4f} -> {xi}")
        psi.isel_({ix: xi})               # project this site onto outcome
        if i < n - 1:
            psi.contract_tags_([psi.site_tag(i), psi.site_tag(i + 1)])
    return bits, prob


def sample_peak(psi, shots=400, seed=0):
    """
    Cross-check / fallback: draw `shots` exact MPS samples and return the
    highest-probability distinct bitstring seen. Catches cases where greedy
    max-marginal would diverge from the true global peak.
    Returns (bits_q0_to_qn-1, prob, n_distinct).
    """
    best_bits, best_p = None, -1.0
    seen = {}
    for config, omega in psi.sample(shots, seed=seed):
        key = tuple(int(x) for x in config)
        seen[key] = omega
        if omega > best_p:
            best_p, best_bits = omega, list(key)
    return best_bits, best_p, len(seen)


def fmt_secret(bits):
    """bits is q[0..n-1]; return both conventions."""
    per_qubit = "".join(str(b) for b in bits)              # q[0..n-1]
    qiskit_str = per_qubit[::-1]                            # q[n-1..0]
    return qiskit_str, per_qubit


def solve(path, max_bond=None, cutoff=1e-10, verbose=False):
    print(f"\n=== {path} ===")
    t0 = time.time()
    circ = load_circuit_mps(path, max_bond=max_bond, cutoff=cutoff)
    psi = circ.psi
    chi = max(psi.bond_sizes()) if psi.L > 1 else 1
    t1 = time.time()
    print(f"  built MPS: L={psi.L}  max_bond(chi)={chi}  "
          f"sim_time={t1-t0:.1f}s  (max_bond_cap={max_bond}, cutoff={cutoff})")
    bits, prob = greedy_peak(psi, verbose=verbose)
    t2 = time.time()
    qiskit_str, per_qubit = fmt_secret(bits)
    print(f"  greedy decode_time={t2-t1:.1f}s")
    print(f"  --> SECRET (greedy peak):  q[n-1..0]={qiskit_str}")
    print(f"                             q[0..n-1]={per_qubit}")
    print(f"      exact prob of this string = {prob:.6e}")

    # cross-check with exact MPS sampling
    sbits, sprob, ndist = sample_peak(psi, shots=400)
    s_qiskit, s_perq = fmt_secret(sbits)
    t3 = time.time()
    agree = (sbits == bits)
    print(f"  sampling cross-check ({ndist} distinct / 400 shots, {t3-t2:.1f}s): "
          f"best q[n-1..0]={s_qiskit} prob={sprob:.6e} "
          f"{'== greedy ✓' if agree else '!= greedy (using higher-prob)'}")
    if not agree and sprob > prob:
        qiskit_str, per_qubit, prob = s_qiskit, s_perq, sprob
        print(f"  --> REVISED SECRET: q[n-1..0]={qiskit_str}  prob={prob:.6e}")
    return qiskit_str, per_qubit


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        args = ["circuits/challenge-8_1.qasm",
                "circuits/challenge-8_11.qasm",
                "circuits/challenge-8_27.qasm"]
    for p in args:
        solve(p, max_bond=None, cutoff=1e-12)
