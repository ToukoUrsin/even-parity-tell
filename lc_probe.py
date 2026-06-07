#!/usr/bin/env python3
"""Probe the contraction WIDTH of single-qubit local_expectation(Z_i) for a
peaked circuit, after ADCRS simplification (which cancels U.Ud pairs).

If width is small (<~28) per qubit, we can compute EXACT per-qubit <Z_i> and
read off the secret bit-by-bit -- the decisive method for near-product circuits
like 56_43 (only 4 qubits carry odd-parity entanglement).

Rehearse-only by default (no contraction) so it's cheap and safe to run.
Usage: lc_probe.py <qasm> [contract=0|1] [qubits=all|comma,list]
"""
import sys, os, time, math
import numpy as np
import quimb as qu
import quimb.tensor as qtn

OPT = 'greedy'
SIMP = 'ADCRS'
_Z = qu.pauli('Z')


def main():
    path = sys.argv[1]
    do_contract = len(sys.argv) > 2 and sys.argv[2] == '1'
    circ = qtn.Circuit.from_openqasm2_file(path)
    n = circ.N
    if len(sys.argv) > 3 and sys.argv[3] != 'all':
        qubits = [int(x) for x in sys.argv[3].split(',')]
    else:
        qubits = list(range(n))
    print(f"=== {os.path.basename(path)} n={n} gates={circ.num_gates} "
          f"contract={do_contract} ===", flush=True)
    widths = []
    bits = {}
    mags = {}
    for i in qubits:
        t0 = time.time()
        # rehearse the local expectation contraction tree to read its width
        info = circ.local_expectation_rehearse(
            _Z, (i,), optimize=OPT, simplify_sequence=SIMP,
            simplify_equalize_norms=False)
        tree = info['tree']
        W = math.log2(tree.max_size())
        widths.append(W)
        line = f"  q{i:2d}: W={W:5.1f} (mem~{16*2**W/1e9:7.3f}GB) rehearse {time.time()-t0:4.1f}s"
        if do_contract and W <= 28:
            z = complex(circ.local_expectation(
                _Z, (i,), optimize=OPT, simplify_sequence=SIMP,
                simplify_equalize_norms=False)).real
            mags[i] = z
            bits[i] = '0' if z >= 0 else '1'
            line += f"  <Z>={z:+.4f} -> bit {bits[i]}"
        print(line, flush=True)
    print(f"\n  width summary: min={min(widths):.1f} max={max(widths):.1f} "
          f"mean={sum(widths)/len(widths):.1f}", flush=True)
    if do_contract and len(bits) == n:
        q0 = "".join(bits[i] for i in range(n))      # q[0..n-1]
        qiskit = q0[::-1]                              # q[n-1..0]
        weak = [i for i in range(n) if abs(mags[i]) < 0.2]
        print(f"  secret q[0..n-1] = {q0}")
        print(f"  secret q[n-1..0] = {qiskit}  (Qiskit submission order)")
        print(f"  min|<Z>| = {min(abs(v) for v in mags.values()):.4f}  weak_qubits={weak}")
        print("LCRESULT:" + qiskit)


if __name__ == "__main__":
    main()
