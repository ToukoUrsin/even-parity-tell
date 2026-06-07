#!/usr/bin/env python3
"""FAST heuristic: the obfuscation is C = U1 U1d P1 U2 U2d ... where U.Ud CX gates
come in matched (even-parity) pairs that cancel; only a few ODD-parity CX pairs are
real "defects". Hypothesis: removing every CX on an even-parity pair (keeping the few
odd-parity CX + ALL single-qubit gates) yields a circuit whose PEAK == the true peak,
but with almost no entanglement -> tiny MPS bond -> simulates in seconds.

We VALIDATE this on known-answer circuits before trusting it on the targets.

Usage: reduce_oddcx.py <id> [max_bond=64] [shots=2000] [--keep-odd-pos]
Prints REDUCED:{...} with secret_qiskit (q[n-1..0]) and secret_q0 (q[0..n-1]).
"""
import sys, re, json, time
from collections import Counter, defaultdict
import numpy as np


def parse_and_reduce(path):
    """Return (n, kept_lines) where kept_lines drops even-parity-pair CX gates."""
    lines = open(path).read().splitlines()
    n = 0
    # first pass: count CX parity per pair
    pair_count = Counter()
    for ln in lines:
        s = ln.strip()
        m = re.match(r'qreg\s+\w+\[(\d+)\]', s)
        if m:
            n = int(m.group(1))
        m = re.match(r'cx\s+\w+\[(\d+)\]\s*,\s*\w+\[(\d+)\]', s)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            pair_count[tuple(sorted((a, b)))] += 1
    odd_pairs = {p for p, c in pair_count.items() if c % 2}
    # second pass: keep header + single-qubit gates + CX whose pair is odd-parity.
    # For an odd pair keep only the LAST occurrence (net one CX) to stay minimal.
    odd_seen = defaultdict(int)
    odd_total = {p: pair_count[p] for p in odd_pairs}
    kept = []
    n_cx_kept = 0
    for ln in lines:
        s = ln.strip()
        m = re.match(r'cx\s+\w+\[(\d+)\]\s*,\s*\w+\[(\d+)\]', s)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            p = tuple(sorted((a, b)))
            if p in odd_pairs:
                odd_seen[p] += 1
                # keep only the final occurrence of this odd pair
                if odd_seen[p] == odd_total[p]:
                    kept.append(ln); n_cx_kept += 1
                # else drop (it pairs with another -> even part)
            # even-parity pair CX: drop entirely
        else:
            kept.append(ln)
    return n, kept, len(odd_pairs), n_cx_kept


def solve(path, max_bond=64, shots=2000, seed=7):
    import quimb.tensor as qtn
    n, kept, n_odd, n_cx = parse_and_reduce(path)
    qasm = "\n".join(kept) + "\n"
    tmp = "/tmp/_reduced.qasm"
    open(tmp, "w").write(qasm)
    t0 = time.time()
    circ = qtn.CircuitMPS.from_openqasm2_file(tmp, gate_opts={"max_bond": max_bond, "cutoff": 1e-10})
    psi = circ.psi
    chi = max(psi.bond_sizes()) if psi.L > 1 else 1
    samples = list(circ.sample(shots, seed=seed))
    arr = np.array([[int(c) for c in ss] for ss in samples])
    bit_probs = arr.mean(axis=0)
    q0 = "".join(str(i) for i in (bit_probs > 0.5).astype(int))   # q[0..n-1]
    margins = np.abs(bit_probs - 0.5) * 2
    rec = dict(id=path.split("challenge-")[-1].replace(".qasm", ""), n=n,
               n_odd_pairs=n_odd, n_cx_kept=n_cx, chi=int(chi),
               secret_q0=q0, secret_qiskit=q0[::-1],
               min_margin=round(float(margins.min()), 3),
               confidence=round(float(margins.mean()), 3),
               n_weak=int((margins < 0.2).sum()), t=round(time.time() - t0, 1))
    print("REDUCED:" + json.dumps(rec), flush=True)
    return rec


if __name__ == "__main__":
    path = f"circuits/challenge-{sys.argv[1]}.qasm"
    mb = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    shots = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    solve(path, mb, shots)
