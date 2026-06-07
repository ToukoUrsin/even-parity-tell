#!/usr/bin/env python3
"""
Pauli backpropagation (sparse Pauli dynamics) peak finder for QMill peaked circuits.

Operator-space dual of the MPS attack. To recover bit i of the secret we compute the
single-qubit marginal  <Z_i> = <0| C^dagger Z_i C |0>.  In the Heisenberg picture we
start from the Pauli Z_i and conjugate it backwards through every gate of C, keeping a
sparse dictionary of Pauli terms and TRUNCATING low-weight terms. For a peaked circuit
C|0> ~ |s>, so <Z_i> ~ (1 - 2 s_i) is a large +/-1 signal and survives truncation.
Bit:  s_i = 0 if <Z_i> > 0 else 1.

Pauli bookkeeping uses the phase-free Weyl basis W(x,z) = X^x Z^z (x,z big integers as
bitmasks).  Group law:  W(0,z) W(x',z') has sign (-1)^{z . x'};  <0| W(x,z) |0> = [x==0].
This avoids tracking i-phases except the ones produced by rotations (handled explicitly).
Verified against exact statevector marginals on the 8-qubit circuits.

Why it can beat MPS on the monsters: it never builds the (highly entangled) intermediate
STATE; it tracks an OPERATOR whose spread is bounded by the circuit's *effective* (post-
cancellation) complexity. The U U^dagger inverse-pair construction means most spreading
cancels. Cost is controlled by the term cap, not by 2^n or by intermediate chi.
"""
import sys, re, time, math, argparse
import numpy as np


def parse_qasm(path):
    """Return (n_qubits, gates) where gates = [(kind, (q,..), angle)] in circuit order."""
    gates = []
    n = None
    for line in open(path):
        line = line.strip()
        m = re.match(r'qreg\s+\w+\[(\d+)\]', line)
        if m:
            n = int(m.group(1)); continue
        m = re.match(r'(rx|rz|ry)\(([^)]+)\)\s+\w+\[(\d+)\]', line)
        if m:
            ang = eval_angle(m.group(2))
            gates.append((m.group(1), (int(m.group(3)),), ang)); continue
        m = re.match(r'cx\s+\w+\[(\d+)\]\s*,\s*\w+\[(\d+)\]', line)
        if m:
            gates.append(('cx', (int(m.group(1)), int(m.group(2))), 0.0)); continue
        m = re.match(r'swap\s+\w+\[(\d+)\]\s*,\s*\w+\[(\d+)\]', line)
        if m:
            gates.append(('swap', (int(m.group(1)), int(m.group(2))), 0.0)); continue
    return n, gates


def eval_angle(s):
    s = s.strip().replace('pi', str(math.pi))
    return float(eval(s, {"__builtins__": {}}, {}))


def backprop_marginal(gates, i, max_terms=200000, thresh=1e-9):
    """Compute <Z_i> = <0|C^dag Z_i C|0> by Heisenberg backprop of Z_i through gates.
    Returns (expectation, n_terms_peak)."""
    bit = 1 << i
    # operator as dict: (x, z) -> complex coeff. Start with Z_i = W(0, bit).
    op = {(0, bit): 1.0 + 0j}
    support = bit                      # union of (x|z) over terms; skip gates outside it
    peak = 1
    for kind, qs, ang in reversed(gates):
        if kind == 'cx':
            a, b = qs; ma, mb = 1 << a, 1 << b
            if not (support & (ma | mb)):
                continue
            new = {}
            for (x, z), c in op.items():
                # CX(a->b): X_a->X_aX_b ; Z_b->Z_aZ_b ; X_b,Z_a unchanged.
                nx, nz = x, z
                if x & ma: nx ^= mb          # propagate X from control to target
                if z & mb: nz ^= ma          # propagate Z from target to control
                k = (nx, nz)
                new[k] = new.get(k, 0j) + c
            op = new
        elif kind == 'swap':
            a, b = qs; ma, mb = 1 << a, 1 << b
            if not (support & (ma | mb)):
                continue
            new = {}
            for (x, z), c in op.items():
                nx = swap_bits(x, a, b); nz = swap_bits(z, a, b)
                k = (nx, nz)
                new[k] = new.get(k, 0j) + c
            op = new
        else:  # rotations rx/rz/ry, single qubit
            (q,) = qs; mq = 1 << q
            if not (support & mq):
                continue
            ct = math.cos(ang); st = math.sin(ang)
            new = {}
            if kind == 'rz':                 # A = Z_q ; anticommute iff x_q==1
                for (x, z), c in op.items():
                    if not (x & mq):
                        new[(x, z)] = new.get((x, z), 0j) + c
                    else:
                        new[(x, z)] = new.get((x, z), 0j) + c * ct
                        k = (x, z ^ mq)       # U^dag W U = ct W - i st Z_q W ; Z_qW=-W(x,z^mq)
                        new[k] = new.get(k, 0j) + c * (-1j * st)
            elif kind == 'rx':               # A = X_q ; anticommute iff z_q==1
                for (x, z), c in op.items():
                    if not (z & mq):
                        new[(x, z)] = new.get((x, z), 0j) + c
                    else:
                        new[(x, z)] = new.get((x, z), 0j) + c * ct
                        k = (x ^ mq, z)       # + i st X_q W ; X_qW = +W(x^mq,z)
                        new[k] = new.get(k, 0j) + c * (1j * st)
            else:  # ry  A=Y_q ; anticommute iff x_q != z_q
                for (x, z), c in op.items():
                    aq = bool(x & mq) ^ bool(z & mq)
                    if not aq:
                        new[(x, z)] = new.get((x, z), 0j) + c
                    else:
                        new[(x, z)] = new.get((x, z), 0j) + c * ct
                        sign = -1.0 if (x & mq) else 1.0   # -(-1)^{x_q} st
                        k = (x ^ mq, z ^ mq)
                        new[k] = new.get(k, 0j) + c * (sign * st)
            op = new
        # prune
        if thresh:
            op = {k: v for k, v in op.items() if abs(v) > thresh}
        if len(op) > max_terms:
            items = sorted(op.items(), key=lambda kv: -abs(kv[1]))[:max_terms]
            op = dict(items)
        support = 0
        for (x, z) in op:
            support |= x | z
        peak = max(peak, len(op))
    exp = sum(c for (x, z), c in op.items() if x == 0).real
    return exp, peak


def swap_bits(v, a, b):
    ba = (v >> a) & 1; bb = (v >> b) & 1
    if ba != bb:
        v ^= (1 << a) | (1 << b)
    return v


_G = None  # module-level gate list for multiprocessing workers


def _worker(args):
    i, max_terms, thresh = args
    e, peak = backprop_marginal(_G, i, max_terms=max_terms, thresh=thresh)
    return i, e, peak


def _init_pool(gates):
    global _G
    _G = gates


def solve(path, qubits=None, max_terms=200000, thresh=1e-9, verbose=True, procs=1):
    n, gates = parse_qasm(path)
    t0 = time.time()
    qs = list(range(n)) if qubits is None else list(qubits)
    bits = [None] * n
    exps = [None] * n
    if procs > 1:
        import multiprocessing as mp
        with mp.Pool(procs, initializer=_init_pool, initargs=(gates,)) as pool:
            for i, e, peak in pool.imap_unordered(
                    _worker, [(i, max_terms, thresh) for i in qs]):
                bits[i] = 0 if e > 0 else 1
                exps[i] = e
                if verbose:
                    print(f"  q[{i:3d}] <Z>={e:+.4f} -> {bits[i]}  (peak terms={peak})", flush=True)
    else:
        for i in qs:
            e, peak = backprop_marginal(gates, i, max_terms=max_terms, thresh=thresh)
            bits[i] = 0 if e > 0 else 1
            exps[i] = e
            if verbose:
                print(f"  q[{i:3d}] <Z>={e:+.4f} -> {bits[i]}  (peak terms={peak})", flush=True)
    perq = "".join(str(b) for b in bits)
    qiskit = perq[::-1]
    conf = float(np.min([abs(e) for e in exps])) if all(e is not None for e in exps) else None
    print(f"\n  file={path}  n={n}  gates={len(gates)}  time={time.time()-t0:.1f}s")
    print(f"  SECRET q[n-1..0]={qiskit}")
    print(f"         q[0..n-1]={perq}")
    print(f"  min|<Z>| (weakest bit confidence) = {conf:.4f}")
    import json, os
    with open("pauli_results.jsonl", "a") as f:
        f.write(json.dumps(dict(file=os.path.basename(path), n=n, secret_qiskit=qiskit,
                secret_perqubit=perq, min_absZ=conf, time=round(time.time()-t0,1))) + "\n")
    return qiskit, perq, exps


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("circuit")
    ap.add_argument("--max-terms", type=int, default=200000)
    ap.add_argument("--thresh", type=float, default=1e-9)
    ap.add_argument("--procs", type=int, default=1)
    args = ap.parse_args()
    solve(args.circuit, max_terms=args.max_terms, thresh=args.thresh, procs=args.procs)
