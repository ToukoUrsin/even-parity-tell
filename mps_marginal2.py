#!/usr/bin/env python3
"""Fiedler-reordered MPS marginal solver -- speed variant of mps_marginal.py for
the DENSE targets (thousands of long-range CX).

Why reorder: CircuitMPS applies a 2-qubit gate on distant sites by SWAP-ing them
adjacent, applying, swapping back -> O(distance) SVDs per gate. On the dense
obfuscated circuits this swap churn dominates the build. If we first relabel the
qubits by a Fiedler (spectral) ordering of the CX-coupling graph, frequently
coupled qubits sit next to each other -> far fewer/shorter swaps -> much faster
build, and the true (tiny) entanglement still compresses to a small bond.

Relabelling is an EXACT permutation of qubit indices (we rename wire q -> pos[q]
in every gate). We compute marginals on the reordered chain, then map each site's
<Z> back to the original qubit index. The secret in original q[0..n-1] order is
then reversed for the Qiskit q[n-1..0] submission string -- identical decode
convention to mps_marginal.py (validated on 48_8).

Validation: run on a known circuit and check the secret matches.

Usage: mps_marginal2.py <qasm> [max_bond=128] [cutoff=1e-7] [shots=300] [dt=128] [order=fiedler|bfs|identity]
"""
import sys, os, time, json
from collections import defaultdict, deque
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import gpu_patch; gpu_patch.apply()
import svd_fix; svd_fix.apply()

import quimb as qu
import quimb.tensor as qt
from qiskit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit_quimb import quimb_circuit
from mps_marginal import _site_rho

_DT = {"128": np.complex128, "64": np.complex64}


def coupling_graph(circuit):
    n = circuit.num_qubits
    pair = defaultdict(int)
    for inst in circuit.data:
        if inst.operation.num_qubits == 2:
            qs = sorted(circuit.find_bit(q).index for q in inst.qubits)
            pair[(qs[0], qs[1])] += 1
    return n, pair


def fiedler_order(n, pair):
    L = np.zeros((n, n))
    for (a, b), w in pair.items():
        L[a, b] -= w; L[b, a] -= w
        L[a, a] += w; L[b, b] += w
    try:
        vals, vecs = np.linalg.eigh(L)
        idx = np.argsort(vals)
        fied = vecs[:, idx[1]]
        return list(np.argsort(fied))
    except Exception:
        return list(range(n))


def bfs_order(n, pair):
    nbr = defaultdict(set)
    for (a, b) in pair:
        nbr[a].add(b); nbr[b].add(a)
    order = []; seen = [False] * n
    for s in range(n):
        if seen[s]:
            continue
        q = deque([s]); seen[s] = True
        while q:
            u = q.popleft(); order.append(u)
            for v in sorted(nbr[u]):
                if not seen[v]:
                    seen[v] = True; q.append(v)
    return order


def relabel(circuit, order):
    """Return a circuit where original qubit `q` is moved to wire `pos[q]`,
    pos[q] = index of q in `order`. Exact same unitary, just renamed wires."""
    n = circuit.num_qubits
    pos = [0] * n
    for new_wire, oldq in enumerate(order):
        pos[oldq] = new_wire
    qr = QuantumRegister(n, "q")
    out = QuantumCircuit(qr)
    for inst in circuit.data:
        if inst.operation.name in ("measure", "barrier"):
            continue
        newqs = [qr[pos[circuit.find_bit(q).index]] for q in inst.qubits]
        out.append(inst.operation, newqs)
    return out, pos


def solve(path, max_bond=128, cutoff=1e-7, shots=300, seed=1234, dt="128",
          order="fiedler"):
    dtype = _DT[str(dt)]
    def to_backend(x):
        return np.asarray(x, dtype=dtype)
    name = os.path.basename(path)
    t0 = time.time()
    circuit = QuantumCircuit.from_qasm_file(path)
    n = circuit.num_qubits
    _, pair = coupling_graph(circuit)
    if order == "fiedler":
        ordr = fiedler_order(n, pair)
    elif order == "bfs":
        ordr = bfs_order(n, pair)
    else:
        ordr = list(range(n))
    rc, pos = relabel(circuit, ordr)   # pos[oldq] = wire on the MPS chain

    # swap-distance proxy before/after, for the log
    def cutwidth(p):
        cut = np.zeros(n + 1)
        for (a, b) in pair:
            l, r = sorted((p[a], p[b]))
            if l == r:
                continue
            cut[l] += 1; cut[r] -= 1
        return int(np.cumsum(cut)[:n - 1].max()) if n > 1 else 0
    cw_id = cutwidth(list(range(n)))
    cw_new = cutwidth(pos)
    print(f"=== {name} n={n} ops={dict(circuit.count_ops())} order={order} "
          f"cutwidth {cw_id}->{cw_new} max_bond={max_bond} cutoff={cutoff} dt={dt} ===",
          flush=True)

    qc = quimb_circuit(rc, quimb_circuit_class=qt.CircuitMPS,
                       to_backend=to_backend, max_bond=max_bond,
                       cutoff=cutoff, progbar=False)
    psi = qc.psi
    try:
        psi.normalize()
    except Exception:
        pass
    t_build = time.time() - t0
    try:
        mb_reached = int(psi.max_bond())
    except Exception:
        mb_reached = None
    print(f"  built in {t_build:.1f}s  max_bond_reached={mb_reached}", flush=True)

    # marginals on the chain, mapped back to original qubit index
    zexp = [None] * n
    bits_q0 = ["?"] * n
    for oldq in range(n):
        wire = pos[oldq]
        try:
            rho = _site_rho(psi, wire)
            r11 = float(np.real(rho[1, 1])); r00 = float(np.real(rho[0, 0]))
            tr = r00 + r11
            if tr > 0:
                r11 /= tr; r00 /= tr
            z = r00 - r11
        except Exception:
            z, r11 = 0.0, 0.5
        zexp[oldq] = round(z, 5)
        bits_q0[oldq] = "1" if r11 > 0.5 else "0"
    voted_q0 = "".join(bits_q0)
    qiskit_str = voted_q0[::-1]

    # sampling cross-check (samples are in chain order -> map back)
    samp_q0 = None
    try:
        samples = list(qc.sample(shots, seed=seed))
        bp = np.array([[int(s) for s in ss] for ss in samples]).mean(axis=0)
        chain_bits = (bp > 0.5).astype(int)
        samp_q0 = "".join(str(int(chain_bits[pos[oldq]])) for oldq in range(n))
    except Exception as e:
        samp_q0 = f"SAMPLE_FAIL:{type(e).__name__}"

    margins = [abs(z) for z in zexp]
    weak = [i for i in range(n) if abs(zexp[i]) < 0.2]
    agree = (samp_q0 == voted_q0) if isinstance(samp_q0, str) and len(samp_q0) == n else None
    rec = dict(name=name, n=n, order=order, cutwidth=[cw_id, cw_new],
               max_bond=max_bond, max_bond_reached=mb_reached, cutoff=cutoff, dt=str(dt),
               secret_q0=voted_q0, secret_qiskit=qiskit_str,
               sample_q0=samp_q0,
               sample_qiskit=(samp_q0[::-1] if isinstance(samp_q0, str) and len(samp_q0) == n else None),
               z_vs_sample_agree=agree, zexp=zexp,
               min_margin=round(min(margins), 4) if margins else 0.0,
               mean_margin=round(float(np.mean(margins)), 4) if margins else 0.0,
               weak_qubits=weak, n_weak=len(weak),
               t_build=round(t_build, 1), t_total=round(time.time() - t0, 1))
    print("MPSZ2:" + json.dumps(rec), flush=True)
    return rec


if __name__ == "__main__":
    path = sys.argv[1]
    kw = {}
    if len(sys.argv) > 2: kw["max_bond"] = int(sys.argv[2])
    if len(sys.argv) > 3: kw["cutoff"] = float(sys.argv[3])
    if len(sys.argv) > 4: kw["shots"] = int(sys.argv[4])
    if len(sys.argv) > 5: kw["dt"] = sys.argv[5]
    if len(sys.argv) > 6: kw["order"] = sys.argv[6]
    solve(path, **kw)
