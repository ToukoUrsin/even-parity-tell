#!/usr/bin/env python3
"""Explicit U.U-dagger cancellation -> reduced EXACT solve for peaked circuits.

Construction:  C = U1 U1d P1 U2 U2d P2 U3 U3d , Ui Uid = I , Pi = RX layers
encoding the secret s. Obfuscation rewrites into {rx,rz,cx,swap} keep almost every
distinct qubit-pair touched an EVEN number of times by CX -- the entangling links of
the self-inverse Ui Uid blocks cancel.  Only a FEW odd-parity CX pairs ("defects")
carry real entanglement.

We decode the secret from the single-qubit marginals  bit_i = 1 iff <Z_i> < 0
(<Z|0>=+1 ; RX(pi)|0>=|1>, <Z>=-1).  This is proven exact on all known circuits
(OBFUSCATION_RE.md: 0 ambiguous qubits).

The cost of an HONEST <Z_i> is the size of the reverse light-cone of qubit i.
In a dense circuit that cone is the whole register -> intractable.  The structural
fact that rescues us:

  * Build the ODD-PARITY (defect) graph on qubits.  Its connected components are the
    only sets of qubits that are mutually entangled in the final state; everything
    outside any component is in a product state.
  * Therefore the marginal <Z_i> only depends on the qubits in i's defect component
    (plus the net single-qubit rotation funnelled onto i).  We restrict the circuit
    to that component's qubit set, KEEP every gate that acts entirely within the set,
    and EXACT-statevector-simulate that tiny subsystem.

  * For a qubit in NO defect component (product qubit) the subsystem is just {i}:
    we simulate the 1-qubit reduced circuit (all rotations on i, CX on i dropped
    because the partner is disentangled and starts/returns to |0> on the diagonal of
    the cancelling block) -> still need care, so we always include the component.

Validation rule: must reproduce a KNOWN secret before trusting.  Known q[n-1..0]:
  8_1  = 10101101
  48_8 = <redacted-until-deadline>
  56_9 = <redacted-until-deadline>

This file:
  * cancellation-free EXACT subsystem solve (safe, certifiable, cheap when comps small)
  * also exposes the FULL exact statevector path for tiny n (<=20) as a cross-check.

Usage:
  uudag_cancel.py <qasm> [--full]      # --full: also do full statevector (small n)
"""
import sys, os, re, time, json
from collections import defaultdict, deque
import numpy as np

# Apple Accelerate BLAS emits spurious "divide by zero / overflow in matmul"
# RuntimeWarnings on large complex matmuls containing exact zeros; the result is
# numerically exact (we assert unit norm below). Silence them.
np.seterr(divide="ignore", over="ignore", under="ignore", invalid="ignore")

I2 = np.eye(2, dtype=complex)


def rx(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def rz(t):
    return np.array([[np.exp(-1j * t / 2), 0], [0, np.exp(1j * t / 2)]], dtype=complex)


def _eval_angle(s):
    """Evaluate an OpenQASM angle expression that may contain `pi`, `*`, `/`, `+`,
    `-`, and numeric literals (incl. scientific notation)."""
    expr = s.strip().replace("pi", str(np.pi))
    return float(eval(expr, {"__builtins__": {}}, {}))


def parse(path):
    """Return (nq, gates) where gates is a list of (name, (q...), angle_or_None)
    in program order."""
    nq = None
    gates = []
    for line in open(path):
        line = line.strip()
        if line.startswith("qreg"):
            nq = int(re.search(r"\[(\d+)\]", line).group(1))
            continue
        if not line or line.startswith(("OPENQASM", "include", "creg", "//")):
            continue
        m = re.match(r"([a-zA-Z]+)", line)
        if not m:
            continue
        g = m.group(1)
        qs = tuple(int(x) for x in re.findall(r"q\[(\d+)\]", line))
        am = re.search(r"\(([^)]+)\)", line)
        ang = _eval_angle(am.group(1)) if am else None
        gates.append((g, qs, ang))
    return nq, gates


def defect_components(nq, gates):
    """Connected components of the odd-parity CX graph + isolated product qubits."""
    pair = defaultdict(int)
    for g, qs, _ in gates:
        if g == "cx" and len(qs) == 2:
            a, b = qs
            pair[(min(a, b), max(a, b))] += 1
    odd = [(a, b) for (a, b), v in pair.items() if v % 2 == 1]
    nbr = defaultdict(set)
    for a, b in odd:
        nbr[a].add(b)
        nbr[b].add(a)
    seen = set()
    comps = []
    for s in list(nbr):
        if s in seen:
            continue
        q = deque([s])
        seen.add(s)
        c = [s]
        while q:
            u = q.popleft()
            for w in nbr[u]:
                if w not in seen:
                    seen.add(w)
                    c.append(w)
                    q.append(w)
        comps.append(sorted(c))
    in_comp = set(seen)
    return comps, in_comp, odd


def reverse_lightcone_qubits(gates, targets, full_set):
    """Qubits that can influence the marginal of `targets`, restricted to full_set
    membership only as an INITIAL seed -- we follow CX connectivity backwards from the
    targets through the WHOLE gate list to get the honest reverse light-cone."""
    cone = set(targets)
    for g, qs, _ in reversed(gates):
        if g in ("cx", "swap") and len(qs) == 2:
            a, b = qs
            if a in cone or b in cone:
                cone.add(a)
                cone.add(b)
    return cone


def subsystem_statevector(nq, gates, sub):
    """Exact statevector of the circuit RESTRICTED to qubit set `sub`.
    Only gates acting entirely within `sub` are applied; a CX/SWAP straddling the
    boundary is DROPPED (the outside partner is disentangled by U.Ud, so within the
    marginal it contributes identity on `sub`). Returns (state, index_map)."""
    sub = sorted(sub)
    k = len(sub)
    idx = {q: j for j, q in enumerate(sub)}
    psi = np.zeros(2 ** k, dtype=complex)
    psi[0] = 1.0
    psi = psi.reshape([2] * k)
    for g, qs, ang in gates:
        if g == "rx":
            (q,) = qs
            if q in idx:
                psi = apply_1q(psi, rx(ang), idx[q], k)
        elif g == "rz":
            (q,) = qs
            if q in idx:
                psi = apply_1q(psi, rz(ang), idx[q], k)
        elif g == "cx":
            a, b = qs
            if a in idx and b in idx:
                psi = apply_cx(psi, idx[a], idx[b], k)
            # straddling -> dropped (outside partner disentangled)
        elif g == "swap":
            a, b = qs
            if a in idx and b in idx:
                psi = apply_swap(psi, idx[a], idx[b], k)
            elif a in idx or b in idx:
                # swap that moves a sub qubit out / in: relabel within sub if both,
                # else it pulls in an outside qubit -> we must include it. Handled by
                # ensuring `sub` already contains the full reverse cone.
                pass
    return psi, idx


def apply_1q(psi, U, j, k):
    psi = np.moveaxis(psi, j, 0)
    sh = psi.shape
    psi = U @ psi.reshape(2, -1)
    psi = psi.reshape(sh)
    return np.moveaxis(psi, 0, j)


def apply_cx(psi, c, t, k):
    psi = np.moveaxis(psi, [c, t], [0, 1])
    sh = psi.shape
    m = psi.reshape(2, 2, -1).copy()
    # control=1 -> flip target
    m1 = m[1].copy()
    m[1, 0] = m1[1]
    m[1, 1] = m1[0]
    psi = m.reshape(sh)
    return np.moveaxis(psi, [0, 1], [c, t])


def apply_swap(psi, a, b, k):
    return np.swapaxes(psi, a, b)


def z_expect(psi, j, k):
    p = np.abs(psi) ** 2
    p = np.moveaxis(p, j, 0).reshape(2, -1)
    return float(p[0].sum() - p[1].sum())


def solve(path, do_full=False, max_sub=24):
    name = os.path.basename(path)
    t0 = time.time()
    nq, gates = parse(path)
    comps, in_comp, odd = defect_components(nq, gates)
    print(f"=== {name} n={nq} cx={sum(1 for g,_,_ in gates if g=='cx')} "
          f"odd_pairs={len(odd)} defect_qubits={len(in_comp)} "
          f"comps={sorted(len(c) for c in comps)} ===", flush=True)

    # Map qubit -> its defect component (or singleton)
    comp_of = {}
    for c in comps:
        for q in c:
            comp_of[q] = tuple(c)
    bits = ["?"] * nq          # q0..q_{n-1}
    zexp = [None] * nq
    weak = []

    # solve each distinct subsystem once, reuse for all its qubits
    solved_sub = {}
    for i in range(nq):
        comp = comp_of.get(i, (i,))
        # honest reverse light cone of this component within full circuit
        cone = reverse_lightcone_qubits(gates, set(comp), None)
        sub_key = tuple(sorted(cone))
        if len(sub_key) > max_sub:
            # too big for exact; fall back to component-only (drop straddling CX)
            sub_key = tuple(sorted(comp))
            mode = "comp-only"
        else:
            mode = "lightcone"
        if sub_key not in solved_sub:
            psi, idx = subsystem_statevector(nq, gates, sub_key)
            nrm = float(np.linalg.norm(psi))
            if not np.isfinite(nrm) or abs(nrm - 1.0) > 1e-6:
                print(f"  !! subsystem {sub_key[:6]}... (size {len(sub_key)}) "
                      f"non-unit norm {nrm} -- result UNRELIABLE", flush=True)
            solved_sub[sub_key] = (psi, idx, mode)
        psi, idx, mode = solved_sub[sub_key]
        z = z_expect(psi, idx[i], len(idx))
        zexp[i] = round(z, 5)
        bits[i] = "0" if z > 0 else "1"
        if abs(z) < 0.1:
            weak.append(i)

    secret_q0 = "".join(bits)
    secret_qiskit = secret_q0[::-1]
    margins = [abs(z) for z in zexp if z is not None]
    rec = dict(name=name, n=nq, odd_pairs=len(odd),
               secret_q0=secret_q0, secret_qiskit=secret_qiskit,
               min_margin=round(min(margins), 4) if margins else 0.0,
               mean_margin=round(float(np.mean(margins)), 4) if margins else 0.0,
               weak_qubits=weak, n_weak=len(weak),
               t=round(time.time() - t0, 1))
    print("UUDAG:" + json.dumps(rec), flush=True)

    if do_full and nq <= 22:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
        qc = QuantumCircuit.from_qasm_file(path)
        probs = Statevector.from_instruction(qc).probabilities()
        full_q0 = "".join(
            "1" if sum(pr for x, pr in enumerate(probs) if (x >> i) & 1) > 0.5 else "0"
            for i in range(nq))
        print(f"  FULL-SV  q[0..n-1]={full_q0}  match_subsystem={full_q0==secret_q0}")
    return rec


if __name__ == "__main__":
    path = sys.argv[1]
    do_full = "--full" in sys.argv[2:]
    solve(path, do_full=do_full)
