#!/usr/bin/env python3
"""Reverse-engineer QMill obfuscation by analyzing actual qasm files."""
import re, sys, math, collections
from math import pi

GATE_RE = re.compile(r'^(rx|rz|cx|swap)\s*(?:\(([^)]*)\))?\s*(.*?);')

def parse(path):
    """Return (nqubits, list of (gate, angle_or_None, [qubits]))."""
    gates = []
    nq = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('qreg'):
                m = re.search(r'q\[(\d+)\]', line)
                nq = int(m.group(1))
                continue
            if not line or line.startswith('OPENQASM') or line.startswith('include'):
                continue
            m = GATE_RE.match(line)
            if not m:
                continue
            gate, ang, qs = m.group(1), m.group(2), m.group(3)
            qubits = [int(x) for x in re.findall(r'q\[(\d+)\]', qs)]
            angle = None
            if ang is not None:
                # eval pi-expressions safely
                expr = ang.replace('pi', str(pi))
                try:
                    angle = float(eval(expr, {'__builtins__': {}}, {}))
                except Exception:
                    angle = float(ang)
            gates.append((gate, angle, qubits))
    return nq, gates

def wrap(a):
    """Wrap angle to (-pi, pi]."""
    a = a % (2*pi)
    if a > pi:
        a -= 2*pi
    return a

def near(a, target, tol=1e-2):
    return abs(wrap(a - target)) < tol

def analyze_secret_alignment(path, known_secret=None, tol=1e-2):
    """Q1: count rx(pi) per qubit, compare to known secret bits."""
    nq, gates = parse(path)
    # rx near pi (mod 2pi) per qubit
    rxpi_per_q = collections.Counter()
    rx_total_per_q = collections.Counter()
    for g, a, qs in gates:
        if g == 'rx':
            rx_total_per_q[qs[0]] += 1
            if near(a, pi, tol):
                rxpi_per_q[qs[0]] += 1
    print(f"\n=== {path} (nq={nq}) ===")
    print("qubit : #rx(pi)  / #rx_total" + ("  | secret_bit  parity(rxpi)" if known_secret else ""))
    parity_str = []
    for q in range(nq):
        line = f"  q[{q}] : {rxpi_per_q[q]:3d}      / {rx_total_per_q[q]:3d}"
        par = rxpi_per_q[q] % 2
        parity_str.append(str(par))
        if known_secret is not None:
            sb = known_secret.get(q, '?')
            match = '<==MATCH' if str(sb) == str(par) else ''
            line += f"      |   {sb}          {par}   {match}"
        print(line)
    parity_bits = ''.join(parity_str)
    print(f"  rx(pi) parity per qubit q[0..{nq-1}]: {parity_bits}")
    if known_secret is not None:
        secret_q0 = ''.join(str(known_secret[q]) for q in range(nq))
        print(f"  known secret        q[0..{nq-1}]: {secret_q0}")
        nmatch = sum(1 for q in range(nq) if str(known_secret[q]) == parity_str[q])
        print(f"  parity matches secret: {nmatch}/{nq} qubits")
    total_rxpi = sum(rxpi_per_q.values())
    total_rx = sum(rx_total_per_q.values())
    print(f"  total rx(pi)={total_rxpi}  total rx={total_rx}  frac={total_rxpi/max(total_rx,1):.3f}")
    return parity_bits

def histogram_angles(path, tol=1e-2):
    """Q2: histogram rx/rz angles; cluster near pi/2 multiples (Clifford) or pi."""
    nq, gates = parse(path)
    buckets = {'rx': collections.Counter(), 'rz': collections.Counter()}
    near_counts = {'rx': {'0': 0, 'pi/2': 0, 'pi': 0, 'random': 0},
                   'rz': {'0': 0, 'pi/2': 0, 'pi': 0, 'random': 0}}
    totals = {'rx': 0, 'rz': 0}
    for g, a, qs in gates:
        if g not in ('rx', 'rz'):
            continue
        totals[g] += 1
        w = wrap(a)
        if near(a, 0, tol):
            near_counts[g]['0'] += 1
        elif near(a, pi, tol) or near(a, -pi, tol):
            near_counts[g]['pi'] += 1
        elif near(a, pi/2, tol) or near(a, -pi/2, tol):
            near_counts[g]['pi/2'] += 1
        else:
            near_counts[g]['random'] += 1
    print(f"\n--- angle histogram {path} ---")
    for g in ('rx', 'rz'):
        t = max(totals[g], 1)
        nc = near_counts[g]
        clifford = nc['0'] + nc['pi/2'] + nc['pi']
        print(f"  {g}: total={totals[g]}  near0={nc['0']} ({nc['0']/t:.1%})  "
              f"nearPi/2={nc['pi/2']} ({nc['pi/2']/t:.1%})  nearPi={nc['pi']} ({nc['pi']/t:.1%})  "
              f"random={nc['random']} ({nc['random']/t:.1%})  | Clifford-frac={clifford/t:.1%}")
    return near_counts, totals

def adjacent_cx_pairs(path):
    """Q3: count back-to-back cx on same (ordered) pair with nothing touching those qubits between."""
    nq, gates = parse(path)
    cancel = 0
    swap_cancel = 0
    # find cx i,j followed later by cx i,j with no gate touching i or j in between
    last_cx = {}  # (a,b) -> index, but need "nothing between on those qubits"
    # simpler: scan, track for each pair the pending cx; a gate on either qubit resets
    pending = {}  # frozenset-> (ordered tuple, index)
    n = len(gates)
    i = 0
    # We'll do: for each cx, look forward for the next gate touching qubit a or b.
    # If it's an identical cx a,b -> cancelling pair.
    adj_examples = []
    for idx, (g, a, qs) in enumerate(gates):
        if g != 'cx':
            continue
        a0, b0 = qs[0], qs[1]
        # scan forward
        for j in range(idx+1, n):
            gg, aa, qq = gates[j]
            touches = any(x in (a0, b0) for x in qq)
            if not touches:
                continue
            # first gate touching a0 or b0
            if gg == 'cx' and qq == [a0, b0]:
                cancel += 1
                if len(adj_examples) < 5:
                    adj_examples.append((idx, j, a0, b0))
            break
    # swaps back to back
    for idx, (g, a, qs) in enumerate(gates):
        if g != 'swap':
            continue
        a0, b0 = qs[0], qs[1]
        for j in range(idx+1, n):
            gg, aa, qq = gates[j]
            if any(x in (a0, b0) for x in qq):
                if gg == 'swap' and set(qq) == {a0, b0}:
                    swap_cancel += 1
                break
    total_cx = sum(1 for g, a, q in gates if g == 'cx')
    print(f"\n--- UU-dagger / cancelling pairs {path} ---")
    print(f"  total cx={total_cx}  adjacent-cancelling cx pairs (same ordered pair, nothing between)={cancel}")
    print(f"  adjacent-cancelling swap pairs={swap_cancel}")
    if adj_examples:
        print(f"  examples (gate idx i, idx j, qa, qb): {adj_examples}")
    return cancel, total_cx

def net_permutation(path):
    """Q4: compute net permutation induced by swap gates (treating only swaps)."""
    nq, gates = parse(path)
    perm = list(range(nq))  # perm[wire] = logical qubit currently on that wire
    nswap = 0
    for g, a, qs in gates:
        if g == 'swap':
            i, j = qs
            perm[i], perm[j] = perm[j], perm[i]
            nswap += 1
    nontrivial = sum(1 for w in range(nq) if perm[w] != w)
    print(f"\n--- net swap permutation {path} ---")
    print(f"  #swaps={nswap}  net permutation moves {nontrivial}/{nq} wires from identity")
    if nswap:
        print(f"  perm[wire]->logical (first 24): {perm[:24]}")
    return nswap, perm

if __name__ == '__main__':
    import glob, os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # Known secret for 8_1: q[n-1..0]=10101101 => q0..q7 = 1,0,1,1,0,1,0,1
    secret_8_1 = {0:1,1:0,2:1,3:1,4:0,5:1,6:0,7:1}
    print("########## Q1: secret-revealing rx(pi) alignment ##########")
    analyze_secret_alignment('circuits/challenge-8_1.qasm', secret_8_1)
    # 8_11 secret q[n-1..0]=01001110 => q0..q7 = 0,1,1,1,0,0,1,0
    secret_8_11 = {0:0,1:1,2:1,3:1,4:0,5:0,6:1,7:0}
    analyze_secret_alignment('circuits/challenge-8_11.qasm', secret_8_11)
    # 16_2 and 24_3 secret unknown exactly; run without
    analyze_secret_alignment('circuits/challenge-16_2.qasm')
    analyze_secret_alignment('circuits/challenge-24_3.qasm')

    print("\n########## Q2: angle histograms ##########")
    for f in ['circuits/challenge-8_1.qasm','circuits/challenge-8_11.qasm',
              'circuits/challenge-16_2.qasm','circuits/challenge-24_3.qasm',
              'circuits/challenge-48_42.qasm','circuits/challenge-104_49.qasm']:
        if os.path.exists(f):
            histogram_angles(f)

    print("\n########## Q3: UU-dagger / cancelling cx pairs ##########")
    for f in ['circuits/challenge-8_1.qasm','circuits/challenge-8_11.qasm',
              'circuits/challenge-16_2.qasm','circuits/challenge-24_3.qasm']:
        adjacent_cx_pairs(f)

    print("\n########## Q4: net swap permutation ##########")
    for f in sorted(glob.glob('circuits/challenge-*.qasm')):
        net_permutation(f)
