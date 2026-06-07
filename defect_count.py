#!/usr/bin/env python3
"""Count odd-parity CX 'defects' (the real entanglement) per circuit + defect-graph
structure. Pure parsing, no deps. This is the difficulty metric that matters."""
import re, sys
from collections import Counter


def defects(path):
    pc = Counter(); n = 0; ncx = 0
    for ln in open(path):
        ln = ln.strip()
        m = re.match(r'qreg\s+\w+\[(\d+)\]', ln)
        if m:
            n = int(m.group(1))
        m = re.match(r'cx\s+\w+\[(\d+)\]\s*,\s*\w+\[(\d+)\]', ln)
        if m:
            ncx += 1
            pc[tuple(sorted((int(m.group(1)), int(m.group(2)))))] += 1
    odd = [p for p, c in pc.items() if c % 2]
    # union-find for defect-graph components
    par = {}
    def find(x):
        par.setdefault(x, x)
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    qs = set()
    for a, b in odd:
        qs.add(a); qs.add(b)
        ra, rb = find(a), find(b)
        if ra != rb:
            par[ra] = rb
    comps = Counter(find(q) for q in qs)
    maxc = max(comps.values()) if comps else 0
    return n, ncx, len(odd), len(qs), maxc


targets = sys.argv[1:] or ['56_38', '48_42', '56_43', '64_44', '72_45', '80_46',
                           '88_47', '96_48', '104_49', '56_9', '48_8', '64_10']
print(f"{'circuit':12s}{'qubits':>7s}{'totCX':>8s}{'defects':>9s}{'defect_q':>10s}{'max_comp':>10s}  verdict")
for c in targets:
    try:
        n, ncx, nd, nq, mc = defects(f"circuits/challenge-{c}.qasm")
        v = 'TRIVIAL' if nd <= 5 else ('easy' if nd <= 15 else ('moderate' if nd <= 40 else 'WALL'))
        print(f"{c:12s}{n:>7d}{ncx:>8d}{nd:>9d}{nq:>10d}{mc:>10d}  {v}")
    except Exception as e:
        print(f"{c:12s}  ERR {e}")
