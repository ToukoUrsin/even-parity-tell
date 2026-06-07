#!/usr/bin/env python3
"""Drill into odd-parity CX pairs (the non-cancelling entanglement) for 42 & 43,
and quantify 38's structure. Pure parsing, no sims."""
import re
from collections import defaultdict, deque
import numpy as np

FILES = {
 "42": "circuits/challenge-48_42.qasm",
 "38": "circuits/challenge-56_38.qasm",
 "43": "circuits/challenge-56_43.qasm",
}

def parse(path):
    nq=None; gates=[]
    for line in open(path):
        line=line.strip()
        if line.startswith("qreg"):
            nq=int(re.search(r"\[(\d+)\]",line).group(1)); continue
        if not line or line.startswith(("OPENQASM","include","creg","//")): continue
        m=re.match(r"([a-zA-Z]+)",line)
        if not m: continue
        g=m.group(1); qs=tuple(int(x) for x in re.findall(r"q\[(\d+)\]",line))
        gates.append((g,qs))
    return nq,gates

def pair_cx(nq,gates):
    p=defaultdict(int)
    for g,qs in gates:
        if g=="cx" and len(qs)==2:
            a,b=qs; p[(min(a,b),max(a,b))]+=1
    return p

for tag,path in FILES.items():
    nq,gates=parse(path)
    p=pair_cx(nq,gates)
    odd=[(k,v) for k,v in p.items() if v%2==1]
    odd.sort()
    print(f"=== {tag} ({path}) nq={nq} ===")
    print(f"  odd-parity pairs ({len(odd)}):")
    for (a,b),v in odd[:40]:
        print(f"    q{a}-q{b}: {v} CX  (|a-b|={abs(a-b)})")
    if len(odd)>40: print(f"    ... +{len(odd)-40} more")
    # graph of ONLY odd-parity pairs: is the non-cancelling part sparse / a forest?
    onbr=defaultdict(set)
    for (a,b),_ in odd:
        onbr[a].add(b); onbr[b].add(a)
    # components of odd-graph
    seen=set(); comps=[]
    nodes=set(onbr)
    for s in nodes:
        if s in seen: continue
        q=deque([s]); seen.add(s); c=[s]
        while q:
            u=q.popleft()
            for w in onbr[u]:
                if w not in seen: seen.add(w); c.append(w); q.append(w)
        comps.append(sorted(c))
    print(f"  odd-graph: {len(nodes)} qubits, {len(odd)} edges, {len(comps)} components "
          f"sizes={sorted(len(c) for c in comps)}")
    # is odd-graph a tree/forest? edges == nodes - comps  => forest
    is_forest = (len(odd) == len(nodes)-len(comps))
    print(f"  odd-graph is forest/acyclic: {is_forest}")
    # min/max CX multiplicity on even pairs (how 'deep' the U Udag stacks are)
    evenv=[v for k,v in p.items() if v%2==0]
    if evenv:
        ev=np.array(evenv)
        print(f"  even-pair CX multiplicity: min={ev.min()} max={ev.max()} mean={ev.mean():.1f} (these fully self-cancel)")
    print()
