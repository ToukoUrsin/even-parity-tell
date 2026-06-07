#!/usr/bin/env python3
"""Probe the EXACT single-amplitude contraction width for each target.

If the circuit's few defects -> low treewidth, an exact |<s|C|0>|^2 contraction is
feasible (truncation-free => CERTAIN answer). We only REHEARSE (find a path, read its
width) -- no contraction, so this is cheap and CPU-only. Width W => peak intermediate
~2^W complex128 = 2^W * 16 bytes. W<=30 ~ 16GB; W<=34 ~ 256GB (with 1.4TB RAM, feasible).

Usage: probe_width.py <id> [<id> ...]   e.g. probe_width.py 48_42 56_43
"""
import sys, time, json
import quimb.tensor as qtn
import cotengra as ctg

# Time-bounded hyperoptimizer: good width estimate fast, single-process so we don't
# spawn 180 workers and starve the GPU host threads. minimize='size' targets WIDTH
# (peak memory), which is what gates exact feasibility.
def make_opt():
    return ctg.HyperOptimizer(minimize="size", max_time=25, max_repeats=256,
                              parallel=False, progbar=False)

for cid in sys.argv[1:]:
    path = f"circuits/challenge-{cid}.qasm"
    t0 = time.time()
    rec = {"id": cid}
    try:
        circ = qtn.Circuit.from_openqasm2_file(path)
        n = circ.N
        s = "0" * n
        reh = circ.amplitude_rehearse(b=s, optimize=make_opt())
        tree = reh["tree"] if isinstance(reh, dict) and "tree" in reh else reh
        W = float(tree.contraction_width())
        C = float(tree.contraction_cost())  # in FLOPs (count); log10 below
        import math
        rec.update(n=n, width=round(W, 2), log10_flops=round(math.log10(C + 1), 2),
                   feasible_exact=bool(W <= 34), t=round(time.time() - t0, 1))
    except Exception as e:
        rec.update(error=f"{type(e).__name__}: {e}", t=round(time.time() - t0, 1))
    print("WIDTH:" + json.dumps(rec), flush=True)
