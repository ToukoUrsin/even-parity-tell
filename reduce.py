#!/usr/bin/env python3
"""Test whether commutation-aware inverse cancellation collapses the U.U-dagger
obfuscation. The construction C = U1 U1d P1 U2 U2d ... telescope-cancels in the
ORIGINAL; the obfuscation hides this behind commutations + rewrites. Qiskit's
CommutativeInverseCancellation explicitly undoes commuted-inverse pairs. If the
gate count collapses, the residual circuit (RX layers + few defects) is easy.

Usage: reduce.py <id> [<id> ...]
"""
import sys, time, json
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    CommutativeInverseCancellation, CommutativeCancellation,
    Optimize1qGatesDecomposition, InverseCancellation, RemoveBarriers,
)


def ops_count(qc):
    d = dict(qc.count_ops())
    cx = d.get("cx", 0) + d.get("cz", 0)
    return sum(d.values()), cx, d


for cid in sys.argv[1:]:
    path = f"circuits/challenge-{cid}.qasm"
    t0 = time.time()
    rec = {"id": cid}
    try:
        qc = QuantumCircuit.from_qasm_file(path)
        n = qc.num_qubits
        tot0, cx0, _ = ops_count(qc)
        # iterate cancellation passes until fixpoint (or cap)
        pm = PassManager([
            RemoveBarriers(),
            Optimize1qGatesDecomposition(),
            CommutativeCancellation(),
            CommutativeInverseCancellation(),
        ])
        prev = None
        for it in range(12):
            qc = pm.run(qc)
            tot, cx, _ = ops_count(qc)
            if prev is not None and tot >= prev:
                break
            prev = tot
        tot1, cx1, d1 = ops_count(qc)
        rec.update(n=n, tot_before=tot0, cx_before=cx0, tot_after=tot1, cx_after=cx1,
                   reduction=round(1 - tot1 / max(tot0, 1), 3), ops_after=d1,
                   t=round(time.time() - t0, 1))
        # save reduced circuit for downstream solving
        with open(f"reduced-{cid}.qasm", "w") as f:
            f.write(qc.qasm() if hasattr(qc, "qasm") else "")
    except Exception as e:
        import traceback
        rec.update(error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-300:],
                   t=round(time.time() - t0, 1))
    print("REDUCE:" + json.dumps(rec), flush=True)
