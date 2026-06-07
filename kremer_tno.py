#!/usr/bin/env python3
"""CPU wrapper for the Kremer-Dupuis two-sided TNO peak finder (./pcsim).
Contracts the circuit from both ends so the U.U-dagger identity padding cancels,
keeping bond bounded by the real (shallow) residual. Reads the peak off per-qubit
marginals. This is the structure-exploiting method for circuits whose single-amplitude
contraction width is too large for solve_tn.

Usage: kremer_tno.py <circuit.qasm> [chunk_size] [max_bond_core] [max_bond_tne] [cutoff]
"""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pcsim"))
from qiskit import QuantumCircuit
from utils import iter_layers, contract_core, tno_to_tne, extract_bitstring, get_bond_sizes

def run(path, chunk_size=2, mb_core=16, mb_tne=8, cutoff=0.01):
    t0 = time.time()
    qc = QuantumCircuit.from_qasm_file(path)
    n = qc.num_qubits
    layers = list(iter_layers(qc))
    res = contract_core(layers, chunk_size=chunk_size, max_bond=mb_core,
                        cutoff=cutoff, to_backend=None)
    tno = res[0] if isinstance(res, tuple) else res
    try:
        bs = get_bond_sizes(tno); maxbond = max(bs) if hasattr(bs, '__iter__') else bs
    except Exception:
        maxbond = None
    tne = tno_to_tne(tno, max_bond=mb_tne, cutoff=cutoff, to_backend=None)
    out = extract_bitstring(tne)
    pred = out[0] if isinstance(out, tuple) else out
    pred = "".join(str(int(b)) for b in pred)
    # report both orderings; validate against known to fix convention
    rec = dict(file=os.path.basename(path), n=n, pred=pred, pred_rev=pred[::-1],
               core_maxbond=str(maxbond), t=round(time.time()-t0, 1),
               chunk=chunk_size, mb_core=mb_core, mb_tne=mb_tne, cutoff=cutoff)
    print("KREMER:" + json.dumps(rec), flush=True)
    return rec

if __name__ == "__main__":
    p = sys.argv[1]
    kw = {}
    if len(sys.argv) > 2: kw['chunk_size'] = int(sys.argv[2])
    if len(sys.argv) > 3: kw['mb_core'] = int(sys.argv[3])
    if len(sys.argv) > 4: kw['mb_tne'] = int(sys.argv[4])
    if len(sys.argv) > 5: kw['cutoff'] = float(sys.argv[5])
    run(p, **kw)
