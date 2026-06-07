#!/usr/bin/env python3
"""Unswapping peak-finder (Kremer/Dupuis 2604.21908) for dense all-to-all peaked
circuits with hidden permutations -- the Hard tier (36,37,39,40,41) and the wall.

Pipeline (from pcsim notebook):
  1. Consolidate rx/rz/cx into 2-qubit "unitary" blocks (Collect2qBlocks+Consolidate).
  2. mpo_compress_unswap: split at midpoint, cancel U.U-dagger into a central MPO,
     greedily UNSWAP to discover the hidden permutation and keep the bond bounded.
  3. mpo_to_mps: apply leftover layers + MPO to |0> -> low-bond MPS.
  4. Sample, majority bitstring, undo the unswap permutation.

Usage: unswap_solve.py <circuit.qasm> [max_bond=8192] [cutoff=0.002] [shots=1000]
"""
import sys, os, time, json
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                       # for gpu_patch
sys.path.insert(0, os.path.join(_HERE, "pcsim"))  # for unswap/utils/circuit_mpo
import gpu_patch
gpu_patch.apply()                                # autoray cupy-Device + swap_back shims
# svd_fix (numba-SVD shim) was written for quimb 1.11 and breaks 1.14 (array_split
# arity). CPU box runs dense-circuit unswap fine without it. Keep SABRE_TRIALS=4.
from collections import Counter
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Collect2qBlocks, ConsolidateBlocks
from utils import to_backend_cuda
from unswap import mpo_compress_unswap, mpo_to_mps


def solve(path, max_bond=8192, cutoff=0.002, shots=1000, seed=123, max_its=20):
    name = os.path.basename(path)
    t0 = time.time()
    circuit = QuantumCircuit.from_qasm_file(path)
    pm = PassManager([Collect2qBlocks(), ConsolidateBlocks(force_consolidate=True)])
    circuit = pm.run(circuit)
    n = circuit.num_qubits
    print(f"=== {name} n={n} unitary_blocks={circuit.count_ops().get('unitary', 0)} "
          f"max_bond={max_bond} cutoff={cutoff} seed={seed} max_its={max_its} ===", flush=True)

    uthr = float(os.environ.get("QMILL_UNSWAP_THRESH", "1e6"))  # raise -> absorb more unitaries per cycle -> FAR fewer (expensive) SabreSwap rewires; bond is defect-bounded so transient growth is cheap on a big-RAM box
    mpo, layers_left, layers_right, _ = mpo_compress_unswap(
        circuit, seed=seed, to_backend=to_backend_cuda, cutoff=cutoff,
        max_bond=max_bond, unswap_threshold=uthr, center_ratio=0.5, equal=False,
        flip_freq=None, max_its=max_its, early_stopping_gates=0,
        hows=("both", "left", "right"))
    mps, perm = mpo_to_mps(mpo, layers_left[:-2], layers_right,
                           cutoff=cutoff, to_backend=to_backend_cuda)

    raw = [p for p, _ in list(mps.sample(shots))]
    samples = ["".join(str(b) for b in bs) for bs in raw]
    csamples = Counter(samples)
    top_bs, cnt = csamples.most_common(1)[0]
    # MAJORITY-VOTE decode per chain position. Heavy obfuscation -> SHALLOW peak:
    # every sample is a distinct near-neighbor, so most_common (peak_count~1) FAILS,
    # but the <Z_i> marginal stays bimodal (OBFUSCATION_RE) -> majority vote recovers
    # the peak. min_margin tells real (decisive) from destroyed (~0).
    import numpy as _np
    arr = _np.array([[int(c) for c in s] for s in samples])  # shots x n (chain order)
    ones = arr.sum(axis=0)
    pred_bs = "".join('1' if o > shots / 2 else '0' for o in ones)
    margins = _np.abs(2 * ones - shots) / shots
    min_margin = round(float(margins.min()), 4)
    mean_margin = round(float(margins.mean()), 4)
    n_weak = int((margins < 0.1).sum())
    try:                       # never lose a 4h run to a decode bug again
        mps.save(os.path.join(_HERE, f"mps_{name}.dump"))
    except Exception:
        pass
    # undo unswap permutation (forward); inverse provided too for convention-fixing
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    perm_pred = "".join(pred_bs[i] for i in perm)
    invperm_pred = "".join(pred_bs[i] for i in inv)

    rec = dict(name=name, n=n,
               raw_pred=pred_bs, perm=list(perm),
               secret_perm=perm_pred, secret_perm_rev=perm_pred[::-1],
               secret_inv=invperm_pred, secret_inv_rev=invperm_pred[::-1],
               min_margin=min_margin, mean_margin=mean_margin, n_weak=n_weak,
               most_common=top_bs, most_common_count=cnt,
               shots=shots, n_distinct=len(csamples), t=round(time.time() - t0, 1))
    print("UNSWAP:" + json.dumps(rec), flush=True)
    return rec


if __name__ == "__main__":
    path = sys.argv[1]
    kw = {}
    if len(sys.argv) > 2: kw["max_bond"] = int(sys.argv[2])
    if len(sys.argv) > 3: kw["cutoff"] = float(sys.argv[3])
    if len(sys.argv) > 4: kw["shots"] = int(sys.argv[4])
    if len(sys.argv) > 5: kw["seed"] = int(sys.argv[5])
    if len(sys.argv) > 6: kw["max_its"] = int(sys.argv[6])
    solve(path, **kw)
