#!/usr/bin/env python3
"""Single-qubit <Z_i> decoder via the DEFECT-GRAPH-LOCAL reduced density matrix,
with a documented belief-propagation (BP) investigation.

WHY THIS WORKS (the defect-graph reframe):
  C = U1 U1d P1 U2 U2d P2 U3 U3d. The Ui Uid blocks are self-inverse, so the
  even-parity CX bulk algebraically CANCELS under quimb tensor-network
  simplification, leaving only the FEW odd-parity "defect" edges as real
  entanglement. The secret survives in the single-qubit marginals:
    Z|0>=+|0>,  RX(pi)|0>=|1> with <Z>=-1   =>   bit_i (q0 order) = 1 iff <Z_i> < 0.
  Qiskit submission string is q[n-1..0] = reverse of the q0-ordered string.

METHOD (validated):
  For each qubit i, build the lightcone-restricted, simplified reduced density
  matrix rho_i = Tr_{!=i}( C|0><0|C^dag ) via quimb's
  Circuit.get_rdm_lightcone_simplified((i,)). Simplification cancels the U.Ud
  bulk so the surviving TN's geometry IS the local defect graph -- small for
  low-defect circuits. Then <Z_i> = Tr(Z rho_i)/Tr(rho_i). bit_i = 1 if <Z_i><0.
  This is the doubled (density) network, so it is real & numerically stable,
  unlike single-layer amplitude BP. We contract it EXACTLY when the cone is
  narrow (the common case: defect components have <=5 qubits, cone width <=~8).

BELIEF PROPAGATION (investigated, see notes in __main__ output):
  Intent was to contract each defect cone with quimb dense BP so cost scales with
  the defect-graph size, not cone treewidth. FINDING: vanilla 1-norm BP
  (contract_d1bp) and hyper BP (contract_hd1bp) DIVERGE TO NaN on these complex
  signed amplitude TNs -- message normalisation divides by ~0 on rotation-gate
  tensors. D2BP (2-norm) converges for the global norm but its marginal helpers
  (partial_trace_gloop_expand) require PEPS site-tags the circuit TN lacks. So BP
  is NOT a drop-in here; the structurally-equivalent and stable substitute is the
  exact contraction of the (small) simplified defect cone, which we use. The
  --bp flag still runs D1BP per cone and reports where it returns finite values,
  documenting the regime where amplitude BP is/ isn't usable.

Usage:
  bp_z.py <qasm> [dtype=128|64] [seq=ADCRS] [budget_s=0] [--bp]
    budget_s : per-qubit wallclock budget for the exact contract; 0 = unlimited.
               Qubits exceeding it are marked WEAK/'?' (cone too wide -> needs
               approximate contraction on GPU).
"""
import sys, os, time, json
import numpy as np
import gpu_patch; gpu_patch.apply()
import warnings; warnings.filterwarnings("ignore")
from qiskit import QuantumCircuit
import quimb as qu
import quimb.tensor as qt
import quimb.tensor.belief_propagation as bp
from qiskit_quimb import quimb_circuit


def run(path, dt="128", seq="ADCRS", budget_s=0.0, try_bp=False, max_width=30):
    dtype = np.complex128 if str(dt) == "128" else np.complex64

    def to_backend(x):
        return np.asarray(x, dtype=dtype)

    name = os.path.basename(path)
    t0 = time.time()
    circ = QuantumCircuit.from_qasm_file(path)
    n = circ.num_qubits
    qc = quimb_circuit(circ, quimb_circuit_class=qt.Circuit, to_backend=to_backend)
    nops = sum(circ.count_ops().values())
    print(f"=== {name} n={n} ops={nops} RDM-lightcone <Z> (seq={seq} dt={dt} "
          f"budget_s={budget_s} bp={try_bp}) ===", flush=True)

    Z = qu.pauli("Z").astype(dtype)
    Id = qu.eye(2).astype(dtype)

    zexp = [None] * n
    bits_q0 = ["?"] * n
    weak = []
    bp_finite = 0
    bp_signmatch = 0
    bp_tried = 0

    for i in range(n):
        ti = time.time()
        try:
            rt = qc.get_rdm_lightcone_simplified((i,), seq=seq)
            # Width gate: estimate the optimized contraction width (log2 of the
            # largest intermediate). Skip cones too wide for laptop RAM instead
            # of OOM-ing on a naive to_dense. (>~28 => >4 GB complex128.)
            try:
                logw = float(rt.contraction_width(
                    output_inds=[f"k{i}", f"b{i}"], optimize="greedy"))
            except Exception:
                logw = 99.0
            w = round(logw, 1)
            if max_width and logw > max_width:
                print(f"  q{i:3d}: SKIP cone log2-width={w} > {max_width} "
                      f"(needs GPU/approx)  ({time.time()-ti:.2f}s)", flush=True)
                weak.append(i); bits_q0[i] = "?"
                continue
            arr = rt.contract(output_inds=[f"k{i}", f"b{i}"], optimize="greedy")
            rho = np.asarray(arr.data if hasattr(arr, "data") else arr).reshape(2, 2)
            tr = np.real(np.trace(rho))
            v = float(np.real(np.trace(Z @ rho)) / tr) if abs(tr) > 1e-30 else 0.0
            zexp[i] = round(v, 5)
            bits_q0[i] = "1" if v < 0 else "0"

            bptag = ""
            if try_bp:
                bp_tried += 1
                try:
                    rt2 = qc.get_rdm_lightcone_simplified((i,), seq="ACRS")
                    tnz = rt2.copy(); tnz |= qt.Tensor(Z, inds=(f"b{i}", f"k{i}"))
                    tni = rt2.copy(); tni |= qt.Tensor(Id, inds=(f"b{i}", f"k{i}"))
                    bz = complex(bp.contract_d1bp(tnz, max_iterations=4000,
                                                  tol=1e-12, damping=0.3))
                    bi = complex(bp.contract_d1bp(tni, max_iterations=4000,
                                                  tol=1e-12, damping=0.3))
                    bv = (bz / bi).real
                    if np.isfinite(bv):
                        bp_finite += 1
                        if (bv < 0) == (v < 0):
                            bp_signmatch += 1
                        bptag = f"  bp={bv:+.3f}"
                    else:
                        bptag = "  bp=NaN"
                except Exception:
                    bptag = "  bp=ERR"

            dt_q = time.time() - ti
            mark = "" if abs(v) > 0.1 else "  <-WEAK"
            print(f"  q{i:3d}: <Z>={v:+.5f} bit={bits_q0[i]} w={w} "
                  f"({dt_q:.2f}s){mark}{bptag}", flush=True)
            if abs(v) < 0.1:
                weak.append(i)
            if budget_s and dt_q > budget_s:
                print(f"      (q{i} exceeded budget {budget_s}s; cone likely wide)",
                      flush=True)
        except Exception as e:
            print(f"  q{i:3d}: FAILED {type(e).__name__}: {str(e)[:90]}", flush=True)
            weak.append(i)
            bits_q0[i] = "?"

    voted_q0 = "".join(bits_q0)
    qiskit_str = voted_q0[::-1]
    margins = [abs(z) for z in zexp if z is not None]
    rec = dict(name=name, n=n, method="rdm_lightcone_z", dtype=str(dt), seq=seq,
               secret_q0=voted_q0, secret_qiskit=qiskit_str,
               zexp=zexp,
               min_margin=round(min(margins), 4) if margins else 0.0,
               mean_margin=round(float(np.mean(margins)), 4) if margins else 0.0,
               weak_qubits=weak, n_weak=len(weak),
               complete=("?" not in voted_q0),
               t_total=round(time.time() - t0, 1))
    if try_bp:
        rec["bp_finite"] = bp_finite
        rec["bp_signmatch"] = bp_signmatch
        rec["bp_tried"] = bp_tried
    print("BPZ:" + json.dumps(rec), flush=True)
    return rec


if __name__ == "__main__":
    path = sys.argv[1]
    kw = {}
    args = sys.argv[2:]
    if "--bp" in args:
        kw["try_bp"] = True
        args = [a for a in args if a != "--bp"]
    if len(args) > 0: kw["dt"] = args[0]
    if len(args) > 1: kw["seq"] = args[1]
    if len(args) > 2: kw["budget_s"] = float(args[2])
    if len(args) > 3: kw["max_width"] = float(args[3])
    run(path, **kw)
