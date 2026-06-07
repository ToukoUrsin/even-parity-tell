#!/usr/bin/env python3
"""Per-qubit <Z_i> via quimb full-TN Circuit.local_expectation with automatic
reverse-light-cone restriction + aggressive simplification (rank-1/diagonal),
which CANCELS the U.Ud even-parity bulk exactly. bit_i = 0 if <Z_i> > 0 else 1
(Z|0>=+|0>, RX(pi)|0>=|1> with <Z>=-1).

For the 42 product qubits the cone collapses after simplify -> exact, cheap.
For the 3 odd pairs the cone is a small 2-qubit subsystem -> still exact.

Width-gate each contraction: skip (mark weak) any qubit whose simplified cone
contraction width exceeds --max-width (default 28) to stay RAM-safe.

Usage: lightcone_z.py <qasm> [max_width=28] [dtype=128|64]
"""
import sys, time, json
import numpy as np
import gpu_patch; gpu_patch.apply()
from qiskit import QuantumCircuit
import quimb as qu
import quimb.tensor as qt
from qiskit_quimb import quimb_circuit


def run(path, max_width=28, dt="128"):
    dtype = np.complex128 if str(dt) == "128" else np.complex64
    def to_backend(x):
        return np.asarray(x, dtype=dtype)
    name = path.split("/")[-1]
    t0 = time.time()
    circ = QuantumCircuit.from_qasm_file(path)
    n = circ.num_qubits
    # Full TN circuit (lazy) -- no MPS, no gate-by-gate bond growth.
    qc = quimb_circuit(circ, quimb_circuit_class=qt.Circuit, to_backend=to_backend)
    print(f"=== {name} n={n} (full-TN local_expectation, max_width={max_width}, dt={dt}) ===",
          flush=True)

    Z = qu.pauli("Z").astype(dtype)
    zexp = [None] * n
    weak = []
    widths = [None] * n
    bits_q0 = ["?"] * n
    import math
    for i in range(n):
        ti = time.time()
        try:
            # Rehearse first: get the contraction tree WITHOUT contracting, read
            # its width. Width-gate so we never blow up RAM on a wide cone.
            info = qc.local_expectation(
                Z, (i,),
                simplify_sequence="ADCRS",
                optimize="greedy",
                rehearse=True,
            )
            # rehearse dict has 'W' = log2 width of largest intermediate tensor.
            w = float(info["W"])
            widths[i] = round(w, 1)
            if w is not None and w > max_width:
                print(f"  q{i:2d}: WIDE cone width={w:.1f} > {max_width} -> skip (weak) "
                      f"({time.time()-ti:.2f}s)", flush=True)
                weak.append(i)
                bits_q0[i] = "?"
                continue
            val = qc.local_expectation(
                Z, (i,),
                simplify_sequence="ADCRS",
                optimize="greedy",
                rehearse=False,
            )
            v = float(np.real(val))
            zexp[i] = round(v, 5)
            bits_q0[i] = "0" if v > 0 else "1"
            mark = "" if abs(v) > 0.1 else "  <-WEAK"
            wtxt = f" w={widths[i]}" if widths[i] is not None else ""
            print(f"  q{i:2d}: <Z>={v:+.5f}  bit={bits_q0[i]}{wtxt}  ({time.time()-ti:.2f}s){mark}",
                  flush=True)
            if abs(v) < 0.1:
                weak.append(i)
        except Exception as e:
            print(f"  q{i:2d}: FAILED {type(e).__name__}: {str(e)[:80]}", flush=True)
            weak.append(i)
            bits_q0[i] = "?"

    voted_q0 = "".join(bits_q0)
    qiskit_str = voted_q0[::-1]
    margins = [abs(z) for z in zexp if z is not None]
    rec = dict(name=name, n=n, dtype=str(dt),
               secret_q0=voted_q0, secret_qiskit=qiskit_str,
               zexp=zexp,
               min_margin=round(min(margins), 4) if margins else 0.0,
               mean_margin=round(float(np.mean(margins)), 4) if margins else 0.0,
               weak_qubits=weak, n_weak=len(weak),
               t_total=round(time.time() - t0, 1))
    print("LIGHTZ:" + json.dumps(rec), flush=True)
    return rec


if __name__ == "__main__":
    path = sys.argv[1]
    mw = int(sys.argv[2]) if len(sys.argv) > 2 else 28
    dt = sys.argv[3] if len(sys.argv) > 3 else "128"
    run(path, max_width=mw, dt=dt)
