#!/usr/bin/env python3
"""
QMill peaked-circuit distiller — Kaggle/Colab GPU edition (SELF-VALIDATING).

Paste this whole file into ONE Kaggle notebook cell (GPU runtime: P100/T4/A100) and run.
It will:
  1. pip-install qiskit / quimb / qiskit-quimb.
  2. Find the challenge-*.qasm files you attached as a Dataset (under /kaggle/input).
  3. VALIDATE the pipeline on three known secrets (8_1, 48_8, 56_9). If any mismatch,
     it ABORTS — so you never trust a scrambled bitstring.
  4. Distill the three targets 56_43, 48_42, 56_38 at increasing bond dimension and
     print, per circuit: secret_qiskit (this is what you SUBMIT, q[n-1..0]), min_margin
     (per-bit decisiveness; near 1 = trust), and any weak/unresolved qubits.

WHY CircuitMPS not CircuitPermMPS: on the validated stack, CircuitPermMPS returns samples
in PERMUTED physical order (wrong string) and crashes on a swap_back kwarg. CircuitMPS keeps
logical qubit order and reproduces every known secret. We keep the swap_back shim defensively.

Decode = single-qubit-marginal majority vote: a peaked circuit cannot hide its <Z_i>, so
voting each bit over a few hundred samples recovers the secret even if the global MPS fidelity
is ~0 from truncation (the OBFUSCATION_RE insight).
"""
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "qiskit>=1.0", "quimb==1.11.2", "qiskit-quimb"], check=False)

import os, glob, json, time
import numpy as np

# ---- swap_back shim: quimb pipes a stray `swap_back` kwarg into the backend SVD on
# ---- long-range gates; strip it so CircuitMPS doesn't crash on dense circuits. ----
def _apply_swap_back_shim():
    try:
        import quimb.tensor.tensor_core as TC
        def _wrap(fn):
            if getattr(fn, "_sb", False): return fn
            def w(*a, **kw):
                kw.pop("swap_back", None)
                return fn(*a, **kw)
            w._sb = True
            return w
        if hasattr(TC, "tensor_split"):
            TC.tensor_split = _wrap(TC.tensor_split)
    except Exception as e:
        print("shim warn:", e)
_apply_swap_back_shim()

import torch
import quimb.tensor as qt
from qiskit import QuantumCircuit
from qiskit_quimb import quimb_circuit

CUDA = torch.cuda.is_available()
print("CUDA:", CUDA, "| device:", (torch.cuda.get_device_name(0) if CUDA else "CPU"))

# complex64 on GPU = ~2x faster/lighter; needs a looser cutoff or SVD noise -> NaN.
def make_backend(dtype, device):
    def to_backend(x):
        return torch.tensor(np.asarray(x), dtype=dtype, device=device)
    return to_backend

def distill(path, max_bond=128, shots=1000, seed=1234, dtype=None, cutoff=None):
    if dtype is None:
        dtype = torch.complex64 if CUDA else torch.complex128
    if cutoff is None:
        cutoff = 1e-6 if dtype == torch.complex64 else 1e-10
    device = "cuda" if CUDA else "cpu"
    name = os.path.basename(path)
    t0 = time.time()
    circ = QuantumCircuit.from_qasm_file(path)
    n = circ.num_qubits
    qc = quimb_circuit(circ, quimb_circuit_class=qt.CircuitMPS,
                       to_backend=make_backend(dtype, device),
                       max_bond=max_bond, cutoff=cutoff, progbar=False)
    t_build = time.time() - t0
    try:    reached = int(qc.psi.max_bond())
    except Exception: reached = None
    samples = list(qc.sample(shots, seed=seed))          # logical order q[0..n-1]
    bp = np.array([[int(s) for s in ss] for ss in samples]).mean(axis=0)
    q0 = "".join(str(i) for i in (bp > 0.5).astype(int).tolist())
    qiskit_str = q0[::-1]                                  # q[n-1..0] = SUBMIT THIS
    margins = np.abs(bp - 0.5) * 2.0
    weak = [int(i) for i in np.where(margins < 0.2)[0]]
    rec = dict(name=name, n=n, max_bond=max_bond, bond_reached=reached, shots=shots,
               dtype=str(dtype).split(".")[-1], secret_qiskit=qiskit_str,
               min_margin=round(float(margins.min()), 3),
               mean_margin=round(float(margins.mean()), 3),
               n_weak_bits=len(weak), weak_qubits=weak,
               t_total=round(time.time() - t0, 1))
    print("DISTILL:" + json.dumps(rec), flush=True)
    return rec

# ---- locate attached qasm files ----
def find(tag):
    hits = glob.glob(f"/kaggle/input/**/challenge-{tag}.qasm", recursive=True) \
         + glob.glob(f"/kaggle/working/**/challenge-{tag}.qasm", recursive=True) \
         + glob.glob(f"./**/challenge-{tag}.qasm", recursive=True)
    return hits[0] if hits else None

KNOWN = {
 "8_1":  "10101101",
 "48_8": "<redacted-until-deadline>",
 "56_9": "<redacted-until-deadline>",
}
print("\n===== STEP 1: VALIDATE pipeline on known secrets (must all PASS) =====")
ok_all = True
for tag, exp in KNOWN.items():
    p = find(tag)
    if not p:
        print(f"  MISSING challenge-{tag}.qasm — attach the dataset!"); ok_all = False; continue
    r = distill(p, max_bond=128, shots=800)
    match = r["secret_qiskit"] == exp
    ok_all &= match
    print(f"  challenge-{tag}: match={match}  min_margin={r['min_margin']}")
    if not match:
        print("    got", r["secret_qiskit"]); print("    exp", exp)

if not ok_all:
    print("\n*** VALIDATION FAILED — do NOT trust target results. Fix bit-order/dataset first. ***")
else:
    print("\nVALIDATION PASSED. Bit convention confirmed on 8q/48q/56q.")
    print("\n===== STEP 2: distill targets (43 -> 42 -> 38), escalating bond =====")
    for tag in ["56_43", "48_42", "56_38"]:
        p = find(tag)
        if not p:
            print(f"  MISSING challenge-{tag}.qasm"); continue
        print(f"\n--- challenge-{tag} ---")
        for mb in [128, 256, 512]:
            r = distill(p, max_bond=mb, shots=1000)
            print(f"  bond={mb}: SUBMIT secret_qiskit={r['secret_qiskit']}")
            print(f"           min_margin={r['min_margin']} weak_qubits={r['weak_qubits']} t={r['t_total']}s")
            if r["n_weak_bits"] == 0 and r["min_margin"] >= 0.3:
                print(f"  >>> challenge-{tag} CLEAN at bond {mb}: submit the string above (try its reverse if rejected).")
                break
