#!/usr/bin/env python3
"""MPS single-qubit-marginal peak solver for the obfuscated peaked circuits.

Why this is the right tool here (per the defect map + OBFUSCATION_RE):
  - The net unitary is ~ a product of RX layers (P1.P2) plus a HANDFUL of
    odd-parity CX "defect" pairs (42: 3 pairs / 6 qubits; 43: 2 pairs / 4 qubits).
    So the true entanglement is tiny: real MPS bond dim ~ 2^(defects on the cut),
    i.e. <= a few. The U.Ud bulk cancels *numerically* under SVD compression even
    though it never cancels symbolically (quimb ADCRS light-cones blow up to W~150).
  - The secret survives in single-qubit marginals <Z_i> (proven exact, 0 ambiguous
    qubits). So we do NOT need fidelity: build a low-bond CircuitMPS and read
    <Z_i> per qubit. bit_i = 1 if <Z_i> < 0 (RX(pi)|0> = |1>, <Z>=-1) else 0.

Robustness fixes baked in (this box, quimb 1.11.2, numba):
  - svd_fix: NaN/Inf-sanitizing scipy SVD so the build never dies on the
    numba "returned a result with an error set" SystemError.
  - gpu_patch: drops the stray swap_back kwarg quimb pipes into the SVD.
  - periodic state renormalization to stop the matmul overflow blowing up tensors.

Decode: read <Z_i> two ways and report both; majority sampling as a cross-check.

Usage: mps_marginal.py <qasm> [max_bond=64] [cutoff=1e-7] [shots=400] [dt=128|64]
"""
import sys, os, time, json
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import gpu_patch; gpu_patch.apply()
import svd_fix; svd_fix.apply()

import quimb as qu
import quimb.tensor as qt
from qiskit import QuantumCircuit
from qiskit_quimb import quimb_circuit

_DT = {"128": np.complex128, "64": np.complex64}


def _site_rho(psi, i):
    """Exact 1-site reduced density matrix rho_i (2x2) of an MPS.

    Canonicalize the orthogonality centre onto site i, then the single on-site
    tensor T (legs: left-bond, phys, right-bond) contracted with its conjugate
    over both bond legs gives rho_i directly (the rest of the chain is the
    identity in the mixed-canonical gauge). Robust across quimb versions because
    it only uses copy/canonicalize and a raw einsum on the site array.
    """
    p = psi.copy()
    p.canonicalize_(i)            # put orthogonality centre at site i
    t = p[i]
    phys_ind = p.site_ind(i)
    # order legs as (phys, *bonds) so we can einsum over the bond legs
    bond_inds = [ix for ix in t.inds if ix != phys_ind]
    arr = t.transpose(phys_ind, *bond_inds).data
    arr = np.asarray(arr)
    d = arr.shape[0]
    M = arr.reshape(d, -1)        # (phys, combined-bond)
    rho = M @ M.conj().T          # (phys, phys)
    return rho


def solve(path, max_bond=64, cutoff=1e-7, shots=400, seed=1234, dt="128",
          consolidate=False):  # quimb_circuit rejects 'unitary' gates -> keep off
    dtype = _DT[str(dt)]
    def to_backend(x):
        return np.asarray(x, dtype=dtype)
    name = os.path.basename(path)
    t0 = time.time()
    circuit = QuantumCircuit.from_qasm_file(path)
    n = circuit.num_qubits
    raw_ops = dict(circuit.count_ops())
    if consolidate:
        # Merge runs of 1q/2q gates into dense 2q "unitary" blocks. This is an
        # exact rewrite (same unitary) that collapses thousands of rx/rz/cx into
        # a few hundred 2-qubit gate applications -> far fewer MPS swap/SVD ops,
        # so the swap-heavy build is feasible on the dense targets.
        from qiskit.transpiler import PassManager
        from qiskit.transpiler.passes import Collect2qBlocks, ConsolidateBlocks
        pm = PassManager([Collect2qBlocks(),
                          ConsolidateBlocks(force_consolidate=True)])
        circuit = pm.run(circuit)
    print(f"=== {name} n={n} raw_ops={raw_ops} "
          f"consolidated_ops={dict(circuit.count_ops()) if consolidate else 'n/a'} "
          f"max_bond={max_bond} cutoff={cutoff} dt={dt} ===", flush=True)

    qc = quimb_circuit(
        circuit,
        quimb_circuit_class=qt.CircuitMPS,
        to_backend=to_backend,
        max_bond=max_bond,
        cutoff=cutoff,
        progbar=False,
    )
    psi = qc.psi
    try:
        psi.normalize()
    except Exception:
        pass
    t_build = time.time() - t0
    try:
        mb_reached = int(psi.max_bond())
    except Exception:
        mb_reached = None
    print(f"  built in {t_build:.1f}s  max_bond_reached={mb_reached}", flush=True)

    # Per-qubit <Z_i> from the MPS via the 1-site reduced density matrix.
    # Gauge-canonicalize at site i, contract the on-site tensor with its conjugate
    # over the bond legs -> 2x2 rho_i; <Z>=rho00-rho11, P(1)=rho11. This is the
    # exact closed-form marginal (the OBFUSCATION_RE decoder) and avoids the
    # version-specific local_expectation signature.
    zexp = [None] * n
    p1 = [None] * n
    bits_q0 = ["?"] * n
    for i in range(n):
        try:
            rho = _site_rho(psi, i)
            r11 = float(np.real(rho[1, 1]))
            r00 = float(np.real(rho[0, 0]))
            tr = r00 + r11
            if tr > 0:
                r11 /= tr; r00 /= tr
            z = r00 - r11
        except Exception:
            z, r11 = 0.0, 0.5
        zexp[i] = round(z, 5)
        p1[i] = round(r11, 5)
        bits_q0[i] = "1" if r11 > 0.5 else "0"
    voted_q0 = "".join(bits_q0)
    qiskit_str = voted_q0[::-1]
    margins = [abs(z) for z in zexp if z is not None]
    weak = [i for i in range(n) if abs(zexp[i]) < 0.2]

    # Cross-check: majority-vote sampling decode (independently validated on 48_8).
    samp_q0 = None
    try:
        samples = list(qc.sample(shots, seed=seed))
        bp = np.array([[int(s) for s in ss] for ss in samples]).mean(axis=0)
        samp_q0 = "".join(str(b) for b in (bp > 0.5).astype(int).tolist())
    except Exception as e:
        samp_q0 = f"SAMPLE_FAIL:{type(e).__name__}"

    agree = (samp_q0 == voted_q0) if isinstance(samp_q0, str) and len(samp_q0) == n else None
    rec = dict(name=name, n=n, max_bond=max_bond, max_bond_reached=mb_reached,
               cutoff=cutoff, dt=str(dt),
               secret_q0=voted_q0, secret_qiskit=qiskit_str,
               sample_q0=samp_q0, sample_qiskit=(samp_q0[::-1] if isinstance(samp_q0, str) and len(samp_q0) == n else None),
               z_vs_sample_agree=agree, zexp=zexp,
               min_margin=round(min(margins), 4) if margins else 0.0,
               mean_margin=round(float(np.mean(margins)), 4) if margins else 0.0,
               weak_qubits=weak, n_weak=len(weak),
               t_build=round(t_build, 1), t_total=round(time.time() - t0, 1))
    print("MPSZ:" + json.dumps(rec), flush=True)
    return rec


if __name__ == "__main__":
    path = sys.argv[1]
    kw = {}
    if len(sys.argv) > 2: kw["max_bond"] = int(sys.argv[2])
    if len(sys.argv) > 3: kw["cutoff"] = float(sys.argv[3])
    if len(sys.argv) > 4: kw["shots"] = int(sys.argv[4])
    if len(sys.argv) > 5: kw["dt"] = sys.argv[5]
    solve(path, **kw)
