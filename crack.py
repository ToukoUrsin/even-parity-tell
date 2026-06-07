#!/usr/bin/env python3
"""
Unified cracker for QMill peaked circuits.

Method auto-selection per circuit:
  - n <= EXACT_MAX  -> exact statevector (qiskit). Certain answer.
  - otherwise       -> capped-MPS chi-sweep (quimb) with early-stop on convergence,
                       greedy max-marginal + sampling cross-check + surviving-norm.

Optionally verifies a candidate bitstring by an EXACT single-amplitude tensor-network
contraction |<s|C|0>|^2 (quimb Circuit + cotengra). This is truncation-free, so when
it succeeds it CERTIFIES the secret regardless of MPS bond cap. Skipped automatically
when the contraction width is too large.

Output: one JSON object per circuit to stdout (prefixed RESULT:) and appended to
results.jsonl. Designed to be run per-circuit in parallel.
"""
import sys, os, json, time, argparse
import numpy as np

EXACT_MAX = 28          # exact statevector qubit ceiling (2^28*16B ~ 4.3GB)


def count_gates(path):
    import re
    txt = open(path).read()
    m = re.search(r'q(?:reg|ubit)\s*\w*\s*\[?(\d+)\]?', txt)
    nq = None
    mm = re.search(r'qreg\s+\w+\[(\d+)\]', txt)
    if mm: nq = int(mm.group(1))
    cx = len(re.findall(r'(?m)^\s*c[xz]\b', txt))
    tot = len(re.findall(r'(?m)^\s*(rx|rz|ry|cx|cz|h|x|y|z|u|u3|swap|ccx|sx|s|t)\b', txt))
    return nq, cx, tot


# ---------- exact statevector ----------
def solve_exact(path, topk=4):
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    qc = QuantumCircuit.from_qasm_file(path)
    qc = qc.remove_final_measurements(inplace=False)
    n = qc.num_qubits
    sv = Statevector.from_instruction(qc)
    probs = np.abs(np.asarray(sv.data)) ** 2
    order = np.argsort(probs)[::-1]
    best = int(order[0])
    qiskit_str = format(best, f"0{n}b")          # q[n-1..0]
    per_qubit = qiskit_str[::-1]                  # q[0..n-1]
    p1 = float(probs[best]); p2 = float(probs[order[1]]) if len(order) > 1 else 0.0
    top = [(format(int(i), f"0{n}b"), float(probs[int(i)])) for i in order[:topk]]
    return dict(method="statevector", n=n, secret_qiskit=qiskit_str,
                secret_perqubit=per_qubit, peak_prob=p1, runnerup=p2,
                gap=p1 - p2, top=top, confident=True)


# ---------- capped MPS sweep ----------
def solve_mps_sweep(path, chis, cutoff=1e-12, shots=200, time_budget=None):
    import quimb.tensor as qtn
    from solve_mps import greedy_peak, sample_peak, fmt_secret
    t_start = time.time()
    history = []
    prev = None
    stable_count = 0
    last = None
    for mb in chis:
        if time_budget and (time.time() - t_start) > time_budget:
            history.append(dict(chi_cap=mb, skipped="time_budget"))
            break
        t0 = time.time()
        circ = qtn.CircuitMPS.from_openqasm2_file(
            path, gate_opts={"max_bond": mb, "cutoff": cutoff})
        psi = circ.psi
        chi = max(psi.bond_sizes()) if psi.L > 1 else 1
        norm = float(psi.norm())
        bits, prob = greedy_peak(psi)
        qk, pq = fmt_secret(bits)
        sbits, sprob, ndist = sample_peak(psi, shots=shots, seed=0)
        agree = (sbits == bits)
        rec = dict(chi_cap=mb, chi=chi, norm=norm, secret_qiskit=qk,
                   secret_perqubit=pq, greedy_prob=prob, sample_prob=sprob,
                   ndistinct=ndist, greedy_eq_sample=agree, t=time.time() - t0)
        history.append(rec)
        last = rec
        if prev is not None and qk == prev:
            stable_count += 1
        else:
            stable_count = 0
        prev = qk
        # early stop: secret stable across 2 chis AND greedy==sampling AND chi not saturating cap
        if stable_count >= 1 and agree and chi < mb:
            break
    confident = bool(last and last["greedy_eq_sample"] and stable_count >= 1
                     and last["norm"] > 1e-3 and last["chi"] < last["chi_cap"])
    return dict(method="mps_sweep", n=last["chi"] and None, secret_qiskit=last["secret_qiskit"],
                secret_perqubit=last["secret_perqubit"], peak_prob=last["greedy_prob"],
                sample_prob=last["sample_prob"], norm=last["norm"], chi=last["chi"],
                history=history, confident=confident)


# ---------- exact single-amplitude verify ----------
def verify_amplitude(path, perqubit_bits, max_width=26, optimize="auto-hq"):
    """Exact |<s|C|0>|^2 via single-amplitude TN contraction, gated on memory.

    Before contracting we 'rehearse' the contraction (find a path WITHOUT executing
    it) and read its width W = log2(largest intermediate tensor). complex128 is 16 B,
    so peak memory ~ 2^W * 16 bytes (W=28 -> ~4 GB). If W > max_width we SKIP rather
    than risk an OOM that takes down the machine. Returns a dict; never raises.
    """
    import quimb.tensor as qtn
    try:
        circ = qtn.Circuit.from_openqasm2_file(path)
    except Exception as e:
        return dict(skipped=f"load:{e}")
    # bitstring for amplitude: quimb wants string over qubits 0..n-1
    s = "".join(str(b) for b in perqubit_bits)
    # 1) cheap rehearsal: find a contraction path, read its width, do NOT contract yet
    try:
        reh = circ.amplitude_rehearse(b=s, optimize=optimize)
        if isinstance(reh, dict) and reh.get("W") is not None:
            width = float(reh["W"])
        else:
            width = float(reh["tree"].contraction_width())
    except Exception as e:
        return dict(skipped=f"rehearse:{e}")
    if width > max_width:
        return dict(skipped="width", width=width, max_width=max_width)
    # 2) width is bounded -> safe to contract exactly
    try:
        amp = circ.amplitude(s, optimize=optimize)
        return dict(amplitude_abs2=float(abs(amp) ** 2), width=width)
    except Exception as e:
        return dict(skipped=f"contract:{e}", width=width)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("circuit")
    ap.add_argument("--chis", type=int, nargs="*", default=[16, 32, 64, 128, 256])
    ap.add_argument("--verify", action="store_true", help="exact single-amplitude verify")
    ap.add_argument("--time-budget", type=float, default=None)
    ap.add_argument("--out", default="results.jsonl")
    args = ap.parse_args()

    path = args.circuit
    nq, cx, tot = count_gates(path)
    t0 = time.time()
    res = dict(file=os.path.basename(path), n=nq, cx=cx, total_gates=tot)
    try:
        if nq is not None and nq <= EXACT_MAX:
            r = solve_exact(path)
        else:
            r = solve_mps_sweep(path, args.chis, time_budget=args.time_budget)
        res.update(r)
        res["n"] = nq
        if args.verify:
            res["verify"] = verify_amplitude(path, [int(c) for c in res["secret_perqubit"]])
    except Exception as e:
        import traceback
        res["error"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()[-800:]
    res["wall_s"] = round(time.time() - t0, 1)
    line = json.dumps(res)
    print("RESULT:" + line, flush=True)
    with open(args.out, "a") as f:
        f.write(line + "\n")


if __name__ == "__main__":
    main()
