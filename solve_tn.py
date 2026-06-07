#!/usr/bin/env python3
"""
Tensor-network peak finder for QMill peaked circuits — the *general* method.

Idea (why this generalizes and is cheap):
  The obfuscated circuit C is, as a unitary, equal to a product of single-qubit
  RX rotations sandwiched between U U^dagger identity pairs. So C|0> is (up to a
  few non-unitary edits + deliberate peak-lowering) a PRODUCT state |s>. We never
  need the full 2^n statevector — we only need:
    (a) a good candidate s   (cheap, approximate), and
    (b) an EXACT certificate |<s|C|0>|^2  (one closed scalar contraction).
  Both are tensor-network contractions whose cost is set by the circuit's
  *treewidth*, not by 2^n or by intermediate MPS entanglement. Sparse circuits
  (even at 64+ qubits) have small treewidth → solved exactly on a laptop.

Pipeline:
  1. candidate via quimb perfect-sampling (contraction-based conditional marginals)
  2. exact amplitude of the candidate (certificate / peak probability)
  3. greedy single-bit hill-climb on the exact amplitude to snap to the local peak

Bit conventions: quimb indexes qubit i by register order. We print
  q[0..n-1]  (per-qubit, quimb/native order)  and
  q[n-1..0]  (Qiskit big-endian string — the submission convention).
"""
import sys, time, json, os, collections, math
import numpy as np
import quimb as qu
import quimb.tensor as qtn


def load(path):
    return qtn.Circuit.from_openqasm2_file(path)


# 'greedy' is deterministic, single-process, and fast — avoids cotengra's
# HyperOptimizer spawning idle worker pools (which hangs on this machine).
OPT = 'greedy'
SIMP = 'ADCRS'          # tensor-network simplifications before contraction


def amp(circ, bits, optimize=OPT):
    """Exact amplitude <bits|C|0>, bits a string in qubit order q[0..n-1]."""
    return complex(circ.amplitude(bits, optimize=optimize, simplify_sequence=SIMP,
                                  simplify_equalize_norms=False))


def width_estimate(circ, optimize=OPT):
    """(W, log10flops) for one amplitude contraction, where
    W = log2(size of largest intermediate tensor)  => peak mem ~ 16 * 2^W bytes
    (complex128); W<=28 ~ 4GB.  log10flops = log10(total contraction cost).
    Big-int safe: huge circuits make max_size()/contraction_cost() return Python
    arbitrary-precision ints that overflow float64 and break numpy ufuncs, so we
    log them directly with math.log2/log10 (which handle big ints)."""
    n = circ.N
    info = circ.amplitude_rehearse('0' * n, optimize=optimize, simplify_sequence=SIMP,
                                   simplify_equalize_norms=False)
    tree = info['tree']
    W = math.log2(tree.max_size())
    cost = tree.contraction_cost()
    log10flops = math.log10(cost) if cost > 0 else 0.0
    return float(W), float(log10flops)


def sample_candidate(circ, shots, group_size=8, seed=0, optimize=OPT):
    """Return (mode_bits_str_q0..n-1, count, n_distinct, all_counter)."""
    ctr = collections.Counter()
    for cfg in circ.sample(shots, group_size=group_size, seed=seed,
                           optimize=optimize, simplify_equalize_norms=False):
        ctr["".join(str(int(x)) for x in cfg)] += 1
    mode, cnt = ctr.most_common(1)[0]
    return mode, cnt, len(ctr), ctr


_Z = qu.pauli('Z')


def marginal_candidate(circ, optimize=OPT):
    """Deterministic candidate from per-qubit magnetization <Z_i>.
    Z|0>=+|0>, Z|1>=-|1>, so <Z_i> > 0  =>  qubit i is more likely 0.
    One closed contraction per qubit (lightcone-restricted, cheap when the
    circuit's treewidth is small). Returns bits string in q[0..n-1] order."""
    bits = []
    mags = []
    for i in range(circ.N):
        z = complex(circ.local_expectation(_Z, (i,), optimize=optimize,
                    simplify_sequence=SIMP, simplify_equalize_norms=False)).real
        mags.append(z)
        bits.append('0' if z >= 0 else '1')
    return "".join(bits), mags


def hillclimb(circ, bits, optimize=OPT, max_passes=6, verbose=True):
    """Greedy single-bit-flip ascent on exact |amp|^2. Returns (bits, prob)."""
    n = len(bits)
    cur = list(bits)
    best_p = abs(amp(circ, "".join(cur), optimize)) ** 2
    for p in range(max_passes):
        improved = False
        for i in range(n):
            trial = cur.copy()
            trial[i] = '1' if cur[i] == '0' else '0'
            pp = abs(amp(circ, "".join(trial), optimize)) ** 2
            if pp > best_p * (1 + 1e-9):
                cur, best_p = trial, pp
                improved = True
                if verbose:
                    print(f"    flip q[{i}] -> prob {best_p:.6e}")
        if not improved:
            break
    return "".join(cur), best_p


def fmt(bits_q0):
    per_qubit = bits_q0            # q[0..n-1]
    qiskit = bits_q0[::-1]         # q[n-1..0]
    return qiskit, per_qubit


def solve(path, do_hillclimb=True, optimize=OPT, wmax=31.0):
    name = os.path.basename(path)
    print(f"\n=== {name} ===", flush=True)
    t0 = time.time()
    circ = load(path)
    n = circ.N
    print(f"  loaded: {n} qubits, {circ.num_gates} gates  ({time.time()-t0:.1f}s)", flush=True)
    W = None
    try:
        W, log10flops = width_estimate(circ, optimize)
        mem_gb = (16 * 2**W / 1e9) if W < 80 else float('inf')
        print(f"  contraction width W={W:.1f} (mem~{mem_gb:.2f} GB), "
              f"log10(flops)~{log10flops:.1f}", flush=True)
        if W > wmax:
            print(f"  !! width too large (W={W:.1f} > {wmax}); needs better path / LUMI", flush=True)
            return {"name": name, "n": n, "status": "too_wide", "W": W}
    except Exception as e:
        print(f"  width estimate failed: {type(e).__name__}: {e}", flush=True)

    t1 = time.time()
    cand, mags = marginal_candidate(circ, optimize=optimize)
    p_cand = abs(amp(circ, cand, optimize)) ** 2
    qk, pq = fmt(cand)
    weakest = min(abs(m) for m in mags)
    print(f"  marginal candidate q[n-1..0]={qk}  exact prob={p_cand:.6e}  "
          f"min|<Z>|={weakest:.3f}  ({time.time()-t1:.1f}s)", flush=True)

    final, pf = cand, p_cand
    if do_hillclimb:
        t2 = time.time()
        final, pf = hillclimb(circ, cand, optimize)
        if final != cand:
            print(f"  hill-climb moved -> prob={pf:.6e} ({time.time()-t2:.1f}s)", flush=True)
    qk, pq = fmt(final)
    print(f"  --> SECRET q[n-1..0]={qk}")
    print(f"             q[0..n-1]={pq}")
    print(f"      exact peak prob |<s|C|0>|^2 = {pf:.6e}   (t_total={time.time()-t0:.1f}s)", flush=True)
    return {"name": name, "n": n, "status": "solved", "secret_qiskit": qk,
            "secret_perqubit": pq, "peak": pf, "W": W}


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        args = ["circuits/challenge-8_1.qasm", "circuits/challenge-8_11.qasm",
                "circuits/challenge-8_27.qasm"]
    out = {}
    for p in args:
        try:
            out[os.path.basename(p)] = solve(p)
        except Exception as e:
            print(f"  FAILED {type(e).__name__}: {e}")
            out[os.path.basename(p)] = {"name": os.path.basename(p), "error": str(e)}
    json.dump(out, open("results_tn.json", "w"), indent=2)
    print("\nsaved results_tn.json")
