#!/usr/bin/env python3
"""
VECTORIZED numpy reimplementation of the Pauli-backpropagation peak finder.

This is a drop-in faster twin of pauli_crack.py. Same algorithm (Heisenberg
backprop of Z_i through the circuit in the Weyl basis W(x,z) = X^x Z^z), but the
operator is stored as THREE parallel numpy arrays (x int64, z int64, coeff
complex128) and every gate update plus the dedup/prune is fully vectorized. This
lets us run with a term cap of 1,000,000+ on dense 56-qubit circuits.

The Pauli / Weyl update rules are copied verbatim from pauli_crack.backprop_marginal,
so the <Z_i> signs MUST match that reference.

  W(0,z) W(x',z') sign = (-1)^{z.x'};   <0| W(x,z) |0> = [x==0].

Per gate (REVERSED circuit order):
  CX(a->b): if x_a: x ^= 1<<b ;  if z_b: z ^= 1<<a
  rz(q):    x_q==1 anticommutes -> keep c*cos in place; partner z^=1<<q, c*(-i sin)
  rx(q):    z_q==1 anticommutes -> partner x^=1<<q, c*(+i sin)
  ry(q):    x_q xor z_q anticommutes -> partner x^=1<<q,z^=1<<q, c*(sign*sin),
            sign = -1 if x_q else +1
  swap(a,b): swap bits a,b in both x and z

After each gate: combine duplicate (x,z) keys (sum coeffs), prune |coeff|<=thresh,
cap to top-`max_terms` by |coeff|. Gates whose qubits are outside the current
support (union of x|z) are skipped.

Final:  <Z_i> = sum(coeff[x==0]).real ;  bit_i = 0 if <Z_i> > 0 else 1.
"""
import sys, re, time, math, argparse
import numpy as np

# reuse the reference parser / angle eval verbatim
from pauli_crack import parse_qasm, eval_angle


def _dedup(x, z, coeff):
    """Combine duplicate (x,z) keys, summing their coeffs. Vectorized.

    Two paths, both exact (n<=63 -> x,z are non-negative int64):
      * FAST: if the (x,z) pair packs into a single 64-bit key
        (x_bits + z_bits <= 64), build one uint64 key and do a single argsort.
        Sorting one uint64 array is markedly faster than lexsort over two int64s.
      * GENERAL: lexsort over (x, z). Used when packing would overflow 64 bits
        (e.g. very wide support on n>32).
    Then run-length group the sorted keys and segment-sum coeffs via reduceat.
    """
    N = x.shape[0]
    if N <= 1:
        return x, z, coeff

    # bits actually needed by each mask right now
    xmax = int(x.max())
    zmax = int(z.max())
    zbits = zmax.bit_length()          # number of low bits z occupies
    xbits = xmax.bit_length()

    if xbits + zbits <= 64:
        # pack into a single uint64 key: (x << zbits) | z
        key = (x.astype(np.uint64) << np.uint64(zbits)) | z.astype(np.uint64)
        order = np.argsort(key)  # order within equal keys is irrelevant to the sum
        ks = key[order]
        cs = coeff[order]
        new_grp = np.empty(N, dtype=bool)
        new_grp[0] = True
        np.not_equal(ks[1:], ks[:-1], out=new_grp[1:])
        starts = np.flatnonzero(new_grp)
        uk = ks[starts]
        out_c = np.add.reduceat(cs, starts)
        # unpack key back into x, z
        out_z = (uk & ((np.uint64(1) << np.uint64(zbits)) - np.uint64(1))).astype(np.int64)
        out_x = (uk >> np.uint64(zbits)).astype(np.int64)
        return out_x, out_z, out_c

    # general path: lexsort over (x, z) (last key is primary)
    order = np.lexsort((z, x))
    xs = x[order]
    zs = z[order]
    cs = coeff[order]
    new_grp = np.empty(N, dtype=bool)
    new_grp[0] = True
    np.not_equal(xs[1:], xs[:-1], out=new_grp[1:])
    new_grp[1:] |= (zs[1:] != zs[:-1])
    starts = np.flatnonzero(new_grp)
    out_x = xs[starts]
    out_z = zs[starts]
    out_c = np.add.reduceat(cs, starts)
    return out_x, out_z, out_c


def _swap_bits_vec(v, a, b):
    """Swap bits a and b in every element of int64 array v (vectorized)."""
    ba = (v >> a) & 1
    bb = (v >> b) & 1
    diff = ba ^ bb
    # where the two bits differ, flip both
    mask = diff << a | diff << b
    return v ^ mask


def backprop_marginal(gates, i, max_terms=200000, thresh=1e-9):
    """Compute <Z_i> = <0|C^dag Z_i C|0> by vectorized Heisenberg backprop.
    Returns (expectation, n_terms_peak)."""
    bit = np.int64(1) << np.int64(i)
    x = np.array([0], dtype=np.int64)
    z = np.array([bit], dtype=np.int64)
    coeff = np.array([1.0 + 0j], dtype=np.complex128)
    support = int(bit)
    peak = 1

    for kind, qs, ang in reversed(gates):
        if kind == 'cx':
            a, b = qs
            ma = np.int64(1) << np.int64(a)
            mb = np.int64(1) << np.int64(b)
            if not (support & (int(ma) | int(mb))):
                continue
            # X_a -> X_a X_b : flip bit b of x where x has bit a
            xa = (x & ma) != 0
            x = np.where(xa, x ^ mb, x)
            # Z_b -> Z_a Z_b : flip bit a of z where z has bit b
            zb = (z & mb) != 0
            z = np.where(zb, z ^ ma, z)
            # CX is a bijection on (x,z) keys: no new collisions, coeff magnitudes
            # unchanged -> dedup/prune/cap are all no-ops. Just refresh support.
            support = int(np.bitwise_or.reduce(x | z)) if x.shape[0] else 0
            continue
        elif kind == 'swap':
            a, b = qs
            ma = 1 << a
            mb = 1 << b
            if not (support & (ma | mb)):
                continue
            x = _swap_bits_vec(x, a, b)
            z = _swap_bits_vec(z, a, b)
            # swap is also a bijection on keys with unchanged coeffs.
            support = int(np.bitwise_or.reduce(x | z)) if x.shape[0] else 0
            continue
        else:  # rotations rx/rz/ry, single qubit
            (q,) = qs
            mq = np.int64(1) << np.int64(q)
            if not (support & int(mq)):
                continue
            ct = math.cos(ang)
            st = math.sin(ang)
            if kind == 'rz':                 # anticommute iff x_q == 1
                anti = (x & mq) != 0
                # in-place coeff for anticommuting terms gets *ct
                new_coeff = np.where(anti, coeff * ct, coeff)
                # partners: only for anticommuting terms
                px = x[anti]
                pz = z[anti] ^ mq
                pc = coeff[anti] * (-1j * st)
                x = np.concatenate((x, px))
                z = np.concatenate((z, pz))
                coeff = np.concatenate((new_coeff, pc))
            elif kind == 'rx':               # anticommute iff z_q == 1
                anti = (z & mq) != 0
                new_coeff = np.where(anti, coeff * ct, coeff)
                px = x[anti] ^ mq
                pz = z[anti]
                pc = coeff[anti] * (1j * st)
                x = np.concatenate((x, px))
                z = np.concatenate((z, pz))
                coeff = np.concatenate((new_coeff, pc))
            else:  # ry : anticommute iff x_q xor z_q
                xq = (x & mq) != 0
                zq = (z & mq) != 0
                anti = xq ^ zq
                new_coeff = np.where(anti, coeff * ct, coeff)
                # partner sign = -1 if x_q else +1  (per reference)
                sign = np.where(xq[anti], -1.0, 1.0)
                px = x[anti] ^ mq
                pz = z[anti] ^ mq
                pc = coeff[anti] * (sign * st)
                x = np.concatenate((x, px))
                z = np.concatenate((z, pz))
                coeff = np.concatenate((new_coeff, pc))

        # combine duplicate (x,z) keys
        x, z, coeff = _dedup(x, z, coeff)

        # prune low-weight terms
        if thresh:
            mag = np.abs(coeff)
            keep = mag > thresh
            if not keep.all():
                x = x[keep]
                z = z[keep]
                coeff = coeff[keep]

        # cap to top-max_terms by |coeff|
        if coeff.shape[0] > max_terms:
            mag = np.abs(coeff)
            # indices of the max_terms largest magnitudes
            idx = np.argpartition(mag, coeff.shape[0] - max_terms)[-max_terms:]
            x = x[idx]
            z = z[idx]
            coeff = coeff[idx]

        # recompute support = union of x|z
        if coeff.shape[0]:
            support = int(np.bitwise_or.reduce(x | z))
        else:
            support = 0
        peak = max(peak, coeff.shape[0])

    # <Z_i> = sum of coeffs of terms with x == 0
    exp = float(coeff[x == 0].sum().real)
    return exp, peak


_G = None  # module-level gate list for multiprocessing workers


def _worker(args):
    i, max_terms, thresh = args
    e, peak = backprop_marginal(_G, i, max_terms=max_terms, thresh=thresh)
    return i, e, peak


def _init_pool(gates):
    global _G
    _G = gates


def solve(path, qubits=None, max_terms=200000, thresh=1e-9, verbose=True, procs=1):
    n, gates = parse_qasm(path)
    assert n <= 63, f"n={n} > 63 does not fit in int64 bitmask; this fast kernel only supports n<=63"
    t0 = time.time()
    qs = list(range(n)) if qubits is None else list(qubits)
    bits = [None] * n
    exps = [None] * n
    if procs > 1:
        import multiprocessing as mp
        with mp.Pool(procs, initializer=_init_pool, initargs=(gates,)) as pool:
            for i, e, peak in pool.imap_unordered(
                    _worker, [(i, max_terms, thresh) for i in qs]):
                bits[i] = 0 if e > 0 else 1
                exps[i] = e
                if verbose:
                    print(f"  q[{i:3d}] <Z>={e:+.4f} -> {bits[i]}  (peak terms={peak})", flush=True)
    else:
        for i in qs:
            e, peak = backprop_marginal(gates, i, max_terms=max_terms, thresh=thresh)
            bits[i] = 0 if e > 0 else 1
            exps[i] = e
            if verbose:
                print(f"  q[{i:3d}] <Z>={e:+.4f} -> {bits[i]}  (peak terms={peak})", flush=True)
    perq = "".join(str(b) for b in bits)
    qiskit = perq[::-1]
    conf = float(np.min([abs(e) for e in exps])) if all(e is not None for e in exps) else None
    print(f"\n  file={path}  n={n}  gates={len(gates)}  time={time.time()-t0:.1f}s")
    print(f"  SECRET q[n-1..0]={qiskit}")
    print(f"         q[0..n-1]={perq}")
    print(f"  min|<Z>| (weakest bit confidence) = {conf:.4f}")
    import json, os
    with open("pauli_results.jsonl", "a") as f:
        f.write(json.dumps(dict(file=os.path.basename(path), n=n, secret_qiskit=qiskit,
                secret_perqubit=perq, min_absZ=conf, time=round(time.time()-t0,1))) + "\n")
    return qiskit, perq, exps


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("circuit")
    ap.add_argument("--max-terms", type=int, default=200000)
    ap.add_argument("--thresh", type=float, default=1e-9)
    ap.add_argument("--procs", type=int, default=1)
    args = ap.parse_args()
    solve(args.circuit, max_terms=args.max_terms, thresh=args.thresh, procs=args.procs)
