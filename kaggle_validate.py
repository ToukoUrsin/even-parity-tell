#!/usr/bin/env python3
"""Stage 0 validation: run unswap_solve.py at DEFAULT (accurate) config on the known
48_8 circuit and verify the answer matches.

Expected qiskit secret (q[n-1..0]):
  <redacted-until-deadline>

This script runs unswap, then checks ALL 4 convention variants (perm/perm_rev/inv/inv_rev)
AND their bit-reversals against the known answer. PASS = exact match found in any variant.
Hamming distance of best match printed for diagnostics.

Usage: kaggle_validate.py
"""
import sys, os, json, subprocess, time

KNOWN_48_8 = "<redacted-until-deadline>"  # q[n-1..0]


def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


def run_unswap():
    """Run unswap_solve.py at DEFAULT config on 48_8, return parsed JSON record."""
    t0 = time.time()
    # default config: max_bond=8192 cutoff=0.002 shots=1000 seed=123 max_its=20
    cmd = [
        sys.executable, "unswap_solve.py", "circuits/challenge-48_8.qasm",
        "8192", "0.002", "1000", "123", "20",
    ]
    print(f"[stage0] running: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    out = proc.stdout + proc.stderr
    print(out[-3000:], flush=True)
    for line in out.splitlines():
        if line.startswith("UNSWAP:"):
            rec = json.loads(line[len("UNSWAP:"):])
            rec["_wall"] = round(time.time() - t0, 1)
            return rec
    raise RuntimeError(f"no UNSWAP: line in output (exit {proc.returncode})")


def judge(rec):
    """Check all 4 variants and their reversals against KNOWN_48_8.
    PASS = exact match anywhere; near-miss (hd<=2) is a strong signal too."""
    variants = {
        "secret_perm":     rec.get("secret_perm", ""),
        "secret_perm_rev": rec.get("secret_perm_rev", ""),
        "secret_inv":      rec.get("secret_inv", ""),
        "secret_inv_rev":  rec.get("secret_inv_rev", ""),
    }
    n = len(KNOWN_48_8)
    rows = []
    best = (n, None, None)
    for name, s in variants.items():
        if len(s) != n:
            rows.append((name, s, -1, -1)); continue
        hd = hamming(s, KNOWN_48_8)
        hdr = hamming(s[::-1], KNOWN_48_8)
        rows.append((name, s, hd, hdr))
        if hd < best[0]:  best = (hd, name, "as-is")
        if hdr < best[0]: best = (hdr, name, "reversed")
    print(f"\n[stage0] known: {KNOWN_48_8}")
    for name, s, hd, hdr in rows:
        marker = " <-- MATCH" if (hd == 0 or hdr == 0) else ""
        print(f"  {name:16s} = {s}  hd={hd:2d}  rev_hd={hdr:2d}{marker}")
    print(f"\n[stage0] peak_prob={rec.get('peak_prob')} n_distinct={rec.get('n_distinct')}/{rec.get('shots')} t={rec.get('t')}s wall={rec.get('_wall')}s")
    print(f"[stage0] best hamming distance: {best[0]} via {best[1]} ({best[2]})")
    return best[0]


if __name__ == "__main__":
    rec = run_unswap()
    best_hd = judge(rec)
    n = len(KNOWN_48_8)
    if best_hd == 0:
        print("\n[stage0] PASS — unswap+default config cracks the known 48_8. "
              "Method validated. Greenlight Stage 1 (rent a 4-GPU box for 42/43).")
        sys.exit(0)
    if best_hd <= 2:
        print(f"\n[stage0] NEAR-MISS (hd={best_hd}/{n}) — the method finds the right peak "
              "but truncation noise flipped a couple of bits. Likely fixable by raising "
              "shots or tightening cutoff. NOT a clean pass; investigate before spending.")
        sys.exit(2)
    print(f"\n[stage0] FAIL — best hd={best_hd}/{n} = essentially random. "
          "Tooling bug (wrong convention, broken gpu_patch, wrong cutoff path, etc.). "
          "DO NOT rent a box until this is fixed.")
    sys.exit(1)
