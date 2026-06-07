#!/usr/bin/env python3
"""Bank all certified secrets that aren't yet submitted+accepted on the platform.

Reads {problem_id: secret} straight from TRACKER.md (no re-typing), fetches the
current accepted set, and submits only the missing ones via verify-samples. If a
submission is rejected, retries with the reversed bit-order (endianness fallback).

Dry-run by default (prints the plan). Pass --go to actually submit.
"""
import argparse, json, re, sys, time, urllib.request, urllib.error
import qmill_auth

BASE = "https://qas.qmill.com/api/v1/challenges/junction-quantum-hack"
BOUNDARY = "----submitBoundary7MA4YWxkTrZu0gW"
DESC = ("Recovered the peak bitstring via approximate MPS peak-finding (greedy "
        "max-marginal decoding), then certified it with an exact single-amplitude "
        "tensor-network contraction |<s|C|0>|^2 (>0.5 proves the unique peak); "
        "smallest circuits cross-checked against exact statevector simulation.")


def parse_tracker(path="TRACKER.md"):
    """Extract {problem_id: secret} from lines mentioning challenge-<id> + a backticked binary string."""
    secrets = {}
    for line in open(path):
        m_id = re.search(r"challenge-(\d+_\d+)", line)
        m_bits = re.search(r"`([01]{8,})`", line)
        if m_id and m_bits:
            secrets[m_id.group(1)] = m_bits.group(1)
    return secrets


def token():
    return qmill_auth.get_access_token()


def accepted_set():
    req = urllib.request.Request(f"{BASE}/submissions?limit=50",
                                 headers={"Authorization": f"Bearer {token()}"})
    data = json.load(urllib.request.urlopen(req))
    return {s["problem_id"] for s in data.get("submissions", []) if s["accepted_as_quantum"]}


def submit_one(problem, bitstring):
    body = b"".join([
        f"--{BOUNDARY}\r\n".encode(),
        f'Content-Disposition: form-data; name="samples_file"; filename="bitstring-{bitstring}.json"\r\n'.encode(),
        b"Content-Type: application/json\r\n\r\n",
        json.dumps([bitstring]).encode(), b"\r\n",
        f"--{BOUNDARY}\r\n".encode(),
        b'Content-Disposition: form-data; name="description"\r\n\r\n',
        DESC.encode(), b"\r\n",
        f"--{BOUNDARY}--\r\n".encode(),
    ])
    req = urllib.request.Request(
        f"{BASE}/problems/{problem}/verify-samples", data=body, method="POST",
        headers={"Authorization": f"Bearer {token()}",
                 "Content-Type": f"multipart/form-data; boundary={BOUNDARY}"})
    resp = json.load(urllib.request.urlopen(req))
    sub_id = resp.get("submission_id")
    for _ in range(30):
        time.sleep(1.2)
        g = urllib.request.Request(f"{BASE}/submissions/{sub_id}",
                                   headers={"Authorization": f"Bearer {token()}"})
        s = json.load(urllib.request.urlopen(g))
        if s.get("status") == "verified" or s.get("completed_at"):
            return bool(s.get("accepted_as_quantum")), s.get("xeb_score")
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true", help="actually submit (default: dry run)")
    args = ap.parse_args()

    secrets = parse_tracker()
    done = accepted_set()
    todo = {pid: bits for pid, bits in secrets.items() if pid not in done}

    print(f"Parsed {len(secrets)} secrets from TRACKER.md; {len(done)} already accepted: {sorted(done)}")
    print(f"\nTO SUBMIT ({len(todo)}):")
    for pid, bits in sorted(todo.items(), key=lambda kv: int(kv[0].split('_')[0])):
        n = int(pid.split('_')[0])
        flag = "" if len(bits) == n else f"  !! LENGTH {len(bits)} != {n} qubits"
        print(f"   {pid:8} ({n}q) {bits}{flag}")

    if not args.go:
        print("\n(dry run — re-run with --go to submit)")
        return

    print("\n=== SUBMITTING ===")
    banked = []
    for pid, bits in sorted(todo.items(), key=lambda kv: int(kv[0].split('_')[0])):
        try:
            acc, xeb = submit_one(pid, bits)
            if acc:
                print(f"   {pid:8} ACCEPTED  xeb={xeb}")
                banked.append(pid); continue
            # endianness fallback
            acc_r, xeb_r = submit_one(pid, bits[::-1])
            if acc_r:
                print(f"   {pid:8} ACCEPTED (reversed)  xeb={xeb_r}")
                banked.append(pid)
            else:
                print(f"   {pid:8} REJECTED both orders — investigate")
        except urllib.error.HTTPError as e:
            print(f"   {pid:8} HTTP {e.code}: {e.read().decode()[:200]}")
    print(f"\nBanked {len(banked)}/{len(todo)} this run: {banked}")


if __name__ == "__main__":
    main()
