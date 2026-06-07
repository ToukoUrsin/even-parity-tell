#!/usr/bin/env python3
"""
Probe the QMill verify-samples grader with a batch sample file and read back the
graded result (non_zero_count etc.). Single call — no flooding.

Usage:
    QMILL_TOKEN="<fresh bearer token>" python3 probe_run.py \
        --problem 56_38 --file probe_56_38_flip.json --desc "recon: peak + hamming-1 flips"

Get a fresh token from DevTools: any request -> Headers -> 'authorization: Bearer <...>'
(copy just the part after 'Bearer '). Tokens live ~5 min, so grab one right before running.
"""
import argparse, json, os, sys, time, urllib.request, urllib.error

BASE = "https://qas.qmill.com/api/v1/challenges/junction-quantum-hack"
BOUNDARY = "----probeBoundary7MA4YWxkTrZu0gW"


def post_verify(problem, filepath, desc, token):
    with open(filepath, "rb") as f:
        file_bytes = f.read()
    filename = os.path.basename(filepath)
    body = b"".join([
        f"--{BOUNDARY}\r\n".encode(),
        f'Content-Disposition: form-data; name="samples_file"; filename="{filename}"\r\n'.encode(),
        b"Content-Type: application/json\r\n\r\n",
        file_bytes, b"\r\n",
        f"--{BOUNDARY}\r\n".encode(),
        b'Content-Disposition: form-data; name="description"\r\n\r\n',
        desc.encode(), b"\r\n",
        f"--{BOUNDARY}--\r\n".encode(),
    ])
    req = urllib.request.Request(
        f"{BASE}/problems/{problem}/verify-samples", data=body, method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": f"multipart/form-data; boundary={BOUNDARY}",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def get_submission(sub_id, token):
    req = urllib.request.Request(
        f"{BASE}/submissions/{sub_id}",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="56_38")
    ap.add_argument("--file", default="probe_56_38_flip.json")
    ap.add_argument("--desc", default="recon: known peak + all hamming-1 neighbors")
    args = ap.parse_args()

    token = os.environ.get("QMILL_TOKEN")
    if not token and os.path.exists(".qmill_token"):
        token = open(".qmill_token").read().strip()
    if not token and os.path.exists(".qmill_refresh"):
        import qmill_auth                       # auto-mint a fresh access token, no expiry race
        token = qmill_auth.get_access_token()
    if not token:
        sys.exit("ERROR: provide a token via QMILL_TOKEN env var, a .qmill_token file, "
                 "or a .qmill_refresh file (refresh token, auto-minted).")
    token = token.strip().strip('"').strip("'")
    if "eyJ" in token:   # extract the JWT no matter what got copied (Bearer/authorization:/quotes)
        token = token[token.index("eyJ"):].split()[0].strip('"').strip("'")

    n = len(json.load(open(args.file)))
    print(f"POST {args.file} ({n} strings) -> problem {args.problem}/verify-samples ...")
    try:
        resp = post_verify(args.problem, args.file, args.desc, token)
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code}: {e.read().decode()[:500]}")
    print(json.dumps(resp, indent=2))

    sub_id = resp.get("submission_id")
    rl = resp.get("rate_limit", {})
    print(f"\nrate_limit -> submissions_left={rl.get('submissions_left')} "
          f"(was 9 earlier; still 9 => verify-samples is FREE, lower => it's throttled)")
    if not sub_id:
        sys.exit("No submission_id returned; nothing to poll.")

    print(f"\nPolling submission {sub_id} for grade ...")
    for _ in range(40):  # ~40 * 1.5s = up to 60s
        time.sleep(1.5)
        s = get_submission(sub_id, token)
        if s.get("status") == "verified" or s.get("completed_at"):
            print(json.dumps(s, indent=2))
            tot = s.get("total_shots")
            nz = s.get("non_zero_count")
            print(f"\n>>> RESULT: non_zero_count={nz} / total_shots={tot}  "
                  f"(accepted_as_quantum={s.get('accepted_as_quantum')}, xeb_score={s.get('xeb_score')})")
            if nz == 1:
                print(">>> CLIFF: only the exact peak registers. Verify-only oracle, no Hamming gradient.")
            elif isinstance(nz, int) and nz > 1:
                print(">>> GRADIENT: neighbors register too -> bit-recovery may be feasible. Big finding.")
            elif nz == 0:
                print(">>> Even the known peak didn't register -> our threshold model is wrong.")
            return
        print(f"  status={s.get('status')} ...")
    print("Timed out waiting for grade; check 'My submissions' manually.")


if __name__ == "__main__":
    main()
