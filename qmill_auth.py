#!/usr/bin/env python3
"""Mint a fresh QMill access token from a stored Keycloak refresh token.

Reads a refresh token from .qmill_refresh, calls the Keycloak token endpoint,
returns a fresh ~5-min access token, and writes back the rotated refresh token
(Keycloak rotates refresh tokens on each use by default). The credential lives
only in .qmill_refresh, which is git-ignored. Nothing is printed except the token
when run directly.
"""
import json, os, urllib.parse, urllib.request, urllib.error

TOKEN_URL = "https://qas.qmill.com/auth/realms/quantum-platform/protocol/openid-connect/token"
CLIENT_ID = "quantum-app"          # public SPA client (azp in the JWT); no secret
REFRESH_FILE = ".qmill_refresh"


def get_access_token():
    if not os.path.exists(REFRESH_FILE):
        raise SystemExit(f"No {REFRESH_FILE}; save your Keycloak refresh_token there first.")
    rt = open(REFRESH_FILE).read().strip()
    if "eyJ" in rt:                # tolerate pasting extra wrapping text
        rt = rt[rt.index("eyJ"):].split()[0]
    data = urllib.parse.urlencode({
        "grant_type": "refresh_token",
        "client_id": CLIENT_ID,
        "refresh_token": rt,
    }).encode()
    req = urllib.request.Request(
        TOKEN_URL, data=data, method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urllib.request.urlopen(req) as r:
            tok = json.load(r)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"Refresh failed HTTP {e.code}: {e.read().decode()[:400]}")
    if tok.get("refresh_token"):   # persist the rotated refresh token for next call
        with open(REFRESH_FILE, "w") as f:
            f.write(tok["refresh_token"])
    return tok["access_token"]


if __name__ == "__main__":
    print(get_access_token())
