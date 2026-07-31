"""bb_odds_collector.py — focused probe: just dump live markets + odds.

Run:  python C:\\tmp\\bb_odds_collector.py 2>&1 | Out-File C:\\tmp\\bb_odds.txt -Encoding utf8

Output: short lines like
  [market 028226] 2.08  2.85  3.40     (radiant - draw - dire decimal odds)
  [market 08489] 1.95  1.85             (totals over/under)
"""
from __future__ import annotations

import base64
import json
import os
import re
import socket
import ssl
import struct
import sys
import time

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ENV_PATH = r"C:\Users\artka\.minimax\workspace\dota_analyst\.env.betboom"
if not os.path.isfile(ENV_PATH):
    print(f"!! {ENV_PATH} not found")
    sys.exit(1)
for line in open(ENV_PATH, encoding="utf-8"):
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    os.environ.setdefault(k.strip(), v.strip())

TOKEN = os.environ.get("BETBOOM_AUTH_TOKEN", "").strip()
COOKIES = os.environ.get("BETBOOM_COOKIES", "").strip()
HOST = "ru-ws2.sporthub.bet"
PORT = 443


def hexdump(b: bytes, n: int = 60) -> str:
    return " ".join(f"{x:02x}" for x in b[:n])


def varint(d: bytes, p: int = 0):
    r, s = 0, 0
    while p < len(d):
        x = d[p]
        r |= (x & 0x7F) << s
        p += 1
        if not (x & 0x80):
            return r, p
        s += 7
    return r, p


def parse_protobuf(data: bytes, p: int = 0, depth: int = 0,
                   max_depth: int = 6, max_fields: int = 200):
    fields = []
    while p < len(data):
        try:
            t, p = varint(data, p)
        except Exception:
            break
        wire, fnum = t & 0x7, t >> 3
        if wire == 0:
            try:
                v, p = varint(data, p)
            except Exception:
                break
            fields.append((fnum, "varint", v))
        elif wire == 1:
            if p + 8 > len(data):
                break
            v = data[p:p + 8]; p += 8
            fields.append((fnum, "fixed64", v.hex()))
        elif wire == 2:
            try:
                l, p2 = varint(data, p)
            except Exception:
                break
            if p2 + l > len(data):
                break
            v = data[p2:p2 + l]; p = p2 + l
            if depth < max_depth and l > 0 and l < 200000:
                try:
                    sub, _ = parse_protobuf(v, 0, depth + 1, max_depth, max_fields)
                    fields.append((fnum, "msg", sub))
                except Exception:
                    fields.append((fnum, "bytes", v))
            else:
                fields.append((fnum, "bytes", v))
        elif wire == 5:
            if p + 4 > len(data):
                break
            v = data[p:p + 4]; p += 4
            fields.append((fnum, "fixed32", v.hex()))
        else:
            break
        if len(fields) >= max_fields:
            break
    return fields, p


def find_odds_and_strings(data: bytes) -> tuple[list[str], list[str]]:
    """Return (decimal_odds, all UTF-8 strings) found in the payload."""
    txt = data.decode("utf-8", errors="replace")
    odds = re.findall(r"\b[1-9]\.\d{1,3}\b", txt)
    strs = re.findall(r"[\x20-\x7e\xc0-\xff]{4,}", txt)
    return odds, strs


# tree_ws subscribes (HAR-captured)
TREE_SEND = [
    base64.b64decode("Gg8KBmYzNjU3ZBIDYWxsGAg="),
    base64.b64decode("Gg8KBjY5OTk0OBIDYWxsGAg="),
    base64.b64decode("Kg4KBmQyOTZiYxABGgIBAg=="),
]


def build_betstats_sub(sport_id: int, market_id: str) -> bytes:
    """Build market_betstats subscribe: outer field 1 { field 1 varint sport, field 2 string market_id }."""
    inner = bytes([0x08])  # tag field 1, varint
    n = sport_id
    while n >= 0x80:
        inner += bytes([(n & 0x7F) | 0x80])
        n >>= 7
    inner += bytes([n & 0x7F])
    mb = market_id.encode("ascii")
    inner += bytes([0x12, len(mb)]) + mb
    return bytes([0x0a, len(inner)]) + inner


def ws_handshake(sock, path, host_port, subprotocol="protobuf"):
    key = base64.b64encode(os.urandom(16)).decode()
    req = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host_port}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        "Origin: https://betboom.ru\r\n"
        "User-Agent: Mozilla/5.0 dota-analyst/0.4\r\n"
        f"Sec-WebSocket-Protocol: {subprotocol}\r\n"
        f"Cookie: {COOKIES}\r\n"
        f"Authorization: Bearer {TOKEN}\r\n"
        "\r\n"
    )
    sock.sendall(req.encode("utf-8"))
    resp = b""
    while b"\r\n\r\n" not in resp:
        chunk = sock.recv(4096)
        if not chunk:
            break
        resp += chunk
    return resp.decode("utf-8", errors="replace")


def ws_send(sock, op, payload):
    mask = os.urandom(4)
    masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    header = bytes([0x80 | op])
    length = len(payload)
    if length < 126:
        header += bytes([0x80 | length])
    elif length < (1 << 16):
        header += bytes([0x80 | 126]) + struct.pack(">H", length)
    else:
        header += bytes([0x80 | 127]) + struct.pack(">Q", length)
    sock.sendall(header + mask + masked)


def ws_recv(sock, timeout=3.0):
    sock.settimeout(timeout)
    try:
        hdr = sock.recv(2)
        if not hdr or len(hdr) < 2:
            return -1, b""
        op = hdr[0] & 0x0F
        masked = hdr[1] & 0x80
        length = hdr[1] & 0x7F
        if length == 126:
            length = struct.unpack(">H", sock.recv(2))[0]
        elif length == 127:
            length = struct.unpack(">Q", sock.recv(8))[0]
        mask_key = sock.recv(4) if masked else b""
        data = b""
        while len(data) < length:
            chunk = sock.recv(min(65536, length - len(data)))
            if not chunk:
                break
            data += chunk
        if masked and mask_key:
            data = bytes(b ^ mask_key[i % 4] for i, b in enumerate(data))
        return op, data
    except socket.timeout:
        return -2, b""


# === STEP 1: tree_ws → list of (market_name, market_id, match_id) ===
print("=" * 78)
print("STEP 1: tree_ws — get list of markets")
print("=" * 78)
markets = []  # list of (name, market_id, match_id)
raw = socket.create_connection((HOST, PORT), timeout=10)
ctx = ssl.create_default_context()
sock = ctx.wrap_socket(raw, server_hostname=HOST)
hs = ws_handshake(sock, "/api/tree_ws/v1", f"{HOST}:{PORT}")
if "101" not in hs.split("\r\n", 1)[0]:
    print("tree_ws HANDSHAKE FAIL"); sock.close(); sys.exit(1)
for f in TREE_SEND:
    ws_send(sock, 0x2, f)
end = time.time() + 8.0
while time.time() < end:
    op, data = ws_recv(sock, timeout=1.5)
    if op < 0: continue
    if op == 0x8: break
    if op == 0x9: ws_send(sock, 0xA, data); continue
    if op != 0x2: continue
    # Parse the data and look for market blocks
    try:
        fields, _ = parse_protobuf(data)
    except Exception:
        continue
    # Find strings that look like market IDs (4-7 digit)
    def walk(fs, parent_match=None):
        for tag, wire, val in fs:
            if wire == "msg":
                walk(val, parent_match)
            elif wire == "bytes" and isinstance(val, (bytes, bytearray)):
                try:
                    s = val.decode("ascii", errors="strict")
                    if s.isdigit() and 4 <= len(s) <= 7 and int(s) > 10000:
                        # Could be a market id
                        markets.append((s, parent_match or "?"))
                except Exception:
                    pass
    walk(fields)
sock.close()

# Also try to extract match_id (d296bc etc) — look for 6-char hex strings
match_ids = set()
raw = socket.create_connection((HOST, PORT), timeout=10)
ctx = ssl.create_default_context()
sock = ctx.wrap_socket(raw, server_hostname=HOST)
ws_handshake(sock, "/api/tree_ws/v1", f"{HOST}:{PORT}")
for f in TREE_SEND:
    ws_send(sock, 0x2, f)
end = time.time() + 8.0
all_text = []
while time.time() < end:
    op, data = ws_recv(sock, timeout=1.5)
    if op < 0: continue
    if op == 0x8: break
    if op == 0x9: ws_send(sock, 0xA, data); continue
    if op == 0x2:
        all_text.append(data)
sock.close()

# Extract all printable strings + match IDs
for data in all_text:
    for m in re.finditer(rb"[a-f0-9]{6}", data):
        s = m.group(0).decode("ascii", errors="replace")
        # match IDs from HAR are 6-char hex (d296bc, 9b7df7, etc.)
        if s[0].isalpha() and any(c.isalpha() for c in s):
            match_ids.add(s)

print(f"Found {len(markets)} market_id(s) and {len(match_ids)} match_id(s)")
print(f"  market_ids (first 30): {[m[0] for m in markets[:30]]}")
print(f"  match_ids: {sorted(match_ids)[:20]}")
print()

# === STEP 2: market_betstats for each market_id → find odds ===
print("=" * 78)
print("STEP 2: market_betstats — try each market_id, dump ODDS")
print("=" * 78)

unique_market_ids = list(dict.fromkeys([m[0] for m in markets]))[:20]
all_odds = {}  # market_id -> list of (decimal_odds, sport_id)
all_odds_raw = {}  # market_id -> raw response

# Try known sport_ids
sport_ids_to_try = [5399556, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for mid in unique_market_ids:
    found_odds = None
    found_sport = None
    for sid in sport_ids_to_try:
        try:
            sub = build_betstats_sub(sid, mid)
            raw = socket.create_connection((HOST, PORT), timeout=10)
            ctx = ssl.create_default_context()
            sock = ctx.wrap_socket(raw, server_hostname=HOST)
            hs = ws_handshake(sock, "/api/market_betstats_ws/v1", f"{HOST}:{PORT}")
            if "101" not in hs.split("\r\n", 1)[0]:
                sock.close()
                continue
            ws_send(sock, 0x2, sub)
            # Read up to 5 frames in 6s
            big_resp = None
            for _ in range(5):
                op, data = ws_recv(sock, timeout=1.5)
                if op < 0: continue
                if op == 0x8: break
                if op == 0x9: ws_send(sock, 0xA, data); continue
                if op == 0x2 and len(data) > 30:
                    big_resp = data
                    break
            sock.close()
            if big_resp:
                odds, strs = find_odds_and_strings(big_resp)
                if odds:
                    found_odds = odds
                    found_sport = sid
                    all_odds_raw[mid] = base64.b64encode(big_resp).decode()
                    break
        except Exception:
            continue
    if found_odds:
        all_odds[mid] = (found_odds, found_sport)
        print(f"  [market {mid}] sport={found_sport} ODDS: {' '.join(found_odds[:20])}")
    else:
        print(f"  [market {mid}] no odds (tried {len(sport_ids_to_try)} sport_ids)")

# === STEP 3: also try market_betstats with HAR market 028226 + a few others ===
print()
print("=" * 78)
print("STEP 3: try HAR-known markets (028226, 8489, 8491, 8490)")
print("=" * 78)
har_markets = ["028226", "8489", "8491", "8490"]
for mid in har_markets:
    if mid in all_odds:
        print(f"  [market {mid}] already have odds: {all_odds[mid][0][:10]}")
        continue
    found_odds = None
    found_sport = None
    for sid in sport_ids_to_try:
        try:
            sub = build_betstats_sub(sid, mid)
            raw = socket.create_connection((HOST, PORT), timeout=10)
            ctx = ssl.create_default_context()
            sock = ctx.wrap_socket(raw, server_hostname=HOST)
            hs = ws_handshake(sock, "/api/market_betstats_ws/v1", f"{HOST}:{PORT}")
            if "101" not in hs.split("\r\n", 1)[0]:
                sock.close(); continue
            ws_send(sock, 0x2, sub)
            big_resp = None
            for _ in range(5):
                op, data = ws_recv(sock, timeout=1.5)
                if op < 0: continue
                if op == 0x8: break
                if op == 0x9: ws_send(sock, 0xA, data); continue
                if op == 0x2 and len(data) > 30:
                    big_resp = data; break
            sock.close()
            if big_resp:
                odds, strs = find_odds_and_strings(big_resp)
                if odds:
                    found_odds = odds
                    found_sport = sid
                    all_odds_raw[mid] = base64.b64encode(big_resp).decode()
                    break
        except Exception:
            continue
    if found_odds:
        all_odds[mid] = (found_odds, found_sport)
        print(f"  [market {mid}] sport={found_sport} ODDS: {' '.join(found_odds[:20])}")
    else:
        print(f"  [market {mid}] no odds")

# === Summary ===
print()
print("=" * 78)
print("SUMMARY")
print("=" * 78)
print(f"Markets found in tree_ws: {len(unique_market_ids)}")
print(f"Match IDs found: {sorted(match_ids)[:20]}")
print(f"Odds captured: {len(all_odds)} / {len(unique_market_ids) + len(har_markets)} markets")
if all_odds:
    print()
    print("=== ODDS ===")
    for mid, (odds, sid) in all_odds.items():
        print(f"  market {mid:8}  sport_id={sid:9}  odds: {' '.join(odds[:15])}")
    print()
    print("=== RAW RESPONSES (base64) ===")
    for mid, b64 in all_odds_raw.items():
        print(f"  market {mid:8}  base64_len={len(b64)}")
        print(f"    {b64[:200]}{'...' if len(b64) > 200 else ''}")
else:
    print()
    print("No odds captured. Possible causes:")
    print("  1) All matches in tree_ws are finished (no live markets now)")
    print("  2) Real odds endpoint is fw-wc2.sportbook.bet (different host, not reachable here)")
    print("  3) Wrong sport_id for these markets")
    print()
    print("Try from Russia (where fw-wc2.sportbook.bet is reachable).")
