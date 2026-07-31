"""bb_odds_dual.py — try BOTH Sporthub backends.  Maybe fw-wc2.sportbook.bet
has the live odds that ru-ws2.sporthub.bet doesn't.

Run:  python C:\\tmp\\bb_odds_dual.py
"""
from __future__ import annotations
import base64
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
for line in open(ENV_PATH, encoding="utf-8"):
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    os.environ.setdefault(k.strip(), v.strip())
TOKEN = os.environ.get("BETBOOM_AUTH_TOKEN", "").strip()
COOKIES = os.environ.get("BETBOOM_COOKIES", "").strip()


def hexdump(b, n=60):
    return " ".join(f"{x:02x}" for x in b[:n])


def varint(d, p=0):
    r, s = 0, 0
    while p < len(d):
        x = d[p]
        r |= (x & 0x7F) << s
        p += 1
        if not (x & 0x80): return r, p
        s += 7
    return r, p


def parse_protobuf(data, p=0, depth=0, max_depth=6):
    fields = []
    while p < len(data):
        try: t, p = varint(data, p)
        except: break
        w, f = t & 7, t >> 3
        if w == 0:
            try: v, p = varint(data, p)
            except: break
            fields.append((f, "varint", v))
        elif w == 2:
            try: l, p2 = varint(data, p)
            except: break
            if p2 + l > len(data): break
            v = data[p2:p2+l]; p = p2 + l
            if depth < max_depth and 0 < l < 200000:
                try:
                    sub, _ = parse_protobuf(v, 0, depth+1, max_depth)
                    fields.append((f, "msg", sub))
                except: fields.append((f, "bytes", v))
            else: fields.append((f, "bytes", v))
        elif w == 1: p += 8
        elif w == 5: p += 4
        else: break
    return fields, p


def find_odds(data):
    txt = data.decode("utf-8", errors="replace")
    odds = re.findall(r"\b[1-9]\.\d{1,3}\b", txt)
    return odds


def build_betstats_sub(sport_id, market_id):
    inner = bytes([0x08])  # tag field 1, varint
    n = sport_id
    while n >= 0x80:
        inner += bytes([(n & 0x7F) | 0x80])
        n >>= 7
    inner += bytes([n & 0x7F])
    mb = market_id.encode("ascii")
    inner += bytes([0x12, len(mb)]) + mb
    return bytes([0x0a, len(inner)]) + inner


def ws_handshake(sock, path, host_port, subprotocol="protobuf", origin="https://betboom.ru"):
    key = base64.b64encode(os.urandom(16)).decode()
    req = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host_port}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        f"Origin: {origin}\r\n"
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
        if not chunk: break
        resp += chunk
    return resp.decode("utf-8", errors="replace")


def ws_send(sock, op, payload):
    mask = os.urandom(4)
    masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    header = bytes([0x80 | op])
    length = len(payload)
    if length < 126: header += bytes([0x80 | length])
    elif length < (1 << 16): header += bytes([0x80 | 126]) + struct.pack(">H", length)
    else: header += bytes([0x80 | 127]) + struct.pack(">Q", length)
    sock.sendall(header + mask + masked)


def ws_recv(sock, timeout=3.0):
    sock.settimeout(timeout)
    try:
        hdr = sock.recv(2)
        if not hdr or len(hdr) < 2: return -1, b""
        op = hdr[0] & 0x0F
        masked = hdr[1] & 0x80
        length = hdr[1] & 0x7F
        if length == 126: length = struct.unpack(">H", sock.recv(2))[0]
        elif length == 127: length = struct.unpack(">Q", sock.recv(8))[0]
        mask_key = sock.recv(4) if masked else b""
        data = b""
        while len(data) < length:
            chunk = sock.recv(min(65536, length - len(data)))
            if not chunk: break
            data += chunk
        if masked and mask_key:
            data = bytes(b ^ mask_key[i % 4] for i, b in enumerate(data))
        return op, data
    except socket.timeout:
        return -2, b""


def try_endpoint(label, host, port, paths, subprotocol="protobuf",
                 origin="https://betboom.ru", no_verify=True,
                 market_ids=None, build_sub=None):
    """Probe one backend across multiple paths.  market_ids + build_sub
    are optional — if given, also try market_betstats sub probes."""
    print(f"\n=== {label} ({host}:{port}, {len(paths)} paths) ===")
    if not no_verify:
        ctx_factory = ssl.create_default_context
    else:
        ctx_factory = lambda: ssl.create_default_context().__class__(
            check_hostname=False, verify_mode=ssl.CERT_NONE)
    for path in paths:
        try:
            raw = socket.create_connection((host, port), timeout=8)
            ctx = ssl.create_default_context()
            if no_verify:
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
            sock = ctx.wrap_socket(raw, server_hostname=host)
            hs = ws_handshake(sock, path, f"{host}:{port}",
                              subprotocol=subprotocol, origin=origin)
            first = hs.split("\r\n", 1)[0]
            if "101" not in first:
                sock.close()
                continue
            # Send a generic subscribe (use the same tree_ws subscribes)
            tree_subs = [
                base64.b64decode("Gg8KBmYzNjU3ZBIDYWxsGAg="),
                base64.b64decode("Gg8KBjY5OTk0OBIDYWxsGAg="),
                base64.b64decode("Kg4KBmQyOTZiYxABGgIBAg=="),
            ]
            for f in tree_subs:
                ws_send(sock, 0x2, f)
            # Read up to 6 frames in 8 seconds
            big = []
            end = time.time() + 8.0
            while time.time() < end and len(big) < 6:
                op, data = ws_recv(sock, timeout=1.0)
                if op < 0: continue
                if op == 0x8: break
                if op == 0x9: ws_send(sock, 0xA, data); continue
                big.append((op, data))
            sizes = ",".join(f"{ln}" for op, ln in [(o, len(d)) for o, d in big])
            any_big = any(ln > 50 for _, ln in [(o, len(d)) for o, d in big])
            mark = " <<<" if any_big else ""
            print(f"  {path:40}  sizes=[{sizes}]{mark}")
            for op, data in big:
                if len(data) > 30:
                    odds = find_odds(data)
                    if odds:
                        print(f"      *** ODDS ({len(data)}b): {' '.join(odds[:15])}")
                        print(f"      base64: {base64.b64encode(data).decode()[:200]}")
                    else:
                        print(f"      {len(data)}b: hex={hexdump(data, 25)}")
            sock.close()
        except Exception as e:
            print(f"  {path:40}  ERR: {type(e).__name__}: {str(e)[:60]}")


# === Probe both backends ===
print("Probing both Sporthub backends for live odds")
print("=" * 78)

# Backend 1: ru-ws2.sporthub.bet (known)
try_endpoint(
    "ru-ws2.sporthub.bet (Russian IP, currently broken — no odds)",
    "ru-ws2.sporthub.bet", 443,
    ["/api/odds_live_ws/v1", "/api/feed_live_ws/v1", "/api/market_live_ws/v1",
     "/api/market_betstats_ws/v1"],
    subprotocol="protobuf",
)

# Backend 2: fw-wc2.sportbook.bet (Namecheap parking for non-RU, may work for RU)
try_endpoint(
    "fw-wc2.sportbook.bet (different backend — may have odds)",
    "fw-wc2.sportbook.bet", 443,
    ["/api/odds_live_ws/v1", "/api/feed_live_ws/v1", "/api/market_live_ws/v1",
     "/api/market_betstats_ws/v1", "/api/feed_live_ws/v2",
     "/api/market_live_ws/v2", "/api/odds_live_ws/v2"],
    subprotocol="protobuf",
)

# Backend 3: sportbook.bet (parent domain, may have wildcard cert)
try_endpoint(
    "sportbook.bet (parent domain)",
    "sportbook.bet", 443,
    ["/api/odds_live_ws/v1", "/api/feed_live_ws/v1", "/api/market_live_ws/v1"],
    subprotocol="protobuf",
)

# Backend 4: betboom-sport.com (separate sportbook SPA)
try_endpoint(
    "betboom-sport.com (separate sportbook SPA)",
    "betboom-sport.com", 443,
    ["/api/odds_live_ws/v1", "/api/feed_live_ws/v1", "/api/market_live_ws/v1"],
    subprotocol="protobuf",
)

# Backend 5: betboom-sport.com with Host=fw-wc2.sportbook.bet (might route to fw-wc2 backend)
try_endpoint(
    "betboom-sport.com with Host=fw-wc2.sportbook.bet",
    "betboom-sport.com", 443,
    ["/api/odds_live_ws/v1", "/api/feed_live_ws/v1"],
    subprotocol="protobuf",
    origin="https://fw-wc2.sportbook.bet",
)

print()
print("=" * 78)
print("If ANY backend returned a frame with decimal odds, paste that")
print("output here.  I'll wire it up in 20-30 minutes.")
print("=" * 78)
