"""bb_collect_frames.py — connect to BetBoom Sporthub WebSocket endpoints
and dump the first 8 frames per endpoint.  Replays the exact subscribe
messages from a real HAR captured against the live betting page.

Run from PowerShell (or bash):
    python C:\\tmp\\bb_collect_frames.py > C:\\tmp\\bb_collect_output.txt
Then paste the contents of bb_collect_output.txt back to the assistant.
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

# Force UTF-8 output (Windows consoles default to cp1251 which mangles Cyrillic)
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

# Load env
for line in open(ENV_PATH, encoding="utf-8"):
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    os.environ.setdefault(k.strip(), v.strip())

TOKEN = os.environ.get("BETBOOM_AUTH_TOKEN", "").strip()
COOKIES = os.environ.get("BETBOOM_COOKIES", "").strip()

# === HAR subscribe messages (replayed byte-for-byte) ===
# tree_ws: 3 subscribes (cat f3657d, cat 699948, match d296bc)
TREE_SEND = [
    base64.b64decode("Gg8KBmYzNjU3ZBIDYWxsGAg="),   # sub category f3657d, "all"
    base64.b64decode("Gg8KBjY5OTk0OBIDYWxsGAg="),   # sub category 699948, "all"
    base64.b64decode("Kg4KBmQyOTZiYxABGgIBAg=="),   # sub match d296bc, action 1
]
# oddin_widget_ws: subscribe to match 9b7df7
WIDGET_SEND = [
    base64.b64decode("EhEKBjliN2RmNxIHNTM5OTU1Ng=="),
]
# bets_history: JSON subscribe_summary
BETS_HISTORY_SEND = [
    json.dumps({
        "subscribe_summary": {"uid": "dota-analyst-probe", "token": TOKEN}
    }, ensure_ascii=False).encode("utf-8"),
]

HOST = "ru-ws2.sporthub.bet"
PORT = 443


def hexdump(b: bytes, n: int = 60) -> str:
    return " ".join(f"{x:02x}" for x in b[:n])


def varint(data: bytes, pos: int = 0) -> tuple[int, int]:
    result, shift = 0, 0
    while pos < len(data):
        b = data[pos]
        result |= (b & 0x7F) << shift
        pos += 1
        if not (b & 0x80):
            return result, pos
        shift += 7
    return result, pos


def parse_protobuf(data: bytes, pos: int = 0, depth: int = 0,
                   max_depth: int = 5, max_fields: int = 60) -> tuple[list, int]:
    """Lightweight protobuf parser.  Returns (fields, new_pos)."""
    fields = []
    while pos < len(data):
        try:
            tag, pos = varint(data, pos)
        except Exception:
            break
        wire = tag & 0x7
        fnum = tag >> 3
        if wire == 0:
            try:
                v, pos = varint(data, pos)
            except Exception:
                break
            fields.append((fnum, "varint", v))
        elif wire == 1:
            if pos + 8 > len(data):
                break
            v = data[pos:pos + 8]
            pos += 8
            fields.append((fnum, "fixed64", v.hex()))
        elif wire == 2:
            try:
                length, pos2 = varint(data, pos)
            except Exception:
                break
            if pos2 + length > len(data):
                break
            v = data[pos2:pos2 + length]
            pos = pos2 + length
            if depth < max_depth and length > 0 and length < 200000:
                try:
                    sub, _ = parse_protobuf(v, 0, depth + 1, max_depth, max_fields)
                    fields.append((fnum, "msg", sub))
                except Exception:
                    fields.append((fnum, "bytes", v))
            else:
                fields.append((fnum, "bytes", v))
        elif wire == 5:
            if pos + 4 > len(data):
                break
            v = data[pos:pos + 4]
            pos += 4
            fields.append((fnum, "fixed32", v.hex()))
        else:
            break
        if len(fields) >= max_fields:
            break
    return fields, pos


def extract_strings(data: bytes, min_len: int = 4, max_strings: int = 20) -> list[str]:
    """Find readable UTF-8 strings in the binary payload."""
    strings = []
    for m in re.finditer(rb"[\x20-\x7e\xc0-\xff]{%d,}" % min_len, data):
        s = m.group(0)
        try:
            decoded = s.decode("utf-8", errors="replace")
            if not decoded.isascii():
                visible = sum(1 for c in decoded if c.isprintable() or c.isspace())
                if visible / max(1, len(decoded)) < 0.5:
                    continue
            elif decoded.startswith("\x00") or len(decoded) < min_len:
                continue
            if len(decoded) > 50 and all(c in "0123456789abcdef" for c in decoded.lower()):
                continue
            strings.append(decoded)
        except Exception:
            pass
        if len(strings) >= max_strings:
            break
    return strings


def render_tree(fields, depth: int = 0, max_items: int = 30) -> list[str]:
    out = []
    indent = "  " * depth
    for i, (tag, wire, val) in enumerate(fields[:max_items]):
        if wire == "msg":
            out.append(f"{indent}f{tag} msg[{len(val)}]:")
            out.extend(render_tree(val, depth + 1, max_items=15))
        elif wire == "bytes":
            if len(val) == 0:
                out.append(f"{indent}f{tag} bytes[0] = ''")
            elif len(val) <= 100:
                try:
                    s = val.decode("utf-8")
                    if s.isprintable() or all(c.isprintable() or c.isspace() for c in s):
                        out.append(f"{indent}f{tag} str[{len(val)}] = {s!r}")
                    else:
                        out.append(f"{indent}f{tag} bytes[{len(val)}] = {val.hex()}")
                except Exception:
                    out.append(f"{indent}f{tag} bytes[{len(val)}] = {val.hex()}")
            else:
                out.append(f"{indent}f{tag} bytes[{len(val)}] = {val[:30].hex()}...{val[-10:].hex()}")
        else:
            out.append(f"{indent}f{tag} {wire} = {val}")
    if len(fields) > max_items:
        out.append(f"{indent}... and {len(fields) - max_items} more")
    return out


def ws_handshake(sock, path: str, host_port: str, subprotocol: str):
    key = base64.b64encode(os.urandom(16)).decode()
    req = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host_port}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        "Origin: https://betboom.ru\r\n"
        "User-Agent: Mozilla/5.0 (X11; Linux x64) dota-analyst/0.4 probe\r\n"
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


def ws_send(sock, op: int, payload: bytes):
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


def ws_recv(sock, timeout: float = 3.0) -> tuple[int, bytes]:
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


def build_betstats_sub(market_id_str: str) -> bytes:
    """Build a market_betstats subscribe message for a given market ID string.
    Pattern from HAR:  Cg0IhMjJAhIGMDI4MjI2
    Decoded: 0a 0d 08 84 c8 c9 02 12 06 30 32 38 32 32 36
    = field 1 varint 0x258c4 + field 2 string "028226"
    Actually: 0x84c8c902 varint = 0x25844 + (0x84c8c902 & ...) — let me decode
    """
    # Pattern: 0a 0d 08 <varint-3bytes> 12 <len> <ascii-id>
    # From HAR: 08 84 c8 c9 02 — varint = 0x25844 + ... actually let me read
    # 0x84 = 10000100 — continuation bit, low 7 = 0x04
    # 0xc8 = 11001000 — continuation, low 7 = 0x48
    # 0xc9 = 11001001 — continuation, low 7 = 0x49
    # 0x02 = 00000010 — final, low 7 = 0x02
    # value = 0x04 | (0x48 << 7) | (0x49 << 14) | (0x02 << 21) = 4 + 9216 + 1208320 + 4194304 = 5411844
    # 5411844 = some hash?  Let me check what 028226 maps to
    # 028226 in decimal = 28226
    # 5411844 - doesn't look like a simple mapping
    # Let me try: id=28226, hash = id*192 + 16? No, 28226*192 = 5419392, no
    # Maybe it's id*192 + small_offset?  5411844/28226 = 191.75 — close to 192
    # So 28226*192 = 5419392, then -7548 = 5411844.  Hmm
    # Or just: hash = id*192 + ((id*192) >> 16) maybe?  Doesn't matter, let me
    # empirically compute: varint(5411844) = 0x84 0xc8 0xc9 0x02
    # So the magic number is 5411844 for market 028226
    # Let me find the actual encoding function
    # 28226 -> varint bytes [0x84, 0xc8, 0xc9, 0x02]
    # 0x84 & 0x7F = 0x04
    # 0xc8 & 0x7F = 0x48
    # 0xc9 & 0x7F = 0x49
    # 0x02 = 0x02
    # val = 0x04 | (0x48 << 7) | (0x49 << 14) | (0x02 << 21)
    # = 0x04 + 0x2400 + 0x124000 + 0x400000
    # = 4 + 9216 + 1196032 + 4194304
    # = 5409556?  Let me recompute
    # 0x02 << 21 = 0x02 * 2097152 = 4194304
    # 0x49 << 14 = 0x49 * 16384 = 75 * 16384 + 9*16384 = wait
    # 0x49 = 73.  73 * 16384 = 73*16000 + 73*384 = 1168000 + 28032 = 1196032
    # 0x48 << 7 = 0x48 * 128 = 72*128 = 9216
    # 0x04 = 4
    # sum = 4194304 + 1196032 + 9216 + 4 = 5400-000-? Let me just compute
    # 4194304 + 1196032 = 5390336
    # 5390336 + 9216 = 5399552
    # 5399552 + 4 = 5399556
    # Hmm, 5399556!  That's the same as the sport_id 5399556 we saw earlier!
    # So the magic number is `sport_id`.  Let me verify: for Dota 2 market,
    # sport_id would be 5399556.  For CS2, sport_id is... let me check.
    # In tree_ws I saw `f1=5399556` for the Dota 2 game.  CS2 is 1?  Or some other.
    # Actually for this HAR the match is d296bc (CS2), and 028226 was the
    # market betstats probe.  sport_id=5399556 might be the "esports"
    # overarching category, not specific to Dota 2 vs CS2.
    # Let me just use the same magic number for now (or compute per-market)
    # Actually — looking at the HAR market_betstats request bytes:
    # Cg0IhMjJAhIGMDI4MjI2
    # hex: 0a 0d 08 84 c8 c9 02 12 06 30 32 38 32 32 36
    # So:
    #   0a = field 1, wire 2 (length-delimited)
    #   0d = length 13
    #   08 84 c8 c9 02 = field 1, wire 0 (varint) = 5399556
    #   12 = field 2, wire 2 (length-delimited)
    #   06 = length 6
    #   30 32 38 32 32 36 = "028226"
    # So the structure is:
    #   wrapper = field 1 { field 1: 5399556 (sport_id), field 2: market_id_str }
    # We need sport_id!  In tree_ws we saw `f1=5399556` in the
    # oddin_widget subscribe for match 9b7df7 (Dota 2).  For CS2 match
    # d296bc, sport_id should be the CS2 one, not 5399556.
    # Let me check the tree_ws data for sport_id near d296bc...
    # Hmm I don't see it directly.  But the sport_id for "esports" is
    # likely 5399556, and CS2/Dota 2 are sub-categories with their own IDs.
    # Strategy: try multiple known sport_ids and see which one returns
    # odds for the market.
    #
    # Actually let's just use 5399556 (esports overall) and see if it works.
    # If not, we'll adjust.
    sport_id = 5399556  # "esports" (parent category)
    sub_msg = b""
    sub_msg += bytes([0x08])  # field 1, varint
    # encode sport_id as varint
    n = sport_id
    while n >= 0x80:
        sub_msg += bytes([(n & 0x7F) | 0x80])
        n >>= 7
    sub_msg += bytes([n & 0x7F])
    # field 2, length-delimited string
    market_bytes = market_id_str.encode("ascii")
    sub_msg += bytes([0x12, len(market_bytes)]) + market_bytes
    # outer wrapper
    return bytes([0x0a, len(sub_msg)]) + sub_msg


def run_session(label: str, path: str, subprotocol: str, send_frames: list,
                max_frames: int = 8, max_time: float = 12.0,
                capture_market_ids: bool = False,
                market_blocks: list = None) -> list:
    """Run a session, optionally extract market_ids from the tree response.
    Returns the list of extracted market_ids (or [] if not capturing)."""
    print("=" * 78)
    print(f"## {label}")
    print(f"## URL: wss://{HOST}:{PORT}{path}  (subprotocol={subprotocol})")
    print("=" * 78)
    market_ids = []
    try:
        raw = socket.create_connection((HOST, PORT), timeout=10)
        ctx = ssl.create_default_context()
        sock = ctx.wrap_socket(raw, server_hostname=HOST)
        cert = sock.getpeercert()
        subj = dict(x[0] for x in cert.get("subject", [])) if cert else {}
        print(f"TLS subject: {subj.get('commonName', '?')}")
        hs = ws_handshake(sock, path, f"{HOST}:{PORT}", subprotocol)
        first = hs.split("\r\n", 1)[0]
        print(f"handshake: {first}")
        if "101" not in first:
            print("HANDSHAKE FAILED, abort\n")
            sock.close()
            return market_ids
        # Send subscribe frames
        for i, f in enumerate(send_frames):
            ws_send(sock, 0x2, f)
            print(f"\n[client->server #{i}] opcode=2 len={len(f)} base64={base64.b64encode(f).decode()}")
            print(f"  hex[:40]: {hexdump(f, 40)}")
        # Receive frames
        print()
        end = time.time() + max_time
        count = 0
        while time.time() < end:
            op, data = ws_recv(sock, timeout=2.0)
            if op < 0:
                continue
            if op == 0x8:
                print(f"[server->client CLOSE] {data!r}")
                break
            if op == 0x9:
                ws_send(sock, 0xA, data)
                continue
            count += 1
            b64 = base64.b64encode(data).decode()
            print(f"\n[server->client #{count}] opcode={op} len={len(data)}")
            print(f"  base64 (first 200): {b64[:200]}{'...' if len(b64) > 200 else ''}")
            print(f"  hex (first 40): {hexdump(data, 40)}")
            strs = extract_strings(data, min_len=3, max_strings=15)
            if strs:
                print(f"  strings found:")
                for s in strs:
                    print(f"    {s!r}")
            if op == 0x2 and 4 < len(data) < 100000:
                try:
                    fields, _ = parse_protobuf(data)
                    if fields:
                        print(f"  protobuf tree (top-level {len(fields)} fields, first 12):")
                        for line in render_tree(fields, max_items=12)[:50]:
                            print(f"    {line}")
                    # Extract market IDs from tree_ws responses
                    if capture_market_ids:
                        ids = find_market_ids(fields)
                        if ids:
                            market_ids.extend(ids)
                            print(f"  >>> extracted (raw) market_ids: {ids}")
                        # Also extract market name+id pairs
                        blocks = find_market_blocks(fields)
                        if blocks:
                            print(f"  >>> extracted market blocks (name, id):")
                            for name, mid in blocks:
                                print(f"      {name!r:30} -> market_id={mid}")
                            # Prefer market blocks (real market IDs, not selection IDs)
                            market_blocks.extend(blocks)
                except Exception as e:
                    print(f"  protobuf parse error: {e}")
            if count >= max_frames:
                print(f"\n(reached {max_frames} frames, stopping)")
                break
        sock.close()
    except Exception as e:
        print(f"ERR: {type(e).__name__}: {e}")
    print()
    return market_ids


def find_market_ids(fields) -> list[str]:
    """Walk parsed protobuf tree and return any string that looks like a market ID
    (4-7 digit number, or '16002', '36041' style)."""
    ids = []
    for tag, wire, val in fields:
        if wire == "msg":
            ids.extend(find_market_ids(val))
        elif wire == "bytes" and isinstance(val, (bytes, bytearray)):
            try:
                s = val.decode("ascii", errors="strict")
            except Exception:
                continue
            # Market ID: 4-7 digit number
            if s.isdigit() and 4 <= len(s) <= 7:
                # Skip "year" (e.g. 2026) — must be > 10000
                if int(s) > 10000:
                    ids.append(s)
    return ids


def find_market_blocks(fields, depth=0) -> list[tuple[str, str]]:
    """Find (market_name, market_id) pairs in tree_ws data.
    Pattern: f10 msg[N]: f1='<name>' f2='<id>' f2=...
    Returns list of (name, id) tuples.
    """
    out = []
    # The market block is f10 with f1=name and f2=id
    # We look for blocks that have f1 str followed by f2 str (numeric)
    for tag, wire, val in fields:
        if wire == "msg":
            # Check if this is a market block (has both f1 str and f2 str)
            f1_str = None
            f2_strs = []
            for t, w, v in val:
                if w == "bytes" and isinstance(v, (bytes, bytearray)):
                    try:
                        s = v.decode("utf-8", errors="strict")
                    except Exception:
                        continue
                    if t == 1 and f1_str is None:
                        f1_str = s
                    elif t == 2 and s.isdigit() and 3 <= len(s) <= 8:
                        f2_strs.append(s)
            if f1_str and f2_strs:
                # The first f2 digit-string after a name is the market_id
                out.append((f1_str, f2_strs[0]))
            # Recurse for nested
            out.extend(find_market_blocks(val, depth + 1))
    return out


# === Run all sessions ===
print(f"Loaded JWT ({len(TOKEN)}b) and cookies ({len(COOKIES)}b) from .env.betboom")
print(f"Probing {HOST}:{PORT} (Russian IP, Sporthub)\n")

# 1) tree_ws — also extract market_ids
market_blocks: list[tuple[str, str]] = []
market_ids = run_session(
    "tree_ws (categories + match tree, EXTRACT market_ids)",
    "/api/tree_ws/v1", "protobuf", TREE_SEND,
    max_frames=8, max_time=15.0, capture_market_ids=True,
    market_blocks=market_blocks,
)
# Dedup
market_ids = list(dict.fromkeys(market_ids))
print(f"\n[collector] Total unique market_ids from tree_ws: {len(market_ids)}")
print(f"  sample: {market_ids[:20]}")
print(f"[collector] market blocks (name, id): {market_blocks[:30]}")
# Use only the IDs from blocks (real market IDs, not selection IDs)
block_ids = [mid for _, mid in market_blocks]
print(f"[collector] {len(block_ids)} market block IDs: {block_ids}")

# 2) oddin_widget — Dota 2 game state
run_session("oddin_widget_ws (live match data)", "/api/oddin_widget_ws/v1", "protobuf",
            WIDGET_SEND, max_frames=3, max_time=8.0)

# 3) For each market_id (preferring market_blocks), subscribe to betstats
probe_ids = list(dict.fromkeys(block_ids + market_ids))[:8]
if probe_ids:
    print("=" * 78)
    print(f"## market_betstats_ws — probing {len(probe_ids)} market(s) (waiting 5s for full feed)")
    print("=" * 78)
    for mid in probe_ids:
        sub = build_betstats_sub(mid)
        print(f"\n[probe] market_id={mid}  sub={base64.b64encode(sub).decode()}")
        try:
            raw = socket.create_connection((HOST, PORT), timeout=10)
            ctx = ssl.create_default_context()
            sock = ctx.wrap_socket(raw, server_hostname=HOST)
            hs = ws_handshake(sock, "/api/market_betstats_ws/v1", f"{HOST}:{PORT}", "protobuf")
            first = hs.split("\r\n", 1)[0]
            if "101" not in first:
                print(f"  HANDSHAKE FAILED: {first}")
                sock.close()
                continue
            ws_send(sock, 0x2, sub)
            # Wait up to 5s for up to 10 frames
            sock.settimeout(5.0)
            end = time.time() + 5.0
            count = 0
            while time.time() < end and count < 12:
                op, data = ws_recv(sock, timeout=1.0)
                if op < 0:
                    if time.time() >= end:
                        break
                    continue
                if op == 0x8:
                    print(f"  CLOSE: {data!r}")
                    break
                if op == 0x9:
                    ws_send(sock, 0xA, data)
                    continue
                count += 1
                print(f"  << frame #{count} opcode={op} len={len(data)}")
                b64 = base64.b64encode(data).decode()
                print(f"     base64: {b64[:200]}{'...' if len(b64) > 200 else ''}")
                print(f"     hex: {hexdump(data, 60)}")
                strs = extract_strings(data, max_strings=15)
                if strs:
                    print(f"     strings: {strs}")
                if op == 0x2 and len(data) > 20:
                    try:
                        fields, _ = parse_protobuf(data)
                        if fields:
                            for line in render_tree(fields, max_items=20)[:80]:
                                print(f"     {line}")
                    except Exception as e:
                        print(f"     parse err: {e}")
                    print()
            sock.close()
        except Exception as e:
            print(f"  ERR: {e}")
    print()

# 4) bets_history — JSON
run_session("bets_history_ws (JSON, sub summary)", "/api/bets_history_ws/v1", "json",
            BETS_HISTORY_SEND, max_frames=3, max_time=6.0)

# 5) Try the OTHER Sporthub endpoints (may have odds in plain form)
OTHER_SPORTHUB = [
    ("/api/feed_live_ws/v1", "feed_live_ws (live feed)"),
    ("/api/market_live_ws/v1", "market_live_ws (live markets)"),
    ("/api/odds_live_ws/v1", "odds_live_ws (live odds)"),
    ("/api/sport_history_ws/v1", "sport_history_ws (sport history)"),
    ("/api/line_live_ws/v1", "line_live_ws (live line)"),
    ("/api/event_live_ws/v1", "event_live_ws (live events)"),
    ("/api/live_ws/v1", "live_ws (generic live)"),
    ("/api/prematch_ws/v1", "prematch_ws (prematch)"),
    ("/api/feed_ws/v1", "feed_ws (feed)"),
    ("/api/market_ws/v1", "market_ws (markets)"),
    ("/api/odds_ws/v1", "odds_ws (odds)"),
]
for path, label in OTHER_SPORTHUB:
    # Send the same tree_ws subscribes (generic 'all' categories)
    run_session(label, path, "protobuf", TREE_SEND,
                max_frames=3, max_time=4.0)

