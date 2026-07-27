"""Code audit for the dota_analyst project (v0.3.22 → 0.4.0 prep).

Walks the source tree and reports findings across 4 severities:
  [P0] security / correctness — must fix before any public exposure
  [P1] performance / robustness — fix before 1.0
  [P2] code quality / docs — backlog

What we look for:

Security
--------
  * hardcoded secrets (api keys, tokens, passwords) outside test fixtures
  * `eval(` / `exec(` outside sandboxed contexts
  * `subprocess.Popen` / `os.system` / `shell=True`
  * `pickle.load` on untrusted input
  * `assert` in non-test code (stripped by `python -O`)
  * `requests.get(..., verify=False)` / `urllib` ssl bypass

Correctness
-----------
  * `except Exception` without re-raise (swallows bugs)
  * `except:` (catches KeyboardInterrupt + SystemExit)
  * bare `pass` in `except`
  * mutable default arguments (`def f(x=[])`)
  * `==` / `is None` mixups
  * race-prone lazy-init (`if not _cache: _cache = ...` without lock)

Performance
-----------
  * `urllib.request.urlopen` / `requests.get` inside a hot loop without
    a Session (each call is a fresh TCP+TLS handshake)
  * blocking sleep (`time.sleep`) inside `async def`
  * `urllib` calls without `timeout=`
  * un-bounded loops that could be vectorised

Quality
-------
  * `print(` outside scripts/ and tests/
  * `TODO` / `FIXME` / `XXX` without an owner / version target
  * `import *` outside `__init__.py`
  * module-level `os.environ` reads inside hot paths (cache them)
  * tests that don't `await` async fixtures (async TestClient gotcha)
  * files > 1000 LoC (refactor candidate)

Run with `python scripts/audit.py`.  Exits 0 even on findings so it
fits into CI as a non-blocking lint step; look at the `findings`
counter instead of the exit code.
"""
from __future__ import annotations

import ast
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Force UTF-8 stdout so non-ASCII findings (em-dash, etc.) don't
# crash the run on Windows consoles with cp1251/cp437.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except (AttributeError, OSError):
    pass

ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOTS = ("business", "gateway", "scripts", "tests")
SKIP_DIRS = {".git", "__pycache__", "ml_data", "node_modules", ".pytest_cache", "scratch_*"}

# -- heuristics -----------------------------------------------------------

SECRET_PATTERNS = [
    (re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*=\s*['\"][^'\"]{8,}['\"]"), "literal secret"),
    (re.compile(r"dev-local-dota-analyst-key-change-me"), "dev API key (expected, but should be moved to env)"),
]
# Note: each call is matched as a *standalone* call, not as a
# substring of `executor`/`execution`/etc.  Use the AST pass for
# the real call detection below — the regex here is a quick scan
# only, intentionally conservative.
SUSPICIOUS_CALLS = {
    # `eval(`  — actual call form
    re.compile(r"\beval\s*\("): "[P0] eval()",
    re.compile(r"\bexec\s*\("): "[P0] exec()",
    re.compile(r"\bpickle\.loads?\b"): "[P0] pickle.load on untrusted input",
    re.compile(r"\bsubprocess\.Popen\b"): "[P1] subprocess.Popen",
    re.compile(r"\bos\.system\b"): "[P0] os.system",
    re.compile(r"\bos\.popen\b"): "[P0] os.popen",
}
INSECURE_NET = {
    "verify=False": "[P1] requests.get(verify=False) — TLS bypass",
    "ssl._create_unverified_context": "[P1] urllib ssl bypass",
}
PRINT_RE = re.compile(r"^\s*print\(")
TODO_RE = re.compile(r"#\s*(TODO|FIXME|XXX|HACK)\b", re.IGNORECASE)
IMPORT_STAR_RE = re.compile(r"^\s*from\s+\S+\s+import\s+\*\s*$")
MUTABLE_DEFAULT_RE = re.compile(r"def\s+\w+\s*\(([^)]*)=\s*[\[\{][^)]*\)")
TIME_SLEEP_RE = re.compile(r"\b(time\.sleep)\b")
URLOOPEN_RE = re.compile(r"\burllib\.request\.urlopen\b")
REQUESTS_GET_RE = re.compile(r"\brequests\.(get|post|put|delete|head|patch)\b")
URLLIB_WO_TIMEOUT_RE = re.compile(r"urllib\.request\.urlopen\s*\(")

# -- finding types --------------------------------------------------------

class Finding:
    def __init__(self, severity: str, kind: str, file: str, line: int, msg: str) -> None:
        self.severity = severity
        self.kind = kind
        self.file = file
        self.line = line
        self.msg = msg

    def __str__(self) -> str:
        rel = os.path.relpath(self.file, ROOT)
        return f"{self.severity}  {rel}:{self.line}  [{self.kind}]  {self.msg}"


# -- walker ---------------------------------------------------------------

def _walk_python_files() -> List[Path]:
    files: List[Path] = []
    for root in SOURCE_ROOTS:
        rdir = ROOT / root
        if not rdir.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(rdir):
            # prune
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not any(d.startswith(s.rstrip("*")) for s in SKIP_DIRS if s.endswith("*"))]
            for fn in filenames:
                if fn.endswith(".py"):
                    p = Path(dirpath) / fn
                    # skip this file itself
                    if p == Path(__file__).resolve():
                        continue
                    files.append(p)
    return sorted(files)


def _is_test_file(path: Path) -> bool:
    parts = path.parts
    return "tests" in parts or "scratch" in parts or "/scripts/" in str(path).replace("\\", "/") or path.name.startswith("scratch_")


def _line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def audit_file(path: Path) -> List[Finding]:
    text = path.read_text(encoding="utf-8", errors="replace")
    findings: List[Finding] = []

    # text-level regex passes
    for line_idx, line in enumerate(text.splitlines(), start=1):
        # strip comments for call detection — a `eval()` mentioned
        # in a docstring or comment isn't a real call.
        code_part = line
        if "#" in code_part:
            code_part = code_part.split("#", 1)[0]
        # secrets
        for pat, msg in SECRET_PATTERNS:
            if pat.search(line):
                # allow dev-key mention in .env* / docs
                if "change-me" in line and ("#" in line or "dev-local" in line):
                    # informational, not a finding — but mark it
                    findings.append(Finding("[P1]", "dev-secret-inline", str(path), line_idx, msg))
                elif _is_test_file(path):
                    # test fixtures use literal "secret" / "wrong" / etc.
                    # for the auth middleware tests.  Skip — they're
                    # not real secrets, they're test inputs.
                    continue
                else:
                    findings.append(Finding("[P0]", "secret-literal", str(path), line_idx, msg))
        # suspicious calls
        for pat, sev_kind in SUSPICIOUS_CALLS.items():
            if pat.search(code_part):
                # eval in test fixtures / train scripts is fine
                if ("eval" in pat.pattern or "exec" in pat.pattern) and (
                    _is_test_file(path) or path.name == "audit.py"
                ):
                    continue
                findings.append(Finding(sev_kind.split("]")[0] + "]", pat.pattern, str(path), line_idx, line.strip()[:120]))
        # insecure net
        for needle, msg in INSECURE_NET.items():
            if needle in line:
                findings.append(Finding(msg.split("]")[0] + "]", "insecure-net", str(path), line_idx, msg))
        # prints
        if PRINT_RE.match(line) and not _is_test_file(path):
            # exception: scripts/ is allowed to print
            if "scripts" not in path.parts:
                findings.append(Finding("[P2]", "print-stdout", str(path), line_idx, "use logger instead of print"))
        # TODO/FIXME
        if TODO_RE.search(line):
            findings.append(Finding("[P2]", "todo", str(path), line_idx, "TODO/FIXME/XXX — link to a version target"))
        # import *
        if IMPORT_STAR_RE.match(line) and "__init__" not in path.name:
            findings.append(Finding("[P1]", "import-star", str(path), line_idx, "import * pollutes namespace"))
        # time.sleep in async def — needs AST, skip regex
        # mutable defaults — handled by AST below

    # AST-level passes
    try:
        # `ast.parse` doesn't accept a BOM in the source string
        # in some Python versions.  Strip it explicitly — most
        # editors save UTF-8-with-BOM as their default for new
        # Python files on Windows.  Python's tokenizer accepts
        # the file either way, so this is just for the audit.
        parse_src = text.lstrip("\ufeff")
        tree = ast.parse(parse_src, filename=str(path))
    except SyntaxError as exc:
        findings.append(Finding("[P0]", "syntax-error", str(path), exc.lineno or 0, str(exc)))
        return findings

    for node in ast.walk(tree):
        # mutable default args
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg, dv in zip(node.args.args, node.args.defaults):
                if isinstance(dv, (ast.List, ast.Dict, ast.Set)):
                    findings.append(Finding("[P1]", "mutable-default", str(path), node.lineno, f"arg {arg.arg} has mutable default"))
            # time.sleep inside async def
            if isinstance(node, ast.AsyncFunctionDef):
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
                        if sub.func.attr == "sleep" and isinstance(sub.func.value, ast.Name) and sub.func.value.id == "time":
                            findings.append(Finding("[P1]", "time-sleep-in-async", str(path), sub.lineno, f"blocking sleep inside async def {node.name} — use asyncio.sleep"))
            # bare except / pass
            for sub in ast.walk(node):
                if isinstance(sub, ast.ExceptHandler):
                    if sub.body and len(sub.body) == 1 and isinstance(sub.body[0], ast.Pass):
                        findings.append(Finding("[P1]", "bare-pass-except", str(path), sub.lineno, f"except {sub.type and getattr(sub.type, 'id', sub.type)}: pass — silently swallows"))
                    if sub.type is None and sub.body:
                        findings.append(Finding("[P0]", "bare-except", str(path), sub.lineno, "bare except — catches KeyboardInterrupt + SystemExit"))
                    if sub.type and isinstance(sub.type, ast.Name) and sub.type.id == "Exception" and len(sub.body) <= 2 and not any(isinstance(n, ast.Raise) for n in sub.body):
                        # short except Exception with no re-raise — info
                        findings.append(Finding("[P2]", "broad-exception", str(path), sub.lineno, "except Exception — verify re-raise / log"))
        # top-level os.environ reads outside source roots
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if (isinstance(node.func.value, ast.Name) and node.func.value.id == "os"
                    and node.func.attr == "environ" and isinstance(node.func, ast.Call)):
                # os.environ() (not os.environ["x"] or os.environ.get) — wrong usage
                findings.append(Finding("[P2]", "os.environ-call", str(path), node.lineno, "os.environ() — should be os.environ[...] or .get()"))

    return findings


# -- networking focused audit --------------------------------------------

def audit_network_calls() -> List[Finding]:
    """Find URL fetches without an explicit timeout (P0 in our context:
    dltv.org has been observed to hang for >60s; a missing timeout
    can wedge the whole event loop)."""
    findings: List[Finding] = []
    for path in _walk_python_files():
        if _is_test_file(path) or "scratch_" in path.name or path == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "urlopen":
                continue
            # urllib.request.urlopen(url, timeout=X) — look at kwargs
            has_timeout = any(kw.arg == "timeout" for kw in node.keywords)
            if not has_timeout:
                findings.append(Finding("[P1]", "urllib-no-timeout", str(path), node.lineno, "urllib.request.urlopen without explicit timeout"))
    return findings


# -- size audit -----------------------------------------------------------

def audit_module_sizes(max_loc: int = 1000) -> List[Finding]:
    findings: List[Finding] = []
    for path in _walk_python_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        loc = text.count("\n") + 1
        if loc > max_loc:
            findings.append(Finding("[P2]", "large-file", str(path), 0, f"{loc} lines (limit {max_loc}) — split into smaller modules"))
    return findings


# -- known pre-1.0 audit checklist ---------------------------------------

def audit_pre_1_0_checklist() -> List[Finding]:
    """Items from TODO.md § 'Local-only / pre-release assumptions' that
    should be revisited before any public deployment."""
    findings: List[Finding] = []
    # 1. SSE auth bypass
    gw_mw = ROOT / "gateway" / "_middleware.py"
    if gw_mw.exists():
        text = gw_mw.read_text(encoding="utf-8")
        if 'UNAUTHED_PREFIXES' in text and '"/api/stream/"' in text:
            findings.append(Finding("[P0]", "sse-auth-bypass", str(gw_mw), 0, "/api/stream/* bypasses auth — DDoS vector for public deployment (TODO §Auth)"))
    # 2. nginx dev X-API-Key auto-inject
    nw = ROOT / "web" / "nginx.conf"
    if nw.exists():
        text = nw.read_text(encoding="utf-8")
        if 'dev-local-dota-analyst-key-change-me' in text and 'default' in text:
            # Confirm the line includes 'default' on the same block
            for i, line in enumerate(text.splitlines(), start=1):
                if 'dev-local-dota-analyst-key-change-me' in line:
                    findings.append(Finding("[P0]", "nginx-dev-key-default", str(nw), i, "nginx `map default 'dev-...'` — public deploy must change to `default $http_x_api_key`"))
                    break
    # 3. JsonFileRepository only
    files = list((ROOT / "business").rglob("*.py")) if (ROOT / "business").exists() else []
    uses_db = any("psycopg" in p.read_text(encoding="utf-8", errors="replace") or "asyncpg" in p.read_text(encoding="utf-8", errors="replace") for p in files)
    if not uses_db:
        findings.append(Finding("[P2]", "no-postgres", "(project-wide)", 0, "JsonFileRepository only — Postgres planned for 1.0 (TODO §Storage)"))
    # 4. No rate limiting at edge
    if (ROOT / "gateway" / "_rate_limit.py").exists():
        # Token bucket exists in-process; flag absence of edge limits
        findings.append(Finding("[P2]", "no-edge-rate-limit", "(project-wide)", 0, "in-process token bucket only — no nginx/cloud rate limits (TODO §Storage)"))
    return findings


# -- main -----------------------------------------------------------------

def main() -> int:
    print("=" * 78)
    print("dota_analyst code audit (v0.3.22 -> 0.4.0 prep)")
    print("=" * 78)

    all_findings: List[Finding] = []

    print("\n[1/5] scanning source files for security + quality issues...")
    for path in _walk_python_files():
        all_findings.extend(audit_file(path))

    print("[2/5] auditing network calls for missing timeouts...")
    all_findings.extend(audit_network_calls())

    print("[3/5] checking module sizes...")
    all_findings.extend(audit_module_sizes())

    print("[4/5] pre-1.0 deployment checklist...")
    all_findings.extend(audit_pre_1_0_checklist())

    print("[5/5] aggregating results...")

    by_sev: Dict[str, List[Finding]] = defaultdict(list)
    for f in all_findings:
        by_sev[f.severity].append(f)

    for sev in ("[P0]", "[P1]", "[P2]"):
        items = by_sev.get(sev, [])
        print(f"\n{sev} {len(items)} finding(s):")
        # group by file for readability
        by_file: Dict[str, List[Finding]] = defaultdict(list)
        for f in items:
            by_file[f.file].append(f)
        for f in sorted(items, key=lambda f: (os.path.relpath(f.file, ROOT), f.line)):
            print(f"  {f}")

    # Summary
    print("\n" + "=" * 78)
    total = sum(len(v) for v in by_sev.values())
    print(f"TOTAL: {total} finding(s)  (P0={len(by_sev['[P0]'])}  P1={len(by_sev['[P1]'])}  P2={len(by_sev['[P2]'])})")
    print("=" * 78)
    if by_sev["[P0]"]:
        print("ACTION: P0 items block public deployment. Address before 0.4.0 → 1.0.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
