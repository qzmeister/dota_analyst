"""Unit tests for gateway/_session.py — stateless HMAC session tokens."""
import time

import pytest

from gateway._session import (
    SESSION_COOKIE_NAME,
    _CLOCK_SKEW_SEC,
    sign_session_token,
    verify_session_token,
)


SECRET = "test-secret-key-do-not-use-in-prod-please"


def test_sign_then_verify_round_trip():
    token = sign_session_token(SECRET, ttl_sec=60)
    assert verify_session_token(SECRET, token) is True


def test_verify_rejects_wrong_secret():
    token = sign_session_token(SECRET, ttl_sec=60)
    assert verify_session_token("not-the-secret", token) is False


def test_verify_rejects_empty_token():
    assert verify_session_token(SECRET, "") is False
    assert verify_session_token(SECRET, None) is False


def test_verify_rejects_malformed_token():
    # No dot separator
    assert verify_session_token(SECRET, "abc") is False
    # No colon in head
    assert verify_session_token(SECRET, "abc.def") is False
    # Non-integer expiry
    assert verify_session_token(SECRET, "abc:nonce.def") is False
    # Missing nonce
    assert verify_session_token(SECRET, "12345.sig") is False


def test_verify_rejects_expired_token():
    # ttl well past the clock-skew tolerance (-120s, skew is 60s)
    token = sign_session_token(SECRET, ttl_sec=-120)
    assert verify_session_token(SECRET, token) is False


def test_verify_accepts_just_expired_within_skew():
    # ttl=-30s: expires 30s ago, within the 60s skew tolerance
    token = sign_session_token(SECRET, ttl_sec=-30)
    assert verify_session_token(SECRET, token) is True


def test_verify_rejects_tampered_signature():
    token = sign_session_token(SECRET, ttl_sec=60)
    # Flip the last hex char of the signature
    head, sig = token.rsplit(".", 1)
    flipped = sig[:-1] + ("0" if sig[-1] != "0" else "1")
    assert verify_session_token(SECRET, f"{head}.{flipped}") is False


def test_verify_rejects_tampered_expiry():
    token = sign_session_token(SECRET, ttl_sec=60)
    head, sig = token.rsplit(".", 1)
    # Bump the expiry by 1s — the HMAC was over the original
    expires_at, nonce = head.split(":", 1)
    tampered_head = f"{int(expires_at) + 1}:{nonce}"
    assert verify_session_token(SECRET, f"{tampered_head}.{sig}") is False


def test_two_tokens_have_different_nonces():
    # Even with the same ttl, two tokens should differ (HMAC is
    # over `expiry:nonce` and nonce is 16 hex chars of randomness).
    t1 = sign_session_token(SECRET, ttl_sec=60)
    t2 = sign_session_token(SECRET, ttl_sec=60)
    assert t1 != t2
    assert verify_session_token(SECRET, t1) is True
    assert verify_session_token(SECRET, t2) is True


def test_cookie_name_is_stable():
    # The cookie name is part of the wire contract — changing
    # it would log every existing user out on deploy.
    assert SESSION_COOKIE_NAME == "dota_analyst_session"


def test_clock_skew_tolerance_is_bounded():
    # The 60s window is a balance: big enough to absorb NTP
    # drift on a laptop, small enough that a stolen cookie's
    # "replay after I log out" window is still 1 minute.
    assert _CLOCK_SKEW_SEC == 60
