import base64
import hashlib
import json
import os
import http.client
import threading
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from openai_compatible_bridge import jetbrains_login
from openai_compatible_bridge.providers.jetbrains_oauth import JetBrainsOAuthError


def test_pkce_and_authorization_parameters():
    verifier, challenge = jetbrains_login.make_pkce()
    expected = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    assert challenge == expected
    assert 43 <= len(verifier) <= 128
    parsed = urlsplit(jetbrains_login.authorization_url(62345, challenge, "state"))
    assert parsed.scheme == "https" and parsed.netloc == "junie.jetbrains.com"
    assert parse_qs(parsed.query) == {
        "client_id": ["junie-cli"], "scope": ["offline_access openid jb-authn-service"],
        "state": ["state"], "code_challenge": [challenge], "redirect_uri": ["http://localhost:62345"],
    }


def test_exchange_uses_code_verifier_and_does_not_expose_token(monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            assert kwargs["follow_redirects"] is False
            assert kwargs["trust_env"] is False
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def post(self, url, *, data, headers):
            calls.append((url, data))
            return httpx.Response(200, json={"access_token": "synthetic-access", "refresh_token": "synthetic-refresh", "expires_in": 3600})

    monkeypatch.setattr(jetbrains_login.httpx, "Client", FakeClient)
    state = jetbrains_login.exchange_code("synthetic-code", "synthetic-verifier", 62345)
    assert calls[0][0] == jetbrains_login.TOKEN_URL
    assert calls[0][1] == {"grant_type": "authorization_code", "code": "synthetic-code",
                           "code_verifier": "synthetic-verifier", "client_id": "junie-cli",
                           "redirect_uri": "http://localhost:62345"}
    assert state["refresh_token"] == "synthetic-refresh"
    assert state["refresh_in_progress"] is False


def test_login_persists_fresh_grant_without_printing_tokens(tmp_path, monkeypatch, capsys):
    os.chmod(tmp_path, 0o700)
    path = tmp_path / "new.json"

    class FakeServer:
        server_address = ("127.0.0.1", 62345)
        oauth_result = None
        def handle_request(self):
            self.oauth_result = ("synthetic-code", False)
        def server_close(self):
            pass

    monkeypatch.setattr(jetbrains_login, "callback_server", lambda state: FakeServer())
    monkeypatch.setattr(jetbrains_login, "exchange_code", lambda *args: {
        "access_token": "synthetic-access", "refresh_token": "synthetic-refresh", "expires_at": 100,
        "client_id": "junie-cli", "token_url": jetbrains_login.TOKEN_URL,
    })
    jetbrains_login.login(str(path))
    assert path.stat().st_mode & 0o777 == 0o600
    assert json.loads(path.read_text())["refresh_token"] == "synthetic-refresh"
    output = capsys.readouterr().out
    assert "synthetic-access" not in output and "synthetic-refresh" not in output
    with pytest.raises(JetBrainsOAuthError):
        jetbrains_login.login(str(path))


def test_failed_exchange_does_not_leak_response(monkeypatch):
    class FakeClient:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def post(self, *args, **kwargs):
            return httpx.Response(400, text="synthetic-secret")

    monkeypatch.setattr(jetbrains_login.httpx, "Client", FakeClient)
    with pytest.raises(JetBrainsOAuthError) as error:
        jetbrains_login.exchange_code("code", "verifier", 62345)
    assert "synthetic-secret" not in str(error.value)


def test_callback_rejects_wrong_or_duplicate_state_before_accepting():
    server = jetbrains_login.callback_server("expected-state")
    worker = threading.Thread(target=lambda: [server.handle_request() for _ in range(3)])
    worker.start()
    try:
        port = server.server_address[1]
        for path, status in [
            ("/?code=bad&state=wrong", 400),
            ("/?code=bad&state=expected-state&state=expected-state", 400),
            ("/?code=good&state=expected-state", 200),
        ]:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("GET", path)
            response = conn.getresponse()
            assert response.status == status
            response.read()
            conn.close()
        worker.join(timeout=5)
        assert server.oauth_result == ("good", False)
    finally:
        server.server_close()
