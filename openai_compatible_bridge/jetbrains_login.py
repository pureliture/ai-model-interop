"""Junie OAuth PKCE 별도 로그인. 사용자가 명시적으로 실행할 때만 네트워크를 사용한다."""

from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import hmac
import math
import os
import secrets
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx

from openai_compatible_bridge.providers.jetbrains_oauth import JetBrainsOAuthError, JetBrainsOAuthStore

AUTH_URL = "https://junie.jetbrains.com/cli-auth"
TOKEN_URL = "https://oauth.account.jetbrains.com/oauth2/token"
CLIENT_ID = "junie-cli"
SCOPE = "offline_access openid jb-authn-service"
PORTS = range(62345, 62365)


def make_pkce() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(32)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return verifier, base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def authorization_url(port: int, challenge: str, state: str) -> str:
    return AUTH_URL + "?" + urlencode({
        "client_id": CLIENT_ID, "scope": SCOPE, "state": state,
        "code_challenge": challenge, "redirect_uri": f"http://localhost:{port}",
    })


def callback_server(state: str) -> HTTPServer:
    class LoopbackServer(HTTPServer):
        def get_request(self):
            connection, address = super().get_request()
            connection.settimeout(5)
            return connection, address

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def do_GET(self) -> None:
            parsed = urlsplit(self.path)
            params = parse_qs(parsed.query, keep_blank_values=True)
            states = params.get("state", [])
            codes = params.get("code", [])
            errors = params.get("error", [])
            if (parsed.path != "/" or len(states) != 1
                    or not hmac.compare_digest(states[0], state)
                    or not ((len(codes) == 1 and codes[0] and not errors)
                            or (len(errors) == 1 and errors[0] and not codes))):
                self.send_error(400)
                return
            self.server.oauth_result = (codes[0] if codes else None, bool(errors))
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write("인증 응답을 받았습니다. 이 창을 닫고 터미널을 확인하세요.".encode())

    for port in PORTS:
        try:
            server = LoopbackServer(("127.0.0.1", port), Handler)
            server.oauth_result = None
            return server
        except OSError:
            continue
    raise JetBrainsOAuthError("OAuth callback 포트를 열 수 없습니다.")


def exchange_code(code: str, verifier: str, port: int) -> dict:
    started = time.time()
    try:
        with httpx.Client(timeout=30, follow_redirects=False, trust_env=False) as client:
            response = client.post(TOKEN_URL, data={
                "grant_type": "authorization_code", "code": code, "code_verifier": verifier,
                "client_id": CLIENT_ID, "redirect_uri": f"http://localhost:{port}",
            }, headers={"Accept": "application/json"})
        if response.status_code != 200:
            raise ValueError
        data = response.json()
        access, refresh, lifetime = data["access_token"], data["refresh_token"], data["expires_in"]
        if (not isinstance(access, str) or not access or not isinstance(refresh, str) or not refresh
                or isinstance(lifetime, bool) or not isinstance(lifetime, (int, float))
                or not math.isfinite(lifetime) or lifetime <= 0 or started + lifetime <= time.time()
                or str(data.get("token_type", "Bearer")).lower() != "bearer"):
            raise ValueError
        return {"token_url": TOKEN_URL, "client_id": CLIENT_ID, "access_token": access,
                "refresh_token": refresh, "expires_at": started + lifetime, "refresh_in_progress": False}
    except Exception:
        raise JetBrainsOAuthError("OAuth 코드 교환에 실패했습니다. 새 로그인이 필요합니다.") from None


def login(path: str, *, wait_seconds: float = 300) -> None:
    store = JetBrainsOAuthStore(path)
    try:
        store._private_stat(store.path.parent, directory=True)
        lock_path = store.path.with_name(store.path.name + ".lock")
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            store._private_stat(lock_path, directory=False)
            fcntl.flock(fd, fcntl.LOCK_EX)
            if os.path.lexists(store.path):
                raise JetBrainsOAuthError("대상 파일이 이미 있습니다. 새 전용 경로를 지정하세요.")
            verifier, challenge = make_pkce()
            state = secrets.token_urlsafe(32)
            server = callback_server(state)
            try:
                port = server.server_address[1]
                print("브라우저에서 다음 주소를 열어 직접 승인하세요:", flush=True)
                print(authorization_url(port, challenge, state), flush=True)
                deadline = time.monotonic() + wait_seconds
                while server.oauth_result is None:
                    server.timeout = max(0, deadline - time.monotonic())
                    if not server.timeout:
                        raise JetBrainsOAuthError("OAuth 승인 대기 시간이 초과됐습니다.")
                    server.handle_request()
                code, failed = server.oauth_result
                if failed:
                    raise JetBrainsOAuthError("OAuth 승인이 거부됐습니다.")
                token_state = exchange_code(code, verifier, port)
                store._write(token_state)
            finally:
                server.server_close()
        finally:
            os.close(fd)
    except JetBrainsOAuthError:
        raise
    except Exception:
        raise JetBrainsOAuthError("OAuth 로그인 또는 저장소 접근에 실패했습니다.") from None


def main() -> int:
    parser = argparse.ArgumentParser(description="JetBrains OAuth의 별도 grant를 발급받습니다.")
    parser.add_argument("--output", required=True, help="전용 0700 디렉터리 안의 아직 존재하지 않는 파일 경로")
    args = parser.parse_args()
    try:
        login(args.output)
    except JetBrainsOAuthError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print("별도 OAuth 자격 증명을 저장했습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
