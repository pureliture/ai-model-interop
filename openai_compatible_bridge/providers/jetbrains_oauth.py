"""명시적으로 제공된 독립 OAuth 자격 증명의 보수적인 갱신 저장소."""

from __future__ import annotations

import asyncio
import fcntl
import json
import math
import os
import stat
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx


class JetBrainsOAuthError(Exception):
    """민감한 원본 응답이나 자격 증명을 포함하지 않는 오류."""


class JetBrainsOAuthStore:
    def __init__(self, path: str, *, timeout: float = 60, skew: float = 60) -> None:
        self.path = Path(path)
        self.timeout = timeout
        self.skew = skew

    async def access_token(self) -> str:
        return await asyncio.to_thread(self._access_token)

    def _private_stat(self, path: Path, *, directory: bool) -> None:
        info = path.lstat()
        kind = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
        if not kind or info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise JetBrainsOAuthError("OAuth 저장소의 소유권 또는 권한이 안전하지 않습니다.")
        if not directory and info.st_nlink != 1:
            raise JetBrainsOAuthError("OAuth 저장소 파일의 하드링크는 허용되지 않습니다.")

    def _write(self, state: dict) -> None:
        fd, name = tempfile.mkstemp(prefix=".oauth-", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(state, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self.path)
            dirfd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(dirfd)
            finally:
                os.close(dirfd)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def _access_token(self) -> str:
        try:
            self._private_stat(self.path.parent, directory=True)
            flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
            lock_path = self.path.with_name(self.path.name + ".lock")
            lock_fd = os.open(lock_path, flags, 0o600)
            try:
                self._private_stat(lock_path, directory=False)
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                self._private_stat(self.path, directory=False)
                with self.path.open(encoding="utf-8") as stream:
                    state = json.load(stream)
                if not isinstance(state, dict) or state.get("refresh_in_progress"):
                    raise JetBrainsOAuthError("OAuth 갱신 상태를 복구할 수 없어 독립 자격 증명을 다시 발급해야 합니다.")
                access = state.get("access_token")
                expiry = state.get("expires_at")
                if (isinstance(access, str) and access and isinstance(expiry, (int, float))
                        and not isinstance(expiry, bool) and math.isfinite(expiry)
                        and expiry > time.time() + self.skew):
                    return access
                endpoint, client_id, refresh = (state.get(k) for k in ("token_url", "client_id", "refresh_token"))
                if not all(isinstance(v, str) and v for v in (endpoint, client_id, refresh)):
                    raise JetBrainsOAuthError("OAuth 설정이 불완전합니다.")
                parsed = urlsplit(endpoint)
                if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
                    raise JetBrainsOAuthError("OAuth token_url은 사용자 정보가 없는 HTTPS URL이어야 합니다.")
                secret = state.get("client_secret")
                if secret is not None and (not isinstance(secret, str) or not secret):
                    raise JetBrainsOAuthError("OAuth client_secret이 올바르지 않습니다.")
                form = {"grant_type": "refresh_token", "client_id": client_id, "refresh_token": refresh}
                if secret is not None:
                    form["client_secret"] = secret
                # 전송 직전 durable 표시: 응답 유실이나 프로세스 중단 후 이전 토큰 재사용을 금지한다.
                state["refresh_in_progress"] = True
                self._write(state)
                requested_at = time.time()
                with httpx.Client(timeout=self.timeout, follow_redirects=False, trust_env=False) as client:
                    response = client.post(endpoint, data=form, headers={"Accept": "application/json"})
                if response.status_code != 200:
                    raise JetBrainsOAuthError("OAuth 갱신이 실패했습니다. 독립 자격 증명을 다시 발급하세요.")
                data = response.json()
                if not isinstance(data, dict):
                    raise JetBrainsOAuthError("OAuth 갱신 응답이 올바르지 않습니다.")
                new_access = data.get("access_token")
                new_refresh = data.get("refresh_token", refresh)
                lifetime = data.get("expires_in")
                if (not isinstance(new_access, str) or not new_access or not isinstance(new_refresh, str)
                        or not new_refresh or isinstance(lifetime, bool) or not isinstance(lifetime, (int, float))
                        or not math.isfinite(lifetime) or lifetime <= 0
                        or str(data.get("token_type", "Bearer")).lower() != "bearer"):
                    raise JetBrainsOAuthError("OAuth 갱신 응답이 올바르지 않습니다.")
                state.update(access_token=new_access, refresh_token=new_refresh,
                             expires_at=requested_at + lifetime, refresh_in_progress=False)
                self._write(state)
                if state["expires_at"] <= time.time():
                    raise JetBrainsOAuthError("OAuth access token이 이미 만료되었습니다.")
                return new_access
            finally:
                os.close(lock_fd)
        except JetBrainsOAuthError:
            raise
        except Exception:
            raise JetBrainsOAuthError("OAuth 저장소 또는 갱신 요청에 실패했습니다. 저장소 상태를 확인하세요.") from None
