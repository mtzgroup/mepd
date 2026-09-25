"""Token login for `mepd web --auth`.

The server can run jobs as you and read your files, so anything beyond
localhost must be behind a login. One secret token per user (kept in
~/.config/mepd/web_token, mode 0600, so a phone stays logged in across
server restarts) is exchanged once for an HttpOnly, SameSite=Lax cookie
(Lax, not Strict, so a login link opened from another app works; nothing in
the app changes state through a GET):

* GET  /login?token=...  (e.g. from a link or QR code) -> cookie, redirect to /
* POST /login            (form with the token)          -> cookie, redirect to /
* everything else without a valid cookie (or `Authorization: Bearer <token>`)
  -> the login page (HTML) or 401 (API)

The cookie holds an HMAC of the token, not the token itself; rotating the
token (`mepd web --new-token`) logs every device out.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import html
import os
import secrets
import time
from pathlib import Path
from typing import Optional

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

COOKIE = "mepd_session"
COOKIE_MAX_AGE = 30 * 24 * 3600
_OPEN_PATHS = {"/login", "/favicon.ico"}
_OPEN_PREFIXES = ("/static/vendor/fonts/",)  # the login page's fonts; public OFL files


def _open_path(path: str) -> bool:
    return path in _OPEN_PATHS or path.startswith(_OPEN_PREFIXES)


def token_path() -> Path:
    base = os.environ.get("MEPD_WEB_STATE_DIR") or os.path.join(
        os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config"), "mepd")
    return Path(base) / "web_token"


def load_or_create_token(rotate: bool = False) -> str:
    fp = token_path()
    if fp.exists() and not rotate:
        tok = fp.read_text().strip()
        if len(tok) >= 20:
            return tok
    tok = secrets.token_urlsafe(24)
    fp.parent.mkdir(parents=True, exist_ok=True)
    tmp = fp.with_suffix(".tmp")
    tmp.write_text(tok + "\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, fp)
    return tok


class Auth:
    def __init__(self, token: str):
        self._token = token
        self._cookie_value = hmac.new(token.encode(), b"mepd-web-session", hashlib.sha256).hexdigest()
        self._failures: dict[str, list[float]] = {}

    # ------------------------------------------------------------ checks
    def _token_ok(self, candidate: Optional[str]) -> bool:
        return bool(candidate) and hmac.compare_digest(candidate.strip(), self._token)

    def authenticated(self, request: Request) -> bool:
        cookie = request.cookies.get(COOKIE)
        if cookie and hmac.compare_digest(cookie, self._cookie_value):
            return True
        header = request.headers.get("authorization", "")
        return header.startswith("Bearer ") and self._token_ok(header[7:])

    def _throttled(self, client: str) -> bool:
        """At most 10 failed logins per client per 10 minutes."""
        now = time.time()
        recent = [t for t in self._failures.get(client, []) if now - t < 600]
        self._failures[client] = recent
        return len(recent) >= 10

    def _fail(self, client: str) -> None:
        self._failures.setdefault(client, []).append(time.time())

    # ---------------------------------------------------------- responses
    def _set_cookie(self, request: Request, response: Response) -> Response:
        secure = request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"
        response.set_cookie(COOKIE, self._cookie_value, max_age=COOKIE_MAX_AGE, httponly=True,
                            samesite="lax", secure=secure, path="/")
        return response

    async def login(self, request: Request) -> Response:
        client = request.client.host if request.client else "?"
        if request.method == "POST":
            form = await request.form()
            candidate = form.get("token")
        else:
            candidate = request.query_params.get("token")
            if candidate is None:
                return login_page()
        if self._throttled(client):
            return login_page("Too many attempts. Wait a few minutes.", status=429)
        if not self._token_ok(candidate):
            self._fail(client)
            await asyncio.sleep(0.5)  # blunt online guessing further
            return login_page("That token is not right.", status=401)
        # Redirect so the token never stays in the address bar or history.
        return self._set_cookie(request, RedirectResponse("/", status_code=303))

    def logout(self) -> Response:
        response = RedirectResponse("/login", status_code=303)
        response.delete_cookie(COOKIE, path="/")
        return response

    async def middleware(self, request: Request, call_next):
        path = request.url.path
        if _open_path(path) or self.authenticated(request):
            return await call_next(request)
        return _unauthenticated(path)


def _unauthenticated(path: str, demo: bool = False) -> Response:
    if path.startswith("/api/"):
        return JSONResponse({"detail": "not logged in"}, status_code=401)
    if path == "/":
        return login_page(demo=demo)
    return Response(status_code=401)


class VisitorAuth:
    """Demo mode: one shared password; every browser that logs in becomes a
    separate visitor (random id in a signed cookie) with its own workspace.
    The signing secret persists in the state dir, so visitors keep their
    workspace across server restarts (until they clear cookies)."""

    COOKIE = "mepd_visitor"

    def __init__(self, password: str, secret: bytes):
        self._password = password
        self._secret = secret
        self._failures: dict[str, list[float]] = {}

    def _sign(self, visitor: str) -> str:
        return hmac.new(self._secret, visitor.encode(), hashlib.sha256).hexdigest()

    def visitor(self, request: Request) -> Optional[str]:
        raw = request.cookies.get(self.COOKIE, "")
        vid, _, sig = raw.partition(".")
        if vid and sig and hmac.compare_digest(sig, self._sign(vid)):
            return vid
        return None

    def _throttled(self, client: str) -> bool:
        now = time.time()
        recent = [t for t in self._failures.get(client, []) if now - t < 600]
        self._failures[client] = recent
        # per client, and a global ceiling in case a proxy hides client IPs
        total = sum(len(v) for v in self._failures.values())
        return len(recent) >= 10 or total >= 200

    async def login(self, request: Request) -> Response:
        client = request.client.host if request.client else "?"
        if request.method != "POST":
            return login_page(demo=True)
        form = await request.form()
        candidate = str(form.get("password") or "")
        if self._throttled(client):
            return login_page("Too many attempts. Wait a few minutes.", status=429, demo=True)
        if not (candidate and hmac.compare_digest(candidate, self._password)):
            self._failures.setdefault(client, []).append(time.time())
            await asyncio.sleep(0.5)
            return login_page("That password is not right.", status=401, demo=True)
        vid = self.visitor(request) or secrets.token_urlsafe(16)
        response = RedirectResponse("/", status_code=303)
        secure = request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"
        response.set_cookie(self.COOKIE, f"{vid}.{self._sign(vid)}", max_age=COOKIE_MAX_AGE, httponly=True,
                            samesite="lax", secure=secure, path="/")
        return response

    def logout(self) -> Response:
        # Drops the visitor cookie: logging in again starts a new private
        # workspace (the old one stays on disk for the admin).
        response = RedirectResponse("/login", status_code=303)
        response.delete_cookie(self.COOKIE, path="/")
        return response

    async def middleware(self, request: Request, call_next):
        path = request.url.path
        vid = self.visitor(request)
        if vid:
            request.state.visitor = vid
            return await call_next(request)
        if _open_path(path):
            return await call_next(request)
        return _unauthenticated(path, demo=True)


def load_or_create_secret(fp: Path) -> bytes:
    if fp.exists():
        return fp.read_bytes()
    fp.parent.mkdir(parents=True, exist_ok=True)
    data = secrets.token_bytes(32)
    tmp = fp.with_suffix(".tmp")
    tmp.write_bytes(data)
    os.chmod(tmp, 0o600)
    os.replace(tmp, fp)
    return data


def login_page(error: str = "", status: int = 200, demo: bool = False) -> HTMLResponse:
    msg = f'<p class="err">{html.escape(error)}</p>' if error else ""
    return HTMLResponse(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>mepd · log in</title>
<style>
  @font-face {{ font-family:"Source Sans 3 Variable"; font-weight:200 900; font-display:swap; src:url(/static/vendor/fonts/source-sans-3-latin-wght-normal.woff2) format("woff2-variations"); }}
  @font-face {{ font-family:"Source Serif 4 Variable"; font-weight:200 900; font-display:swap; src:url(/static/vendor/fonts/source-serif-4-latin-wght-normal.woff2) format("woff2-variations"); }}
  :root {{ color-scheme: light dark; --bg:#f8f7f3; --panel:#fcfcfa; --text:#202a33; --heading:#17324a; --muted:#5e6d78; --accent:#3868b8; --accent-strong:#17324a; --border:#c9d1d8; --err:#b23b3b; }}
  @media (prefers-color-scheme: dark) {{ :root {{ --bg:#0f1d2a; --panel:#132434; --text:#e9edf1; --heading:#fcfcfa; --muted:#a9bacb; --accent:#7fa6df; --accent-strong:#b1cbef; --border:#3a536b; --err:#ee8a84; }} }}
  body {{ margin:0; min-height:100vh; display:grid; place-items:center; background:var(--bg); color:var(--text);
         font:16px/1.5 "Source Sans 3 Variable", "Segoe UI", system-ui, sans-serif; padding:16px; box-sizing:border-box; }}
  form {{ background:var(--panel); border:1px solid var(--border); border-radius:4px; padding:28px; width:min(380px,100%); box-sizing:border-box; }}
  h1 {{ font:450 30px/1.1 "Source Serif 4 Variable", Georgia, serif; letter-spacing:-.035em; color:var(--heading); margin:0 0 10px; }}
  p {{ color:var(--muted); margin:0 0 16px; font-size:15px; }}
  input {{ width:100%; box-sizing:border-box; font:inherit; padding:11px 12px; border-radius:3px; border:1px solid var(--border);
          background:transparent; color:var(--text); margin-bottom:12px; }}
  input:focus {{ outline:none; border-color:var(--accent); }}
  button {{ width:100%; font:inherit; font-weight:600; padding:11px; border:1px solid var(--accent); border-radius:3px; background:var(--accent); color:#fff; cursor:pointer; }}
  button:hover {{ background:var(--accent-strong); border-color:var(--accent-strong); }}
  .err {{ color:var(--err); font-weight:600; }}
</style></head><body>
<form method="post" action="/login">
  <h1>mepd</h1>
  <p>{"Enter the demo password you were given. You get your own private workspace." if demo else "Enter the access token printed by <code>mepd web --auth</code>."}</p>
  {msg}
  <input name="{"password" if demo else "token"}" type="password" autocomplete="current-password" autofocus placeholder="{"password" if demo else "access token"}" required>
  <button type="submit">Log in</button>
</form></body></html>""", status_code=status)
