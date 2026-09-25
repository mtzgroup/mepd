"""Sessions = workspaces the server has open.

Owner mode: one session is *current* (what the page shows); the others stay
loaded so their running jobs keep being monitored and finish normally. A
small per-user file remembers recently opened workspaces for the "Open
session" dialog (override its directory with MEPD_WEB_STATE_DIR).

Demo mode: every visitor has their own sessions under
DEMO_ROOT/visitors/<visitor>/<name>/ and their own current one; nothing
else is reachable. Running jobs across all visitors share one concurrency
limit.

Events carry the key (workspace path) of the session they belong to, and
each browser only receives its own session's events.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Optional

from mepd.web.jobs import Broadcaster, JobManager
from mepd.web.workspace import Workspace, WorkspaceError, _atomic_write

_VISITOR_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
_SESSION_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,48}$")


def _state_dir() -> Path:
    base = os.environ.get("MEPD_WEB_STATE_DIR") or os.path.join(
        os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config"), "mepd")
    return Path(base)


class _SessionBus:
    """A session's handle on the broadcaster: tags its events with its key."""

    def __init__(self, bus: Broadcaster, key: str):
        self._bus, self._key = bus, key

    def bind(self, loop) -> None:
        self._bus.bind(loop)

    def publish(self, event: str, data: Any) -> None:
        self._bus.publish(event, data, key=self._key)


class Session:
    def __init__(self, ws: Workspace, jobs: JobManager):
        self.ws, self.jobs = ws, jobs

    def describe(self) -> dict:
        snap = self.ws.snapshot()
        active = sum(1 for j in self.jobs.jobs.values() if j["status"] in ("queued", "running"))
        return {"path": str(self.ws.root), "name": self.ws.root.name,
                "structures": len(snap["structures"]), "edges": len(snap["edges"]),
                "jobs": len(self.jobs.jobs), "active_jobs": active}


class Sessions:
    def __init__(self, bus: Broadcaster, *, max_concurrent: int,
                 on_finished: Callable[[JobManager, dict], Any], demo=None, demo_root: Optional[Path] = None,
                 on_opened: Optional[Callable[[JobManager], Any]] = None):
        self.bus = bus
        self._on_opened = on_opened
        self.max_concurrent = max_concurrent
        self._on_finished = on_finished
        self._open: dict[str, Session] = {}
        self.current_key: Optional[str] = None
        self._recent_fp = _state_dir() / "web_sessions.json"
        self.demo = demo                        # DemoPolicy or None
        self.demo_root = Path(demo_root).resolve() if demo_root else None
        self._visitor_current: dict[str, str] = {}

    # ------------------------------------------------------------ lookup
    def session(self, key: Optional[str]) -> Session:
        return self._open[key or self.current_key]

    def key_for(self, visitor: Optional[str]) -> Optional[str]:
        """The session a request/event stream belongs to right now."""
        if visitor is None:
            return self.current_key
        return self._visitor_current.get(visitor)

    @property
    def current(self) -> Session:
        return self._open[self.current_key]

    # ------------------------------------------------------------ recent
    def _read_recent(self) -> list[dict]:
        try:
            return json.loads(self._recent_fp.read_text())
        except Exception:
            return []

    def _remember(self, path: Path) -> None:
        recent = [r for r in self._read_recent() if r.get("path") != str(path)]
        recent.insert(0, {"path": str(path), "opened": time.time()})
        try:
            self._recent_fp.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(self._recent_fp, json.dumps(recent[:30], indent=1))
        except OSError:
            pass  # a read-only home just means no recent list

    def _describe_path(self, p: Path) -> Optional[dict]:
        if str(p) in self._open:
            return self._open[str(p)].describe()
        if not (p / "workspace.json").exists():
            return None
        try:
            snap = json.loads((p / "workspace.json").read_text())
        except Exception:
            return None
        return {"path": str(p), "name": p.name, "structures": len(snap.get("structures", {})),
                "edges": len(snap.get("edges", {})),
                "jobs": len(list((p / "jobs").glob("*/job.json"))), "active_jobs": 0}

    def listing(self, visitor: Optional[str] = None) -> dict:
        current = self.key_for(visitor)
        if visitor is not None:
            vdir = self._visitor_dir(visitor)
            dirs = [d for d in vdir.iterdir() if d.is_dir()] if vdir.exists() else []
            rows = [(d, d.stat().st_mtime) for d in sorted(dirs, key=lambda d: -d.stat().st_mtime)]
        else:
            rows = [(Path(r["path"]), r.get("opened")) for r in self._read_recent()]
        items = []
        for p, opened in rows:
            info = self._describe_path(p)
            if info is None:
                continue  # moved or deleted since
            if visitor is not None:
                info["path"] = p.name  # visitors never see server paths
            items.append({**info, "opened": opened, "loaded": str(p) in self._open, "current": str(p) == current})
        root = "your demo space" if visitor is not None else str(self.default_root())
        shown_current = Path(current).name if (visitor is not None and current) else current
        return {"current": shown_current, "root": root, "sessions": items}

    def default_root(self) -> Path:
        """Where a new session given only a name is created: next to the
        current workspace."""
        return Path(self.current_key).parent if self.current_key else Path.cwd()

    # -------------------------------------------------------------- demo
    def _visitor_dir(self, visitor: str) -> Path:
        if not self.demo_root or not _VISITOR_RE.match(visitor):
            raise WorkspaceError("invalid visitor")
        return self.demo_root / "visitors" / visitor

    def visitor_session_path(self, visitor: str, name: str) -> Path:
        name = name.strip()
        if not _SESSION_NAME_RE.match(name):
            raise WorkspaceError("session names use letters, digits, spaces, '.', '_' or '-' (max 49)")
        return self._visitor_dir(visitor) / name

    async def ensure_visitor(self, visitor: str) -> str:
        """Open (creating on first visit) the visitor's current session."""
        key = self._visitor_current.get(visitor)
        if key and key in self._open:
            return key
        vdir = self._visitor_dir(visitor)
        existing = sorted((d for d in vdir.glob("*") if (d / "workspace.json").exists()),
                          key=lambda d: -d.stat().st_mtime) if vdir.exists() else []
        await self.open(existing[0] if existing else vdir / "main", create=True, visitor=visitor)
        return self._visitor_current[visitor]

    def _sync_demo_profiles(self, ws: Workspace) -> None:
        """Visitors get exactly the admin's profiles (read-only for them)."""
        src = self.demo_root / "profiles"
        wanted = {fp.name for fp in src.glob("*.toml")}
        for fp in ws.profiles_dir.glob("*.toml"):
            if fp.name not in wanted:
                fp.unlink()
        for fp in src.glob("*.toml"):
            shutil.copyfile(fp, ws.profiles_dir / fp.name)

    # ------------------------------------------------------ concurrency
    def total_running(self) -> int:
        return sum(s.jobs.running_count() for s in self._open.values())

    def _global_slot_free(self) -> bool:
        limit = self.demo.global_concurrency if self.demo else None
        return limit is None or self.total_running() < limit

    def _wake_all(self) -> None:
        for s in self._open.values():
            s.jobs.wake()

    # -------------------------------------------------------------- open
    async def open(self, path: Path, *, create: bool = False, must_be_new: bool = False,
                   visitor: Optional[str] = None) -> Session:
        path = Path(path).expanduser().resolve()
        shown = path.name if visitor is not None else str(path)
        if visitor is not None and self._visitor_dir(visitor).resolve() not in path.parents:
            raise WorkspaceError("not your session")
        exists = (path / "workspace.json").exists()
        if must_be_new and exists:
            raise WorkspaceError(f"{shown} already holds a workspace; open it instead")
        if not exists and not create:
            raise WorkspaceError(f"{shown} is not an mepd workspace (no workspace.json)")
        if must_be_new and path.exists() and not exists and any(path.iterdir()):
            raise WorkspaceError(f"{shown} exists and is not empty; pick a new or empty directory")
        if visitor is not None and not exists and self.demo:
            vdir = self._visitor_dir(visitor)
            n = len([d for d in vdir.glob("*") if (d / "workspace.json").exists()]) if vdir.exists() else 0
            if n >= self.demo.max_sessions:
                raise WorkspaceError(f"the demo allows {self.demo.max_sessions} sessions per visitor")
        key = str(path)
        if key in self._open and not exists:
            # Its folder was removed on disk while loaded: forget the stale copy.
            await self._open.pop(key).jobs.stop()
        if key not in self._open:
            ws = Workspace(path)
            if self.demo_root is not None and visitor is not None:
                self._sync_demo_profiles(ws)
            elif must_be_new and self.current_key:
                # Carry the user's tuned compute profiles over to the new session.
                for fp in self.current.ws.profiles_dir.glob("*.toml"):
                    target = ws.profiles_dir / fp.name
                    if not target.exists():
                        target.write_text(fp.read_text())
            jobs = JobManager(
                ws, _SessionBus(self.bus, key), max_concurrent=self.max_concurrent,
                global_slot_free=self._global_slot_free, on_slot_freed=self._wake_all,
                max_runtime=self.demo.job_timeout_s if self.demo else None,
                op_runtime=self.demo.op_timeout_s if self.demo else None,
            )
            jobs.on_finished = lambda job, _jobs=jobs: self._on_finished(_jobs, job)
            if visitor is None:
                await asyncio.to_thread(ws.ensure_default_profile)
            await jobs.start()
            self._open[key] = Session(ws, jobs)
            if self._on_opened:
                self._on_opened(jobs)
        if visitor is not None:
            self._visitor_current[visitor] = key
        else:
            self.current_key = key
            self._remember(path)
        self.bus.publish("session", {"path": shown}, key=key)
        return self._open[key]

    async def stop_all(self) -> None:
        for s in self._open.values():
            await s.jobs.stop()
