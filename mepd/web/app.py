"""FastAPI app: REST for state changes, one SSE stream for live updates.

The client model is "fetch /api/state once, then apply events": every
mutation (REST or a job finishing) is broadcast as an event, so several
browser tabs on one workspace stay in sync and a reconnecting tab just
refetches /api/state.
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
import re
import secrets
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import asynccontextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from mepd.web import chem
from mepd.web.jobs import Broadcaster, JobManager, _reduce_chain_payload
from mepd.web.sessions import Sessions
from mepd.web.operations import OPERATIONS
from mepd.web.results import MINIMA_KINDS, collect_cached, find_entry, summarize
from mepd.web.workspace import HARTREE_TO_KCAL, Workspace, WorkspaceError, is_ts, validate_profile_text
from mepd.web.workspace import find_duplicate as _find_duplicate

STATIC = Path(__file__).parent / "static"


# ------------------------------------------------------------ bodies

class StructureIn(BaseModel):
    text: str = Field(..., description="SMILES (one per line) or xyz text (may be multi-frame)")
    name: Optional[str] = None
    charge: Optional[int] = None
    multiplicity: Optional[int] = None
    optimize: bool = Field(True, description="Minimize at the workspace level of theory once added")


class IdsIn(BaseModel):
    structures: list[str]


class LevelIn(BaseModel):
    profile: Optional[str] = None
    validate_minima: Optional[bool] = None


class StructurePatch(BaseModel):
    name: Optional[str] = None
    charge: Optional[int] = None
    multiplicity: Optional[int] = None
    notes: Optional[str] = None


class EdgeIn(BaseModel):
    source: str
    target: str
    label: str = ""


class EdgePatch(BaseModel):
    label: Optional[str] = None
    reverse: bool = False
    conformers: Optional[dict[str, Optional[str]]] = None   # {structure id: conformer id, or None = lowest}


class JobIn(BaseModel):
    op: str
    structures: list[str] = []
    edges: list[str] = []
    params: dict = {}
    profile: Optional[str] = None
    label: str = ""
    dry_run: bool = False
    source_job: Optional[str] = None  # follow-up operations: the job they build on
    conformers: Optional[dict[str, str]] = None  # {structure id: conformer id} to use instead of the lowest


class ProfileFormIn(BaseModel):
    text: str
    path: str
    value: Any = None


class ImportIn(BaseModel):
    path: str
    op: Optional[str] = None
    charge: int = 0
    multiplicity: int = 1
    title: str = ""


class EntriesImportIn(BaseModel):
    entries: list[str] = Field(..., min_length=1, max_length=200)


class EntryImportIn(BaseModel):
    entry: str
    frames: Literal["endpoints", "all", "ts", "one"] = "endpoints"
    frame: Optional[int] = None
    connect: bool = True


class ProfileIn(BaseModel):
    text: str


class DeleteIn(BaseModel):
    structures: list[str] = []
    edges: list[str] = []


class SessionIn(BaseModel):
    path: Optional[str] = None
    name: Optional[str] = None


def _die_with_parent(parent_pid: int) -> None:
    """Pool-worker initializer: exit when the server does. A server killed
    outright (SIGKILL, `fuser -k`) never shuts its pool down, and spawned
    workers would otherwise linger, idle, holding memory."""
    import signal

    if sys.platform.startswith("linux"):
        try:
            import ctypes

            PR_SET_PDEATHSIG = 1
            ctypes.CDLL("libc.so.6", use_errno=True).prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
        except Exception:
            pass
    if os.getppid() != parent_pid:  # the server died before we got here
        os._exit(0)


class ResultParser:
    """Parses job outputs in worker processes, not the server's threads.

    Reading a big output folder is seconds of pure-Python/numpy work that
    holds the GIL; in a thread it would stall the event loop (SSE, every
    other request) for that long. Spawned (not forked: the server has
    threads) and kept warm, so mepd is imported once per worker."""

    def __init__(self, workers: int = 2):
        self.workers = workers
        self._pool: Optional[ProcessPoolExecutor] = None

    def _get(self) -> ProcessPoolExecutor:
        if self._pool is None:
            self._pool = ProcessPoolExecutor(self.workers, mp_context=multiprocessing.get_context("spawn"),
                                             initializer=_die_with_parent, initargs=(os.getpid(),))
        return self._pool

    async def __call__(self, job: dict, job_dir: Path) -> dict:
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(self._get(), collect_cached, job, job_dir)
        except BrokenProcessPool:
            self._pool = None  # a worker died (e.g. OOM); retry once on a fresh pool
            return await loop.run_in_executor(self._get(), collect_cached, job, job_dir)

    def warm(self) -> None:
        self._get().submit(_warm_worker)

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)


def _warm_worker() -> None:
    import mepd.web.results  # noqa: F401  (pull in mepd/rdkit before the first real request)
    import mepd.chain  # noqa: F401


def create_app(workspace_root: Path, *, max_concurrent: int = 2, auth_token: Optional[str] = None,
               demo=None, demo_password: Optional[str] = None) -> FastAPI:
    """`auth_token`: require a login (see mepd.web.auth) -- mandatory for
    anything reachable beyond localhost.

    `demo` (a mepd.web.demo.DemoPolicy) + `demo_password`: public demo mode.
    `workspace_root` is then the demo root; every visitor gets private
    workspaces under it and the policy's restrictions apply."""
    if demo is not None and not demo_password:
        raise ValueError("demo mode needs a password")
    bus = Broadcaster()
    parse_result = ResultParser()

    def on_finished(manager: JobManager, job: dict) -> None:
        # Parse results off the event loop, then attach the headline/barrier
        # to the job record so list views and edge badges need no extra fetch.
        async def attach() -> None:
            if job["op"] == "optimize" and not job.get("external"):
                await run_in_threadpool(apply_optimization, manager.ws, job)
                bus.publish("workspace", manager.ws.snapshot(), key=str(manager.ws.root))
            try:
                result = await parse_result(job, manager.job_dir(job["id"]))
                job["summary"] = summarize(result)
                if job["status"] == "done" and not job.get("external"):
                    if await run_in_threadpool(attach_conformers, manager.ws, job, result):
                        bus.publish("workspace", manager.ws.snapshot(), key=str(manager.ws.root))
            except Exception as exc:
                job["summary"] = {"headline": f"result not readable: {type(exc).__name__}: {exc}",
                                  "barrier_kcal": None, "counts": {}}
            manager._update(job)
            source = manager.jobs.get(job.get("source_job") or "")
            if source is not None:
                # A follow-up (e.g. vri-check) changed the source job's result:
                # re-read it, and let every open view of it refresh.
                (manager.job_dir(source["id"]) / "result.json").unlink(missing_ok=True)
                try:
                    source["summary"] = summarize(await parse_result(source, manager.job_dir(source["id"])))
                except Exception:
                    pass
                source["result_rev"] = int(source.get("result_rev") or 0) + 1
                manager._update(source)
        asyncio.get_running_loop().create_task(attach())

    def refresh_stale_summaries(manager: JobManager) -> None:
        """Jobs finished under an older result reader keep an outdated
        summary (e.g. a barrier from before IRC-route verification): redo
        those from the files on disk, in the background."""
        stale = [j for j in manager.jobs.values()
                 if j["status"] == "done" and j.get("summary") and ("barrier_verified" not in j["summary"]
                     # ...or from before the edge's TS was recorded (route_ts, for VRI on edges)
                     or (j["op"] in ("ts", "channels") and "route_ts" not in j["summary"]))]
        if stale:
            async def redo() -> None:
                for job in stale:
                    try:
                        job["summary"] = summarize(await parse_result(job, manager.job_dir(job["id"])))
                        manager._update(job)
                    except Exception:
                        continue
            asyncio.get_running_loop().create_task(redo())

    sessions = Sessions(bus, max_concurrent=max_concurrent, on_finished=on_finished,
                        demo=demo, demo_root=Path(workspace_root) if demo is not None else None,
                        on_opened=refresh_stale_summaries)

    # Which session this request works on: the owner's current one, or (demo)
    # the visitor's. Set per request by the middleware below.
    request_key: ContextVar[Optional[str]] = ContextVar("mepd_session_key", default=None)

    def W() -> Workspace:
        return sessions.session(request_key.get()).ws

    def J() -> JobManager:
        return sessions.session(request_key.get()).jobs

    def visitor_of(request: Request) -> Optional[str]:
        return getattr(request.state, "visitor", None)

    def deny_in_demo(what: str) -> None:
        if demo is not None:
            raise HTTPException(403, f"{what} is not available in the demo")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if demo is None:
            await sessions.open(Path(workspace_root), create=True)
        else:
            (Path(workspace_root) / "visitors").mkdir(parents=True, exist_ok=True)
        parse_result.warm()
        yield
        await sessions.stop_all()
        parse_result.shutdown()

    app = FastAPI(title="mepd", lifespan=lifespan)
    app.state.sessions = sessions

    @app.middleware("http")
    async def _bind_session(request: Request, call_next):
        # Runs inside the auth middleware (added after this one), so a demo
        # visitor is already identified here.
        visitor = visitor_of(request)
        if visitor is not None:
            request_key.set(await sessions.ensure_visitor(visitor))
        elif demo is not None and request.url.path.startswith("/api/"):
            return JSONResponse({"detail": "not logged in"}, status_code=401)
        return await call_next(request)

    if demo is not None:
        from mepd.web.auth import VisitorAuth, load_or_create_secret

        vauth = VisitorAuth(demo_password, load_or_create_secret(Path(workspace_root) / ".demo_secret"))
        app.middleware("http")(vauth.middleware)
        app.add_api_route("/login", vauth.login, methods=["GET", "POST"], include_in_schema=False)
        app.add_api_route("/logout", lambda: vauth.logout(), methods=["GET", "POST"], include_in_schema=False)
    elif auth_token:
        from mepd.web.auth import Auth

        auth = Auth(auth_token)
        app.middleware("http")(auth.middleware)
        app.add_api_route("/login", auth.login, methods=["GET", "POST"], include_in_schema=False)
        app.add_api_route("/logout", lambda: auth.logout(), methods=["GET", "POST"], include_in_schema=False)

    @app.middleware("http")
    async def _cache_policy(request: Request, call_next):
        # The app's own modules must be revalidated on every load (cheap 304s
        # via ETag): a browser that heuristically reuses a stale module next
        # to fresh ones fails to link them and shows a blank page after an
        # upgrade. Vendored libraries are pinned by version, so they may be
        # cached.
        response = await call_next(request)
        path = request.url.path
        if path.startswith("/static/vendor/"):
            response.headers.setdefault("Cache-Control", "public, max-age=86400")
        elif path.startswith("/static/") or path == "/":
            response.headers["Cache-Control"] = "no-cache"
        return response

    if demo is not None:
        # Registered last = outermost: must run before the login check, which
        # otherwise answers a plain-http request itself.
        @app.middleware("http")
        async def _https_only(request: Request, call_next):
            # Behind the public proxy: a plain-http visit can be tampered with
            # by the network (injected redirects), so bounce it to https and
            # tell browsers to stay on https (HSTS).
            # Cloudflare reports the visitor's scheme in CF-Visitor (its hop to
            # cloudflared is always https); other proxies use X-Forwarded-Proto.
            cf = request.headers.get("cf-visitor", "")
            visitor_scheme = ("http" if '"http"' in cf else "https") if cf else request.headers.get("x-forwarded-proto")
            if visitor_scheme == "http":
                return RedirectResponse(str(request.url.replace(scheme="https")), status_code=301)
            response = await call_next(request)
            if visitor_scheme == "https":
                response.headers["Strict-Transport-Security"] = "max-age=31536000"
            return response

    @app.exception_handler(WorkspaceError)
    async def _ws_error(_: Request, exc: WorkspaceError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(ValidationError)
    async def _params_error(_: Request, exc: ValidationError):
        msgs = [f"{'.'.join(str(p) for p in e['loc']) or 'params'}: {e['msg']}" for e in exc.errors()]
        return JSONResponse({"detail": "; ".join(msgs)}, status_code=400)

    def publish_ws() -> None:
        bus.publish("workspace", W().snapshot(), key=str(W().root))

    # ------------------------------------------------------------ state
    @app.get("/api/state")
    def state():
        return {
            # Snapshot time (same clock as job 'rev'): the page keeps jobs it
            # heard about after this, which the snapshot cannot contain yet.
            "now_ns": time.time_ns(),
            # Demo visitors see their session's name, never server paths.
            "workspace": {**W().snapshot(), "root": W().root.name if demo is not None else str(W().root)},
            "jobs": J().list(),
            "operations": [demo.adapt_operation(op.describe()) if demo is not None else op.describe()
                           for op in OPERATIONS.values()],
            "profiles": W().profile_names(),
            "level_profile": W().level_profile,
            "validate_minima": W().validate_minima,
            "levels": W().levels(),
            "path_summaries": W().path_summaries(),
            "max_concurrent": J().max_concurrent,
            "auth": bool(auth_token) or demo is not None,
            "demo": demo.public() if demo is not None else None,
            "cpus": os.cpu_count(),
        }

    @app.get("/api/events")
    async def events(request: Request):
        q = bus.subscribe()
        visitor = visitor_of(request)

        async def stream():
            try:
                # `hello` first: a client that doesn't see it within a few
                # seconds is behind a proxy that buffers streams and falls
                # back to /api/poll.
                yield "retry: 2000\n\nevent: hello\ndata: {}\n\n"
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        event, data, key = await asyncio.wait_for(q.get(), timeout=15)
                        # Only this browser's own session (re-resolved each
                        # time: the owner may have switched sessions).
                        if key is not None and key != sessions.key_for(visitor):
                            continue
                        yield f"event: {event}\ndata: {json.dumps(data)}\n\n"
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
            finally:
                bus.unsubscribe(q)

        # no-transform: proxies such as Cloudflare otherwise compress the
        # stream, which means buffering it -- events then never arrive.
        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no",
                                          "Content-Encoding": "identity"})

    # Long-poll fallback for clients whose proxy buffers SSE (e.g. Cloudflare
    # quick tunnels): same events, delivered as ordinary request/responses.
    pollers: dict[str, dict] = {}

    def _gc_pollers() -> None:
        now = time.time()
        for cid in [c for c, p in pollers.items() if now - p["seen"] > 90]:
            bus.unsubscribe(pollers.pop(cid)["queue"])

    @app.get("/api/poll")
    async def poll(request: Request, cid: Optional[str] = None, wait: float = 20.0):
        _gc_pollers()
        visitor = visitor_of(request)
        p = pollers.get(cid) if cid else None
        if p is None or p["visitor"] != visitor:
            cid = secrets.token_urlsafe(12)
            p = pollers[cid] = {"queue": bus.subscribe(), "visitor": visitor, "seen": time.time()}
            return {"cid": cid, "events": [{"event": "hello", "data": {}}]}
        p["seen"] = time.time()
        out = []

        def take(item):
            event, data, key = item
            if key is None or key == sessions.key_for(visitor):
                out.append({"event": event, "data": data})

        try:
            take(await asyncio.wait_for(p["queue"].get(), timeout=max(0.0, min(wait, 25.0))))
            while not p["queue"].empty() and len(out) < 500:
                take(p["queue"].get_nowait())
        except asyncio.TimeoutError:
            pass
        p["seen"] = time.time()
        return {"cid": cid, "events": out}

    # ------------------------------------------------------- structures
    def _add_from_text(text: str, name: Optional[str], charge, mult) -> list[dict]:
        text = text.strip()
        if not text:
            raise WorkspaceError("nothing to add")
        if demo is not None:
            demo.check_text(text)
        todo = []  # (structure, add_structure kwargs): everything parsed before anything is stored
        if chem.looks_like_xyz(text):
            frames = chem.structures_from_xyz_text(text, charge, mult)
            for i, s in enumerate(frames):
                label = name if len(frames) == 1 else (f"{name} {i}" if name else None)
                todo.append((s, {"name": label, "origin": {"kind": "xyz"}}))
        else:
            lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
            for line in lines:
                # "SMILES name" per line, like a .smi file
                smi, _, line_name = line.partition(" ")
                try:
                    s = chem.structure_from_smiles(smi, charge, mult)
                except Exception as exc:
                    raise WorkspaceError(f"could not embed {smi!r}: {exc}") from None
                todo.append((s, {"name": (line_name.strip() or name if len(lines) == 1 else line_name.strip()) or smi,
                                 "origin": {"kind": "smiles", "input": smi}, "smiles": smi}))
        if demo is not None:
            demo.check_structures(len(W().snapshot()["structures"]), [len(s.symbols) for s, _ in todo])
        out = []
        for s, kw in todo:
            added = W().add_or_merge(s, **kw)
            out.append({**added["rec"], "merged": added["merged"], "duplicate": added["duplicate"],
                        "added_conformer": added["conformer"]})
        return out

    def queue_optimization(sids: list[str], conformers: Optional[list] = None) -> list[dict]:
        """Minimize structures at the workspace level of theory: one
        `mepd optimize` job per charge/multiplicity group. Their geometry and
        energy are replaced in place when it finishes (apply_optimization).
        `conformers` (parallel to `sids`): which conformer of each; default
        the representative."""
        ws = W()
        groups: dict[tuple, list[tuple]] = {}
        for i, sid in enumerate(sids):
            rec = ws.structure(sid)
            if is_ts(rec):
                continue  # minimizing a saddle point would destroy it
            cid = conformers[i] if conformers and i < len(conformers) else None
            groups.setdefault((rec["charge"], rec["multiplicity"]), []).append((sid, cid))
        if not groups:
            raise WorkspaceError("nothing to optimize: transition-state structures are never minimized")
        created = []
        for pairs in groups.values():
            ids = [sid for sid, _ in pairs]
            names = ", ".join(dict.fromkeys(ws.structure(i)["name"] for i in ids[:3])) + (
                f" +{len(ids) - 3}" if len(ids) > 3 else "")
            created += J().submit("optimize", structure_ids=ids, edge_ids=[],
                                  params={"validate_minima_with_hessian": ws.validate_minima},
                                  profile=ws.level_profile, label=f"Optimize: {names}",
                                  conformers=[cid for _, cid in pairs])
            ws.set_status(list(dict.fromkeys(ids)), "optimizing")
        return created

    @app.post("/api/structures")
    async def add_structures(body: StructureIn):
        added = await run_in_threadpool(_add_from_text, body.text, body.name, body.charge, body.multiplicity)
        fresh = [a for a in added if not a["duplicate"]]   # a geometry we already had needs no minimization
        if body.optimize and fresh:
            queue_optimization([a["id"] for a in fresh], [a["added_conformer"] for a in fresh])
            added = [{**a, **W().structure(a["id"])} for a in added]
        publish_ws()
        return added

    @app.post("/api/structures/reoptimize")
    async def reoptimize(body: IdsIn):
        if not body.structures:
            raise WorkspaceError("nothing selected")
        jobs_created = queue_optimization(body.structures)
        publish_ws()
        return jobs_created

    @app.put("/api/level")
    async def set_level(body: LevelIn):
        if "profile" in body.model_fields_set:
            W().set_level_profile(body.profile)
        if body.validate_minima is not None:
            W().set_validate_minima(body.validate_minima)
        publish_ws()
        bus.publish("level", {"level_profile": W().level_profile, "levels": W().levels()}, key=str(W().root))
        return {"level_profile": W().level_profile, "levels": W().levels()}

    @app.post("/api/structures/upload")
    async def upload_structures(files: list[UploadFile] = File(...), charge: Optional[int] = Form(None),
                                multiplicity: Optional[int] = Form(None), optimize: bool = Form(True)):
        added = []
        for f in files:
            raw = await f.read(demo.max_upload_bytes + 1 if demo is not None else -1)
            if demo is not None and len(raw) > demo.max_upload_bytes:
                raise WorkspaceError("file too large for the demo")
            text = raw.decode("utf-8", "replace")
            stem = Path(f.filename or "structure").stem
            if f.filename and f.filename.endswith((".smi", ".txt")) and not chem.looks_like_xyz(text):
                added += await run_in_threadpool(_add_from_text, text, None, charge, multiplicity)
            else:
                added += await run_in_threadpool(_add_from_text, text, stem, charge, multiplicity)
        if optimize and added:
            queue_optimization([a["id"] for a in added])
            added = [W().structure(a["id"]) for a in added]
        publish_ws()
        return added

    @app.patch("/api/structures/{sid}")
    def patch_structure(sid: str, body: StructurePatch):
        rec = W().update_structure(sid, **body.model_dump())
        publish_ws()
        return rec

    @app.delete("/api/structures/{sid}")
    def delete_structure(sid: str):
        W().delete_structure(sid)
        publish_ws()
        return {"ok": True}

    @app.get("/api/structures/{sid}/xyz", response_class=PlainTextResponse)
    def structure_xyz(sid: str, conformer: Optional[str] = None):
        return (W().conformer_path(sid, conformer) if conformer else W().structure_path(sid)).read_text()

    @app.delete("/api/structures/{sid}/conformers/{cid}")
    def delete_conformer(sid: str, cid: str):
        rec = W().delete_conformer(sid, cid)
        publish_ws()
        return rec

    @app.post("/api/structures/merge-duplicates")
    def merge_duplicates():
        """Fold nodes that are the same molecule into one node with all
        their conformers (for graphs built before conformers were merged)."""
        out = W().merge_duplicates()
        publish_ws()
        return out

    @app.get("/api/depict")
    def depict(smiles: str, w: int = 220, h: int = 160):
        if demo is not None and len(smiles) > demo.max_smiles_length:
            raise HTTPException(400, "SMILES too long")
        w, h = min(max(w, 40), 800), min(max(h, 40), 800)
        svg = chem.depict_svg(smiles, w, h)
        if svg is None:
            raise HTTPException(404, "cannot depict")
        return Response(svg, media_type="image/svg+xml", headers={"Cache-Control": "max-age=86400"})

    # ------------------------------------------------------------ edges
    @app.post("/api/edges")
    def add_edge(body: EdgeIn):
        rec = W().add_edge(body.source, body.target, label=body.label)
        publish_ws()
        return rec

    @app.patch("/api/edges/{eid}")
    def patch_edge(eid: str, body: EdgePatch):
        rec = W().update_edge(eid, **body.model_dump())
        publish_ws()
        return rec

    @app.delete("/api/edges/{eid}")
    def delete_edge(eid: str):
        W().delete_edge(eid)
        publish_ws()
        return {"ok": True}

    @app.put("/api/positions")
    def put_positions(positions: dict[str, dict]):
        W().set_positions(positions)
        return {"ok": True}  # not broadcast: layout is per-drag noise

    # --------------------------------------------------------- profiles
    @app.get("/api/profiles/{name}", response_class=PlainTextResponse)
    def get_profile(name: str):
        return W().read_profile(name)

    @app.put("/api/profiles/{name}")
    def put_profile(name: str, body: ProfileIn):
        deny_in_demo("Editing compute profiles")
        W().write_profile(name, body.text)
        bus.publish("profiles", W().profile_names(), key=str(W().root))
        return {"ok": True}

    @app.delete("/api/profiles/{name}")
    def delete_profile(name: str):
        deny_in_demo("Deleting compute profiles")
        W().delete_profile(name)
        bus.publish("profiles", W().profile_names(), key=str(W().root))
        return {"ok": True}

    @app.post("/api/profiles/form")
    def profile_form_view(body: ProfileIn):
        """The settings form for a profile's TOML (big choices + the
        Advanced settings those choices use)."""
        from mepd.web import profile_form

        return profile_form.form(body.text)

    @app.post("/api/profiles/form/apply")
    def profile_form_apply(body: ProfileFormIn):
        """Change one form setting; returns the new TOML (not saved) and form."""
        deny_in_demo("Editing compute profiles")
        from mepd.web import profile_form

        return profile_form.apply(body.text, body.path, body.value)

    @app.post("/api/profiles/validate")
    async def validate_profile(body: ProfileIn):
        deny_in_demo("Validating custom profiles")
        return await run_in_threadpool(validate_profile_text, body.text)

    # ------------------------------------------------------------- jobs
    @app.post("/api/jobs")
    async def submit(body: JobIn):
        if demo is not None:
            demo.check_op(body.op, body.params)
            if not body.dry_run:
                demo.check_capacity(J().list())
        # On the loop (not a threadpool): submit wakes the asyncio scheduler.
        # It only snapshots files and builds argv, so it is quick.
        created = J().submit(body.op, structure_ids=body.structures, edge_ids=body.edges,
                             params=body.params, profile=body.profile, label=body.label,
                             dry_run=body.dry_run, source_job_id=body.source_job, conformers=body.conformers)
        if body.op == "optimize" and not body.dry_run:
            W().set_status(body.structures, "optimizing")
            publish_ws()
        return created

    @app.get("/api/jobs/{jid}/vri-viewer", response_class=HTMLResponse)
    def vri_viewer(jid: str):
        """The interactive VRI explorer (`mepd visualize` of a VRI folder),
        with the app's own 3Dmol instead of the CDN copy. Rebuilt only when
        the folder's results change (a follow-up adds checks or a surface)."""
        from mepd.viz import _3DMOL_CDN_SCRIPT

        try:
            from mepd.viz_vri import is_vri_output, load_vri_result, render_vri_html
        except ImportError:
            raise HTTPException(404, "this mepd has no VRI explorer (mepd.viz_vri) yet")

        job = J().get(jid)
        out = Path(job["output_dir"])
        if not is_vri_output(out):
            raise HTTPException(404, "no VRI result in this job (yet)")
        stamp = max((p.stat().st_mtime for p in out.glob("*.json")), default=0.0)
        cache = J().job_dir(jid) / "vri_viewer.html"
        if cache.exists() and cache.stat().st_mtime >= stamp:
            return HTMLResponse(cache.read_text())
        page = render_vri_html(load_vri_result(out), title=job["title"])
        page = page.replace(_3DMOL_CDN_SCRIPT, '<script src="/static/vendor/3Dmol-min.js"></script>')
        try:
            cache.write_text(page)
        except OSError:
            pass
        return HTMLResponse(page)

    @app.post("/api/jobs/import")
    async def import_job(body: ImportIn):
        deny_in_demo("Opening output folders on the server")
        return J().import_external(Path(body.path), op_key=body.op, charge=body.charge,
                                    multiplicity=body.multiplicity, title=body.title)

    @app.get("/api/jobs/{jid}")
    def get_job(jid: str):
        return {"job": J().get(jid), "progress": J().progress_snapshot(jid)}

    @app.get("/api/jobs/{jid}/live/{stream}")
    def get_live_stream(jid: str, stream: str):
        """One live stream in full (a finished minimization's whole replay,
        which the progress updates leave out to stay small)."""
        J().get(jid)
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", stream):
            raise HTTPException(404)
        fp = J().job_dir(jid) / "live" / f"{stream}.json"
        if not fp.is_file():
            raise HTTPException(404)
        return _reduce_chain_payload(json.loads(fp.read_text()), full=True)

    @app.get("/api/jobs/{jid}/log", response_class=PlainTextResponse)
    def job_log(jid: str, which: Literal["stdout", "progress"] = "stdout", offset: int = 0):
        J().get(jid)
        fp = J().job_dir(jid) / ("stdout.log" if which == "stdout" else "progress.log")
        if not fp.exists():
            return PlainTextResponse("", headers={"X-Log-Size": "0"})
        size = fp.stat().st_size
        if offset < 0:
            offset = max(0, size + offset)
        with fp.open("rb") as fh:
            fh.seek(min(offset, size))
            data = fh.read(2_000_000)
        return PlainTextResponse(data.decode("utf-8", "replace"), headers={"X-Log-Size": str(size)})

    @app.get("/api/jobs/{jid}/result")
    async def job_result(jid: str):
        job = J().get(jid)
        return await parse_result(job, J().job_dir(jid))

    @app.post("/api/jobs/{jid}/cancel")
    async def cancel(jid: str):
        return J().cancel(jid)

    @app.post("/api/jobs/{jid}/retry")
    async def retry(jid: str):
        return J().retry(jid)

    @app.delete("/api/jobs/{jid}")
    async def delete_job(jid: str):
        J().delete(jid)
        return {"ok": True}

    @app.get("/api/jobs/{jid}/files")
    def list_files(jid: str):
        out = Path(J().get(jid)["output_dir"])
        if not out.is_dir():
            return []
        return sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())[:5000]

    @app.get("/api/jobs/{jid}/files/{rel:path}")
    def get_file(jid: str, rel: str):
        out = Path(J().get(jid)["output_dir"]).resolve()
        fp = (out / rel).resolve()
        if out not in fp.parents or not fp.is_file():
            raise HTTPException(404)
        return FileResponse(fp)

    def _import_picks(job: dict, group: dict, entry: dict, picks: list[int], connect: bool) -> dict:
        """Add chosen frames of one result entry to the Graph (reusing any
        structure already there); with `connect` and two picks, join them by
        an edge that remembers the result it came from."""
        jid = job["id"]
        frames = entry["frames"]
        added, reused, in_order = [], [], []
        for k in picks:
            f = frames[k]
            (s,) = chem.structures_from_xyz_text(f["xyz"], job.get("charge"), job.get("multiplicity"))
            smiles = chem.perceive_smiles(s)
            ts_frame = (k == entry["ts_index"] and len(frames) > 1) or \
                (len(frames) == 1 and group["kind"] in ("ts", "ts_other", "channel", "alternate", "offtarget"))
            existing = _find_duplicate(W(), smiles, f["energy_hartree"], job.get("level"), jid) if ts_frame else None
            if existing is not None:
                reused.append(existing)
                in_order.append(existing)
                continue
            name = (smiles or chem.formula(s)) + (" [TS]" if ts_frame else "")
            # A minimum of a molecule already in the graph becomes one more
            # conformer of that node (or is recognized as one it has).
            before = set(W().snapshot()["structures"])
            rec = W().add_structure(
                s, name=name, energy=f["energy_hartree"], smiles=smiles,
                # A minimum only counts as one if its Hessian check (when
                # one was run) passed.
                optimized=group["kind"] in MINIMA_KINDS and len(frames) == 1
                and (entry.get("validation") or {}).get("is_minimum", True),
                level=job.get("level"), role="ts" if ts_frame else "minimum",
                validation=entry.get("validation"),
                origin={"kind": "job", "job": jid, "entry": entry["id"], "label": entry["label"], "frame": k})
            (added if rec["id"] not in before else reused).append(rec)
            in_order.append(rec)
        edge = None
        if connect and len(picks) == 2:
            ids = [r["id"] for r in in_order]
            if len(set(ids)) == 2:
                edge = W().add_edge(ids[0], ids[1], label=entry["label"], origin={
                    "kind": "job", "job": jid, "entry": entry["id"], "barrier_kcal": entry.get("barrier_kcal"),
                    "headline": entry.get("note", ""),
                    # IRC-derived edges know their TS (a VRI search can start from it).
                    "group": group.get("kind"), "has_ts": entry.get("ts_index") is not None})
        return {"added": added, "reused": reused, "edge": edge}

    @app.post("/api/jobs/{jid}/import-entry")
    async def import_entry(jid: str, body: EntryImportIn):
        """Pull structures out of a result into the Graph, optionally
        connecting them with an edge that remembers where it came from."""
        job = J().get(jid)
        result = await parse_result(job, J().job_dir(jid))
        try:
            group, entry = find_entry(result, body.entry)
        except KeyError:
            raise HTTPException(404, f"no entry {body.entry!r}") from None
        frames = entry["frames"]
        if body.frames == "endpoints":
            picks = [0, len(frames) - 1] if len(frames) > 1 else [0]
        elif body.frames == "all":
            picks = list(range(len(frames)))
        elif body.frames == "ts":
            picks = [entry["ts_index"] if entry["ts_index"] is not None else 0]
        else:
            if body.frame is None or not 0 <= body.frame < len(frames):
                raise HTTPException(400, "frame out of range")
            picks = [body.frame]
        connect = body.connect and body.frames == "endpoints"
        out = await run_in_threadpool(_import_picks, job, group, entry, picks, connect)
        publish_ws()
        return out

    @app.post("/api/jobs/{jid}/import-entries")
    async def import_entries(jid: str, body: EntriesImportIn):
        """Bulk "Add to Graph": the structure of each chosen entry -- its only
        frame, or the TS of a path -- in one request and one update."""
        job = J().get(jid)
        result = await parse_result(job, J().job_dir(jid))
        chosen = []
        for eid in dict.fromkeys(body.entries):
            try:
                chosen.append(find_entry(result, eid))
            except KeyError:
                raise HTTPException(404, f"no entry {eid!r}") from None
        if demo is not None:
            demo.check_structures(len(W().snapshot()["structures"]), [0] * len(chosen))

        def run() -> dict:
            total = {"added": [], "reused": [], "edge": None}
            for group, entry in chosen:
                k = 0 if len(entry["frames"]) == 1 else (entry["ts_index"] if entry["ts_index"] is not None else 0)
                out = _import_picks(job, group, entry, [k], connect=False)
                total["added"] += out["added"]
                total["reused"] += out["reused"]
            return total

        out = await run_in_threadpool(run)
        publish_ws()
        return out

    # ------------------------------------------------------- bulk / export
    @app.post("/api/delete")
    def delete_many(body: DeleteIn):
        removed = W().delete_many(body.structures, body.edges)
        publish_ws()
        return removed

    @app.get("/api/structures-export", response_class=PlainTextResponse)
    def export_structures(ids: str):
        """Selected structures as one multi-frame xyz (library order kept)."""
        frames = []
        for sid in [i for i in ids.split(",") if i]:
            rec = W().structure(sid)
            s = W().load_structure(sid)
            lines = s.to_xyz().splitlines()
            energy = f" E={rec['energy']:.8f} Eh" if rec.get("energy") is not None else ""
            lines[1] = f"{rec['name']} | {rec.get('smiles') or rec['formula']} | charge={rec['charge']} mult={rec['multiplicity']}{energy}"
            frames.append("\n".join(lines) + "\n")
        return PlainTextResponse("".join(frames), headers={
            "Content-Disposition": 'attachment; filename="structures.xyz"'})

    @app.get("/api/jobs/{jid}/archive")
    def job_archive(jid: str):
        """The whole output folder (plus the job's inputs and logs) as a zip."""
        import tempfile
        import zipfile

        job = J().get(jid)
        out = Path(job["output_dir"])
        jdir = J().job_dir(jid)
        tmp = tempfile.SpooledTemporaryFile(max_size=64 * 1024 * 1024)
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
            if out.is_dir():
                for fp in sorted(out.rglob("*")):
                    if fp.is_file():
                        zf.write(fp, f"output/{fp.relative_to(out)}")
            if not job["external"]:
                for name in ("job.json", "stdout.log", "progress.log"):
                    if (jdir / name).exists():
                        zf.write(jdir / name, name)
                for fp in sorted((jdir / "inputs").glob("*")):
                    zf.write(fp, f"inputs/{fp.name}")
        tmp.seek(0)
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in job["title"])[:60] or jid
        return StreamingResponse(tmp, media_type="application/zip",
                                 headers={"Content-Disposition": f'attachment; filename="{safe}.zip"'})

    # ------------------------------------------------------------ sessions
    @app.get("/api/sessions")
    def list_sessions(request: Request):
        return sessions.listing(visitor_of(request))

    @app.post("/api/sessions/new")
    async def new_session(body: SessionIn, request: Request):
        visitor = visitor_of(request)
        if demo is not None:
            if body.path or not (body.name and body.name.strip()):
                raise WorkspaceError("give the new session a name")
            await sessions.open(sessions.visitor_session_path(visitor, body.name), create=True,
                                must_be_new=True, visitor=visitor)
            return sessions.listing(visitor)
        if body.path:
            path = Path(body.path)
        elif body.name and body.name.strip():
            name = body.name.strip()
            if "/" in name or name.startswith("."):
                raise WorkspaceError("a session name cannot contain '/' or start with '.'")
            path = sessions.default_root() / name
        else:
            raise WorkspaceError("give the new session a name or a directory")
        await sessions.open(path, create=True, must_be_new=True)
        return sessions.listing()

    @app.post("/api/sessions/open")
    async def open_session(body: SessionIn, request: Request):
        visitor = visitor_of(request)
        if demo is not None:
            # Visitors open their own sessions by name only.
            await sessions.open(sessions.visitor_session_path(visitor, body.path or body.name or ""), visitor=visitor)
            return sessions.listing(visitor)
        if not body.path:
            raise WorkspaceError("which workspace directory?")
        await sessions.open(Path(body.path))
        return sessions.listing()

    # ----------------------------------------------------------- static
    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def attach_conformers(ws: Workspace, job: dict, result: dict) -> int:
    """Conformers a finished job sampled (RDKit/CREST) become conformers of
    the node they belong to: a `conformers` job's of its structure, a
    `channels` job's reactant/product pools of its start/end. Energies count
    only when the job minimized them at its level of theory. Returns how
    many were new."""
    targets = job["targets"]["structures"]
    params = job.get("params") or {}
    if job["op"] == "conformers":
        owner = {"conf_": targets[0] if targets else None}
        minimized = params.get("minimize", True)
    elif job["op"] == "channels":
        owner = {"start_conf_": targets[0] if targets else None,
                 "end_conf_": targets[1] if len(targets) > 1 else None}
        minimized = params.get("minimize_ends", True)
    else:
        return 0
    added = 0
    for group in result.get("groups", []):
        if group.get("kind") != "conformers":
            continue
        for entry in group["entries"]:
            sid = next((s for prefix, s in owner.items() if entry["id"].startswith(prefix)), None)
            if sid is None or sid not in ws.snapshot()["structures"] or not entry.get("frames"):
                continue
            frame = entry["frames"][0]
            (s,) = chem.structures_from_xyz_text(frame["xyz"], job.get("charge"), job.get("multiplicity"))
            rec = ws.structure(sid)
            smiles = chem.perceive_smiles(s)
            if smiles and rec.get("smiles") and chem.canonical_key(smiles) != chem.canonical_key(rec["smiles"]):
                continue   # minimized into a different molecule: not a conformer of this one
            energy = frame.get("energy_hartree") if minimized else None
            validation = entry.get("validation")
            _, duplicate = ws.add_conformer(
                sid, s, energy=energy, level=job.get("level") if energy is not None else None,
                optimized=energy is not None and (validation or {}).get("is_minimum", True),
                validation=validation, origin={"kind": "job", "job": job["id"], "entry": entry["id"],
                                               "label": entry["label"], "frame": 0})
            added += not duplicate
    return added


def apply_optimization(ws: Workspace, job: dict) -> None:
    """Write a finished `optimize` job back onto its structures (in place)."""
    sids = job["targets"]["structures"]
    out = Path(job["output_dir"])
    summary = {}
    try:
        summary = {r["index"]: r for r in json.loads((out / "summary.json").read_text())["structures"]}
    except Exception:
        pass
    confs = job.get("target_conformers") or [None] * len(sids)
    for i, sid in enumerate(sids):
        rec = summary.get(i)
        if job["status"] == "done" and rec and rec["converged"] and (out / f"opt_{i}.xyz").exists():
            (s,) = chem.structures_from_xyz_text((out / f"opt_{i}.xyz").read_text())
            validation = {k: rec[k] for k in ("is_minimum", "min_frequency", "rescued", "validation") if k in rec} or None
            ws.replace_geometry(sid, s, energy=rec.get("energy"), level=job.get("level"), validation=validation,
                                conformer=confs[i] if i < len(confs) else None)
        else:
            why = (rec or {}).get("error") or job.get("error") or job["status"]
            ws.set_status([sid], "opt_failed", f"optimization {job['status']}: {str(why).splitlines()[-1][:300]}")

