"""FastAPI app: REST for state changes, one SSE stream for live updates.

The client model is "fetch /api/state once, then apply events": every
mutation (REST or a job finishing) is broadcast as an event, so several
browser tabs on one workspace stay in sync and a reconnecting tab just
refetches /api/state.
"""

from __future__ import annotations

import asyncio
import logging
import contextlib
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

from mepd.reaction_smiles import ReactionSmilesError, is_reaction_smiles, reaction_structures

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


class DesignNew(BaseModel):
    scratch: bool = False                 # an empty design, built atom by atom
    smiles: Optional[str] = None
    xyz: Optional[str] = None
    name: Optional[str] = None
    charge: Optional[int] = None          # of an xyz (bond orders are assigned for this total charge)
    multiplicity: Optional[int] = None


class DesignLoad(BaseModel):
    structure: str
    conformer: Optional[str] = None


class DesignEdit(BaseModel):
    op: dict
    side: Literal["reactant", "product"] = "reactant"   # reaction mode: which molecule was clicked
    linked: bool = True                                 # reaction mode: repeat the edit on the other side


class DesignReactionPut(BaseModel):
    product_molblock: str
    amap: list[int]


class DesignPut(BaseModel):
    molblock: Optional[str] = None
    reaction: Optional[DesignReactionPut] = None        # reaction mode undo/redo: the product side too
    name: Optional[str] = None
    charge: Optional[int] = None
    multiplicity: Optional[int] = None


class DesignSearch(BaseModel):
    op: Literal["ts", "channels"] = "ts"
    profile: Optional[str] = None
    params: Optional[dict] = None


class DesignMinimize(BaseModel):
    profile: Optional[str] = None
    params: dict = {}


class EdgeIn(BaseModel):
    source: str
    target: str
    label: str = ""


class EdgePatch(BaseModel):
    label: Optional[str] = None
    reverse: bool = False
    conformers: Optional[dict[str, Optional[str]]] = None   # {structure id: conformer id, or None = lowest}


def _solvent_list() -> list[dict]:
    from mepd.solvation import SOLVENTS

    return [{"key": v.key, "label": v.label, "kind": v.kind, "epsilon": v.epsilon, "bp_c": v.bp_c}
            for v in SOLVENTS.values()]


class SetupsIn(BaseModel):
    setups: list[dict] = []
    active: Optional[str] = None


class SetupCompareIn(BaseModel):
    base: str
    other: str


class SetupFillIn(BaseModel):
    mode: str = "single-point"


class JobIn(BaseModel):
    op: str
    structures: list[str] = []
    edges: list[str] = []
    params: dict = {}
    profile: Optional[str] = None
    label: str = ""
    dry_run: bool = False
    source_job: Optional[str] = None  # follow-up operations: the job they build on
    extends: Optional[str] = None     # Sample more paths from a TS search: the page it adds to
    # {structure id: conformer id} to use instead of the lowest; None = the lowest
    # (the run form sends every endpoint, None for the ones left on 'lowest').
    conformers: Optional[dict[str, Optional[str]]] = None


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
    profile: Optional[str] = None     # the workspace profile it was computed at (else the one kept with it)


class EntriesImportIn(BaseModel):
    entries: list[str] = Field(..., min_length=1, max_length=200)


class EntryImportIn(BaseModel):
    entry: str
    frames: Literal["endpoints", "all", "ts", "one"] = "endpoints"
    frame: Optional[int] = None
    connect: bool = True


class ProfileIn(BaseModel):
    text: str


class KineticsIn(BaseModel):
    initial: dict = {}                  # structure id -> M
    held: list[str] = []                # constant feed
    temperature: float = Field(298.15, gt=0)
    time_s: float = Field(3600.0, gt=0)
    target: Optional[str] = None
    include_unverified: bool = False
    control: bool = True
    control_what: Literal["amount", "rate"] = "amount"
    sweep: list[float] = []


class ProposeIn(BaseModel):
    reactants: list[str]
    n_break: int = Field(2, ge=0, le=3)
    n_form: int = Field(2, ge=0, le=3)


class ComposeIn(BaseModel):
    reactants: list[str]
    products: list[str] = []
    proposal: Optional[dict] = None   # one of /api/reactions/propose's proposals, with its complex_xyz


class SandboxIn(BaseModel):
    counts: dict[str, int] = {}       # species id -> how many in the interactive reactor (packed)...
    structure: Optional[str] = None   # ...or one structure's geometry as it is (a complex)
    temperature: float = 800.0
    radius: Optional[float] = None


class ComplexIn(BaseModel):
    counts: dict[str, int]            # species id -> how many
    method: str = "packed"            # mepd.complexes.METHODS: packed at once, the others as a job
    keep: int = 3
    profile: Optional[str] = None


class DeleteIn(BaseModel):
    structures: list[str] = []
    edges: list[str] = []
    reactions: list[str] = []


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
            procs = list((getattr(self._pool, "_processes", None) or {}).values())
            self._pool.shutdown(wait=False, cancel_futures=True)
            # A worker still importing mepd (or stuck) would otherwise keep
            # the exiting process waiting on it forever.
            for p in procs:
                p.terminate()
            self._pool = None


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

    async def page_result(manager: JobManager, job: dict) -> dict:
        """A job's result as its page shows it: a TS search or channels run
        with Sample more paths runs includes theirs."""
        view = manager.result_view(job) if manager.page_job(job) is job else job
        return await parse_result(view, manager.job_dir(job["id"]))

    def on_finished(manager: JobManager, job: dict) -> None:
        # Parse results off the event loop, then attach the headline/barrier
        # to the job record so list views and edge badges need no extra fetch.
        async def attach() -> None:
            if job["op"] == "optimize" and not job.get("external"):
                await run_in_threadpool(apply_optimization, manager.ws, job)
                if job.get("qmmm"):
                    from mepd.web import qmmm as web_qmmm

                    await run_in_threadpool(web_qmmm.refresh_edges_of, manager.ws, manager.jobs,
                                            job["targets"]["structures"])
                bus.publish("workspace", manager.ws.snapshot(), key=str(manager.ws.root))
            if job["op"] == "tsopt" and job.get("qmmm") and job["status"] == "done" and not job.get("external"):
                from mepd.web import qmmm as web_qmmm

                if await run_in_threadpool(web_qmmm.attach_ts, manager.ws, job, manager.jobs):
                    bus.publish("workspace", manager.ws.snapshot(), key=str(manager.ws.root))
            if job["op"] == "design-optimize":
                await run_in_threadpool(apply_design_optimization, manager.ws, job)
                bus.publish("workspace", manager.ws.snapshot(), key=str(manager.ws.root))
            if job["op"] in ("nanoreactor", "nanoreactor-more"):
                from mepd.web.nanoreactor import adopt_nanoreactor

                base = manager.page_job(job)   # Run longer: its nanoreactor's reactions, updated in place
                if await run_in_threadpool(adopt_nanoreactor, manager.ws, base, final=True):
                    manager._update(base)
                    bus.publish("workspace", manager.ws.snapshot(), key=str(manager.ws.root))
                if job["status"] == "done" and not job.get("external"):
                    from mepd.web.nanoreactor import spawn_ts_searches

                    spawn_ts_searches(manager, base)
            if job["op"] == "retrosynthesis" and job["status"] == "done":
                from mepd.web.retro import adopt_best_route

                # The best route only: every species of every route was a
                # wall of unoptimized nodes; the others are a click away.
                if await run_in_threadpool(adopt_best_route, manager.ws, job):
                    manager._update(job)
                    bus.publish("workspace", manager.ws.snapshot(), key=str(manager.ws.root))
            if job["op"] == "complex" and job["status"] == "done" and not job.get("external"):
                made = await run_in_threadpool(adopt_complexes, manager.ws, job)
                if made:
                    try:
                        queue_optimization(made)
                    except WorkspaceError:
                        pass
                    bus.publish("workspace", manager.ws.snapshot(), key=str(manager.ws.root))
            if job["op"] == "qmmm-build" and job["status"] == "done" and not job.get("external"):
                from mepd.web import qmmm as web_qmmm

                sid = await run_in_threadpool(web_qmmm.adopt_build, manager.ws, job)
                if sid:
                    try:
                        queue_optimization([sid])
                    except WorkspaceError:
                        pass
                    bus.publish("workspace", manager.ws.snapshot(), key=str(manager.ws.root))
            if job["op"] in ("qmmm-reaction", "qmmm-embed") and job["status"] == "done" and not job.get("external"):
                from mepd.web import qmmm as web_qmmm

                adopt = web_qmmm.adopt_reaction if job["op"] == "qmmm-reaction" else web_qmmm.adopt_embed
                made = await run_in_threadpool(adopt, manager.ws, job)
                minima = [made.get(k) for k in ("start", "end") if made.get(k)]
                if made.get("role") == "minimum" and made.get("structure"):
                    minima.append(made["structure"])
                ts = made.get("ts") or (made.get("structure") if made.get("role") == "ts" else None)
                with contextlib.suppress(WorkspaceError):
                    if minima:
                        queue_optimization(minima)
                with contextlib.suppress(WorkspaceError):
                    if ts:   # the gas-phase TS, re-optimized as a TS in the solvent, with its IRC
                        manager.submit("tsopt", structure_ids=[ts], edge_ids=[], params={"irc": True},
                                       profile=manager.ws.level_profile, label="Re-optimize the TS in solvent")
                bus.publish("workspace", manager.ws.snapshot(), key=str(manager.ws.root))
            if job["op"] == "graph-enumeration" and job["status"] == "done":
                if await run_in_threadpool(adopt_expansion_steps, manager, job):
                    bus.publish("workspace", manager.ws.snapshot(), key=str(manager.ws.root))
            if job["op"] == "design-tsopt":
                try:
                    result = await parse_result(job, manager.job_dir(job["id"]))
                except Exception:
                    result = {"groups": [], "headline": "result not readable"}
                await run_in_threadpool(apply_design_tsopt, manager.ws, job, result)
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
            page = manager.page_job(job)
            if page is not job and page.get("id") != job.get("source_job"):
                # Sample more paths from a TS search: its page now shows these paths too.
                (manager.job_dir(page["id"]) / "result.json").unlink(missing_ok=True)
                try:
                    page["summary"] = summarize(await page_result(manager, page))
                except Exception:
                    pass
                page["result_rev"] = int(page.get("result_rev") or 0) + 1
                manager._update(page)
            source = manager.jobs.get(job.get("source_job") or "")
            if source is not None:
                # A follow-up (e.g. vri-check) changed the source job's result:
                # re-read it, and let every open view of it refresh.
                (manager.job_dir(source["id"]) / "result.json").unlink(missing_ok=True)
                try:
                    source["summary"] = summarize(await page_result(manager, source))
                except Exception:
                    pass
                source["result_rev"] = int(source.get("result_rev") or 0) + 1
                manager._update(source)
        asyncio.get_running_loop().create_task(attach())

    def refresh_stale_summaries(manager: JobManager) -> None:
        """Jobs finished under an older result reader keep an outdated
        summary (e.g. a barrier from before IRC-route verification): redo
        those from the files on disk, in the background."""
        # Nanoreactor reactions imported by an older reader lack what the reaction card shows.
        from mepd.web.nanoreactor import adopt_nanoreactor

        nano = [j for j in manager.jobs.values() if j["op"] == "nanoreactor" and j["status"] == "done"
                and any(r.get("origin", {}).get("job") == j["id"]
                        and ("ladder" not in r
                             # ends that changed bonds on optimization, from before those were kept when they differ
                             or (not r.get("edge") and r.get("complex_reason")))
                        for r in manager.ws.snapshot().get("reactions", {}).values())]
        for job in nano:
            try:
                if adopt_nanoreactor(manager.ws, job, final=True):
                    manager._update(job)
            except Exception:
                logging.getLogger(__name__).exception("could not refresh the reactions of %s", job["id"])
        # ...and each finished run's reactions with the MD events that are them
        # (adopted before events carried their reaction's key, a card could
        # play another reaction's event).
        from mepd.web.nanoreactor import _sync_events

        synced = False
        for job in [j for j in manager.jobs.values() if j["op"] == "nanoreactor" and j["status"] == "done"]:
            try:
                synced |= _sync_events(manager.ws, job, Path(job.get("output_dir") or ""))
            except Exception:
                logging.getLogger(__name__).exception("could not match the MD events of %s", job["id"])
        if nano or synced:
            bus.publish("workspace", manager.ws.snapshot(), key=str(manager.ws.root))
        # QM/MM TSs re-optimized before their edges learned their barriers.
        from mepd.web import qmmm as web_qmmm

        edges = manager.ws.snapshot()["edges"].values()
        known = {(e.get("origin") or {}).get("qmmm_ts", {}).get("job") for e in edges}
        attached = False
        for job in [j for j in manager.jobs.values() if j["op"] == "tsopt" and j.get("qmmm")
                    and j["status"] == "done" and j["id"] not in known]:
            try:
                attached |= bool(web_qmmm.attach_ts(manager.ws, job, manager.jobs))
            except Exception:
                logging.getLogger(__name__).exception("could not attach the QM/MM TS of %s", job["id"])
        # ...and QM/MM edges with a TS but no barrier yet (their ends may have
        # gained energies at its level since, e.g. IRC ends added from a result).
        pending = [eid for eid, e in manager.ws.snapshot()["edges"].items()
                   if (e.get("origin") or {}).get("qmmm_ts") and (e.get("origin") or {}).get("barrier_kcal") is None]
        for eid in pending:
            try:
                attached |= web_qmmm.edge_barrier(manager.ws, manager.jobs, eid) is not None
            except Exception:
                logging.getLogger(__name__).exception("could not compute the QM/MM barrier of %s", eid)
        if attached:
            bus.publish("workspace", manager.ws.snapshot(), key=str(manager.ws.root))
        stale = [j for j in manager.jobs.values()
                 if j["status"] == "done" and j.get("summary") and ("barrier_verified" not in j["summary"]
                     # ...or from before the edge's TS was recorded (route_ts, for VRI on edges)
                     or (j["op"] in ("ts", "channels") and "route_ts" not in j["summary"])
                     # ...or from before multistep routes were told from direct steps (Explore's edges)
                     or (j["op"] in ("ts", "channels", "channels-more") and "n_steps" not in j["summary"])
                     # ...or a solvent comparison from before reaction energies were kept
                     or (j["op"] == "solvent" and "gas_reaction_kcal" not in (j["summary"].get("conditions") or {})))]
        if stale:
            async def redo() -> None:
                for job in stale:
                    try:
                        job["summary"] = summarize(await page_result(manager, job))
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
            "solvents": _solvent_list(),
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
                if is_reaction_smiles(smi):
                    # Both ends, atoms in the same order, joined by an edge.
                    try:
                        start, end, pair = reaction_structures(smi, charge, mult)
                    except ReactionSmilesError as exc:
                        raise WorkspaceError(str(exc)) from None
                    rs, ps = smi.partition(">")[0], smi.rpartition(">")[2]
                    origin = {"kind": "reaction", "input": smi, "mapping": pair.source, "mapped": pair.mapped_smiles}
                    todo.append((start, {"name": rs, "origin": origin, "smiles": rs, "_edge": "start",
                                         "_label": line_name.strip() or name or ""}))
                    todo.append((end, {"name": ps, "origin": origin, "smiles": ps, "_edge": "end"}))
                    continue
                try:
                    s = chem.structure_from_smiles(smi, charge, mult)
                except Exception as exc:
                    why = chem.smiles_problem(smi)
                    raise WorkspaceError(f"Not a valid SMILES: {smi} ({why})" if why else
                                         f"Could not build a 3D structure for {smi}: "
                                         f"{str(exc).strip().splitlines()[0][:200]}") from None
                todo.append((s, {"name": (line_name.strip() or name if len(lines) == 1 else line_name.strip())
                                 or chem.common_name(smi) or smi,
                                 "origin": {"kind": "smiles", "input": smi}, "smiles": smi}))
        if demo is not None:
            demo.check_structures(len(W().snapshot()["structures"]), [len(s.symbols) for s, _ in todo])
        out, start, label = [], None, ""
        skip = set()
        for k, (s, kw) in enumerate(todo):
            # A reaction with several molecules on a side: its molecules as species,
            # the two ends as the reaction's complexes (see compose.reaction_from_endpoints).
            if kw.get("_edge") == "start" and k + 1 < len(todo) and todo[k + 1][1].get("_edge") == "end":
                from mepd.web.compose import reaction_from_endpoints

                made = reaction_from_endpoints(W(), s, todo[k + 1][0], origin=kw["origin"], label=kw.get("_label") or "")
                if made is not None:
                    for res in made["added"]:
                        out.append({**res["rec"], "merged": res["merged"], "duplicate": res["duplicate"],
                                    "added_conformer": res["conformer"]})
                    out[-1]["edge"], out[-1]["reaction"] = made["edge"], made["reaction"]["id"]
                    skip.update({k, k + 1})
        for k, (s, kw) in enumerate(todo):
            if k in skip:
                continue
            role = kw.pop("_edge", None)
            label = kw.pop("_label", label)
            added = W().add_or_merge(s, **kw)
            out.append({**added["rec"], "merged": added["merged"], "duplicate": added["duplicate"],
                        "added_conformer": added["conformer"]})
            if role == "start":
                start = added
            elif role == "end" and start is not None:
                ws = W()
                if start["rec"]["id"] != added["rec"]["id"]:
                    edge = ws.add_edge(start["rec"]["id"], added["rec"]["id"], label=label,
                                       origin={"kind": "reaction", "input": kw.get("origin", {}).get("input")})
                    # The conformers built for this reaction, whose atoms line up
                    # (not an older conformer these geometries turned out to repeat).
                    if start["conformer"] and added["conformer"] and not start["duplicate"] and not added["duplicate"]:
                        with contextlib.suppress(Exception):
                            ws.update_edge(edge["id"], conformers={start["rec"]["id"]: start["conformer"],
                                                                    added["rec"]["id"]: added["conformer"]})
                    out[-1]["edge"] = edge["id"]
                start = None
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
            groups.setdefault((rec["charge"], rec["multiplicity"], rec.get("qmmm")), []).append((sid, cid))
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

    # ------------------------------------------------------------ QM/MM
    @app.get("/api/qmmm/systems/{sysid}")
    def qmmm_system(sysid: str):
        from mepd.web.qmmm import region_view

        ws = W()
        sysrec = ws.snapshot()["qmmm_systems"].get(sysid)
        if sysrec is None:
            raise HTTPException(404, f"no QM/MM system {sysid!r}")
        return region_view(ws.qmmm_region(sysid), sysrec)

    @app.post("/api/qmmm/preview")
    async def qmmm_preview(body: dict):
        from mepd.web import qmmm as web_qmmm

        try:
            return await run_in_threadpool(web_qmmm.preview, W(), body.get("structure", ""), body)
        except ValueError as exc:
            raise WorkspaceError(str(exc)) from None

    @app.post("/api/qmmm/systems")
    async def qmmm_create(body: dict):
        """A QM/MM system: from a structure in the graph (body.structure),
        or a TeraChem input on this machine (body.terachem)."""
        from mepd.web import qmmm as web_qmmm

        if body.get("terachem"):
            deny_in_demo("Reading files on the server")
        try:
            if body.get("terachem"):
                out = await run_in_threadpool(web_qmmm.from_terachem, W(), body["terachem"],
                                              body.get("mm") or "amber", body.get("active_radius"))
                sid = out["structure"]
            else:
                out = await run_in_threadpool(web_qmmm.create, W(), body.get("structure", ""), body,
                                              demo.qmmm_limits() if demo is not None else None)
                sid = body.get("structure")
        except ValueError as exc:
            raise WorkspaceError(str(exc)) from None
        if body.get("optimize") and sid:
            with contextlib.suppress(WorkspaceError):
                queue_optimization([sid])
        publish_ws()
        return out

    @app.post("/api/qmmm/upload")
    async def qmmm_upload(file: UploadFile = File(...), qm_atoms: str = Form(...),
                          charge: Optional[int] = Form(None), multiplicity: Optional[int] = Form(None),
                          qm_charge: Optional[int] = Form(None), active_radius: Optional[float] = Form(6.0),
                          mm: str = Form("gfnff"), optimize: bool = Form(True)):
        from mepd.web import qmmm as web_qmmm

        raw = await file.read(demo.max_upload_bytes + 1 if demo is not None else -1)
        if demo is not None and len(raw) > demo.max_upload_bytes:
            raise WorkspaceError("file too large for the demo")
        if demo is not None:      # (its atoms: qmmm_limits, below)
            demo.check_structures(len(W().snapshot()["structures"]), [0])
        body = {"qm_atoms": qm_atoms, "charge": charge, "multiplicity": multiplicity,
                "qm_charge": qm_charge if qm_charge is not None else charge, "active_radius": active_radius or None,
                "mm": mm, "qm_multiplicity": multiplicity}
        try:
            out = await run_in_threadpool(web_qmmm.from_upload, W(), raw.decode("utf-8", "replace"),
                                          file.filename or "system.xyz", body,
                                          demo.qmmm_limits() if demo is not None else None)
        except ValueError as exc:
            raise WorkspaceError(str(exc)) from None
        if optimize:
            with contextlib.suppress(WorkspaceError):
                queue_optimization([out["structure"]])
        publish_ws()
        return out

    @app.put("/api/qmmm/systems/{sysid}")
    async def qmmm_update(sysid: str, body: dict):
        from mepd.web import qmmm as web_qmmm

        try:
            out = await run_in_threadpool(web_qmmm.update, W(), sysid, body,
                                          demo.qmmm_limits() if demo is not None else None)
        except ValueError as exc:
            raise WorkspaceError(str(exc)) from None
        publish_ws()
        return out

    @app.get("/api/qmmm/check")
    async def qmmm_check_structure(structure: str, conformer: Optional[str] = None):
        """Checks of one QM/MM structure against its system's reference."""
        from mepd.web import qmmm as web_qmmm

        ws = W()
        rec = ws.structure(structure)
        if not rec.get("qmmm"):
            raise WorkspaceError("not a QM/MM structure")
        fp = ws.conformer_path(structure, conformer) if conformer else ws.structure_path(structure)
        region = ws.qmmm_region(rec["qmmm"])
        frames = [region.reference or fp.read_text(), fp.read_text()]
        return await run_in_threadpool(web_qmmm.check_frames, ws, rec["qmmm"], frames)

    @app.get("/api/jobs/{jid}/qmmm-check")
    async def qmmm_check_job(jid: str, entry: str):
        """Per-frame QM/MM checks of one result entry (and its energy split,
        when a QM/MM energy split follow-up has run on it)."""
        from mepd.web import qmmm as web_qmmm

        job = J().get(jid)
        if not job.get("qmmm"):
            raise WorkspaceError("not a QM/MM job")
        result = await page_result(J(), job)
        try:
            _, e = find_entry(result, entry)
        except KeyError:
            raise HTTPException(404, f"no entry {entry!r}") from None
        frames = [f["xyz"] for f in e["frames"]]
        report = await run_in_threadpool(web_qmmm.check_frames, W(), job["qmmm"], frames,
                                         [f.get("energy_hartree") for f in e["frames"]])
        report["split"] = web_qmmm.energy_split(W(), J().jobs, jid, entry)
        return report

    # ------------------------------------------------------------ design
    def _store_design(info: dict, *, name=None, source=None, charge=None, multiplicity=None, keep=None,
                      reaction=None) -> dict:
        """Save a molecule as the design. An edit clears the energy/level (it
        is no longer the minimized geometry) and keeps name/source/overrides."""
        old = keep or {}
        new_rx = reaction if reaction is not None else old.get("reaction")

        def auto_name(smiles, rx):
            return f"{smiles}>>{rx['product'].get('smiles')}" if rx and smiles else smiles

        if name is None and old.get("name") in (None, old.get("smiles"), (old.get("source") or {}).get("input"),
                                                auto_name(old.get("smiles"), old.get("reaction"))):
            name = auto_name(info.get("smiles"), new_rx) or old.get("name")   # an automatic name follows the molecule
        design = {
            "molblock": info["molblock"], "smiles": info.get("smiles"), "formula": info.get("formula"),
            "natoms": info.get("natoms"),
            "name": name if name is not None else old.get("name"),
            "source": source if source is not None else old.get("source"),
            # Charge and spin follow the formal charges / radicals, plus whatever
            # was set by hand (or came with a loaded structure) kept as an
            # offset: adding Mg2+ to a neutral structure makes it +2.
            "charge_offset": (charge - info["charge"]) if charge is not None else int(old.get("charge_offset") or 0),
            "multiplicity_offset": (multiplicity - info["multiplicity"]) if multiplicity is not None
            else int(old.get("multiplicity_offset") or 0),
            "energy": None, "level": None, "warnings": info.get("warnings", []),
        }
        design["charge"] = info["charge"] + design["charge_offset"]
        design["multiplicity"] = max(1, info["multiplicity"] + design["multiplicity_offset"])
        design["coverage"] = _coverage_warnings(info["molblock"])
        rx = reaction if reaction is not None else old.get("reaction")
        if rx:
            # Reaction mode: the product shares the charge offset and spin.
            prod = rx["product"]
            rx = {**rx, "product": {**prod, "charge": prod["formal_charge"] + design["charge_offset"]}}
            rx["balanced"] = rx.get("balanced", True)
            design["coverage"] = sorted(set(design["coverage"]) | set(_coverage_warnings(prod["molblock"])))
        design["reaction"] = rx or None
        return W().set_design(design)

    def _reaction_part(info: dict, amap: list, mapping=None, balanced=True, note=None) -> dict:
        prod = {k: info.get(k) for k in ("molblock", "smiles", "formula", "natoms")}
        prod["formal_charge"] = info["charge"]
        prod["warnings"] = info.get("warnings", [])
        return {"product": prod, "amap": list(amap), "mapping": mapping, "balanced": balanced, "note": note}

    def _level_text() -> Optional[str]:
        ws = W()
        try:
            return ws.read_profile(ws.level_profile) if ws.level_profile else None
        except Exception:
            return None

    def _coverage_warnings(molblock: str) -> list[str]:
        from mepd.web import coverage, design

        try:
            symbols = [a.GetSymbol() for a in design.read(molblock).GetAtoms()]
            return coverage.element_warnings(_level_text(), symbols)
        except Exception:
            return []

    def _current_design(*, atoms: bool = False) -> dict:
        d = W().design
        if not d:
            raise WorkspaceError("there is no design yet: start one from scratch, from SMILES, or from the graph")
        if atoms and not d.get("natoms"):
            raise WorkspaceError("the design has no atoms yet: click in the 3D view to drop the first one")
        return d

    @app.get("/api/design/groups")
    def design_groups():
        from mepd.web import design

        return list(design.GROUPS)

    @app.post("/api/design/new")
    def design_new(body: DesignNew):
        from mepd.web import design

        if body.scratch:
            out = _store_design(design.empty(), name=body.name or "New molecule", source={"kind": "scratch"},
                                keep={}, charge=body.charge, multiplicity=body.multiplicity)
            publish_ws()
            return out
        if body.smiles and is_reaction_smiles(body.smiles.strip()):
            from mepd.web import design_reaction

            pair = design_reaction.new_pair(body.smiles.strip())
            out = _store_design(pair["reactant"], name=body.name or body.smiles.strip(),
                                source={"kind": "reaction", "input": body.smiles.strip()}, keep={},
                                reaction=_reaction_part(pair["product"], pair["amap"], pair["mapping"]))
            publish_ws()
            return out
        if body.xyz and not body.smiles:
            from mepd.web import chem

            try:
                frames = [s.to_xyz() for s in chem.structures_from_xyz_text(body.xyz)]
            except Exception:
                frames = [body.xyz]   # bare "El x y z" lines, or malformed: the single-structure path says why
            if len(frames) > 2:
                raise WorkspaceError(f"that xyz holds {len(frames)} structures: Design opens one, or two as a "
                                     "reaction (reactant, then product). Add many at once in Explore (+ Add).")
            if len(frames) == 2:
                from mepd.web import design_reaction

                if body.multiplicity is not None and body.multiplicity < 1:
                    raise WorkspaceError("multiplicity must be at least 1")
                pair = design_reaction.pair_from_xyz(frames, body.charge or 0)
                out = _store_design(pair["reactant"], name=body.name or "reaction", source={"kind": "reaction-xyz"},
                                    keep={}, charge=body.charge, multiplicity=body.multiplicity,
                                    reaction=_reaction_part(pair["product"], pair["amap"], pair["mapping"]))
                publish_ws()
                return out
        if body.smiles:
            info, source = design.from_smiles(body.smiles), {"kind": "smiles", "input": body.smiles.strip()}
            name = body.name or body.smiles.strip()
        elif body.xyz:
            info, warnings = design.from_xyz(body.xyz, body.charge or 0)
            info["warnings"] = warnings
            source, name = {"kind": "xyz"}, body.name or "design"
        else:
            raise WorkspaceError("give a SMILES or an xyz")
        if body.multiplicity is not None and body.multiplicity < 1:
            raise WorkspaceError("multiplicity must be at least 1")
        out = _store_design(info, name=name, source=source, keep={},
                            charge=body.charge if body.xyz else None,
                            multiplicity=body.multiplicity if body.xyz else None)
        publish_ws()
        return out

    @app.post("/api/design/load")
    def design_load(body: DesignLoad):
        """A graph structure (a conformer of it, or its lowest) into the Design tab."""
        from mepd.web import design

        ws = W()
        rec = ws.structure(body.structure)
        xyz = (ws.conformer_path(rec["id"], body.conformer) if body.conformer else ws.structure_path(rec["id"])).read_text()
        info, warnings = design.from_xyz(xyz, rec["charge"])
        info["warnings"] = warnings
        out = _store_design(info, name=f"{rec['name']} (edited)", charge=rec["charge"],
                            multiplicity=rec["multiplicity"], keep={},
                            source={"kind": "graph", "structure": rec["id"], "conformer": body.conformer,
                                    "name": rec["name"], "ts": is_ts(rec)})
        publish_ws()
        return out

    @app.post("/api/design/edit")
    def design_edit(body: DesignEdit):
        from mepd.web import design

        d = _current_design()
        if d.get("reaction"):
            from mepd.web import design_reaction

            rx = d["reaction"]
            res = design_reaction.linked_edit(d["molblock"], rx["product"]["molblock"], rx["amap"], body.side,
                                              body.op, body.linked)
            part = _reaction_part(res["product"], res["amap"], rx.get("mapping"), res["balanced"], res["note"])
            out = _store_design(res["reactant"], keep=d, reaction=part)
            publish_ws()
            return {**out, "changed": res["changed"].get("reactant", []),
                    "changed_product": res["changed"].get("product", []), "note": res["note"]}
        info = design.edit(d["molblock"], body.op)
        charge = None
        if body.op.get("op") == "place" and body.op.get("xyz") and body.op.get("charge") is not None:
            # A placed xyz brings the charge it was given, whatever formal
            # charges its bonds could be assigned.
            count = max(1, min(int(body.op.get("count", 1)), 30))
            charge = int(d["charge"]) + int(body.op["charge"]) * count
        out = _store_design(info, keep=d, charge=charge)
        publish_ws()
        return out

    @app.post("/api/design/clean")
    def design_clean():
        from mepd.web import design

        d = _current_design(atoms=True)
        rx = d.get("reaction")
        if rx:
            p = design.clean(rx["product"]["molblock"])   # same atoms, new coordinates: the map holds
            out = _store_design(design.clean(d["molblock"]), keep=d,
                                reaction={**rx, **_reaction_part(p, rx["amap"], rx.get("mapping"), rx.get("balanced", True))})
        else:
            out = _store_design(design.clean(d["molblock"]), keep=d)
        publish_ws()
        return out

    @app.put("/api/design")
    def design_put(body: DesignPut):
        """Set the molblock (undo/redo), the name, or a charge/multiplicity
        set by hand (fields left out stay as they are)."""
        from mepd.web import design

        d = _current_design()
        info = design.describe(design.read(body.molblock or d["molblock"]), sanitized=False)
        if body.multiplicity is not None and body.multiplicity < 1:
            raise WorkspaceError("multiplicity must be at least 1")
        reaction = None
        if body.reaction is not None and d.get("reaction"):
            rx = d["reaction"]
            p = design.describe(design.read(body.reaction.product_molblock), sanitized=False)
            ok = -1 not in body.reaction.amap and sorted(body.reaction.amap) == list(range(p["natoms"]))
            reaction = _reaction_part(p, body.reaction.amap, rx.get("mapping"), ok and p["natoms"] == info["natoms"])
        out = _store_design(info, name=body.name, keep=d, charge=body.charge, multiplicity=body.multiplicity,
                            reaction=reaction)
        if body.molblock is None and d.get("energy") is not None and body.charge is None and body.multiplicity is None:
            out = W().set_design({**out, "energy": d["energy"], "level": d["level"]})   # a rename keeps the energy
        publish_ws()
        return out

    @app.delete("/api/design")
    def design_clear():
        W().set_design(None)
        publish_ws()
        return {"ok": True}

    @app.get("/api/design/xyz", response_class=PlainTextResponse)
    def design_xyz(side: str = "reactant"):
        from mepd.web import design

        d = _current_design()
        if side == "product":
            if not d.get("reaction"):
                raise HTTPException(404, "this design is one molecule, not a reaction")
            return design.to_xyz(_product_in_reactant_order(d), d["charge"], d["multiplicity"])
        return design.to_xyz(d["molblock"], d["charge"], d["multiplicity"])

    def _product_in_reactant_order(d: dict) -> str:
        """The product molblock with its atoms in the reactant's order."""
        from rdkit import Chem

        from mepd.web import design, design_reaction

        order = design_reaction.product_order(d["reaction"]["amap"])
        return Chem.MolToMolBlock(Chem.RenumberAtoms(design.read(d["reaction"]["product"]["molblock"]), order))

    @app.post("/api/design/minimize")
    async def design_minimize(body: DesignMinimize):
        if _current_design(atoms=True).get("reaction"):
            raise WorkspaceError("a reaction design is minimized when its path search starts: use Search TS")
        params = {"validate_minima_with_hessian": W().validate_minima, **(body.params or {})}
        created = J().submit("design-optimize", structure_ids=[], edge_ids=[], params=params,
                             profile=body.profile if body.profile is not None else W().level_profile)
        return created

    @app.get("/api/design/coverage")
    def design_coverage():
        """Which elements the workspace level of theory covers (None = unknown)."""
        from mepd.web import coverage

        name, covered = coverage.coverage(_level_text())
        return {"method": name, "elements": sorted(covered) if covered is not None else None}

    @app.get("/api/design/species")
    def design_species():
        from mepd.web import design

        return list(design.SPECIES)

    @app.post("/api/design/tsopt")
    async def design_tsopt(body: DesignMinimize):
        if _current_design(atoms=True).get("reaction"):
            raise WorkspaceError("Optimize as TS takes one structure (a TS guess), not a reaction")
        return J().submit("design-tsopt", structure_ids=[], edge_ids=[], params={"irc": True, **(body.params or {})},
                          profile=body.profile if body.profile is not None else W().level_profile)

    @app.post("/api/design/ts-to-graph")
    async def design_ts_to_graph():
        """The TS the design converged to, and its IRC's reactant and product
        (which carry whatever catalyst/solvent the design added), into the
        graph: a TS node and the two ends joined by an edge with the barrier."""
        d = _current_design()
        last = d.get("last_ts") or {}
        if not last.get("ok"):
            raise WorkspaceError("the design has no converged TS yet: run Optimize as TS first")
        job = J().get(last["job"])
        result = await parse_result(job, J().job_dir(job["id"]))

        def run() -> dict:
            out = {"ts": None, "edge": None, "added": [], "reused": []}
            for group in result["groups"]:
                if group["kind"] == "ts" and group["entries"]:
                    r = _import_picks(job, group, group["entries"][0], [0], connect=False)
                    out["ts"] = (r["added"] + r["reused"])[0]
                if group["kind"] == "irc" and group["entries"]:
                    e = group["entries"][0]
                    from mepd.web.compose import reaction_from_endpoints

                    ends = [chem.structures_from_xyz_text(e["frames"][k]["xyz"], job.get("charge"), job.get("multiplicity"))[0]
                            for k in (0, len(e["frames"]) - 1)]
                    made = reaction_from_endpoints(
                        W(), ends[0], ends[1], origin={"kind": "design", "label": "Design TS"},
                        energies=(e["frames"][0].get("energy_hartree"), e["frames"][-1].get("energy_hartree")),
                        level=job.get("level"), ts={"job": job["id"], "entry": e["id"], "barrier_kcal": e.get("barrier_kcal")})
                    if made is not None:   # the catalyst/solvent the design added: a reaction between complexes
                        out["edge"], out["reaction"] = made["edge"], made["reaction"]["id"]
                        continue
                    r = _import_picks(job, group, e, [0, len(e["frames"]) - 1], connect=True)
                    out["edge"] = r["edge"]
                    out["added"] += r["added"]
                    out["reused"] += r["reused"]
            return out

        out = await run_in_threadpool(run)
        publish_ws()
        return out

    @app.post("/api/design/to-graph")
    def design_to_graph():
        """The design as a graph node (or, if the graph has that molecule, as
        one more conformer of it)."""
        from mepd.web import design

        d = _current_design(atoms=True)
        if d.get("reaction"):
            out = _reaction_to_graph(d)
            publish_ws()
            return out
        (s,) = chem.structures_from_xyz_text(design.to_xyz(d["molblock"], d["charge"], d["multiplicity"]),
                                              d["charge"], d["multiplicity"])
        ts = d.get("role") == "ts"   # a converged TS stays a saddle point (never merged, never minimized)
        name = d.get("name") or d.get("smiles") or None
        if ts and name and not name.endswith("[TS]"):
            name += " [TS]"
        added = W().add_or_merge(
            s, name=name, energy=d.get("energy"), level=d.get("level"),
            optimized=d.get("energy") is not None and not ts, smiles=None, role="ts" if ts else "minimum",
            origin={"kind": "design", "label": "Design", "source": d.get("source")})
        publish_ws()
        return {**added["rec"], "merged": added["merged"], "duplicate": added["duplicate"],
                "added_conformer": added["conformer"]}

    def _reaction_to_graph(d: dict) -> dict:
        """Both ends of a reaction design (the product's atoms in the
        reactant's order), joined by an edge that uses these conformers."""
        from mepd.web import design

        rx = d["reaction"]
        if not rx.get("balanced", True):
            raise WorkspaceError("the reactant and the product no longer have the same atoms (an edit was made on one "
                                 "side only): make them match before adding the reaction")
        geoms = []
        for mb in (d["molblock"], _product_in_reactant_order(d)):
            (s,) = chem.structures_from_xyz_text(design.to_xyz(mb, d["charge"], d["multiplicity"]),
                                                  d["charge"], d["multiplicity"])
            geoms.append(s)
        # Several molecules on a side: its molecules as species, the two ends as the reaction's complexes.
        from mepd.web.compose import reaction_from_endpoints

        made = reaction_from_endpoints(W(), geoms[0], geoms[1], label=d.get("name") or "",
                                       origin={"kind": "design", "label": "Design (reaction)", "source": d.get("source")})
        if made is not None:
            return {"reaction": made["reaction"]["id"], "edge": made["edge"], "species": made["species"],
                    "existing": bool(made.get("existing"))}
        ends = []
        for s, smiles in zip(geoms, (d.get("smiles"), rx["product"].get("smiles"))):
            ends.append(W().add_or_merge(s, name=smiles, smiles=None, role="minimum",
                                         origin={"kind": "design", "label": "Design (reaction)", "source": d.get("source")}))
        (a, b), ws = ends, W()
        if a["rec"]["id"] == b["rec"]["id"]:
            raise WorkspaceError("both sides are the same molecule, so there is no reaction to search")
        edge = ws.add_edge(a["rec"]["id"], b["rec"]["id"], label=d.get("name") or "",
                           origin={"kind": "reaction", "input": (d.get("source") or {}).get("input")})
        ws.update_edge(edge["id"], conformers={a["rec"]["id"]: a["conformer"], b["rec"]["id"]: b["conformer"]})
        return {"reactant": a["rec"], "product": b["rec"], "edge": edge["id"],
                "new": [x["rec"]["id"] for x in ends if not x["merged"]]}

    @app.post("/api/design/search")
    async def design_search(body: DesignSearch):
        """Reaction mode: add both ends and their edge, then run a TS search
        (or a channels run) on it at the workspace level."""
        d = _current_design(atoms=True)
        if not d.get("reaction"):
            raise WorkspaceError("Search needs a reaction design (start one from a reaction SMILES)")
        out = await run_in_threadpool(_reaction_to_graph, d)
        publish_ws()
        ws = W()
        # The edge carries these conformers, so the search starts from exactly this pair.
        created = await run_in_threadpool(
            J().submit, body.op, structure_ids=[], edge_ids=[out["edge"]], params=body.params or {},
            profile=body.profile if body.profile is not None else ws.level_profile)
        return {**out, "jobs": created}

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

    @app.get("/api/references")
    def get_references():
        """The methods behind each feature, with citations (References tab)."""
        from mepd.web.references import references

        return references()

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
                             dry_run=body.dry_run, source_job_id=body.source_job, conformers=body.conformers,
                             extends_job_id=body.extends)
        if body.op == "optimize" and not body.dry_run:
            W().set_status(body.structures, "optimizing")
            publish_ws()
        return created

    @app.put("/api/setups")
    def put_setups(body: SetupsIn):
        """Experimental setups for Explore (mepd.web.setups), and which one it shows."""
        W().set_setups(body.setups, body.active)
        publish_ws()
        return {"setups": W().snapshot().get("setups", []), "active": W().snapshot().get("active_setup")}

    @app.post("/api/setups/{sid}/predict")
    async def predict_setup(sid: str):
        """Run the graph forward in time under a setup: what the flask holds at the end."""
        from mepd.web.setups import predict

        return await run_in_threadpool(predict, W().snapshot(), dict(J().jobs), W().setup(sid))

    @app.post("/api/setups/{sid}/analyze")
    async def analyze_setup(sid: str):
        """Whole-network properties under a setup (kinetics, equilibrium,
        timescales, bottlenecks) and the TSs and species that control them."""
        from mepd.web.setups import analyze

        return await run_in_threadpool(analyze, W().snapshot(), dict(J().jobs), W().setup(sid), W().levels())

    @app.post("/api/setups/compare")
    async def compare_setup(body: SetupCompareIn):
        """How the network's properties change from one setup to another, and
        which steps' energy changes explain it."""
        from mepd.web.setups import compare_setups

        return await run_in_threadpool(compare_setups, W().snapshot(), dict(J().jobs), W().setup(body.base),
                                       W().setup(body.other), W().levels())

    @app.post("/api/setups/{sid}/fill")
    async def fill_setup(sid: str, body: SetupFillIn):
        """Queue 'Solvent effects' on every edge that has a gas-phase TS but
        no barrier in this setup's solvent yet."""
        from mepd.web.setups import fill_requests

        setup = W().setup(sid)
        todo = fill_requests(W().snapshot(), dict(J().jobs), setup)
        if demo is not None and todo:
            demo.check_capacity(J().list())
        created = []
        for req in todo:
            created += J().submit("solvent", structure_ids=[], edge_ids=[], params={
                "solvents": [setup["solvent"]], "mode": body.mode, "temperature": setup["temperature"]},
                profile=None, source_job_id=req["source_job"])
        return {"queued": [j["id"] for j in created], "edges": [r["edge"] for r in todo]}

    _channels_map_cache: dict = {}

    @app.get("/api/jobs/{jid}/channels-map")
    async def channels_map_view(jid: str, mode: str = "bonds", ts: Optional[str] = None):
        """Every path a channels run relaxes, placed on one approximate energy
        map (mepd/web/channels_pes.py); cheap enough to poll while it runs."""
        from mepd.web.channels_pes import channels_map

        if mode not in ("bonds", "distance", "irc"):
            raise HTTPException(400, "mode is 'bonds', 'distance' or 'irc'")
        # Sample more paths runs add to their page's run: one map for all of them.
        family = J().family(J().get(jid))
        base = family[0]
        jdirs = [J().job_dir(j["id"]) for j in family]
        view = J().result_view(base)
        jdir, out = jdirs[0], Path((view.get("extension") or view)["output_dir"])
        running = any(j["status"] == "running" for j in family)
        search = None
        if base["op"] == "ts" and out != Path(base["output_dir"]):
            # The first search's TSs, as its merged result classified them.
            kind = {"channel": "direct", "alternate": "multi-step"}
            res = await page_result(J(), base)
            kinds = {e["id"].removeprefix("first_").removesuffix("_irc"): kind.get(g["kind"], "other")
                     for g in res["groups"] if g["kind"] in ("channel", "alternate", "offtarget")
                     for e in g["entries"] if e["id"].startswith("first_")}
            search = (Path(base["output_dir"]), kinds)

        def signature():
            files = [f for d in jdirs for f in (d / "live").glob("*.json")] + list((out / "ts").glob("ts_pair_*"))
            return (mode, ts, running, len(files),
                    max((f.stat().st_mtime_ns for f in files), default=0))

        # Rebuild only when the run's files moved on (several open tabs, or a
        # poll between two writes, reuse the last map).
        key = (str(jdir), mode, ts)
        sig = await run_in_threadpool(signature)
        hit = _channels_map_cache.get(key)
        if hit and hit[0] == sig:
            return hit[1]
        result = await run_in_threadpool(channels_map, jdirs, mode, 64, running, out, ts, search)
        _channels_map_cache[key] = (sig, result)
        if len(_channels_map_cache) > 32:
            _channels_map_cache.pop(next(iter(_channels_map_cache)))
        return result

    def _tree_family(jid: str):
        """(job dirs, output dirs, running) of a job and the follow-ups that share its folder
        or its page (Sample more paths)."""
        job = J().get(jid)
        family = J().family(job)
        if len(family) == 1:
            base = J().get(job["source_job"]) if job.get("source_job") in J().jobs and \
                J().jobs[job["source_job"]].get("output_dir") == job.get("output_dir") else job
            family = [base] + sorted((j for j in J().jobs.values() if j.get("source_job") == base["id"]
                                      and j.get("output_dir") == base.get("output_dir")), key=lambda j: j["created"])
        outs = list(dict.fromkeys(Path(j["output_dir"]) for j in family))
        return ([J().job_dir(j["id"]) for j in family], outs,
                any(j["status"] == "running" for j in family))

    def _stream(tid: str) -> str:
        from mepd.web.opt_tree import STREAM

        if not STREAM.match(tid) or tid in (".", ".."):
            raise HTTPException(400, "bad tree id")
        return tid

    @app.get("/api/jobs/{jid}/trees")
    async def job_trees(jid: str):
        """The MSMEP split trees in a job (one per path search), finished or live."""
        from mepd.web.opt_tree import list_trees

        dirs, out, running = _tree_family(jid)
        return await run_in_threadpool(list_trees, dirs, out, running)

    @app.get("/api/jobs/{jid}/trees/{tid}")
    async def job_tree(jid: str, tid: str):
        from mepd.web.opt_tree import load_tree

        dirs, out, running = _tree_family(jid)
        try:
            return await run_in_threadpool(load_tree, dirs, out, _stream(tid), running)
        except KeyError:
            raise HTTPException(404, f"no optimization tree '{tid}' in this job")

    @app.get("/api/jobs/{jid}/trees/{tid}/nodes/{key}")
    async def job_tree_node(jid: str, tid: str, key: int, step: Optional[int] = None):
        """One node's optimization, step by step (geometries of one step)."""
        from mepd.web.opt_tree import node_detail, running_node

        dirs, out, running = _tree_family(jid)
        try:
            detail = await run_in_threadpool(node_detail, dirs, out, _stream(tid), key, running, step)
        except KeyError:
            raise HTTPException(404, f"no optimization tree '{tid}' in this job")
        if running:
            for d in dirs:
                live = await run_in_threadpool(running_node, d, tid, key)
                if live:
                    detail["live"] = live
                    break
        return detail

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
                                    multiplicity=body.multiplicity, title=body.title, profile=body.profile)

    @app.get("/api/jobs/{jid}")
    def get_job(jid: str):
        return {"job": J().get(jid), "progress": J().progress_snapshot(jid)}

    @app.get("/api/jobs/{jid}/route-ts", response_class=PlainTextResponse)
    def job_route_ts(jid: str, reaction: Optional[int] = None):
        """The structure behind a job's barrier (xyz): its IRC-verified TS;
        else its first optimized TS; else the highest point of its path (an
        unverified barrier). A nanoreactor job: that reaction's own TS."""
        job = J().get(jid)
        fp = J().job_dir(jid) / "route_ts.xyz"
        if fp.is_file():
            return PlainTextResponse(fp.read_text())
        if job["op"] == "nanoreactor":
            from mepd.web.nanoreactor import _json, _read

            data = _json(Path(job["output_dir"]) / "network.json") or {}
            rx = next((r for r in data.get("reactions") or [] if r["id"] == reaction), None)
            text = _read(Path(((rx or {}).get("ts") or {}).get("files", {}).get("ts") or "")) if rx else None
            if text:
                return PlainTextResponse(text)
            raise HTTPException(404, "no TS for this reaction in the run")
        result = collect_cached(job, J().job_dir(jid))
        if (result.get("route_ts") or {}).get("xyz"):
            return PlainTextResponse(result["route_ts"]["xyz"])
        for kind in ("ts", "path"):
            for g in result.get("groups") or []:
                if g.get("kind") != kind or not g.get("entries"):
                    continue
                e = g["entries"][0]
                k = e.get("ts_index") or 0 if kind == "path" else 0
                if e.get("frames"):
                    return PlainTextResponse(e["frames"][min(k, len(e["frames"]) - 1)]["xyz"])
        raise HTTPException(404, "this job has no TS or path")

    @app.get("/api/jobs/{jid}/reactor")
    def job_reactor(jid: str, start: int = 0, run: Optional[int] = None):
        """A nanoreactor job's trajectory from raw frame `start` on, and its events (live view)."""
        from mepd.web.nanoreactor import reactor_view

        return reactor_view(_reactor_out(J().get(jid), run), start)

    def _reactor_out(job: dict, run: Optional[int] = None) -> Path:
        """Where a job's reactor MD is: a nanoreactor job's output, or one
        run (the newest unless `run` is given) of a network expansion with the
        nanoreactor generator (output/reactor/run_kk, one per species)."""
        out = Path(job["output_dir"])
        if job.get("op") in ("nanoreactor", "nanoreactor-more"):
            return out
        runs = sorted((out / "reactor").glob("run_*")) if job.get("op") == "graph-enumeration" else []
        if not runs:
            raise HTTPException(404, "this job has no reactor MD (yet)")
        if run is not None:
            return runs[max(0, min(int(run), len(runs) - 1))]
        return max(runs, key=lambda d: max((f.stat().st_mtime for f in (d / "md").glob("*")), default=0.0))

    @app.get("/api/jobs/{jid}/complex-live")
    def job_complex_live(jid: str):
        """A running `complex` job's frames so far (mepd.web.complex_live)."""
        from mepd.web.complex_live import complex_live

        job = J().get(jid)
        if job.get("op") != "complex":
            raise HTTPException(404, "not a complex job")
        counts = [int(c) for c in str((job.get("params") or {}).get("counts", "")).replace(",", " ").split() if c.isdigit()]
        return complex_live(Path(job["output_dir"]), (job.get("params") or {}).get("method", ""), sum(counts))

    @app.get("/api/jobs/{jid}/reactor/events/{k}")
    def job_reactor_event(jid: str, k: int, run: Optional[int] = None):
        """One reaction event of a nanoreactor job, at full time resolution, cut to its atoms."""
        from mepd.web.nanoreactor import event_view

        try:
            return event_view(_reactor_out(J().get(jid), run), k)
        except KeyError:
            raise HTTPException(404, f"no event {k} (yet)") from None

    @app.get("/api/jobs/{jid}/live/{stream}")
    def get_live_stream(jid: str, stream: str):
        """One live stream in full (a finished minimization's whole replay,
        which the progress updates leave out to stay small)."""
        J().get(jid)
        # (Every deliberate 404 says what is missing: a bare "Not Found" then
        # means only an unknown route -- a server older than the page.)
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", stream):
            raise HTTPException(404, f"no live stream {stream!r}")
        # On a page with Sample more paths runs, any of them (the newest first).
        job = J().get(jid)
        runs = reversed(J().family(job)) if J().page_job(job) is job else [job]
        fp = next((f for f in (J().job_dir(j["id"]) / "live" / f"{stream}.json" for j in runs) if f.is_file()), None)
        if fp is None:
            raise HTTPException(404, f"no live stream {stream!r} for this job")
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
        return await page_result(J(), J().get(jid))

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
            raise HTTPException(404, f"no file {rel!r} in this job's output")
        return FileResponse(fp)

    def _import_picks(job: dict, group: dict, entry: dict, picks: list[int], connect: bool) -> dict:
        """Add chosen frames of one result entry to the Graph (reusing any
        structure already there); with `connect` and two picks, join them by
        an edge that remembers the result it came from."""
        jid = job["id"]
        if job.get("external") and not job.get("level"):
            # Imported before imports had a level: the profile kept with the output.
            level = J()._import_level(Path(job["output_dir"]), None, job.get("qmmm"))
            if level:
                J()._update(job, level=level)
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
        if job.get("qmmm") and in_order:
            # Ends that gained an energy at a TS's level: their edges' barriers.
            from mepd.web import qmmm as web_qmmm

            web_qmmm.refresh_edges_of(W(), J().jobs, [r["id"] for r in in_order])
        if edge is not None and job.get("external") and edge["id"] not in job["targets"]["edges"]:
            # An imported result belongs to the edge it was added on (also an
            # edge that was already there): the edge lists it and its barrier.
            J()._update(job, targets={"structures": sorted(set(job["targets"]["structures"]) |
                                                           {r["id"] for r in in_order}),
                                      "edges": job["targets"]["edges"] + [edge["id"]]})
        return {"added": added, "reused": reused, "edge": edge}

    @app.post("/api/jobs/{jid}/import-entry")
    async def import_entry(jid: str, body: EntryImportIn):
        """Pull structures out of a result into the Graph, optionally
        connecting them with an edge that remembers where it came from."""
        job = J().get(jid)
        result = await page_result(J(), job)
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

    @app.post("/api/jobs/{jid}/retro/routes/{route}/explore")
    async def retro_route_to_explore(jid: str, route: int):
        """One retrosynthesis route into Explore: whatever of it is no longer
        there (deleted since) is added back, the rest reused. Returns the
        route's structures and edges, to select."""
        from mepd.web.retro import adopt_retro, route_elements

        job = J().get(jid)
        if job.get("op") != "retrosynthesis" or job.get("status") != "done":
            raise WorkspaceError("only a finished retrosynthesis job has routes to add")
        if await run_in_threadpool(adopt_retro, W(), job, route):
            J()._update(job)
            publish_ws()
        return await run_in_threadpool(route_elements, W(), job, route)

    @app.post("/api/jobs/{jid}/import-entries")
    async def import_entries(jid: str, body: EntriesImportIn):
        """Bulk "Add to Graph": the structure of each chosen entry -- its only
        frame, or the TS of a path -- in one request and one update."""
        job = J().get(jid)
        result = await page_result(J(), job)
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
    @app.post("/api/kinetics")
    async def kinetics(body: KineticsIn):
        """Microkinetics of the workspace's network (Analyze › Kinetics)."""
        from mepd.web.kinetics import analyze

        ws = W()
        level = ws.level_of(ws.level_profile).get("key")
        try:
            return await run_in_threadpool(
                analyze, ws.snapshot(), dict(J().jobs), level, initial=body.initial, held=body.held,
                temperature=body.temperature, time_s=body.time_s, target=body.target,
                include_unverified=body.include_unverified, control=body.control, control_what=body.control_what,
                sweep=body.sweep[:12])
        except RuntimeError as exc:
            raise WorkspaceError(str(exc)) from None

    @app.post("/api/reactions/propose")
    async def reactions_propose(body: ProposeIn):
        """What these species could become together (bond rules on their complex)."""
        from mepd.web.compose import propose

        return await run_in_threadpool(propose, W(), body.reactants, n_break=body.n_break, n_form=body.n_form)

    @app.post("/api/reactions/compose")
    async def reactions_compose(body: ComposeIn):
        """A new reaction from species in the graph; its complexes (and new
        product species) are minimized at the workspace level."""
        from mepd.web.compose import compose

        out = await run_in_threadpool(compose, W(), body.reactants, products=body.products or None,
                                      proposal=body.proposal)
        try:
            queue_optimization(out["optimize"])
        except WorkspaceError:
            pass
        publish_ws()
        return out["reaction"]

    # ---------------------------------------------------- interactive reactor
    def _sandbox(sid: str):
        from mepd.web import sandbox

        try:
            box = sandbox.get(sid)
        except KeyError:
            raise HTTPException(404, "this interactive reactor is not running (stopped, or the server restarted)")
        if demo is not None and box.owner != str(W().root):
            raise HTTPException(404, "this interactive reactor is not running (stopped, or the server restarted)")
        return box

    @app.post("/api/sandbox")
    def sandbox_start(body: SandboxIn):
        """A live MD of these species (so many of each) to steer by hand (mepd.web.sandbox)."""
        from mepd.web import sandbox
        from mepd.web.compose import species_structures

        limits = demo.sandbox if demo is not None else None
        max_atoms = min(120, demo.max_atoms * 3) if demo is not None else 120
        if body.structure:
            from mepd.web.workspace import new_id

            rec = W().structure(body.structure)
            if rec["natoms"] > max_atoms:
                raise WorkspaceError(f"{rec['natoms']} atoms is more than the interactive reactor keeps up with "
                                     f"(at most {max_atoms})")
            try:
                box = sandbox.start_from(new_id("sb_"), W().load_structure(body.structure), temperature=body.temperature,
                                         radius=body.radius, names=rec["name"], owner=str(W().root),
                                         sources=[body.structure], limits=limits)
            except (ValueError, RuntimeError) as exc:
                raise WorkspaceError(str(exc)) from None
            return box.describe()
        ids = [sid for sid, n in body.counts.items() if int(n) > 0]
        mols = [s for s, n in zip(species_structures(W(), ids), [int(body.counts[i]) for i in ids]) for _ in range(n)]
        if not mols:
            raise WorkspaceError("pick at least one molecule")
        natoms = sum(len(m.symbols) for m in mols)
        if natoms > max_atoms:
            raise WorkspaceError(f"{natoms} atoms is more than the interactive reactor keeps up with (at most {max_atoms})")
        names = " + ".join(f"{body.counts[i]} {W().structure(i)['name']}" if int(body.counts[i]) > 1
                           else W().structure(i)["name"] for i in ids)
        try:
            from mepd.web.workspace import new_id

            box = sandbox.start(new_id("sb_"), mols, temperature=body.temperature, radius=body.radius, names=names,
                                owner=str(W().root), sources=ids, limits=limits)
        except (ValueError, RuntimeError) as exc:
            raise WorkspaceError(str(exc)) from None
        return box.describe()

    @app.get("/api/sandboxes")
    def sandboxes_running():
        """The interactive reactors started from this workspace that still run."""
        from mepd.web import sandbox

        return sandbox.running(str(W().root))

    @app.get("/api/sandbox/{sid}")
    def sandbox_info(sid: str):
        return _sandbox(sid).describe()

    @app.get("/api/sandbox/{sid}/frame")
    def sandbox_frame(sid: str, since: int = 0):
        """The newest frame after `since` (waits briefly for one)."""
        return _sandbox(sid).frame(since)

    @app.get("/api/sandbox/{sid}/events")
    def sandbox_events(sid: str):
        """Its reaction events so far, found as the nanoreactor finds them."""
        return _sandbox(sid).events_view()

    @app.post("/api/sandbox/{sid}/command")
    def sandbox_command(sid: str, body: dict):
        """pull {atom, target [x,y,z] A, k kcal/mol/A^2} | release {atom} | release_all |
        radius {value A} | temperature {value K} | pause {value}."""
        if body.get("op") not in ("pull", "release", "release_all", "radius", "temperature", "pause"):
            raise WorkspaceError(f"unknown command {body.get('op')!r}")
        try:
            _sandbox(sid).command(body)
        except RuntimeError as exc:
            raise WorkspaceError(str(exc)) from None
        return {"ok": True}

    @app.post("/api/sandbox/{sid}/snapshot")
    def sandbox_snapshot(sid: str):
        """The current frame into Explore (several molecules: a complex), not minimized."""
        import numpy as np
        from qcconst.constants import ANGSTROM_TO_BOHR
        from qcdata import Structure

        box = _sandbox(sid)
        f = box.frame(0, wait=0)
        if not f.get("pos"):
            raise WorkspaceError("no frame yet")
        s = Structure(symbols=box.symbols, geometry=np.asarray(f["pos"]) * ANGSTROM_TO_BOHR, charge=box.charge,
                      multiplicity=box.multiplicity)
        res = W().add_or_merge(s, optimized=False, origin={"kind": "sandbox", "label": f"interactive reactor, {f.get('t_fs', 0)} fs"})
        publish_ws()
        return res["rec"]

    @app.post("/api/sandbox/{sid}/analyze")
    def sandbox_analyze(sid: str):
        """Stop it and analyze what it did like a nanoreactor run: its
        trajectory saved in the workspace, then a nanoreactor job on it (the
        same event detection; species and reaction complexes refined at the
        workspace level; its result page, reactor replay, and Explore)."""
        from mepd.web import sandbox

        box = _sandbox(sid)
        if demo is not None:
            demo.check_capacity(J().list())
        try:
            traj, start = box.save(W().root / "sandbox" / sid)
        except RuntimeError as exc:
            raise WorkspaceError(str(exc)) from None
        sandbox.forget(sid)
        jobs = J().submit("nanoreactor", structure_ids=box.sources, edge_ids=[], profile=W().level_profile,
                          params={"trajectory": str(traj), "trajectory_start": str(start)},
                          label=f"Interactive reactor: {box.names}")
        return {"job": jobs[0]["id"]}

    @app.delete("/api/sandbox/{sid}")
    def sandbox_stop(sid: str):
        from mepd.web import sandbox

        if demo is not None:
            _sandbox(sid)      # (a visitor stops only their own)
        sandbox.stop(sid)
        return {"ok": True}

    @app.post("/api/complexes")
    async def complexes_make(body: ComplexIn):
        """A complex of species in the graph (so many of each), minimized at
        the workspace level; Explore draws it as a complex node. Placed side
        by side or packed at once; docked, an NCI ensemble or a solvation
        shell as a `complex` job (its geometries join when it ends:
        adopt_complexes)."""
        from mepd.complexes import INSTANT, METHODS, missing_programs
        from mepd.web.compose import complex_members, make_complex

        if body.method not in METHODS:
            raise WorkspaceError(f"unknown method {body.method!r}")
        members = complex_members(W(), body.counts)
        if demo is not None:   # the complex is one structure: the demo's atom and structure limits apply
            demo.check_op("complex", {"method": body.method, "keep": body.keep})
            demo.check_structures(len(W().snapshot()["structures"]),
                                  [sum(W().structure(sid)["natoms"] for sid in members)])
        if body.method not in INSTANT:
            if missing_programs(body.method):
                raise WorkspaceError(f"{body.method} needs {', '.join(missing_programs(body.method))} on PATH")
            ids = [sid for sid, n in body.counts.items() if int(n) > 0]
            if demo is not None:
                demo.check_capacity(J().list())
            jobs = J().submit("complex", structure_ids=ids, edge_ids=[], profile=body.profile,
                              params={"method": body.method, "keep": body.keep,
                                      "counts": ", ".join(str(int(body.counts[sid])) for sid in ids)},
                              label=f"Complex ({body.method}): " + " + ".join(
                                  f"{body.counts[sid]} {W().structure(sid)['name']}" if body.counts[sid] > 1
                                  else W().structure(sid)["name"] for sid in ids))
            return {"job": jobs[0]["id"]}
        out = await run_in_threadpool(make_complex, W(), body.counts, body.method)
        if not out["existing"]:
            try:
                queue_optimization([out["complex"]["id"]])
            except WorkspaceError:
                pass
        publish_ws()
        return out

    @app.post("/api/delete")
    def delete_many(body: DeleteIn):
        removed = W().delete_many(body.structures, body.edges, body.reactions)
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
    if job["op"] == "ts":
        return _attach_path_conformers(ws, job, result)
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


def adopt_complexes(ws: Workspace, job: dict) -> list[str]:
    """A finished `complex` job's geometries as complexes (each one a
    geometry of its complex node; the same one again is recognized).
    Geometries whose molecules bonded on the way are dropped: a geometry is
    kept when it holds the molecules asked for (by formula, a species that
    is itself several molecules counting as those). What happened is written
    to output/adopted.json for the result page. Returns the new complexes
    (to minimize)."""
    from collections import Counter

    from mepd.web.compose import _fragments
    from mepd.web.operations import _parse_copies

    out = Path(job["output_dir"])
    fp = out / "complexes.xyz"
    if not fp.exists():
        return []
    known = ws.snapshot()["structures"]
    recs = [known[sid] for sid in job["targets"]["structures"] if sid in known]
    if len(recs) != len(job["targets"]["structures"]):
        return []    # a species was deleted meanwhile
    counts = _parse_copies((job.get("params") or {}).get("counts", "1"), recs)
    method = (job.get("params") or {}).get("method", "")
    want = Counter()
    for rec, n in zip(recs, counts):
        for f in _fragments(ws.load_structure(rec["id"])):
            want[chem.formula(f)] += n
    made, duplicates, reacted = [], 0, 0
    frames = chem.structures_from_xyz_text(fp.read_text())
    for k, s in enumerate(frames):
        if Counter(chem.formula(f) for f in _fragments(s)) != want:
            reacted += 1     # its molecules bonded (or fell apart) on the way: not a complex of these
            continue
        res = ws.add_or_merge(s, optimized=False, origin={"kind": "job", "job": job["id"], "entry": f"complex_{k}",
                                                          "label": f"Complex {k + 1} ({method})"})
        if res["rec"].get("role") != "complex":
            reacted += 1
        elif res["duplicate"]:
            duplicates += 1
        else:
            made.append(res["rec"]["id"])
    (out / "adopted.json").write_text(json.dumps({"kept": len(made), "duplicates": duplicates, "reacted": reacted,
                                                  "total": len(frames), "complexes": made}))
    return made


def adopt_expansion_steps(manager: JobManager, job: dict) -> bool:
    """A finished expansion's verified steps (summary.json) onto its graph
    edges -- the same as the live events do while it runs; this covers runs
    without a live view and steps whose events were missed."""
    try:
        summary = json.loads((Path(job["output_dir"]) / "summary.json").read_text())
    except Exception:
        return False
    nodes = dict(job.get("live_nodes") or {})
    known = manager.ws.snapshot()["structures"]
    # Each pair's lowest step, with its final barriers (they may differ from
    # the live ones: species energies can drop later in a run).
    best: dict[frozenset, dict] = {}
    for st in summary.get("steps") or []:
        key = frozenset((st["a"], st["b"]))
        if key not in best or (st.get("ts_energy") is not None and best[key].get("ts_energy") is not None
                               and st["ts_energy"] < best[key]["ts_energy"]):
            best[key] = st   # the lower TS: the lower barrier both ways
    changed = False
    for st in best.values():
        ev = {"event": "step", "a": st["a"], "b": st["b"], "label": st.get("label"),
              "barrier_kcal": st.get("barrier_kcal"), "warning": st.get("warning")}
        changed |= manager._adopt_step(job, ev, nodes, known, final=True)
    return changed


def _attach_path_conformers(ws: Workspace, job: dict, result: dict) -> int:
    """A path search (`mepd run`) minimizes its endpoints and, recursively,
    every point it splits at. Those that are the start's or the end's
    molecule -- often a conformer neither RDKit nor CREST produced -- join
    that node's conformers when new."""
    targets = [sid for sid in job["targets"]["structures"] if sid in ws.snapshot()["structures"]]
    if not targets:
        return 0
    keys = {sid: chem.canonical_key(ws.structure(sid)["smiles"]) for sid in targets if ws.structure(sid).get("smiles")}
    added = 0
    for group in result.get("groups", []):
        if group.get("kind") != "path":
            continue
        for entry in group["entries"]:
            frames = entry.get("frames") or []
            for k in {0, len(frames) - 1}:
                if not frames or frames[k].get("energy_hartree") is None:
                    continue
                (s,) = chem.structures_from_xyz_text(frames[k]["xyz"], job.get("charge"), job.get("multiplicity"))
                smiles = chem.perceive_smiles(s)
                owner = next((sid for sid, key in keys.items() if smiles and chem.canonical_key(smiles) == key), None)
                if owner is None:
                    continue   # an intermediate: a different molecule
                _, duplicate = ws.add_conformer(
                    owner, s, energy=frames[k]["energy_hartree"], level=job.get("level"), optimized=True,
                    origin={"kind": "job", "job": job["id"], "entry": entry["id"],
                            "label": f"{entry['label']} ({'start' if k == 0 else 'end'})", "frame": k})
                added += not duplicate
    return added


def apply_design_optimization(ws: Workspace, job: dict) -> None:
    """A finished minimization of the design: its geometry and energy
    replace the design's -- unless the design was edited meanwhile."""
    from mepd.web import design

    d = ws.design
    out = Path(job["output_dir"])
    if job["status"] != "done" or not d or d.get("rev") != job.get("design_rev") or not (out / "opt_0.xyz").exists():
        return
    rec = (_read_summary(out) or [{}])[0]
    frames = (out / "opt_0.xyz").read_text()
    info = design.with_coordinates(d["molblock"], frames)
    validation = {k: rec[k] for k in ("is_minimum", "min_frequency", "rescued", "validation") if k in rec} or None
    ws.set_design({**d, "molblock": info["molblock"], "energy": rec.get("energy"), "level": job.get("level"),
                   "validation": validation, "minimized_by": job["id"], "warnings": []})


def apply_design_tsopt(ws: Workspace, job: dict, result: dict) -> None:
    """A finished TS search from the design. Converged: the design becomes
    the TS (energy, level, barrier from its IRC). Not converged (or failed):
    the design goes back to exactly what was submitted, so it can be edited
    into something that converges. Either way `design.last_ts` says what
    happened."""
    from mepd.web import design

    d = ws.design
    snap = job.get("design_snapshot")
    if not d or d.get("rev") != job.get("design_rev"):
        return   # edited meanwhile: leave the user's newer design alone
    ts_group = next((g for g in result.get("groups", []) if g.get("kind") == "ts"), None)
    irc_group = next((g for g in result.get("groups", []) if g.get("kind") == "irc"), None)
    entry = ts_group["entries"][0] if ts_group and ts_group.get("entries") else None
    if job["status"] == "done" and entry is not None:
        frame = entry["frames"][0]
        info = design.with_coordinates(d["molblock"], frame["xyz"])
        irc = irc_group["entries"][0] if irc_group and irc_group.get("entries") else None
        ws.set_design({**d, "molblock": info["molblock"], "energy": frame.get("energy_hartree"),
                       "level": job.get("level"), "role": "ts", "warnings": [],
                       "last_ts": {"job": job["id"], "ok": True, "headline": result.get("headline"),
                                   "barrier_kcal": entry.get("barrier_kcal"), "irc": bool(irc),
                                   "irc_note": irc.get("note") if irc else None}})
        return
    why = (job.get("error") or "").strip().splitlines()[-1:] or [result.get("headline") or "no TS converged"]
    base = snap or d
    ws.set_design({**base, "last_ts": {"job": job["id"], "ok": False, "headline": result.get("headline"),
                                       "error": why[0][:300]}})


def _read_summary(out: Path) -> Optional[list]:
    try:
        return json.loads((out / "summary.json").read_text())["structures"]
    except Exception:
        return None


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
            if ws.structure(sid).get("members"):   # a complex: did it hold together?
                from mepd.web.compose import complex_intact
                try:
                    ws.set_intact(sid, complex_intact(ws, sid))
                except Exception:
                    pass
        else:
            why = (rec or {}).get("error") or job.get("error") or job["status"]
            ws.set_status([sid], "opt_failed", f"optimization {job['status']}: {str(why).splitlines()[-1][:300]}")

