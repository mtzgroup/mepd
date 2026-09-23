"""Job queue: every calculation is a `python -m mepd.cli ...` subprocess.

Why subprocesses rather than calling mepd in-process:
* mepd's worker helpers `typer.Exit` on bad input, fork process pools
  (`_fork_map`, unsafe under a threaded server), and flip process-global
  settings when RunInputs is built;
* a crash, a hung QM call or CREST misbehaving only takes down its own job;
* cancel = kill the process group (reaches `_fork_map` children and CREST);
* resume = rerun the same argv into the same --output directory, which the
  CLI commands already treat as "skip finished pairs / TSs".

Progress comes from the hooks mepd already has: stdout (stage messages),
MEPD_DRIVE_PROGRESS_LOG (per-branch status lines), MEPD_DRIVE_CHAIN_JSON
(latest energy profile) and, for `channels`, stats.json.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import signal
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

from mepd.web.operations import JobContext, get_operation
from mepd.web.workspace import Workspace, WorkspaceError, _atomic_write, new_id, remove_tree

TERMINAL = {"done", "failed", "cancelled", "interrupted"}


class Broadcaster:
    """Fan-out of server events to every connected SSE client."""

    def __init__(self) -> None:
        self._queues: set[asyncio.Queue] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._queues.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._queues.discard(q)

    def publish(self, event: str, data: Any, key: Optional[str] = None) -> None:
        """Safe to call from the event loop or from a threadpool worker (sync
        FastAPI endpoints run in one). `key` names the workspace the event
        belongs to; each SSE client only receives its own workspace's events
        (key None = everyone)."""
        try:
            on_loop = asyncio.get_running_loop() is self._loop
        except RuntimeError:
            on_loop = False
        if self._loop is not None and not on_loop:
            data = json.loads(json.dumps(data))  # snapshot before crossing threads
            self._loop.call_soon_threadsafe(self._deliver, event, data, key)
        else:
            self._deliver(event, data, key)

    def _deliver(self, event: str, data: Any, key: Optional[str] = None) -> None:
        for q in list(self._queues):
            try:
                q.put_nowait((event, data, key))
            except asyncio.QueueFull:
                # A client that stopped reading (e.g. a phone with the page in
                # the background) would otherwise silently miss events such as
                # "job finished". Drop its backlog and tell it to reload state.
                while not q.empty():
                    q.get_nowait()
                q.put_nowait(("resync", {}, None))


def _tail_lines(fp: Path, n: int = 1, max_bytes: int = 8192) -> list[str]:
    try:
        with fp.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - max_bytes))
            chunk = fh.read().decode("utf-8", "replace")
    except OSError:
        return []
    # A carriage return is a progress redraw: keep only the latest state.
    lines = [ln.rsplit("\r", 1)[-1].strip() for ln in chunk.splitlines()]
    # Skip rows that are only table borders/box drawing (rich tables, ASCII
    # energy plots): as a one-line status they are just noise.
    return [ln for ln in lines if ln and not set(ln) <= _DECORATION][-n:]


_DECORATION = set("│┃|╭╮╰╯┌┐└┘├┤┬┴┼─━═╇╈╉╊┡┩┣┫┳┻╋ -_=+*·.:")


class JobManager:
    def __init__(self, ws: Workspace, broadcaster: Broadcaster, *, max_concurrent: int = 2,
                 on_finished: Optional[Callable[[dict], Any]] = None,
                 global_slot_free: Optional[Callable[[], bool]] = None,
                 on_slot_freed: Optional[Callable[[], Any]] = None,
                 max_runtime: Optional[float] = None):
        """`global_slot_free`/`on_slot_freed`: a concurrency limit shared by
        several managers (one per demo visitor); `max_runtime`: seconds after
        which a running job is killed."""
        self.ws = ws
        self.bus = broadcaster
        self.max_concurrent = max_concurrent
        self.on_finished = on_finished
        self.global_slot_free = global_slot_free or (lambda: True)
        self.on_slot_freed = on_slot_freed
        self.max_runtime = max_runtime
        self.jobs: dict[str, dict] = {}
        self._procs: dict[str, asyncio.subprocess.Process] = {}
        self._cancel_requested: set[str] = set()
        self._wake = asyncio.Event()
        self._watch: dict[str, dict] = {}
        self._tasks: list[asyncio.Task] = []
        self._timed_out: set[str] = set()

    # ---------------------------------------------------------- lifecycle
    def load(self) -> None:
        for fp in sorted(self.ws.jobs_dir.glob("*/job.json")):
            try:
                job = json.loads(fp.read_text())
            except Exception:
                continue
            if job.get("status") == "running":
                # The server went away while this was running; its process is
                # gone (it was in our session) or orphaned. Either way the
                # user decides whether to resume.
                job["status"] = "interrupted"
                job["error"] = "server stopped while the job was running; resume to continue where it left off"
                self._write(job)
            self.jobs[job["id"]] = job

    async def start(self) -> None:
        self.bus.bind(asyncio.get_running_loop())
        self.load()
        self._tasks = [asyncio.create_task(self._scheduler()), asyncio.create_task(self._monitor())]
        self._wake.set()

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        for jid in list(self._procs):
            self._kill(jid, signal.SIGTERM)

    # ------------------------------------------------------------ records
    def job_dir(self, jid: str) -> Path:
        return self.ws.jobs_dir / jid

    def _write(self, job: dict) -> None:
        d = self.job_dir(job["id"])
        d.mkdir(parents=True, exist_ok=True)
        _atomic_write(d / "job.json", json.dumps(job, indent=1))

    def _update(self, job: dict, **fields: Any) -> None:
        job.update(fields)
        self._write(job)
        self.bus.publish("job", job)

    def get(self, jid: str) -> dict:
        try:
            return self.jobs[jid]
        except KeyError:
            raise WorkspaceError(f"unknown job {jid!r}") from None

    def list(self) -> list[dict]:
        return sorted(self.jobs.values(), key=lambda j: j["created"], reverse=True)

    # ------------------------------------------------------------- submit
    def _resolve_targets(self, op, structure_ids: list[str], edge_ids: list[str]) -> list[tuple[list[str], list[str]]]:
        """(structure ids, edge ids) per job to create."""
        if op.target == "pair":
            if edge_ids:
                out = []
                for eid in edge_ids:
                    e = self.ws.edge(eid)
                    out.append(([e["source"], e["target"]], [eid]))
                return out
            if len(structure_ids) == 2:
                edge = self.ws.add_edge(*structure_ids, origin={"kind": "job-target"})
                self.bus.publish("workspace", self.ws.snapshot())
                # Selection order (first clicked = start) wins over the
                # stored edge direction.
                return [(list(structure_ids), [edge["id"]])]
            raise WorkspaceError(f"{op.title} needs an edge, or exactly two structures")
        if op.target == "structure":
            if not structure_ids:
                raise WorkspaceError(f"{op.title} needs at least one structure")
            return [([sid], []) for sid in structure_ids]
        if len(structure_ids) < op.min_structures:
            raise WorkspaceError(f"{op.title} needs at least {op.min_structures} structures")
        return [(list(structure_ids), [])]

    def submit(self, op_key: str, *, structure_ids: list[str], edge_ids: list[str],
               params: Optional[dict], profile: Optional[str], label: str = "",
               dry_run: bool = False) -> list[dict]:
        """Create one job per target set. `dry_run` validates and returns
        the would-be records (with their `command`) without keeping anything."""
        op = get_operation(op_key)
        parsed = op.parse_params(params)
        if profile and profile not in self.ws.profile_names():
            raise WorkspaceError(f"unknown profile {profile!r}")
        if dry_run and op.target == "pair" and not edge_ids and len(structure_ids) == 2:
            target_sets = [(list(structure_ids), [])]  # don't create an edge just to preview
        else:
            target_sets = self._resolve_targets(op, structure_ids, edge_ids)
        batch = new_id("b_") if len(target_sets) > 1 else None

        created, dirs = [], []
        try:
            for sids, eids in target_sets:
                jid = new_id("j_")
                jdir = self.job_dir(jid)
                jdir.mkdir(parents=True)
                dirs.append(jdir)
                recs = [self.ws.structure(s) for s in sids]
                if op.target == "structure" and op.key != "tsopt":
                    busy = [r["name"] for r in recs if r.get("status") == "optimizing"]
                    if busy:
                        raise WorkspaceError(f"{', '.join(busy)} is still being optimized; run this once it is done")
                ctx = JobContext(self.ws, jdir, jdir / "output", recs, profile)
                argv = op.build(ctx, parsed)  # raises on invalid combinations
                names = " → ".join(r["name"] for r in recs) if op.target == "pair" else ", ".join(r["name"] for r in recs)
                created.append({
                    "id": jid, "op": op.key, "title": label or f"{op.title}: {names}",
                    "status": "queued", "created": time.time(), "started": None, "finished": None,
                    "returncode": None, "argv": argv,
                    "command": "mepd " + " ".join(shlex.quote(a) for a in argv),
                    "targets": {"structures": sids, "edges": eids},
                    "charge": recs[0]["charge"], "multiplicity": recs[0]["multiplicity"],
                    "params": parsed.model_dump(), "profile": profile, "level": ctx.level(), "batch": batch,
                    "output_dir": str(jdir / "output"), "external": False,
                    "error": None, "last_line": "", "summary": None,
                })
        except Exception:
            for d in dirs:
                remove_tree(d)
            raise
        if dry_run:
            for d in dirs:
                remove_tree(d)
            return created
        for job in created:
            self.jobs[job["id"]] = job
            self._write(job)
            self.bus.publish("job", job)
        self._wake.set()
        return created

    def import_external(self, path: Path, *, op_key: Optional[str], charge: int, multiplicity: int,
                        title: str = "") -> dict:
        from mepd.web.results import detect_operation

        path = path.expanduser().resolve()
        if not path.exists():
            raise WorkspaceError(f"{path} does not exist")
        op_key = op_key or detect_operation(path)
        if op_key is None:
            raise WorkspaceError(f"could not tell which mepd command wrote {path}")
        jid = new_id("j_")
        now = time.time()
        job = {
            "id": jid, "op": op_key, "title": title or f"Imported: {path.name}", "status": "done",
            "created": now, "started": now, "finished": now, "returncode": 0, "argv": [],
            "command": f"(existing output) {path}", "targets": {"structures": [], "edges": []},
            "charge": charge, "multiplicity": multiplicity, "params": {}, "profile": None, "batch": None,
            "output_dir": str(path), "external": True, "error": None, "last_line": "", "summary": None,
        }
        self.jobs[jid] = job
        self._write(job)
        self.bus.publish("job", job)
        if self.on_finished:
            self.on_finished(job)
        return job

    # ------------------------------------------------------------ control
    def cancel(self, jid: str) -> dict:
        job = self.get(jid)
        if job["status"] == "queued":
            self._update(job, status="cancelled", finished=time.time())
        elif job["status"] == "running":
            self._cancel_requested.add(jid)
            self._kill(jid, signal.SIGTERM)
            asyncio.get_running_loop().call_later(8, self._kill, jid, signal.SIGKILL)
        return job

    def _kill(self, jid: str, sig: int) -> None:
        proc = self._procs.get(jid)
        if proc is None or proc.returncode is not None:
            return
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass

    def retry(self, jid: str) -> dict:
        job = self.get(jid)
        if job["external"]:
            raise WorkspaceError("imported results cannot be rerun")
        if job["status"] not in TERMINAL:
            raise WorkspaceError(f"job is {job['status']}")
        self._update(job, status="queued", error=None, returncode=None, finished=None, summary=None)
        (self.job_dir(jid) / "result.json").unlink(missing_ok=True)
        self._wake.set()
        return job

    def delete(self, jid: str) -> None:
        job = self.get(jid)
        if job["status"] == "running":
            raise WorkspaceError("cancel the job before deleting it")
        del self.jobs[jid]
        remove_tree(self.job_dir(jid))  # never touches an external output dir
        self.bus.publish("job_deleted", {"id": jid})

    # ----------------------------------------------------------- running
    async def _scheduler(self) -> None:
        while True:
            await self._wake.wait()
            self._wake.clear()
            queued = sorted((j for j in self.jobs.values() if j["status"] == "queued"), key=lambda j: j["created"])
            while queued and len(self._procs) < self.max_concurrent and self.global_slot_free():
                job = queued.pop(0)
                try:
                    await self._launch(job)
                except Exception as exc:  # e.g. fork failure
                    self._update(job, status="failed", error=f"could not start: {exc}", finished=time.time())

    async def _launch(self, job: dict) -> None:
        jdir = self.job_dir(job["id"])
        env = dict(os.environ)
        env.update({
            "PYTHONUNBUFFERED": "1",
            "NEB_DISCOVERY_SILENT_TERMINAL": "1",
            "MEPD_DRIVE_PROGRESS_LOG": str(jdir / "progress.log"),
            "MEPD_DRIVE_CHAIN_JSON": str(jdir / "chain.json"),
            # One file per concurrent path search (e.g. per channels pair).
            "MEPD_DRIVE_CHAIN_DIR": str(jdir / "live"),
            "COLUMNS": "160",
            "TERM": "dumb",
            "NO_COLOR": "1",
        })
        log = open(jdir / "stdout.log", "ab")
        log.write(f"\n$ {job['command']}\n".encode())
        log.flush()
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "mepd.cli", *job["argv"],
                stdout=log, stderr=asyncio.subprocess.STDOUT, stdin=asyncio.subprocess.DEVNULL,
                cwd=str(jdir), env=env, start_new_session=True,
            )
        finally:
            log.close()
        self._procs[job["id"]] = proc
        self._update(job, status="running", started=time.time(), finished=None)
        asyncio.create_task(self._wait(job, proc))

    async def _wait(self, job: dict, proc: asyncio.subprocess.Process) -> None:
        rc = await proc.wait()
        self._procs.pop(job["id"], None)
        jid = job["id"]
        if jid in self._timed_out:
            self._timed_out.discard(jid)
            self._cancel_requested.discard(jid)
            status, error = "failed", f"stopped: exceeded the {self.max_runtime / 60:.0f}-minute run-time limit"
        elif jid in self._cancel_requested:
            self._cancel_requested.discard(jid)
            status, error = "cancelled", "cancelled by user; resume to continue where it left off"
        elif rc == 0:
            status, error = "done", None
        else:
            status = "failed"
            error = "\n".join(_tail_lines(self.job_dir(jid) / "stdout.log", 12)) or f"exit code {rc}"
        if jid in self.jobs:
            self._update(job, status=status, returncode=rc, finished=time.time(), error=error)
            self._publish_progress(job, force=True)
            if self.on_finished:
                self.on_finished(job)
        self._wake.set()
        if self.on_slot_freed:
            self.on_slot_freed()

    def wake(self) -> None:
        self._wake.set()

    def running_count(self) -> int:
        return len(self._procs)

    def _publish_progress(self, job: dict, force: bool = False) -> None:
        payload = self._collect_progress(job, force)
        if payload is not None:
            self.bus.publish("progress", payload)

    def _collect_progress(self, job: dict, force: bool = False) -> Optional[dict]:
        """What changed since the last poll (or everything, with `force`)."""
        jdir = self.job_dir(job["id"])
        out = Path(job["output_dir"])
        state = self._watch.setdefault(job["id"], {})
        payload: dict[str, Any] = {"id": job["id"]}
        changed = force

        sizes = {}
        for name in ("stdout.log", "progress.log"):
            try:
                sizes[name] = (jdir / name).stat().st_size
            except OSError:
                sizes[name] = 0
        if sizes != state.get("sizes"):
            state["sizes"] = sizes
            newest = max(("progress.log", "stdout.log"), key=lambda n: _mtime(jdir / n))
            last = _tail_lines(jdir / newest) or _tail_lines(jdir / "stdout.log")
            payload["last_line"] = last[-1] if last else ""
            payload["log_size"] = sizes["stdout.log"]
            job["last_line"] = payload["last_line"]
            changed = True

        # Live path streams: forward only the ones whose file changed.
        seen = state.setdefault("streams", {})
        streams = {}
        for fp in sorted((jdir / "live").glob("*.json")):
            m = _mtime(fp)
            if not m or seen.get(fp.stem) == m:
                continue
            try:
                streams[fp.stem] = _reduce_chain_payload(json.loads(fp.read_text()))
            except Exception:
                continue  # mid-write on a filesystem without atomic rename; next poll
            seen[fp.stem] = m
        if streams:
            payload["streams"] = streams
            changed = True

        for key, fp in (("chain", jdir / "chain.json"), ("stats", out / "stats.json")):
            m = _mtime(fp)
            if m and m != state.get(key):
                state[key] = m
                try:
                    data = json.loads(fp.read_text())
                except Exception:
                    continue
                if key == "chain":
                    data = _reduce_chain_payload(data)
                payload[key] = data
                changed = True
        return payload if changed else None

    def progress_snapshot(self, jid: str) -> dict:
        """Current progress state in full, for a client that opens a job's
        detail view mid-run (or after it finished)."""
        job = self.get(jid)
        self._watch.pop(jid, None)
        return self._collect_progress(job, force=True) or {"id": jid}

    async def _monitor(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            for jid in list(self._procs):
                job = self.jobs.get(jid)
                if job is not None:
                    if (self.max_runtime and job.get("started") and jid not in self._timed_out
                            and time.time() - job["started"] > self.max_runtime):
                        self._timed_out.add(jid)
                        self._cancel_requested.add(jid)
                        self._kill(jid, signal.SIGTERM)
                        asyncio.get_running_loop().call_later(8, self._kill, jid, signal.SIGKILL)
                    try:
                        self._publish_progress(job)
                    except Exception:
                        pass


def _reduce_chain_payload(data: dict) -> dict:
    """What the live view needs from a progress file (drops the 120-step
    plot history and ASCII art)."""
    return {
        "plot": data.get("plot"),
        "caption": data.get("caption"),
        # Current path geometry + TS-guess index (live viewer).
        "geometry": data.get("geometry"),
        "monitor_id": data.get("monitor_id"),
        "stream": data.get("stream"),
        "updated": data.get("updated"),
        "finished": data.get("finished", False),
        "status": data.get("status"),
        "monitors": {
            mid: {"plot": mon.get("plot"), "caption": mon.get("caption"),
                  "status": mon.get("status_message"), "active": mon.get("active"),
                  "geometry": mon.get("geometry")}
            for mid, mon in (data.get("monitors") or {}).items()
        },
    }


def _mtime(fp: Path) -> float:
    try:
        return fp.stat().st_mtime
    except OSError:
        return 0.0
