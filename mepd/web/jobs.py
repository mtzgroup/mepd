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
import logging
import os
import re
import shlex
import signal
import sys
import time
import shutil
from pathlib import Path
from typing import Any, Callable, Optional

from mepd.web.operations import OPERATIONS, JobContext, get_operation
from mepd.web.workspace import Workspace, WorkspaceError, _atomic_write, new_id, remove_tree

TERMINAL = {"done", "failed", "cancelled", "interrupted"}


def extends(job: dict) -> Optional[str]:
    """The job a Sample more paths (or Run longer) run adds to (its result shows there)."""
    if job.get("extends"):
        return job["extends"]
    return job.get("source_job") if job.get("op") in ("channels-more", "nanoreactor-more") else None


EXIT_FILE = "exit_code"   # written by mepd.web.job_runner when the run ends


def _alive(pid: int, jdir: Path) -> bool:
    """Whether `pid` is still the runner of the job in `jdir` (not a new
    process that reused the number)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return False
    try:
        return str(jdir) in Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
    except OSError:
        return True   # no /proc (macOS): trust the pid


def _find_run(jdir: Path) -> Optional[int]:
    """The pid of a mepd run whose command line names `jdir` (Linux)."""
    for d in Path("/proc").glob("[0-9]*") if Path("/proc").is_dir() else []:
        try:
            cmd = (d / "cmdline").read_bytes().decode(errors="replace")
        except OSError:
            continue
        if str(jdir) in cmd and "mepd" in cmd:
            return int(d.name)
    return None



def _profile_engine(ws, profile: Optional[str]) -> str:
    """A profile's engine_name (the default, g-xTB, without one)."""
    import tomli

    try:
        data = tomli.loads(ws.read_profile(profile)) if profile else {}
    except Exception:
        data = {}
    return str(data.get("engine_name") or "gxtb")

class _Adopted:
    """A run started by an earlier server (not our child): the
    asyncio.subprocess.Process interface _wait, _kill and the monitor use,
    by polling its pid and reading the exit code its runner wrote."""

    def __init__(self, pid: int, jdir: Path):
        self.pid, self.jdir, self.returncode = pid, jdir, None

    async def wait(self) -> Optional[int]:
        while _alive(self.pid, self.jdir) and not (self.jdir / EXIT_FILE).exists():
            await asyncio.sleep(2.0)
        for _ in range(10):   # the runner writes the code just before it exits
            try:
                self.returncode = int((self.jdir / EXIT_FILE).read_text().strip())
                break
            except (OSError, ValueError):
                await asyncio.sleep(0.2)
        return self.returncode


def _bump_rev(job: dict) -> None:
    """Every change to a job record gets a larger `rev`, so a browser that
    receives two copies out of order (a live event and a full-state reload
    racing each other) keeps the newer one."""
    job["rev"] = max(time.time_ns(), job.get("rev", 0) + 1)

log = logging.getLogger(__name__)


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


_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_BOX = set("│┃║╭╮╰╯┏┓┗┛┡┩└┘┌┐├┤┬┴┼─━═╔╗╚╝╠╣╦╩╬ ")


def _status_line(lines: list[str]) -> str:
    """The last line worth showing as a job's live status: not blank, and
    not a row or border of a table or box a CLI printed (e.g. the settings
    table `mepd discovery expand` prints first, whose last row would
    otherwise stand for the job while a silent step such as CREST runs).
    progress.log's "[main] " prefix is dropped."""
    for line in reversed(lines or []):
        text = _ANSI.sub("", line).strip()
        if not text or text[0] in _BOX or set(text) <= _BOX:
            continue
        return text.removeprefix("[main] ")
    return ""


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


def explore_limit(job: dict) -> Optional[float]:
    """How far above the seed (kcal/mol) a network expansion's species may
    be to become Explore nodes: the job's `explore_within`, else its
    expansion window (when it grows by window), else no limit."""
    p = job.get("params") or {}
    if p.get("explore_within") is not None:
        return float(p["explore_within"])
    rounds = int(p.get("rounds") or 1)
    steer = p.get("steer") or "auto"
    by_window = steer == "window" or (steer == "auto" and rounds <= 1)
    if by_window and p.get("energy_window") is not None:
        return float(p["energy_window"])
    return None


class JobManager:
    def __init__(self, ws: Workspace, broadcaster: Broadcaster, *, max_concurrent: int = 2,
                 on_finished: Optional[Callable[[dict], Any]] = None,
                 global_slot_free: Optional[Callable[[], bool]] = None,
                 on_slot_freed: Optional[Callable[[], Any]] = None,
                 max_runtime: Optional[float] = None, op_runtime: Optional[dict] = None):
        """`global_slot_free`/`on_slot_freed`: a concurrency limit shared by
        several managers (one per demo visitor); `max_runtime`: seconds after
        which a running job is killed (`op_runtime`: per-operation overrides)."""
        self.ws = ws
        self.bus = broadcaster
        self.max_concurrent = max_concurrent
        self.on_finished = on_finished
        self.global_slot_free = global_slot_free or (lambda: True)
        self.on_slot_freed = on_slot_freed
        self.max_runtime = max_runtime
        self.op_runtime = dict(op_runtime or {})
        self.jobs: dict[str, dict] = {}
        self._procs: dict[str, asyncio.subprocess.Process] = {}
        self._cancel_requested: set[str] = set()
        self._wake = asyncio.Event()
        self._watch: dict[str, dict] = {}
        self._tasks: list[asyncio.Task] = []
        self._timed_out: set[str] = set()
        self._adopt: list[str] = []   # running when the last server went away (see load)

    # ---------------------------------------------------------- lifecycle
    def load(self) -> None:
        migration = self.ws.level_migration()
        for fp in sorted(self.ws.jobs_dir.glob("*/job.json")):
            try:
                job = json.loads(fp.read_text())
            except Exception:
                continue
            if job.get("status") == "running" and self._still_running(job):
                # The server went away while this ran, and the run carried on
                # (or ended since, leaving its exit code): watch it again.
                self._adopt.append(job["id"])
            elif job.get("status") == "running":
                # Its process is gone, without an exit code (stopped with the
                # server): the user decides whether to resume.
                job["status"] = "interrupted"
                job["error"] = "server stopped while the job was running; resume to continue where it left off"
                self._write(job)
            if self.ws.migrate_level(job.get("level"), migration):
                self._write(job)   # recorded under the old level fingerprint
            self.jobs[job["id"]] = job

    async def start(self) -> None:
        self.bus.bind(asyncio.get_running_loop())
        self.load()
        for jid in self._adopt:
            proc = _Adopted(self.jobs[jid]["pid"], self.job_dir(jid))
            self._procs[jid] = proc
            asyncio.create_task(self._wait(self.jobs[jid], proc))
        self._adopt = []
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
        _bump_rev(job)
        self._write(job)
        self.bus.publish("job", job)

    def get(self, jid: str) -> dict:
        try:
            return self.jobs[jid]
        except KeyError:
            raise WorkspaceError(f"unknown job {jid!r}") from None

    def list(self) -> list[dict]:
        return sorted(self.jobs.values(), key=lambda j: j["created"], reverse=True)

    # ----------------------------------------------------------- families
    # "Sample more paths" runs extend the result they start from: a
    # channels run on a TS search's pair (`extends`), a channels-more run in
    # its channels run's folder. They all show on one page, the first run's.
    def page_job(self, job: dict) -> dict:
        """The job whose page shows `job` (itself, unless it extends one)."""
        seen = {job["id"]}
        while (up := extends(job)) in self.jobs and up not in seen:
            seen.add(up)
            job = self.jobs[up]
        return job

    def family(self, job: dict) -> list[dict]:
        """The page job and every run extending it, oldest first."""
        base = self.page_job(job)
        return sorted((j for j in self.jobs.values() if j is base or self.page_job(j) is base),
                      key=lambda j: j["created"])

    def result_view(self, job: dict) -> dict:
        """The record the page's result is read from: the page job, plus
        (`extension`) the channels folder its Sample more paths runs share.
        Its status is the family's: running while any of them runs."""
        family = self.family(job)
        base = family[0]
        runs = [j for j in family if j is not base]
        if not runs:
            return base
        chans = [j for j in runs if j["op"] == "channels"]
        # The newest that finished (a cancelled retry must not hide it), else the newest.
        chans = [j for j in chans if j["status"] == "done"] or chans
        folder = chans[-1]["output_dir"] if chans else base["output_dir"]
        sharing = [j for j in family if j["output_dir"] == folder]
        active = [j for j in family if j["status"] in ("queued", "running")]
        return {**base, "extension": {
            "output_dir": folder, "charge": sharing[-1].get("charge"), "multiplicity": sharing[-1].get("multiplicity"),
            "status": "running" if active else "done",
            "finished": None if active else max((j.get("finished") or 0) for j in family)}}

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
               dry_run: bool = False, source_job_id: Optional[str] = None,
               conformers=None, extends_job_id: Optional[str] = None) -> list[dict]:
        """Create one job per target set. `dry_run` validates and returns
        the would-be records (with their `command`) without keeping anything.
        A follow-up operation (target "job") works on `source_job_id`'s
        finished output, in that job's output folder, at its level of theory.
        `conformers`: which conformer of each structure to use -- a list in
        target order, or {structure id: conformer id}; otherwise an edge's
        chosen conformers, otherwise each node's lowest-energy one.
        `extends_job_id`: a finished TS search whose page this channels run
        (Sample more paths) adds its paths to."""
        op = get_operation(op_key)
        parsed = op.parse_params(params)
        source = None
        if extends_job_id:
            base = self.get(extends_job_id)
            if op.key != "channels" or base["op"] != "ts":
                raise WorkspaceError("only a Reaction channels run can add paths to a Transition state search")
            if sorted(structure_ids) != sorted(base["targets"]["structures"]):
                raise WorkspaceError("Sample more paths runs on the same two structures as the search it extends")
        if op.target == "job":
            source = self.get(source_job_id or "")
            if source["op"] not in op.source_ops:
                raise WorkspaceError(f"{op.title} follows up on {', '.join(op.source_ops)} results, not {source['op']}")
            if source["status"] != "done":
                raise WorkspaceError(f"{op.title} needs the {source['title']} job to have finished")
            profile = source.get("profile")
            known = self.ws.snapshot()
            structure_ids = [sid for sid in source["targets"]["structures"] if sid in known["structures"]]
            edge_ids = [eid for eid in source["targets"]["edges"] if eid in known["edges"]]
        if profile and profile not in self.ws.profile_names():
            raise WorkspaceError(f"unknown profile {profile!r}")
        if profile:
            # Refuse a profile the settings form already knows is incomplete
            # (e.g. the ASE engine with no calculator), rather than queue a
            # job that can only fail on loading it.
            from mepd.web import profile_form

            pf = profile_form.form(self.ws.read_profile(profile))
            issues = pf["issues"]
            if op.key in ("optimize", "tsopt"):   # these never read the path method
                issues = [i for i in issues if i not in pf["path_issues"]]
            if issues:
                raise WorkspaceError(f"profile {profile!r} is incomplete: " + " ".join(issues))
        if op.target == "design":
            if not self.ws.design:
                raise WorkspaceError("there is no design")
            target_sets = [([], [])]
        elif source is not None:
            target_sets = [(structure_ids, edge_ids)]
        elif dry_run and op.target == "pair" and not edge_ids and len(structure_ids) == 2:
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
                chosen = {}
                for eid in eids:
                    chosen.update((self.ws.edge(eid).get("conformers") or {}))
                if isinstance(conformers, dict):
                    for k, v in conformers.items():   # None = explicitly the lowest, over an edge's saved pick
                        if v:
                            chosen[k] = v
                        else:
                            chosen.pop(k, None)
                picks = [(conformers[i] if isinstance(conformers, list) and i < len(conformers) else None)
                         or chosen.get(s) for i, s in enumerate(sids)]
                recs = [self.ws.structure_view(s, c) for s, c in zip(sids, picks)]
                if isinstance(conformers, dict) and eids and not dry_run and op.target == "pair":
                    # The edge remembers the endpoint conformers chosen for it.
                    self.ws.update_edge(eids[0], conformers={s: conformers.get(s) for s in sids if s in conformers})
                if op.target == "structure" and op.key != "tsopt":
                    busy = [r["name"] for r in recs if r.get("status") == "optimizing"]
                    if busy:
                        raise WorkspaceError(f"{', '.join(busy)} is still being optimized; run this once it is done")
                ctx = JobContext(self.ws, jdir, jdir / "output", recs, profile, source=source,
                                 edge_ids=list(eids), jobs=self.jobs)
                from mepd.web.operations import QMMM_OPS

                from mepd.web.operations import QMMM_MIXED_OPS, _qmmm_embed_pair

                if op.key in QMMM_MIXED_OPS:
                    qmmm = _qmmm_embed_pair(ctx)[0]["qmmm"]
                else:
                    qmmm = ctx.qmmm if op.target != "design" else None
                if qmmm and op.key not in QMMM_OPS:
                    raise WorkspaceError(f"{op.title} does not run on QM/MM systems (it needs whole molecules)")
                if qmmm:
                    from mepd.qmmm import embedding_problem

                    problem = embedding_problem(self.ws.qmmm_region(qmmm).embedding,
                                                _profile_engine(self.ws, profile))
                    if problem:
                        raise WorkspaceError(problem)
                argv = op.build(ctx, parsed)  # raises on invalid combinations
                names = " → ".join(r["name"] for r in recs) if op.target == "pair" else ", ".join(r["name"] for r in recs)
                if source is not None:
                    names = source["title"]
                design = self.ws.design if op.target == "design" else None
                if design is not None:
                    names = design.get("name") or design.get("formula") or "design"
                created.append({
                    "id": jid, "op": op.key, "title": label or f"{op.title}: {names}",
                    "status": "queued", "created": time.time(), "started": None, "finished": None,
                    "returncode": None, "argv": argv,
                    "command": "mepd " + " ".join(shlex.quote(a) for a in argv),
                    "targets": {"structures": sids, "edges": eids},
                    "target_conformers": [r.get("conformer_id") or r.get("conformer") for r in recs],
                    "charge": recs[0]["charge"] if recs else (design or source or {}).get("charge", 0),
                    "multiplicity": recs[0]["multiplicity"] if recs else (design or source or {}).get("multiplicity", 1),
                    "design_rev": design.get("rev") if design is not None else None,
                    # What was submitted, to go back to if a TS search fails.
                    "design_snapshot": dict(design) if design is not None else None,
                    "params": parsed.model_dump(), "profile": profile, "level": ctx.level(), "batch": batch,
                    "qmmm": qmmm,
                    "output_dir": source["output_dir"] if source is not None and not op.own_output
                    else str(jdir / "output"),
                    "external": False, "source_job": source["id"] if source is not None else None,
                    "extends": extends_job_id or None,
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
            _bump_rev(job)
            self.jobs[job["id"]] = job
            self._write(job)
            self.bus.publish("job", job)
        self._wake.set()
        return created

    def _qmmm_of_output(self, path: Path) -> Optional[str]:
        """The workspace's QM/MM system an output folder's structures belong
        to (by their atoms), if any."""
        from qcdata import Structure

        files = [path] if path.is_file() else sorted(path.glob("*.xyz")) + sorted(path.glob("*/*.xyz"))
        for fp in files[:20]:
            try:
                text = fp.read_text()
                n = int(text.split("\n", 1)[0])
                s = Structure.from_xyz("\n".join(text.splitlines()[:n + 2]))
            except Exception:
                continue
            sysid = self.ws.qmmm_system_of(s)
            if sysid:
                return sysid
        return None

    def _profile_of_output(self, path: Path) -> Optional[Path]:
        """The profile an output was computed with, if one was kept with it:
        a .toml with an engine_name in the output folder or the one above
        it, or a web job's inputs/profile.toml."""
        base = path if path.is_dir() else path.parent
        for fp in [base.parent / "inputs" / "profile.toml", *sorted(base.glob("*.toml")),
                   *sorted(base.parent.glob("*.toml"))]:
            try:
                if fp.is_file() and "engine_name" in fp.read_text():
                    return fp
            except OSError:
                continue
        return None

    def _import_level(self, path: Path, profile: Optional[str], qmmm: Optional[str]) -> Optional[dict]:
        """The level of theory of an imported output: a workspace profile
        chosen for it, else the profile kept with it; None if unknown (its
        structures then sit at no level and never merge into results at one)."""
        import tomli

        from mepd.web.workspace import level_key, level_label

        if profile:
            text, source = self.ws.read_profile(profile), None
        else:
            fp = self._profile_of_output(path)
            if fp is None:
                return None
            text, source = fp.read_text(), str(fp)
        level = {"profile": profile or None, "key": level_key(text), "label": level_label(text)}
        if source:
            level["source"] = source
        if qmmm:
            sig = self.ws.snapshot()["qmmm_systems"][qmmm]["sig"]
            label = f"{level['label']} / QM/MM"
            try:   # the region the output was computed with, if its profile names one
                from mepd.qmmm import region_from_inputs

                table = dict(tomli.loads(text).get("qmmm") or {})
                if source and table.get("file") and not Path(table["file"]).is_absolute():
                    table["file"] = str(Path(source).parent / table["file"])
                region = region_from_inputs(table)
                if region is not None and region.signature() != sig:
                    sig, label = region.signature(), f"{label} (another region)"
            except Exception:
                pass
            level = {**level, "key": f"{level['key']}+qmmm:{sig}", "label": label, "qmmm": qmmm}
        return level

    def import_external(self, path: Path, *, op_key: Optional[str], charge: int, multiplicity: int,
                        title: str = "", profile: Optional[str] = None) -> dict:
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
        job["qmmm"] = self._qmmm_of_output(path)
        job["level"] = self._import_level(path, profile, job["qmmm"])
        if job["level"]:
            job["profile"] = profile or None
        if job["qmmm"]:
            # So results read QM/MM structures by their QM region, and the
            # energy-split follow-up finds the region.
            inputs = self.job_dir(jid) / "inputs"
            inputs.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.ws.qmmm_region_path(job["qmmm"]), inputs / "qmmm_region.json")
        _bump_rev(job)
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

    def _still_running(self, job: dict) -> bool:
        """A job left 'running' by a server that went away: is its run
        still going, or did it end since and record its exit code?"""
        jdir = self.job_dir(job["id"])
        if job.get("pid") is None:   # started before pids were recorded: find its process
            job["pid"] = _find_run(jdir)
        return (jdir / EXIT_FILE).exists() or (job.get("pid") is not None and _alive(job["pid"], jdir))

    def _kill(self, jid: str, sig: int) -> None:
        proc = self._procs.get(jid)
        if proc is None or proc.returncode is not None:
            return
        try:
            # Its group (a run found by its folder after a restart need not lead it).
            os.killpg(os.getpgid(proc.pid), sig)
        except ProcessLookupError:
            pass

    def retry(self, jid: str) -> dict:
        job = self.get(jid)
        if job["external"]:
            raise WorkspaceError("imported results cannot be rerun")
        if job["status"] not in TERMINAL:
            raise WorkspaceError(f"job is {job['status']}")
        op = OPERATIONS.get(job.get("op"))
        if op is not None and not op.available:   # e.g. switched off in the web UI since it ran
            raise WorkspaceError(op.unavailable_reason or f"{op.title} is not available")
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
        (jdir / EXIT_FILE).unlink(missing_ok=True)
        try:
            # Through the runner (it records the exit code), so a server that
            # restarts meanwhile can pick the run back up.
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "mepd.web.job_runner", str(jdir / EXIT_FILE), *job["argv"],
                stdout=log, stderr=asyncio.subprocess.STDOUT, stdin=asyncio.subprocess.DEVNULL,
                cwd=str(jdir), env=env, start_new_session=True,
            )
        finally:
            log.close()
        self._procs[job["id"]] = proc
        self._update(job, status="running", started=time.time(), finished=None, pid=proc.pid)
        asyncio.create_task(self._wait(job, proc))

    async def _wait(self, job: dict, proc: asyncio.subprocess.Process) -> None:
        rc = await proc.wait()
        self._procs.pop(job["id"], None)
        jid = job["id"]
        if isinstance(proc, _Adopted) and (rc is None or rc < 0) and jid not in self._cancel_requested:
            # A run picked up after a restart that was then stopped (with the
            # old server, or killed): resumable, like any run a server stopped.
            status, error = "interrupted", "server stopped while the job was running; resume to continue where it left off"
        elif jid in self._timed_out:
            self._timed_out.discard(jid)
            self._cancel_requested.discard(jid)
            status, error = "failed", f"stopped: exceeded the {self._runtime_limit(job) / 60:.0f}-minute run-time limit"
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
            last = _tail_lines(jdir / newest, n=40) or _tail_lines(jdir / "stdout.log", n=40)
            payload["last_line"] = _status_line(last)
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
        try:
            self._adopt_live_events(job)
        except Exception:   # a bad line must never stop the progress feed
            log.exception("could not add live results of %s to the graph", job["id"])
        if job.get("op") in ("nanoreactor", "nanoreactor-more"):
            from mepd.web.nanoreactor import adopt_nanoreactor

            # A Run longer run writes into its nanoreactor's folder: adopted as that run (one set of reactions).
            base = self.page_job(job)
            try:
                if adopt_nanoreactor(self.ws, base):
                    self._write(base)
                    self.bus.publish("workspace", self.ws.snapshot())
                    if (base.get("params") or {}).get("refine_live"):   # reactions refined live: their TS now
                        from mepd.web.nanoreactor import spawn_ts_searches

                        spawn_ts_searches(self, base)
            except Exception:
                log.exception("could not add the nanoreactor network of %s to the graph", job["id"])

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

    def _adopt_live_events(self, job: dict) -> None:
        """New species a running job reports (live/events.jsonl, written by
        e.g. a network expansion) go straight into the Graph: each one a
        node spawned from the species it came from, joined to it by an edge,
        ready for a path search. Species 0 is the job's own structure."""
        fp = self.job_dir(job["id"]) / "live" / "events.jsonl"
        done = int(job.get("live_offset") or 0)
        try:
            if fp.stat().st_size <= done:
                return
            with open(fp, "rb") as fh:
                fh.seek(done)
                chunk = fh.read()
        except OSError:
            return
        end = chunk.rfind(b"\n") + 1          # only whole lines; the rest next time
        if not end:
            return
        from mepd.web import chem

        seeds = job.get("targets", {}).get("structures") or []
        nodes = dict(job.get("live_nodes") or {})
        # Species left out of Explore (above the job's threshold): index ->
        # parent index, so their descendants can still be placed.
        skipped = dict(job.get("live_skipped") or {})
        limit = explore_limit(job)
        if seeds:
            nodes.setdefault("0", seeds[0])
        known = self.ws.snapshot()["structures"]
        changed = False
        for raw in chunk[:end].splitlines():
            try:
                ev = json.loads(raw)
            except ValueError:
                continue
            if ev.get("event") == "species":
                if str(ev["index"]) in nodes or str(ev["index"]) in skipped:
                    continue    # already handled
                rel = ev.get("rel_energy_kcal")
                if limit is not None and rel is not None and rel > limit:
                    skipped[str(ev["index"])] = str(ev.get("parent"))
                    changed = True
                    continue
                # Found from a species left out: grow it from the nearest one shown, with no edge
                # (the reaction it came from starts at a species that is not in Explore).
                via, hops = str(ev.get("parent")), 0
                while via in skipped and hops < 1000:
                    via, hops = skipped[via], hops + 1
                parent = nodes.get(via)
                if parent not in known:
                    continue    # its parent was deleted
                (s,) = chem.structures_from_xyz_text(ev["xyz"], job.get("charge"), job.get("multiplicity"))
                smiles = self.ws.perceive(s) or ev.get("smiles") or None
                validation = ev.get("validation")
                # A molecule already in the graph gets this geometry as one
                # more conformer rather than a second node.
                rec = self.ws.add_structure(
                    s, name=smiles or chem.formula(s), energy=ev.get("energy_hartree"), smiles=smiles,
                    optimized=(validation or {}).get("is_minimum", True), level=job.get("level"),
                    validation=validation,
                    origin={"kind": "job", "job": job["id"], "entry": f"min_{ev['index']}",
                            "label": f"Minimum {ev['index']}", "frame": 0, "parent": parent, "live": True})
                known = self.ws.snapshot()["structures"]
                nodes[str(ev["index"])] = rec["id"]
                if hops:
                    changed = True
                    continue
                a, b = parent, rec["id"]
            elif ev.get("event") == "step":
                changed |= self._adopt_step(job, ev, nodes, known)
                continue
            elif ev.get("event") == "reaction":
                a, b = nodes.get(str(ev.get("source"))), nodes.get(str(ev.get("target")))
                if a not in known or b not in known:
                    continue
            else:
                continue
            if a != b and self.ws.find_edge(a, b) is None:
                try:
                    self.ws.add_edge(a, b, origin={
                        "kind": "job", "job": job["id"], "live": True, "proposed": True,
                        "headline": "proposed reaction" + (f" ({ev['caption']})" if ev.get("caption") else "")
                        + ": run a path search to connect it"})
                except WorkspaceError:
                    pass
            changed = True
        job["live_offset"] = done + end
        job["live_nodes"] = nodes
        job["live_skipped"] = skipped
        self._write(job)
        if changed:
            self.bus.publish("workspace", self.ws.snapshot())

    def _adopt_step(self, job: dict, ev: dict, nodes: dict, known: dict, final: bool = False) -> bool:
        """A verified step (TS + IRC between species a and b) of a running
        expansion: the edge between their nodes stops being "proposed" and
        carries the barrier (in the edge's direction), and points at the
        step's TS/IRC in the job's result. The lowest barrier wins when
        several TSs join the same pair. `final` (the finished run's summary,
        already reduced to each pair's lowest step) replaces what the live
        events set, whose barriers may predate a species' final energy."""
        a, b = nodes.get(str(ev.get("a"))), nodes.get(str(ev.get("b")))
        if a not in known or b not in known or a == b:
            return False
        edge = self.ws.find_edge(a, b)
        if edge is None:
            try:
                edge = self.ws.add_edge(a, b, origin={"kind": "job", "job": job["id"]})
            except WorkspaceError:
                return False
        fwd, rev = (list(ev.get("barrier_kcal") or []) + [None, None])[:2]
        barrier = fwd if edge["source"] == a else rev
        old = edge.get("origin") or {}
        if final:
            if old.get("job") == job["id"] and old.get("barrier_kcal") == barrier and not old.get("proposed") \
                    and old.get("label") == ev.get("label"):
                return False   # already exactly this
        elif not old.get("proposed") and old.get("barrier_kcal") is not None and barrier is not None \
                and old["barrier_kcal"] <= barrier:
            return False   # already the lower barrier of this pair
        label = ev.get("label") or f"step_{ev.get('a')}_{ev.get('b')}"
        warning = ev.get("warning")
        if barrier is not None and barrier < -0.1:
            from mepd.web.results import negative_barrier_text

            # In the edge's direction: this end is above its TS. Shown as it is, never as 0.
            warning = negative_barrier_text(barrier, "This edge's barrier")
        with self.ws._lock:
            edge["origin"] = {"kind": "job", "job": job["id"], "entry": f"{label}_irc", "group": "irc",
                              "has_ts": True, "barrier_kcal": barrier, "label": label,
                              "headline": "TS + IRC from the flux-steered expansion",
                              # A negative barrier is shown as it is, with why it is a problem.
                              "barrier_warning": warning}
            self.ws._save()
        return True

    def progress_snapshot(self, jid: str) -> dict:
        """Current progress state in full, for a client that opens a job's
        detail view mid-run (or after it finished)."""
        job = self.get(jid)
        self._watch.pop(jid, None)
        return self._collect_progress(job, force=True) or {"id": jid}

    def _runtime_limit(self, job: dict) -> Optional[float]:
        if not self.max_runtime:
            return None
        return self.op_runtime.get(job.get("op"), self.max_runtime)

    async def _monitor(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            for jid in list(self._procs):
                job = self.jobs.get(jid)
                if job is not None:
                    limit = self._runtime_limit(job)
                    if (limit and job.get("started") and jid not in self._timed_out
                            and time.time() - job["started"] > limit):
                        self._timed_out.add(jid)
                        self._cancel_requested.add(jid)
                        self._kill(jid, signal.SIGTERM)
                        asyncio.get_running_loop().call_later(8, self._kill, jid, signal.SIGKILL)
                    try:
                        self._publish_progress(job)
                    except Exception:
                        pass


def _reduce_chain_payload(data: dict, full: bool = False) -> dict:
    """What the live view needs from a progress file (drops the 120-step
    plot history and ASCII art). A finished minimization keeps only its
    final frame unless `full` (the page fetches the replay on demand)."""
    geometry = data.get("geometry")
    if (not full and data.get("kind") in ("minimization", "morph", "sampling") and data.get("finished")
            and geometry and len(geometry.get("frames") or []) > 1):
        geometry = {**geometry, "frames": geometry["frames"][-1:],
                    "frame_steps": (geometry.get("frame_steps") or [])[-1:], "truncated": True}
    return {
        "plot": data.get("plot"),
        "caption": data.get("caption"),
        # Current path geometry + TS-guess index (live viewer).
        "geometry": geometry,
        "monitor_id": data.get("monitor_id"),
        "stream": data.get("stream"),
        "updated": data.get("updated"),
        "finished": data.get("finished", False),
        "status": data.get("status"),
        # Geometry minimizations (Hessian sampling candidates) vs path searches.
        "kind": data.get("kind", "path"),
        "label": data.get("label"),
        "outcome": data.get("outcome"),
        "reference": data.get("reference"),
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
