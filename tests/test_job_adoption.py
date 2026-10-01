"""A web server that restarts while jobs run picks them back up: a run
still going stays 'running' and finishes normally; one that ended
meanwhile is finished from the exit code its runner wrote; one that was
stopped with the old server is 'interrupted' (resumable)."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time

import pytest

pytest.importorskip("fastapi")

from mepd.web.jobs import EXIT_FILE, JobManager  # noqa: E402
from mepd.web.workspace import Workspace  # noqa: E402


class Bus:
    def bind(self, loop):
        pass

    def publish(self, *a, **k):
        pass


def _record(ws, jid, **fields):
    jdir = ws.jobs_dir / jid
    (jdir / "output").mkdir(parents=True)
    job = {"id": jid, "op": "optimize", "title": jid, "status": "running", "created": time.time(),
           "started": time.time(), "finished": None, "returncode": None, "argv": [], "command": "mepd x",
           "targets": {"structures": [], "edges": []}, "output_dir": str(jdir / "output"), "external": False,
           "source_job": None, "error": None, "last_line": "", "summary": None, **fields}
    (jdir / "job.json").write_text(json.dumps(job))
    (jdir / "stdout.log").write_text("")
    return jdir


async def _run(jobs, until, timeout=20.0):
    await jobs.start()
    t0 = time.time()
    while not until() and time.time() - t0 < timeout:
        await asyncio.sleep(0.1)
    await jobs.stop()


def test_a_run_still_going_is_picked_up_and_finishes(tmp_path):
    ws = Workspace(tmp_path / "ws")
    jdir = ws.jobs_dir / "j_live"
    # A stand-in for the runner: its command line names the job folder, and
    # it writes the exit code when it ends.
    proc = subprocess.Popen([sys.executable, "-c", f"import time; time.sleep(1.5); "
                             f"open({str(jdir / EXIT_FILE)!r}, 'w').write('0')", str(jdir)],
                            start_new_session=True)
    _record(ws, "j_live", pid=proc.pid)
    jobs = JobManager(ws, Bus())
    seen = []

    async def go():
        await jobs.start()
        seen.append(jobs.jobs["j_live"]["status"])
        t0 = time.time()
        while jobs.jobs["j_live"]["status"] == "running" and time.time() - t0 < 20:
            await asyncio.sleep(0.1)
        await jobs.stop()

    asyncio.run(go())
    proc.wait()
    assert seen == ["running"]
    assert jobs.jobs["j_live"]["status"] == "done" and jobs.jobs["j_live"]["returncode"] == 0


@pytest.mark.parametrize("code, status", [("0", "done"), ("2", "failed"), ("-15", "interrupted")])
def test_a_run_that_ended_meanwhile_is_finished_from_its_exit_code(tmp_path, code, status):
    ws = Workspace(tmp_path / "ws")
    jdir = _record(ws, "j_ended", pid=999999)
    (jdir / EXIT_FILE).write_text(code)
    jobs = JobManager(ws, Bus())
    asyncio.run(_run(jobs, lambda: jobs.jobs["j_ended"]["status"] != "running"))
    assert jobs.jobs["j_ended"]["status"] == status


def test_a_run_gone_without_an_exit_code_is_interrupted(tmp_path):
    ws = Workspace(tmp_path / "ws")
    _record(ws, "j_old")            # from before pids were recorded
    jobs = JobManager(ws, Bus())
    jobs.load()
    assert jobs.jobs["j_old"]["status"] == "interrupted"


def test_a_run_from_before_pids_were_recorded_is_found_by_its_folder(tmp_path):
    ws = Workspace(tmp_path / "ws")
    jdir = ws.jobs_dir / "j_older"
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", "mepd", str(jdir / "inputs")],
                            start_new_session=True)
    try:
        _record(ws, "j_older")
        jobs = JobManager(ws, Bus())
        jobs.load()
        assert jobs.jobs["j_older"]["status"] == "running" and jobs.jobs["j_older"]["pid"] == proc.pid
    finally:
        proc.kill()
        proc.wait()


def test_the_runner_records_the_exit_code(tmp_path):
    exit_file = tmp_path / EXIT_FILE
    rc = subprocess.run([sys.executable, "-m", "mepd.web.job_runner", str(exit_file), "--help"],
                        capture_output=True, timeout=120).returncode
    assert rc == 0 and exit_file.read_text() == "0"
