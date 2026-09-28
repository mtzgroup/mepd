"""g-xTB processes never outlive their job: each call has a wall-time limit
(gxtb_engine_kwds.timeout_s), and on Linux it is killed when the process
that started it dies (setpriv --pdeathsig)."""

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from mepd.engines.gxtb import _SETPRIV, _run_gxtb_process
from mepd.errors import ElectronicStructureError


def _hanging_gxtb(tmp_path: Path) -> Path:
    """A stand-in g-xTB that records its PID and never finishes."""
    exe = tmp_path / "fake-gxtb"
    exe.write_text(f"#!/bin/sh\necho $$ > {tmp_path}/gxtb.pid\nexec sleep 300\n")
    exe.chmod(0o755)
    return exe


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A killed child that nobody has reaped yet is a zombie, not running.
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except OSError:
        return False


def _wait_for(pred, timeout=10.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.05)
    return False


def test_a_hanging_call_is_killed_at_the_time_limit(tmp_path):
    exe = _hanging_gxtb(tmp_path)
    t0 = time.monotonic()
    with pytest.raises(ElectronicStructureError, match="did not finish within 1 s"):
        _run_gxtb_process([str(exe)], cwd=tmp_path, env=dict(os.environ), timeout_s=1)
    assert time.monotonic() - t0 < 10
    assert not _alive(int((tmp_path / "gxtb.pid").read_text()))


def test_a_missing_executable_still_says_so(tmp_path):
    with pytest.raises(ElectronicStructureError, match="was not found"):
        _run_gxtb_process([str(tmp_path / "no-such-gxtb")], cwd=tmp_path, env=dict(os.environ), timeout_s=0)


@pytest.mark.skipif(_SETPRIV is None, reason="needs setpriv (util-linux)")
def test_gxtb_dies_with_the_job_that_started_it(tmp_path):
    """A job killed outright (SIGKILL: no cleanup code runs) must not leave
    g-xTB running on its own -- the orphans that were clogging the machine."""
    exe = _hanging_gxtb(tmp_path)
    job = subprocess.Popen([sys.executable, "-c", textwrap.dedent(f"""
        import os
        from pathlib import Path
        from mepd.engines.gxtb import _run_gxtb_process
        _run_gxtb_process([{str(exe)!r}], cwd=Path({str(tmp_path)!r}), env=dict(os.environ), timeout_s=0)
    """)])
    pid_file = tmp_path / "gxtb.pid"
    assert _wait_for(pid_file.exists), "fake g-xTB never started"
    gxtb_pid = int(pid_file.read_text())
    assert _alive(gxtb_pid)
    job.send_signal(signal.SIGKILL)
    job.wait()
    assert _wait_for(lambda: not _alive(gxtb_pid)), "g-xTB outlived the job that started it"
