"""Regression tests for the macOS fork hazard that killed `--workers > 1`.

`mepd.cli_common._fork_map` forks its workers, and each worker's first act is
to fork again to launch the engine's subprocess. If Tcl's `pthread_atfork`
handlers are registered in the parent (which `import tkinter` does, and which
`openbabel.pybel` used to pull in transitively), that second fork hits a
`notifierInitLock` owned by a thread that does not exist in the child and the
child is SIGKILLed -- surfacing only as `BrokenProcessPool`, with every pair in
the run lost. `mepd/__init__.py` keeps Tk out of the interpreter to prevent it.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


def _run(script: str, **env_extra) -> subprocess.CompletedProcess:
    import os

    env = {**os.environ, **env_extra}
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True, text=True, env=env, timeout=300,
    )


darwin_only = pytest.mark.skipif(
    sys.platform != "darwin", reason="the Tk/atfork hazard is macOS-specific"
)


@darwin_only
def test_importing_the_cli_does_not_load_tcl():
    """The import used to arrive via `openbabel.pybel`; pybel itself must
    still import, just without Tk."""
    proc = _run(
        """
        import sys
        import mepd.cli_channels  # noqa: F401  (pulls in openbabel.pybel)
        from openbabel import pybel

        assert sys.modules.get("_tkinter") is None, "Tcl got loaded"
        assert pybel.tk is None, "pybel kept a live Tk handle"
        print("OK")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_fork_map_children_can_launch_subprocesses():
    """The actual failure: every child died the moment it spawned the engine,
    so `_fork_map` raised `BrokenProcessPool` instead of returning."""
    proc = _run(
        """
        import subprocess
        from mepd.cli_common import _fork_map

        def job(i):
            # stands in for an engine call: a subprocess, launched with a cwd
            # (which is what pushes CPython onto its fork/exec path)
            return subprocess.run(
                ["/bin/echo", str(i)], capture_output=True, cwd="/tmp"
            ).returncode

        assert _fork_map(job, [0, 1, 2, 3], workers=4) == [0, 0, 0, 0]
        print("OK")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


@darwin_only
def test_fork_map_falls_back_to_serial_when_tcl_is_loaded():
    """MEPD_ALLOW_TKINTER re-enables Tk; `_fork_map` must then notice that
    forking is unsafe and run serially rather than lose the run."""
    proc = _run(
        """
        import subprocess, sys
        import mepd.cli_channels  # noqa: F401
        import tkinter  # noqa: F401  (registers Tcl's atfork handlers)
        from mepd.cli_common import _fork_map, _unsafe_to_fork

        assert _unsafe_to_fork() is not None

        def job(i):
            return subprocess.run(
                ["/bin/echo", str(i)], capture_output=True, cwd="/tmp"
            ).returncode

        assert _fork_map(job, [0, 1, 2], workers=3) == [0, 0, 0]
        print("OK")
        """,
        MEPD_ALLOW_TKINTER="1",
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
    assert "serially" in proc.stdout


def test_fork_map_retries_serially_when_a_worker_dies():
    """A child killed outright surfaces only as `BrokenProcessPool` for the
    whole pool, which used to abort the run. It must fall back to serial."""
    proc = _run(
        """
        import os
        from mepd.cli_common import _fork_map

        PARENT = os.getpid()

        def job(i):
            if os.getpid() != PARENT:
                os._exit(1)   # die like a SIGKILLed worker: no exception
            return i

        assert _fork_map(job, [0, 1, 2], workers=3) == [0, 1, 2]
        print("OK")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
    assert "died without reporting an error" in proc.stdout


@darwin_only
def test_fork_probe_says_yes_on_a_guarded_process():
    """The probe is consulted on every parallel stage, so a false alarm would
    silently drop the run to serial."""
    proc = _run(
        """
        import mepd.cli_common as cc
        assert cc._fork_probe_ok() is True, "probe cried wolf on a clean process"
        assert cc._unsafe_to_fork() is None
        print("OK")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


@darwin_only
def test_fork_probe_detects_a_parent_whose_children_would_be_killed():
    """The probe is what catches hazards the `sys.modules` check cannot see,
    so it has to stand on its own against a parent that really does kill its
    children -- here, one with Tk let back in. (This deliberately kills one
    child, so macOS writes one crash report.)"""
    proc = _run(
        """
        import sys
        import mepd.cli_common as cc

        # MEPD_ALLOW_TKINTER let pybel pull Tcl in on the way here.
        assert sys.modules.get("_tkinter") is not None
        assert cc._fork_probe_ok() is False, "the probe missed a lethal parent"
        print("OK")
        """,
        MEPD_ALLOW_TKINTER="1",
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
