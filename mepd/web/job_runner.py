"""Runs one web job: `python -m mepd.web.job_runner <exit file> <mepd argv...>`.

It runs `mepd <argv>` as its child and writes the child's exit code to
`<exit file>` when it ends, then exits with that code. The web server
launches every job through it (in the job's own process group), so a
server that restarts while a job runs can pick the job back up
(jobs.JobManager.load) and still learn how it ended: the job is no longer
the new server's child, so only this file can tell it.

SIGTERM/SIGINT reach the whole process group (cancel, run-time limit);
the runner ignores them itself, so that it outlives its child and records
how it ended.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys


def main(argv: list[str]) -> int:
    exit_file, args = argv[0], argv[1:]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, signal.SIG_IGN)
    child = subprocess.Popen([sys.executable, "-m", "mepd.cli", *args],
                             preexec_fn=lambda: [signal.signal(s, signal.SIG_DFL) for s in (signal.SIGTERM, signal.SIGINT)])
    code = child.wait()
    tmp = f"{exit_file}.tmp"
    with open(tmp, "w") as fh:
        fh.write(str(code))
    os.replace(tmp, exit_file)
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
