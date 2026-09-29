"""CREST's msreact fragment generator as a network-expansion generator.

`crest <xyz> --msreact` (Pracht, Grimme et al., CREST 3) finds the
fragments and isomers a molecule can reach: it adds repulsive potentials
on bonds (and attractive ones between H and lone pairs), optimizes with
GFN2-xTB, and keeps the distinct results. Every product keeps the input's
atoms in the same order, dissociated fragments included, so they plug
into network_expansion like any other product guess: they are
re-optimized at the run's level of theory, classified, and connected by
path searches. Useful for likely fragments (precursors, in reverse) and
nearby isomers of a molecule, without enumerating bond changes.

msreact runs the `xtb` program for its constrained optimizations, so both
`crest` and `xtb` must be on PATH. CREST runs single-threaded (-T 1);
parallelize across species instead.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
import threading
import re
import tempfile
from pathlib import Path

MODES = ("all", "fragments", "isomers")


def missing_programs() -> list[str]:
    return [p for p in ("crest", "xtb") if shutil.which(p) is None]


def install_hint() -> str:
    return ("needs the CREST (3.x) and xtb programs on PATH: e.g. `conda install -c conda-forge crest xtb`, or the "
            "release binaries from github.com/crest-lab/crest and github.com/grimme-lab/xtb")


_DISTORTIONS = re.compile(r"# of distortions\s+(\d+)\s*\n((?:[ \t]*\d+)*)")


def _run_with_progress(argv: list[str], cwd: Path, env: dict, timeout: float) -> tuple[int, str]:
    """Run CREST, reporting its msreact progress as a status line (the
    terminal spinner, and the web UI's live line via progress.log) about
    once a second. Returns (exit code, combined output)."""
    from mepd.progress import update_status

    proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    chunks: list[bytes] = []
    done_reading = threading.Event()

    def read() -> None:
        while True:
            data = proc.stdout.read1(65536)
            if not data:
                break
            chunks.append(data)
        done_reading.set()

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    update_status("CREST MSReact: distorting the molecule and re-optimizing each distortion...")
    deadline = time.monotonic() + float(timeout) if timeout else None
    last = None
    while not done_reading.wait(1.0):
        if deadline is not None and time.monotonic() > deadline:
            proc.kill()
            reader.join(5)
            raise subprocess.TimeoutExpired(argv, timeout)
        m = _DISTORTIONS.findall(b"".join(chunks).decode(errors="replace"))
        if m:
            total, finished = int(m[-1][0]), len(m[-1][1].split())
            if (finished, total) != last:
                last = (finished, total)
                update_status(f"CREST MSReact: {finished}/{total} distorted structures optimized")
    proc.wait()
    reader.join(5)
    return proc.returncode, b"".join(chunks).decode(errors="replace")


def msreact_products(structure, *, max_products: int = 50, mode: str = "all", nbonds: int = 3, nshifts: int = 0,
                     nshifts2: int = 0, timeout: float = 3600.0, keep_dir=None) -> list:
    """Product Structures (the input's atom order, charge and spin) of
    `crest --msreact`, lowest GFN2-xTB energy first, at most
    `max_products`. `mode`: all, fragments (dissociated only) or isomers
    (non-dissociated only). `nbonds`: bonds apart for the repulsive
    potential (CREST default 3); `nshifts`/`nshifts2`: extra optimizations
    from randomly shifted atoms (without / with the repulsive potential)."""
    from qcdata import Structure

    from mepd.qcdata_structure_helpers import read_multiple_structure_from_file

    missing = missing_programs()
    if missing:
        raise RuntimeError(f"CREST msreact: {', '.join(missing)} not found; {install_hint()}")
    if mode not in MODES:
        raise ValueError(f"msreact mode is one of {', '.join(MODES)}, not {mode!r}")
    work = Path(tempfile.mkdtemp(prefix="msreact_"))
    try:
        (work / "input.xyz").write_text(structure.to_xyz())
        argv = ["crest", "input.xyz", "--msreact", "-T", "1", "-chrg", str(int(structure.charge)),
                "-uhf", str(int(structure.multiplicity) - 1), "-msnbonds", str(int(nbonds))]
        if nshifts:
            argv += ["-msnshifts", str(int(nshifts))]
        if nshifts2:
            argv += ["-msnshifts2", str(int(nshifts2))]
        if mode == "fragments":
            argv.append("-msnoiso")
        elif mode == "isomers":
            argv.append("-msiso")
        # Unbuffered (gfortran buffers a pipe otherwise), so its progress --
        # "# of distortions N", then 1 2 3 ... as each distorted structure is
        # optimized -- arrives while it runs and reaches the live view.
        env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OMP_STACKSIZE": "1G",
               "GFORTRAN_UNBUFFERED_ALL": "y"}
        returncode, log = _run_with_progress(argv, work, env, timeout)
        (work / "crest.log").write_text(log)
        out = work / "crest_msreact_products.xyz"
        if returncode != 0 or not out.exists():
            tail = "\n".join(log.strip().splitlines()[-8:])
            raise RuntimeError(f"crest --msreact failed (exit {returncode}):\n{tail}")
        frames = read_multiple_structure_from_file(out, int(structure.charge), int(structure.multiplicity))
        products = []
        for s in frames[: max(1, int(max_products))]:
            if list(s.symbols) != list(structure.symbols):
                raise RuntimeError("crest --msreact returned a product with its atoms reordered")
            products.append(Structure(symbols=list(structure.symbols), geometry=s.geometry,
                                      charge=structure.charge, multiplicity=structure.multiplicity))
        return products
    finally:
        if keep_dir is not None:
            shutil.copytree(work, Path(keep_dir), dirs_exist_ok=True)
        shutil.rmtree(work, ignore_errors=True)
