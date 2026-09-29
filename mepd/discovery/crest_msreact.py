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
import tempfile
from pathlib import Path

MODES = ("all", "fragments", "isomers")


def missing_programs() -> list[str]:
    return [p for p in ("crest", "xtb") if shutil.which(p) is None]


def install_hint() -> str:
    return ("needs the CREST (3.x) and xtb programs on PATH: e.g. `conda install -c conda-forge crest xtb`, or the "
            "release binaries from github.com/crest-lab/crest and github.com/grimme-lab/xtb")


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
        env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OMP_STACKSIZE": "1G"}
        proc = subprocess.run(argv, cwd=work, env=env, capture_output=True, text=True, timeout=timeout)
        (work / "crest.log").write_text(proc.stdout + proc.stderr)
        out = work / "crest_msreact_products.xyz"
        if proc.returncode != 0 or not out.exists():
            tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-8:])
            raise RuntimeError(f"crest --msreact failed (exit {proc.returncode}):\n{tail}")
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
