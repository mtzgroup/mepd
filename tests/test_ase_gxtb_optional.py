from __future__ import annotations

import subprocess
import sys
import textwrap

_BLOCK_ASE_PREAMBLE = textwrap.dedent(
    """
    import sys
    import importlib.abc

    class _BlockAseImports(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name == "ase" or name.startswith("ase.") or name == "sella":
                raise ImportError(f"blocked for test: {name}")
            return None

    sys.meta_path.insert(0, _BlockAseImports())
    """
)


def _run_blocked(script: str) -> subprocess.CompletedProcess:
    """Runs `script` in a brand-new interpreter with `ase`/`sella` imports
    blocked -- simulating a base `pip install mepd` (no `ase`/`gxtb` extra)
    environment. A fresh subprocess (rather than juggling sys.modules in the
    current pytest process) sidesteps corrupting shared classes like Chain/
    StructureNode that other tests in this suite already hold references to."""
    return subprocess.run(
        [sys.executable, "-c", _BLOCK_ASE_PREAMBLE + textwrap.dedent(script)],
        capture_output=True, text=True, timeout=60,
    )


def test_mepd_chain_and_cli_import_without_ase():
    """The core import chain (mepd.cli -> mepd.chain -> mepd.nodes.node ->
    mepd.qcdata_structure_helpers) must not require `ase` -- regression test
    for the eager `from ase import Atoms` that used to live at the top of
    qcdata_structure_helpers.py, and the unused `ASEEngine` import that used
    to live at the top of neb.py."""
    result = _run_blocked(
        """
        import mepd.cli
        assert "ase" not in sys.modules
        print("OK")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_ase_engine_gives_friendly_error_without_ase_extra():
    result = _run_blocked(
        """
        from mepd.inputs import RunInputs
        try:
            RunInputs(engine_name="ase", ase_engine_kwds={})
        except ImportError as exc:
            print(f"CAUGHT: {exc}")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "pip install mepd[ase]" in result.stdout


def test_gxtb_engine_gives_friendly_error_without_gxtb_extra():
    result = _run_blocked(
        """
        from mepd.inputs import RunInputs
        try:
            RunInputs(engine_name="gxtb", gxtb_engine_kwds={"executable": "gxtb"})
        except ImportError as exc:
            print(f"CAUGHT: {exc}")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "pip install mepd[gxtb]" in result.stdout


def test_gxtb_engine_still_works_when_ase_is_available():
    from mepd.inputs import RunInputs

    eng = RunInputs(engine_name="gxtb", gxtb_engine_kwds={"executable": "gxtb"}).engine
    assert type(eng).__name__ == "GXTBCalculator"
