from __future__ import annotations

import importlib.util
import sys

import mepd.cli as _real_mepd_cli
import pytest
from typer.testing import CliRunner


class _BlockDiscoveryImports:
    """A meta-path finder that makes `import mepd.discovery` (and submodules)
    fail, simulating an environment where the `discovery` extra is not
    installed -- without touching anything on disk."""

    def find_spec(self, name, path=None, target=None):
        if name == "mepd.discovery" or name.startswith("mepd.discovery."):
            raise ImportError(f"blocked for test: {name}")
        return None


@pytest.fixture
def mepd_cli_without_discovery():
    """Loads a fresh copy of `mepd/cli.py` under a throwaway module name,
    with `mepd.discovery` blocked, simulating a `pip install mepd` (no
    `discovery` extra) environment. Uses a throwaway name (rather than
    reloading the real `mepd.cli` in place) so the real `sys.modules["mepd.cli"]`
    that other test modules already hold references into is never touched.

    Other test modules in this suite import `mepd.discovery.*` at collection
    time, so by the time any test runs it is already cached in `sys.modules`
    -- a plain `sys.meta_path` blocker has no effect on an already-cached
    import. So this fixture also evicts any cached `mepd.discovery*` entries
    for the duration of the test, then restores the exact same module objects
    afterwards (not a re-import) so other tests keep the identical objects
    their own module-level imports already bound names to."""
    blocker = _BlockDiscoveryImports()
    saved_modules = {
        name: mod for name, mod in sys.modules.items()
        if name == "mepd.discovery" or name.startswith("mepd.discovery.")
    }
    for name in saved_modules:
        del sys.modules[name]

    sys.meta_path.insert(0, blocker)
    try:
        spec = importlib.util.spec_from_file_location(
            "mepd_cli_without_discovery_test_copy", _real_mepd_cli.__file__
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
            yield module
        finally:
            del sys.modules[spec.name]
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(saved_modules)


def test_mepd_cli_imports_without_mepd_discovery(mepd_cli_without_discovery):
    command_names = {c.name for c in mepd_cli_without_discovery.app.registered_commands}
    assert {"run", "ts", "network-build", "defaults", "visualize"} <= command_names
    assert "discovery" in command_names
    group_names = {g.name for g in mepd_cli_without_discovery.app.registered_groups}
    assert "discovery" not in group_names


def test_discovery_command_reports_friendly_error_without_mepd_discovery(mepd_cli_without_discovery):
    runner = CliRunner()
    result = runner.invoke(mepd_cli_without_discovery.app, ["discovery", "hessian-sample", "O"])

    assert result.exit_code == 1
    assert "mepd discovery is unavailable" in result.output
    assert "pip install mepd[discovery]" in result.output


def test_mepd_cli_registers_real_discovery_app_when_available():
    group_names = {g.name for g in _real_mepd_cli.app.registered_groups}
    assert "discovery" in group_names
