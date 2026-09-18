from __future__ import annotations

import re

import numpy as np
import pytest
from qcdata import Structure

from mepd.chain import Chain
from mepd.nodes.node import StructureNode
from mepd.pathminimizers.gsm import GSM, _inpfileq_text
from qcconst.constants import ANGSTROM_TO_BOHR
from types import SimpleNamespace


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(captured: str) -> str:
    """Flatten a rich-rendered warning back to its plain message.

    `ProgressPrinter.print_warning` goes through a rich Console, which wraps at
    the terminal width and re-emits its colour codes on every wrapped line.
    Splitting on whitespace alone therefore leaves escape sequences wedged
    between words, so any asserted phrase that straddles a wrap point fails --
    and where it wraps depends on the terminal the suite happens to run in.
    """
    return " ".join(_ANSI_RE.sub("", captured).split())


def _hcn_node(c_pos: float, n_pos: float, energy: float) -> StructureNode:
    """A toy linear HCN/HNC-migration node: H fixed at the origin, C and N
    on the z-axis at the given (Angstrom) positions."""
    node = StructureNode(
        structure=Structure(
            symbols=["H", "C", "N"],
            geometry=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, c_pos],
                    [0.0, 0.0, n_pos],
                ],
                dtype=float,
            )
            * ANGSTROM_TO_BOHR,
            charge=0,
            multiplicity=1,
        )
    )
    node._cached_energy = energy
    return node


def test_write_string_blocks_format_round_trips(tmp_path):
    """The block format `_write_string_blocks` writes must be exactly what
    the compiled `gsm` binary's `read_string` (GSM/gstring.cpp) expects, and
    what `_parse_string_blocks` already reads for `stringfile.xyz` output --
    no blank lines between blocks, each block is `<natoms>` /
    `<energy kcal/mol>` / `<natoms> "SYMBOL x y z"` lines."""
    reactant = _hcn_node(1.064, 2.220, energy=-0.5)
    midpoint = _hcn_node(1.6135, 1.6070, energy=-0.3)
    product = _hcn_node(2.163, 0.994, energy=-0.52)
    nodes = [reactant, midpoint, product]
    e_reference = float(reactant.energy)

    fp = tmp_path / "restart.xyz0000"
    GSM._write_string_blocks(fp, nodes, e_reference)

    blocks = GSM._parse_string_blocks(fp.read_text(), natoms_expected=3)
    assert len(blocks) == 3

    _KCAL_PER_HARTREE = 627.5
    for node, (v_kcal, coords) in zip(nodes, blocks):
        expected_kcal = (float(node.energy) - e_reference) * _KCAL_PER_HARTREE
        assert abs(v_kcal - expected_kcal) < 1e-6

        coord_angstrom = np.asarray(node.coords, dtype=float) / ANGSTROM_TO_BOHR
        assert np.allclose(np.asarray(coords), coord_angstrom, atol=1e-6)


def test_write_string_blocks_reactant_energy_is_zero_reference(tmp_path):
    """The reactant's own energy line should read as 0.0 kcal/mol when it is
    used as the reference energy (mirrors how the caller's energy
    reconstruction expects node 0 to sit at `e_reactant + 0`)."""
    reactant = _hcn_node(1.064, 2.220, energy=-0.5)
    product = _hcn_node(2.163, 0.994, energy=-0.5)
    e_reference = float(reactant.energy)

    fp = tmp_path / "restart.xyz0000"
    GSM._write_string_blocks(fp, [reactant, product], e_reference)

    v_kcal, _ = GSM._parse_string_blocks(fp.read_text(), natoms_expected=3)[0]
    assert abs(v_kcal) < 1e-8


class _FakeEnergyEngine:
    def compute_energies(self, nodes):
        for node in nodes:
            if node._cached_energy is None:
                node._cached_energy = -0.4


def test_build_geodesic_seed_reuses_initial_chain_nodes_without_recomputing():
    """`_build_geodesic_seed` must reuse the geodesic-interpolated chain
    already built upstream (`self.initial_chain`, i.e. `chain_trajectory[0]`)
    as GSM's RESTART seed as-is -- not silently recompute a second, possibly
    differently-sized interpolation between the same two endpoints, which
    would make `chain_trajectory[0]` (what the user sees) and what GSM
    actually starts from silently diverge."""
    reactant = _hcn_node(1.064, 2.220, energy=-0.5)
    mid1 = _hcn_node(1.4, 1.9, energy=None)
    mid2 = _hcn_node(1.8, 1.5, energy=None)
    product = _hcn_node(2.163, 0.994, energy=-0.52)
    chain = Chain.model_validate(
        {"nodes": [reactant, mid1, mid2, product], "parameters": {}}
    )

    gsm = GSM(initial_chain=chain, engine=_FakeEnergyEngine())
    seed_nodes = gsm._build_geodesic_seed(chain, reactant, product)

    assert len(seed_nodes) == len(chain.nodes)
    # Interior nodes are the exact same objects already on `chain` -- proof
    # no fresh interpolation was built.
    assert seed_nodes[1] is mid1
    assert seed_nodes[2] is mid2
    assert all(n._cached_energy is not None for n in seed_nodes)


def test_print_live_chain_reports_restart_seeded_target_not_nnodes(monkeypatch):
    """The live Monitor's node-count denominator must reflect the actual
    RESTART seed length (`len(seed_nodes)`), not `path_min_inputs.nnodes`
    (which only governs from-scratch growth and can silently mismatch the
    seed's own size, e.g. default nnodes=9 vs default gi_inputs.nimages=10)
    -- and the caption must say it's RESTART-seeded so a partially-reported
    live string isn't mistaken for "seeding didn't work"."""
    import mepd.progress as progress_module

    captured = {}
    monkeypatch.setattr(
        progress_module,
        "print_chain_step",
        lambda chain, caption: captured.setdefault("captions", []).append(caption),
    )

    reactant = _hcn_node(1.064, 2.220, energy=-0.5)
    mid = _hcn_node(1.6135, 1.6070, energy=-0.3)
    product = _hcn_node(2.163, 0.994, energy=-0.52)
    chain = Chain.model_validate({"nodes": [reactant, mid, product], "parameters": {}})

    gsm = GSM(
        initial_chain=chain,
        engine=None,
        parameters=SimpleNamespace(nnodes=9, verbosity=1),
    )

    gsm._print_live_chain(chain, nnodes_target=3, seeded=True)
    assert "RESTART-seeded at 3 nodes" in captured["captions"][0]
    assert "3/3 have reported gradients" in captured["captions"][0]

    gsm._print_live_chain(chain, nnodes_target=9, seeded=False)
    assert "3/9 nodes seen" in captured["captions"][1]


def test_execute_gsm_attempt_falls_back_to_native_growth_after_seeded_crash(monkeypatch, capsys):
    """If molecularGSM fails on a RESTART-seeded (geodesic-interpolated)
    attempt -- e.g. its internal-coordinate builder segfaulting on a
    transient near-degenerate geometry along the interpolation -- retry
    once from scratch (seed_nodes=None) rather than failing the whole
    attempt, and warn that this happened."""
    from mepd.errors import ElectronicStructureError

    import mepd.progress as progress

    # `get_progress_printer()` hands out a module-level singleton. The `mepd
    # run` CLI tests call the command functions directly, so nothing tears down
    # the rich `Live` those runs start, and the first `print_warning` after one
    # stops it -- which makes rich restore the stdout that was current when the
    # Live started, over capsys's. A printer of our own has no Live to stop.
    monkeypatch.setattr(progress, "_default_printer", progress.ProgressPrinter())
    # Pin the rich Console width so the warning wraps the same way no matter
    # what terminal the suite runs in; `_plain` handles the colour codes.
    monkeypatch.setenv("COLUMNS", "200")

    reactant = _hcn_node(1.064, 2.220, energy=-0.5)
    product = _hcn_node(2.163, 0.994, energy=-0.52)
    chain = Chain.model_validate({"nodes": [reactant, product], "parameters": {}})

    gsm = GSM(initial_chain=chain, engine=_FakeEnergyEngine(), parameters=SimpleNamespace(verbosity=1))

    calls = []

    def fake_execute(chain_, reactant_, product_, e_reactant_, seed_nodes_, allow_early_stop_):
        calls.append(seed_nodes_)
        if seed_nodes_:
            raise ElectronicStructureError(msg="GSM calculation failed with exit code -11.")
        return "final_chain", ["history"], False

    monkeypatch.setattr(gsm, "_execute_gsm_attempt", fake_execute)

    result = gsm._execute_gsm_attempt_with_fallback(
        chain, reactant, product, -0.5, [reactant, product], False
    )

    assert result == ("final_chain", ["history"], False)
    assert len(calls) == 2
    assert calls[0] == [reactant, product]
    assert calls[1] is None

    out = _plain(capsys.readouterr().out)
    assert "RESTART-seeded" in out
    assert "falling back to its native from-scratch internal-coordinate growth" in out


def test_execute_gsm_attempt_no_fallback_when_not_seeded(monkeypatch):
    """A crash on an unseeded (native growth) attempt must propagate
    normally -- there's no seed to blame, so no retry."""
    from mepd.errors import ElectronicStructureError

    reactant = _hcn_node(1.064, 2.220, energy=-0.5)
    product = _hcn_node(2.163, 0.994, energy=-0.52)
    chain = Chain.model_validate({"nodes": [reactant, product], "parameters": {}})

    gsm = GSM(initial_chain=chain, engine=_FakeEnergyEngine(), parameters=SimpleNamespace(verbosity=1))

    def fake_execute(chain_, reactant_, product_, e_reactant_, seed_nodes_, allow_early_stop_):
        raise ElectronicStructureError(msg="GSM calculation failed with exit code -11.")

    monkeypatch.setattr(gsm, "_execute_gsm_attempt", fake_execute)

    with pytest.raises(ElectronicStructureError):
        gsm._execute_gsm_attempt_with_fallback(chain, reactant, product, -0.5, None, False)


def test_inpfileq_text_restart_and_nnodes_override():
    params = SimpleNamespace(nnodes=9)
    text = _inpfileq_text(params, restart=1, nnodes_override=5)
    assert "RESTART                 1" in text
    assert "NNODES                  5" in text

    default_text = _inpfileq_text(params)
    assert "RESTART                 0" in default_text
    assert "NNODES                  9" in default_text
