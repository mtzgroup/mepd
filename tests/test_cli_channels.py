from __future__ import annotations

import subprocess

import numpy as np
import pytest
import typer
from qcdata.models.structure import Structure
from rdkit import Chem
from rdkit.Chem import AllChem

from mepd.cli import channels as cli_channels
from mepd.chain import Chain
from mepd.inputs import ChainInputs, NEBInputs, RunInputs
from mepd.neb import NEB
from mepd.nodes.node import StructureNode
from mepd.optimizers.vpo import VelocityProjectedOptimizer
from mepd.pot import Pot
from mepd.TreeNode import TreeNode


def _install_fake_gxtb_with_coordinate_dependent_energy(monkeypatch):
    """Same rationale as the identically-named helper in test_cli_run.py /
    test_cli_network_splits.py: tying energy to geometry keeps genuinely
    different structures distinguishable instead of all reading as
    "identical" to MSMEP's endpoint check."""

    def fake_run(cmd, cwd, env, text, capture_output, check):
        xyz_path = cwd / cmd[1]
        lines = xyz_path.read_text().splitlines()
        coords = np.array([
            [float(x) for x in line.split()[1:4]] for line in lines[2:2 + int(lines[0])]
        ])
        energy = -76.0 + 0.01 * float(np.sum(coords**2))
        (cwd / "energy").write_text(f"$energy\n     1   {energy:.8f}   {energy:.8f}   {energy:.8f}\n$end\n")
        grad_lines = "\n".join(
            f"   {1.0E-03:.4E}   {0.0:.4E}   {2.0E-03:.4E}" for _ in coords
        )
        (cwd / "gradient").write_text(grad_lines + "\n")
        return subprocess.CompletedProcess(cmd, 0, stdout="normal termination", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)


def _run_inputs_for_test() -> RunInputs:
    return RunInputs(
        engine_name="gxtb",
        path_min_method="NEB",
        gxtb_engine_kwds={"executable": "gxtb"},
        gi_inputs={"nimages": 4},
        path_min_inputs={"max_steps": 2, "v": False, "do_elem_step_checks": False},
    )


def _call_channels(**overrides):
    """Calling a Typer-decorated function directly does not resolve
    `typer.Option(...)` defaults, so every parameter must be given an
    explicit value (same rationale as the other CLI tests)."""
    kwargs = dict(
        start=None,
        end=None,
        method="conformers",
        inputs=None,
        charge=0,
        multiplicity=1,
        atom_mapping=False,
        debug_dump=False,
        atom_mapping_candidates=5,
        atom_mapping_metric="gi-energy",
        atom_mapping_veto_margin=0.0,
        atom_mapping_recheck_splits=False,
        backend="rdkit",
        n_conformers=10,
        n_embed=30,
        rmsd_cutoff=0.5,
        random_seed=0,
        minimize_ends=False,
        max_pairs=100,
        parallel=False,
        parallel_workers=None,
        validate_minima_with_hessian=False,
        hessian_minimum_frequency_cutoff=0.0,
        hessian_minima_rescue_displacement=0.1,
        output=None,
    )
    kwargs.update(overrides)
    return cli_channels(**kwargs)


def test_channels_rejects_unknown_method(tmp_path):
    with pytest.raises(typer.BadParameter):
        _call_channels(start="CCCC", end="CC(C)C", method="not-a-method", output=tmp_path / "out")


def test_channels_rejects_unknown_atom_mapping_metric(tmp_path):
    with pytest.raises(typer.BadParameter):
        _call_channels(
            start="CCCC", end="CC(C)C", atom_mapping_metric="not-a-metric",
            output=tmp_path / "out",
        )


def test_channels_rejects_nonpositive_atom_mapping_candidates(tmp_path):
    with pytest.raises(typer.BadParameter):
        _call_channels(
            start="CCCC", end="CC(C)C", atom_mapping_candidates=0,
            output=tmp_path / "out",
        )


def test_channels_rejects_nonpositive_n_conformers(tmp_path):
    with pytest.raises(typer.BadParameter):
        _call_channels(start="CCCC", end="CC(C)C", n_conformers=0, output=tmp_path / "out")


def test_channels_rejects_nonpositive_n_embed(tmp_path):
    with pytest.raises(typer.BadParameter):
        _call_channels(start="CCCC", end="CC(C)C", n_embed=0, output=tmp_path / "out")


def test_channels_rejects_nonpositive_max_pairs(tmp_path):
    with pytest.raises(typer.BadParameter):
        _call_channels(start="CCCC", end="CC(C)C", max_pairs=0, output=tmp_path / "out")


def test_channels_runs_all_pairs_and_builds_network(tmp_path, monkeypatch):
    """Only the plumbing (generation -> pairing -> MSMEP -> network) is under
    test here, not real chemistry or real TS-opt/IRC (the fake gxtb engine
    below has no real transition-state optimizer, so every leaf's TS-opt is
    expected to fail and land in the "failed" bucket -- that's fine, this
    test only cares that the byproduct network still gets built)."""
    _install_fake_gxtb_with_coordinate_dependent_energy(monkeypatch)

    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    output_dir = tmp_path / "out"
    _call_channels(
        start="CCCC", end="CC(C)C", inputs=inputs_fp,
        n_conformers=2, n_embed=30, output=output_dir,
    )

    pairs_dir = output_dir / "pairs"
    completed = [p for p in pairs_dir.iterdir() if (p / "tree" / "adj_matrix.txt").exists()]
    assert len(completed) >= 1

    network_path = output_dir / "network.json"
    assert network_path.exists()
    pot = Pot.read_from_disk(network_path)
    assert pot.number_of_nodes >= 1


def test_channels_caps_at_max_pairs(tmp_path, monkeypatch, capsys):
    _install_fake_gxtb_with_coordinate_dependent_energy(monkeypatch)

    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    output_dir = tmp_path / "out"
    _call_channels(
        start="CCCC", end="CC(C)C", inputs=inputs_fp,
        n_conformers=2, n_embed=30, max_pairs=1, output=output_dir,
    )

    pairs_dir = output_dir / "pairs"
    completed = [p for p in pairs_dir.iterdir() if (p / "tree" / "adj_matrix.txt").exists()]
    assert len(completed) == 1
    out = capsys.readouterr().out
    assert "capping at --max-pairs=1" in out


def test_channels_is_resumable_skips_completed_pairs(tmp_path, monkeypatch, capsys):
    _install_fake_gxtb_with_coordinate_dependent_energy(monkeypatch)

    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    output_dir = tmp_path / "out"
    _call_channels(
        start="CCCC", end="CC(C)C", inputs=inputs_fp,
        n_conformers=1, n_embed=10, output=output_dir,
    )
    _call_channels(
        start="CCCC", end="CC(C)C", inputs=inputs_fp,
        n_conformers=1, n_embed=10, output=output_dir,
    )
    out = capsys.readouterr().out
    assert "Skipping pair (0, 1): already completed." in out


# --- classification helpers -------------------------------------------------


def _molecule_node(smiles: str, seed: int = 1) -> StructureNode:
    """A real, RDKit-bonded structure (embed + MMFF) -- cheap real chemistry
    for exercising connectivity-based comparisons, unlike a synthetic/toy
    geometry that would need its bonding faked."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(mol)
    from qcconst.constants import ANGSTROM_TO_BOHR

    positions = mol.GetConformer().GetPositions()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    structure = Structure(
        geometry=positions * ANGSTROM_TO_BOHR, symbols=symbols, charge=0, multiplicity=1
    )
    node = StructureNode(structure=structure)
    node._cached_energy = 0.0
    node._cached_gradient = np.zeros((len(symbols), 3))
    return node


def test_connectivity_matches_ignores_conformer_but_not_molecule():
    import mepd.cli as cli_module

    butane_a = _molecule_node("CCCC", seed=1)
    butane_b = _molecule_node("CCCC", seed=2)  # different conformer, same molecule
    isobutane = _molecule_node("CC(C)C", seed=1)

    assert cli_module._connectivity_matches(butane_a, butane_b)
    assert not cli_module._connectivity_matches(butane_a, isobutane)


def test_cluster_by_ts_identity_separates_distinct_ts_geometries():
    import mepd.cli as cli_module

    run_inputs = RunInputs()
    ts_a = _molecule_node("CCCC", seed=1)
    ts_a_again = ts_a.copy()  # identical structure and energy
    ts_b = _molecule_node("CC(C)C", seed=1)  # different graph -> different TS

    candidates = [(ts_a, None, "one"), (ts_a_again, None, "two"), (ts_b, None, "three")]
    clusters = cli_module._cluster_by_ts_identity(candidates, run_inputs)

    assert len(clusters) == 2
    labels_by_cluster = [sorted(label for _, _, label in cluster) for cluster in clusters]
    assert ["one", "two"] in labels_by_cluster
    assert ["three"] in labels_by_cluster


# --- _discover_channels end-to-end classification ---------------------------


def _water_chain_for_guess() -> Chain:
    """A throwaway 3-node chain to satisfy `_collect_ts_guess_tasks`'s
    `chain_trajectory[-1].get_ts_node()` call -- its actual content never
    matters here since `_optimize_ts_and_irc` is monkeypatched away below."""
    water = Structure(
        symbols=["O", "H", "H"],
        geometry=np.array(
            [[0.0, 0.0, 0.0], [1.43, 0.0, 0.95], [-1.43, 0.0, 0.95]], dtype=float
        ),
        charge=0, multiplicity=1,
    )
    nodes = []
    for i, e in enumerate([-76.0, -75.6, -76.05]):
        node = StructureNode(structure=water.model_copy(update={"geometry": water.geometry + 0.01 * i}))
        node._cached_energy = e
        node._cached_gradient = np.zeros((3, 3))
        nodes.append(node)
    return Chain.model_validate({"nodes": nodes, "parameters": ChainInputs()})


def _write_fake_pair_tree(output, pair_name: str) -> None:
    chain = _water_chain_for_guess()
    neb = NEB(
        initial_chain=chain,
        optimizer=VelocityProjectedOptimizer(),
        parameters=NEBInputs(),
        engine=None,
        optimized=chain,
        chain_trajectory=[chain],
    )
    tree = TreeNode(data=neb, children=[], index=0)
    tree_dir = output / "pairs" / pair_name / "tree"
    tree_dir.parent.mkdir(parents=True, exist_ok=True)
    tree.write_to_disk(tree_dir)


def test_discover_channels_classifies_channel_vs_alternate_route(tmp_path, monkeypatch):
    import mepd.cli as cli_module

    output = tmp_path / "out"
    _write_fake_pair_tree(output, "pair_0_1")
    _write_fake_pair_tree(output, "pair_2_3")

    start_node = _molecule_node("CCCC", seed=1)
    end_node = _molecule_node("CC(C)C", seed=1)

    # This leaf's IRC reconnects (a different conformer of) the requested
    # start/end pair -- a genuine channel.
    channel_irc = Chain.model_validate({
        "nodes": [_molecule_node("CCCC", seed=5), _molecule_node("CC(C)C", seed=5)],
        "parameters": ChainInputs(),
    })
    # This leaf's IRC reconnects a completely different pair of minima
    # (two C5H12 isomers, same atom count as each other but unrelated to
    # --start/--end) -- an alternate route, not a channel of the requested
    # pair.
    route_irc = Chain.model_validate({
        "nodes": [_molecule_node("CCCCC", seed=1), _molecule_node("CC(C)CC", seed=1)],
        "parameters": ChainInputs(),
    })

    def fake_optimize_ts_and_irc(guess_node, run_inputs, out_dir, *, run_irc, label):
        irc_chain = channel_irc if "pair_0_1" in label else route_irc
        ts_node = irc_chain[0].copy()
        from mepd.inputs import ChainInputs as CI
        Chain.model_validate({"nodes": [ts_node], "parameters": CI()}).write_to_disk(out_dir / f"{label}.xyz")
        irc_path = out_dir / ("irc.xyz" if label == "ts" else f"{label}_irc.xyz")
        irc_chain.write_to_disk(irc_path)
        return cli_module.TsIrcResult(ts_node=ts_node, irc_chain=irc_chain)

    monkeypatch.setattr(cli_module, "_optimize_ts_and_irc", fake_optimize_ts_and_irc)

    run_inputs = RunInputs()
    cli_module._discover_channels(output, start_node, end_node, run_inputs, charge=0, multiplicity=1)

    channels_dir = output / "channels"
    alt_dir = output / "alternate-routes"

    channel_folders = sorted(channels_dir.iterdir())
    assert len(channel_folders) == 1
    channel_members = (channel_folders[0] / "members.txt").read_text()
    assert "pair_0_1" in channel_members

    route_folders = sorted(alt_dir.iterdir())
    assert len(route_folders) == 1
    route_members = (route_folders[0] / "members.txt").read_text()
    assert "pair_2_3" in route_members
    assert "connects:" in route_members


def test_discover_channels_resumes_from_disk_without_recomputing(tmp_path, monkeypatch):
    import mepd.cli as cli_module

    output = tmp_path / "out"
    _write_fake_pair_tree(output, "pair_0_1")

    start_node = _molecule_node("CCCC", seed=1)
    end_node = _molecule_node("CC(C)C", seed=1)
    channel_irc = Chain.model_validate({
        "nodes": [_molecule_node("CCCC", seed=5), _molecule_node("CC(C)C", seed=5)],
        "parameters": ChainInputs(),
    })

    calls = []

    def fake_optimize_ts_and_irc(guess_node, run_inputs, out_dir, *, run_irc, label):
        calls.append(label)
        ts_node = channel_irc[0].copy()
        from mepd.inputs import ChainInputs as CI
        Chain.model_validate({"nodes": [ts_node], "parameters": CI()}).write_to_disk(out_dir / f"{label}.xyz")
        irc_path = out_dir / ("irc.xyz" if label == "ts" else f"{label}_irc.xyz")
        channel_irc.write_to_disk(irc_path)
        return cli_module.TsIrcResult(ts_node=ts_node, irc_chain=channel_irc)

    monkeypatch.setattr(cli_module, "_optimize_ts_and_irc", fake_optimize_ts_and_irc)

    run_inputs = RunInputs()
    cli_module._discover_channels(output, start_node, end_node, run_inputs, charge=0, multiplicity=1)
    assert len(calls) == 1

    cli_module._discover_channels(output, start_node, end_node, run_inputs, charge=0, multiplicity=1)
    assert len(calls) == 1, "already-optimized leaf must be reloaded from disk, not recomputed"


def test_channels_wires_atom_mapping_flags_into_run_inputs(tmp_path, monkeypatch):
    _install_fake_gxtb_with_coordinate_dependent_energy(monkeypatch)

    import mepd.cli as cli_module

    seen_run_inputs = []
    real_check = cli_module._check_endpoint_atom_mapping

    def spying_check(start_structure, end_structure, atom_mapping, run_inputs, **kwargs):
        seen_run_inputs.append(run_inputs)
        return real_check(start_structure, end_structure, atom_mapping, run_inputs, **kwargs)

    monkeypatch.setattr(cli_module, "_check_endpoint_atom_mapping", spying_check)

    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    _call_channels(
        start="CCCC", end="CC(C)C", inputs=inputs_fp, output=tmp_path / "out",
        n_conformers=1, n_embed=10,
        atom_mapping_candidates=3, atom_mapping_metric="geodesic-distance",
        atom_mapping_veto_margin=1.5, atom_mapping_recheck_splits=True,
    )

    assert len(seen_run_inputs) == 1
    atom_mapping_inputs = seen_run_inputs[0].atom_mapping_inputs
    assert atom_mapping_inputs.n_candidates == 3
    assert atom_mapping_inputs.recheck_on_split is True
    assert atom_mapping_inputs.metric == "geodesic-distance"
    assert atom_mapping_inputs.veto_margin == 1.5
