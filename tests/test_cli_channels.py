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
        crest_method="--gfn2",
        crest_threads=1,
        crest_ewin=6.0,
        crest_timeout=3600.0,
        crest_nci=True,
        complex_energy_tol=0.5,
        n_conformers=10,
        n_embed=30,
        rdkit_ewin=None,
        rdkit_torsion_prefs="both",
        rmsd_cutoff=0.5,
        random_seed=0,
        minimize_ends=False,
        max_pairs=100,
        conformers_only=False,
        parallel=False,
        parallel_workers=None,
        workers=1,
        pairs_per_mechanism=0,
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


def test_channels_rejects_negative_n_conformers(tmp_path):
    with pytest.raises(typer.BadParameter):
        _call_channels(start="CCCC", end="CC(C)C", n_conformers=-1, output=tmp_path / "out")


def test_channels_rejects_nonpositive_rdkit_ewin(tmp_path):
    with pytest.raises(typer.BadParameter):
        _call_channels(start="CCCC", end="CC(C)C", rdkit_ewin=0.0, output=tmp_path / "out")


def test_channels_rejects_nonpositive_crest_threads(tmp_path):
    with pytest.raises(typer.BadParameter):
        _call_channels(
            start="CCCC", end="CC(C)C", backend="crest", crest_threads=0,
            output=tmp_path / "out",
        )


def test_channels_rejects_nonpositive_crest_timeout(tmp_path):
    with pytest.raises(typer.BadParameter):
        _call_channels(
            start="CCCC", end="CC(C)C", backend="crest", crest_timeout=0,
            output=tmp_path / "out",
        )


def test_channels_rejects_negative_n_embed(tmp_path):
    with pytest.raises(typer.BadParameter):
        _call_channels(start="CCCC", end="CC(C)C", n_embed=-1, output=tmp_path / "out")


def test_channels_rejects_negative_max_pairs(tmp_path):
    with pytest.raises(typer.BadParameter):
        _call_channels(start="CCCC", end="CC(C)C", max_pairs=-1, output=tmp_path / "out")


def test_channels_conformers_only_uncapped_writes_stats_and_stops(tmp_path, monkeypatch, capsys):
    """n_conformers=0 / n_embed=0 / max_pairs=0 mean "no cap" / "auto" / "no
    cap"; --conformers-only must stop before any path search but still leave
    the pools and a stats.json describing every stage."""
    import json

    _install_fake_gxtb_with_coordinate_dependent_energy(monkeypatch)
    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)
    output_dir = tmp_path / "out"

    _call_channels(
        start="CCCCCC", end="CC(C)CCC", inputs=inputs_fp,
        n_conformers=0, n_embed=0, max_pairs=0, conformers_only=True, output=output_dir,
    )

    assert not (output_dir / "pairs").exists()
    assert (output_dir / "conformers" / "start.xyz").is_file()
    stats = json.loads((output_dir / "stats.json").read_text())
    start = stats["conformers"]["start"]
    assert stats["backend"] == "rdkit"
    assert start["n_embed"] == 50  # hexane: 3 rotatable bonds -> auto budget 50
    assert start["n_generated"] == 2 * 50  # with and without torsion preferences
    assert start["n_kept"] > 10  # more than the old default cap of 10
    # no --minimize-ends here, so only mirror-image merging can shrink a pool
    assert start["n_final"] == start["n_kept"] - start["n_mirror_images_merged"]
    assert stats["n_pairs"] == stats["n_pairs_possible"] == (
        stats["conformers"]["start"]["n_final"] * stats["conformers"]["end"]["n_final"]
    )
    assert "stopping before path search" in capsys.readouterr().out


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


def test_channels_workers_runs_pairs_in_parallel_processes(tmp_path, monkeypatch, capsys):
    """--workers > 1 must complete the same pairs as a serial run, each pair
    in its own forked process."""
    _install_fake_gxtb_with_coordinate_dependent_energy(monkeypatch)
    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    output_dir = tmp_path / "out"
    _call_channels(
        start="CCCC", end="CC(C)C", inputs=inputs_fp,
        n_conformers=2, n_embed=30, workers=3, output=output_dir,
    )

    out = capsys.readouterr().out
    assert "worker process(es)" in out
    import json

    n_pairs = json.loads((output_dir / "stats.json").read_text())["n_pairs"]
    completed = [p for p in (output_dir / "pairs").iterdir() if (p / "tree" / "adj_matrix.txt").exists()]
    assert n_pairs > 1 and len(completed) == n_pairs


def test_channels_maps_every_pair_once_per_mechanism(tmp_path, monkeypatch):
    """A Claisen's SLAPMapper mappings tie a [3,3] and a [1,3] shift: every
    conformer pair must become one path search per mechanism, and
    --pairs-per-mechanism K must keep only each mechanism's K best pairs."""
    import json

    _install_fake_gxtb_with_coordinate_dependent_energy(monkeypatch)
    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    def run(k, out):
        _call_channels(
            start="C=CCOC=C", end="C=CCCC=O", inputs=inputs_fp, atom_mapping=True,
            atom_mapping_metric="geodesic-distance", n_conformers=3, n_embed=30,
            pairs_per_mechanism=k, conformers_only=True, output=out,
        )
        return (json.loads((out / "stats.json").read_text()),
                json.loads((out / "pair_mechanisms.json").read_text()))

    stats, table = run(0, tmp_path / "all")
    assert stats["n_mechanisms"] == 2
    assert stats["n_path_searches"] == 2 * stats["n_pairs"]
    assert {row["mechanism"] for row in table} == set(stats["path_searches_per_mechanism"])

    stats, table = run(1, tmp_path / "best1")
    assert stats["n_path_searches"] == stats["n_mechanisms"] == 2
    assert len({row["mechanism"] for row in table}) == 2


def test_channels_rejects_negative_pairs_per_mechanism(tmp_path):
    with pytest.raises(typer.BadParameter):
        _call_channels(start="CCCC", end="CC(C)C", pairs_per_mechanism=-1, output=tmp_path / "out")


def test_completed_tree_dirs_skips_a_tree_whose_root_failed(tmp_path):
    """A pair whose first NEB failed leaves adj_matrix.txt plus
    node_0_failed.xyz; it must not reach network construction."""
    import mepd.cli as cli_module

    good = tmp_path / "pair_0_1" / "tree"
    bad = tmp_path / "pair_0_2" / "tree"
    for tree, root in ((good, "node_0.xyz"), (bad, "node_0_failed.xyz")):
        tree.mkdir(parents=True)
        (tree / "adj_matrix.txt").write_text("0\n")
        (tree / root).write_text("")
    assert cli_module._completed_tree_dirs(tmp_path) == [good]


# IRC product ends of two Diels-Alder TSs from a real `mepd channels` run
# (butadiene + ethylene, g-xTB): ordinary cyclohexene, and the trans-cyclohexene
# (ring C-C=C-C ~90 deg, ~54 kcal/mol higher) that an antarafacial TS leads to.
# Geometries in bohr.
_CIS_CYCLOHEXENE = [
        ('C', -0.964970, 2.500854, 0.067112),
        ('C', -2.510837, 0.660541, -1.420296),
        ('C', -2.172335, -1.768340, -0.951705),
        ('C', -0.274427, -2.461415, 1.024464),
        ('C', 1.824229, 1.741579, -0.030231),
        ('C', 2.217245, -1.085592, 0.519323),
        ('H', -1.586644, 2.478729, 2.041391),
        ('H', -1.192477, 4.428092, -0.616097),
        ('H', -3.824761, 1.344032, -2.822377),
        ('H', -3.189888, -3.238903, -1.934297),
        ('H', 0.046239, -4.491869, 1.103737),
        ('H', -0.982243, -1.883672, 2.879834),
        ('H', 2.541341, 2.178794, -1.909116),
        ('H', 2.893348, 2.895821, 1.294267),
        ('H', 3.127454, -1.976359, -1.099601),
        ('H', 3.483545, -1.344621, 2.119986),
    ]
_TRANS_CYCLOHEXENE = [
        ('C', -0.694177, 2.766601, -0.971615),
        ('C', -2.668342, 0.892934, -0.283218),
        ('C', -2.121889, -1.391600, -1.248404),
        ('C', -0.594103, -2.901562, 0.558923),
        ('C', 1.557999, 1.632008, 0.632986),
        ('C', 1.909843, -1.271114, 0.478882),
        ('H', -0.232169, 2.697960, -2.985890),
        ('H', -1.021312, 4.731217, -0.426043),
        ('H', -3.204863, 0.951160, 1.690367),
        ('H', -1.405893, -1.432031, -3.164541),
        ('H', -0.172416, -4.855311, 0.039980),
        ('H', -1.396983, -2.871723, 2.464220),
        ('H', 3.330650, 2.530934, 0.089446),
        ('H', 1.213323, 2.146209, 2.602866),
        ('H', 2.865759, -1.720919, -1.294868),
        ('H', 3.176661, -1.854423, 1.996044),
    ]


def _node_from_atoms(atoms):
    return StructureNode(structure=Structure(
        symbols=[a[0] for a in atoms], geometry=np.array([a[1:] for a in atoms]),
        charge=0, multiplicity=1,
    ))


def test_connectivity_matches_tells_cis_from_trans_cyclohexene():
    """Stereo SMILES can't encode E/Z in a six-membered ring, so cis- and
    trans-cyclohexene used to count as the same molecule, and an
    antarafacial Diels-Alder TS to trans-cyclohexene was classified as a
    channel to ordinary cyclohexene."""
    import mepd.cli as cli_module

    cis = _node_from_atoms(_CIS_CYCLOHEXENE)
    trans = _node_from_atoms(_TRANS_CYCLOHEXENE)
    cis_other_conformer = _molecule_node("C1=CCCCC1", seed=5)

    assert cli_module._n_trans_small_ring_alkenes(cis) == 0
    assert cli_module._n_trans_small_ring_alkenes(trans) == 1
    assert cli_module._connectivity_matches(cis, cis_other_conformer)
    assert not cli_module._connectivity_matches(cis, trans)


def test_fork_map_matches_serial_map():
    from mepd.cli import _fork_map

    offset = 7  # closed over, reaches the children through fork
    assert _fork_map(lambda x: x * x + offset, list(range(6)), 3) == [x * x + offset for x in range(6)]


def test_channels_rejects_nonpositive_workers(tmp_path):
    with pytest.raises(typer.BadParameter):
        _call_channels(start="CCCC", end="CC(C)C", workers=0, output=tmp_path / "out")


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


def test_cluster_by_ts_identity_merges_relabeled_and_mirrored_copies():
    """Per-pair atom mapping hands back the same TS with equivalent atoms
    relabeled, and mirror-image conformer pairs hand back its mirror image;
    both are the same saddle point and must land in one class, while a
    same-graph TS at a different energy must not."""
    import mepd.cli as cli_module
    from mepd.conformers import mirror_image

    run_inputs = RunInputs()
    ts = _molecule_node("CCCC", seed=1)
    symbols = list(ts.structure.symbols)
    h = [i for i, sym in enumerate(symbols) if sym == "H"]
    order = list(range(len(symbols)))
    order[h[0]], order[h[1]] = order[h[1]], order[h[0]]  # swap two methyl H
    geom = np.asarray(ts.structure.geometry)[order]

    def _node(structure, energy):
        node = StructureNode(structure=structure)
        node._cached_energy = energy
        node._cached_gradient = np.zeros((len(symbols), 3))
        return node

    relabeled = _node(ts.structure.model_copy(update={"geometry": geom}), 0.0)
    mirrored = _node(mirror_image(ts.structure), 0.0)
    higher = _node(ts.structure, 5.0 / 627.5)  # same geometry, 5 kcal/mol up

    clusters = cli_module._cluster_by_ts_identity(
        [(ts, None, "ts"), (relabeled, None, "relabeled"), (mirrored, None, "mirrored"),
         (higher, None, "higher")],
        run_inputs,
    )
    labels = [sorted(label for _, _, label in cluster) for cluster in clusters]
    assert ["mirrored", "relabeled", "ts"] in labels
    assert ["higher"] in labels


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


def test_discover_channels_classifies_channel_vs_offtarget_exit(tmp_path, monkeypatch):
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
    # --start/--end). Neither of them is reachable from --start, so no
    # sequence of discovered steps gets from it to --end -- an off-target
    # exit channel, not a channel of the requested pair.
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

    channel_folders = sorted((output / "channels").iterdir())
    assert len(channel_folders) == 1
    channel_members = (channel_folders[0] / "members.txt").read_text()
    assert "pair_0_1" in channel_members

    offtarget_folders = sorted((output / "offtarget-exit-channels").iterdir())
    assert len(offtarget_folders) == 1
    offtarget_members = (offtarget_folders[0] / "members.txt").read_text()
    assert "pair_2_3" in offtarget_members
    assert "connects:" in offtarget_members

    # A one-step start->end result is a channel, never an alternate channel.
    assert not (output / "alternate-channels").exists()


def test_discover_channels_chains_steps_into_multistep_alternate_channel(tmp_path, monkeypatch):
    """No single TS connects --start to --end, but two discovered steps do
    when composed through a shared intermediate: that is an *alternate
    channel*, and it must be reported as one multistep route rather than as
    two unrelated off-target steps. A third step that leaves --start for a
    minimum nothing leads onward from stays off-target.

    Everything here is C5H12 or its cracking products, so that each IRC is
    a well-formed equal-atom-count chain; what the classifier actually keys
    on is which IRC endpoints fall into the same connectivity class."""
    import mepd.cli as cli_module

    output = tmp_path / "out"
    _write_fake_pair_tree(output, "pair_0_1")
    _write_fake_pair_tree(output, "pair_2_3")
    _write_fake_pair_tree(output, "pair_4_5")

    start_node = _molecule_node("CCCCC", seed=1)        # n-pentane
    end_node = _molecule_node("CC(C)(C)C", seed=1)      # neopentane

    # --start -> isopentane, then isopentane -> --end. Note the two
    # isopentane structures are different conformers (different seeds) of
    # the same species -- they still have to land in one class for the two
    # legs to join up.
    first_leg = Chain.model_validate({
        "nodes": [_molecule_node("CCCCC", seed=5), _molecule_node("CC(C)CC", seed=1)],
        "parameters": ChainInputs(),
    })
    second_leg = Chain.model_validate({
        "nodes": [_molecule_node("CC(C)CC", seed=2), _molecule_node("CC(C)(C)C", seed=5)],
        "parameters": ChainInputs(),
    })
    # --start cracks to methane + 1-butene, and nothing leads onward from
    # there.
    dead_end = Chain.model_validate({
        "nodes": [_molecule_node("CCCCC", seed=7), _molecule_node("C.C=CCC", seed=1)],
        "parameters": ChainInputs(),
    })

    def fake_optimize_ts_and_irc(guess_node, run_inputs, out_dir, *, run_irc, label):
        if "pair_0_1" in label:
            irc_chain = first_leg
        elif "pair_2_3" in label:
            irc_chain = second_leg
        else:
            irc_chain = dead_end
        ts_node = irc_chain[0].copy()
        from mepd.inputs import ChainInputs as CI
        Chain.model_validate({"nodes": [ts_node], "parameters": CI()}).write_to_disk(out_dir / f"{label}.xyz")
        irc_path = out_dir / ("irc.xyz" if label == "ts" else f"{label}_irc.xyz")
        irc_chain.write_to_disk(irc_path)
        return cli_module.TsIrcResult(ts_node=ts_node, irc_chain=irc_chain)

    monkeypatch.setattr(cli_module, "_optimize_ts_and_irc", fake_optimize_ts_and_irc)

    cli_module._discover_channels(output, start_node, end_node, RunInputs(), charge=0, multiplicity=1)

    # Nothing bridges --start to --end in a single step.
    assert not (output / "channels").exists()

    route_folders = sorted((output / "alternate-channels").iterdir())
    assert len(route_folders) == 1, "the two composable legs are one route, not two"
    route = route_folders[0]

    path_text = (route / "path.txt").read_text()
    assert "2 step(s), 1 intermediate(s)" in path_text

    step_dirs = sorted(d.name for d in route.iterdir() if d.is_dir())
    assert step_dirs == ["step_0", "step_1"]
    # Ordered from --start outwards, not in discovery order.
    assert "pair_0_1" in (route / "step_0" / "members.txt").read_text()
    assert "pair_2_3" in (route / "step_1" / "members.txt").read_text()
    for step in step_dirs:
        assert (route / step / "ts.xyz").is_file()
        assert (route / step / "irc.xyz").is_file()

    offtarget_folders = sorted((output / "offtarget-exit-channels").iterdir())
    assert len(offtarget_folders) == 1
    assert "pair_4_5" in (offtarget_folders[0] / "members.txt").read_text()


def test_alternate_channel_keeps_every_distinct_ts_for_a_leg(tmp_path, monkeypatch):
    """Two genuinely different TSs for the same leg of a route are two real
    mechanisms for that step. The cheaper one defines the route, but the
    other must survive next to it instead of being silently dropped -- it is
    neither off-target (it is on the route) nor a duplicate."""
    import mepd.cli as cli_module

    output = tmp_path / "out"
    for pair in ("pair_0_1", "pair_2_3", "pair_6_7"):
        _write_fake_pair_tree(output, pair)

    start_node = _molecule_node("CCCCC", seed=1)
    end_node = _molecule_node("CC(C)(C)C", seed=1)

    legs = {
        # Two distinct TSs, both for --start -> isopentane.
        "pair_0_1": (["CCCCC", "CC(C)CC"], (5, 1), 0.0),
        "pair_6_7": (["CCCCC", "CC(C)CC"], (3, 4), 0.05),
        "pair_2_3": (["CC(C)CC", "CC(C)(C)C"], (2, 5), 0.0),
    }
    chains = {
        name: Chain.model_validate({
            "nodes": [_molecule_node(smi, seed=sd) for smi, sd in zip(smiles, seeds)],
            "parameters": ChainInputs(),
        })
        for name, (smiles, seeds, _) in legs.items()
    }

    def fake_optimize_ts_and_irc(guess_node, run_inputs, out_dir, *, run_irc, label):
        name = next(n for n in legs if n in label)
        irc_chain = chains[name]
        ts_node = irc_chain[0].copy()
        # Separate the two --start -> isopentane TSs by more than
        # `node_ene_thre`, so they cluster as distinct classes.
        ts_node._cached_energy = legs[name][2]
        from mepd.inputs import ChainInputs as CI
        Chain.model_validate({"nodes": [ts_node], "parameters": CI()}).write_to_disk(out_dir / f"{label}.xyz")
        irc_chain.write_to_disk(out_dir / f"{label}_irc.xyz")
        return cli_module.TsIrcResult(ts_node=ts_node, irc_chain=irc_chain)

    monkeypatch.setattr(cli_module, "_optimize_ts_and_irc", fake_optimize_ts_and_irc)

    cli_module._discover_channels(output, start_node, end_node, RunInputs(), charge=0, multiplicity=1)

    routes = sorted((output / "alternate-channels").iterdir())
    assert len(routes) == 1
    route = routes[0]

    # The cheaper of the two first-leg TSs defines the route itself.
    assert "pair_0_1" in (route / "step_0" / "members.txt").read_text()
    alternates = sorted((route / "step_0").glob("alternate_ts_*"))
    assert len(alternates) == 1, "the second distinct TS for leg 0 must be kept"
    assert "pair_6_7" in (alternates[0] / "members.txt").read_text()
    assert "step_0: 1 further distinct TS class(es)" in (route / "path.txt").read_text()

    # An on-route step is never also reported as an off-target exit.
    assert not (output / "offtarget-exit-channels").exists()


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
