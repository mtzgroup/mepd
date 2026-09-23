from __future__ import annotations

import shutil

import numpy as np
import pytest
from qcdata.models.structure import Structure
from rdkit import Chem
from rdkit.Chem import AllChem

from mepd.conformers import ConformerInputs, generate_conformers
from mepd.helper_functions import RMSD
from mepd.nodes.node import StructureNode


def _butane_node(seed: int = 1, smiles: str = "CCCC") -> StructureNode:
    """Butane has real conformational freedom (anti/gauche about the central
    C-C bond) -- a good, cheap real molecule for exercising dedup, unlike a
    rigid toy system that would only ever produce one distinct conformer."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(mol)
    from qcconst.constants import ANGSTROM_TO_BOHR

    positions = mol.GetConformer().GetPositions()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    structure = Structure(
        geometry=positions * ANGSTROM_TO_BOHR, symbols=symbols, charge=0, multiplicity=1
    )
    return StructureNode(structure=structure)


def test_generate_conformers_rdkit_finds_multiple_distinct_conformers():
    node = _butane_node()
    confs = generate_conformers(
        node, ConformerInputs(n_conformers=10, n_embed=30, random_seed=0)
    )
    assert 1 < len(confs) <= 10


def test_generate_conformers_rdkit_default_seed_still_samples_distinct_conformers():
    """Regression: RDKit derives per-conformer seeds multiplicatively from
    `randomSeed`, so passing 0 straight through made every embedding
    identical and the pool collapsed to the input plus one geometry."""
    node = _butane_node(smiles="CCCCCC")
    confs = generate_conformers(
        node, ConformerInputs(n_conformers=10, n_embed=30, random_seed=0)
    )
    assert len(confs) >= 4


def test_generate_conformers_rdkit_energy_window_filters_and_reports_stats():
    node = _butane_node(smiles="CCCCCC")
    wide, narrow = {}, {}
    n_wide = len(generate_conformers(
        node, ConformerInputs(n_conformers=None, n_embed=60, rdkit_ewin_kcal=50.0), wide
    ))
    n_narrow = len(generate_conformers(
        node, ConformerInputs(n_conformers=None, n_embed=60, rdkit_ewin_kcal=0.5), narrow
    ))
    assert wide["n_generated"] == narrow["n_generated"] == 2 * 60  # both torsion-pref modes
    assert narrow["n_in_window"] < wide["n_in_window"]
    assert n_narrow < n_wide
    assert narrow["n_kept"] == n_narrow and narrow["seconds"] >= 0


def test_generate_conformers_rdkit_keeps_input_geometry_first():
    node = _butane_node(smiles="CCCCC")
    confs = generate_conformers(node, ConformerInputs(n_conformers=None, n_embed=30))
    assert np.allclose(confs[0].structure.geometry, node.structure.geometry)


def test_generate_conformers_rdkit_samples_a_two_fragment_complex():
    """Regression: ETKDG embeds the fragments of a non-covalent complex on
    top of each other, snap-RMSD rejected every such embedding as a
    different molecule, and the pool silently collapsed to the input. Each
    fragment must keep its sampled internal conformation but be put back
    where it sits in the input complex."""
    from qcconst.constants import ANGSTROM_TO_BOHR

    mol = Chem.AddHs(Chem.MolFromSmiles("CCCCC.O"))
    AllChem.EmbedMolecule(mol, randomSeed=3)
    positions = mol.GetConformer().GetPositions()
    oxygen = next(a for a in mol.GetAtoms() if a.GetSymbol() == "O")
    water = [oxygen.GetIdx()] + [n.GetIdx() for n in oxygen.GetNeighbors()]
    pentane = [i for i in range(mol.GetNumAtoms()) if i not in water]
    positions[water] += (
        positions[pentane].mean(axis=0) + np.array([6.0, 0.0, 0.0]) - positions[water].mean(axis=0)
    )
    node = StructureNode(structure=Structure(
        geometry=positions * ANGSTROM_TO_BOHR,
        symbols=[a.GetSymbol() for a in mol.GetAtoms()], charge=0, multiplicity=1,
    ))

    stats = {}
    confs = generate_conformers(node, ConformerInputs(n_conformers=None, n_embed=30), stats)

    assert stats["n_rejected_not_isomorphic"] == 0
    assert len(confs) >= 3
    ref = np.asarray(node.structure.geometry)
    for conf in confs:
        geom = np.asarray(conf.structure.geometry)
        # the water stays where it was relative to the pentane
        assert np.linalg.norm(geom[water].mean(axis=0) - ref[water].mean(axis=0)) < 1.0


def test_merge_mirror_images_drops_mirror_conformers_only():
    """Mirror-image conformers of an achiral molecule aren't superimposable
    by rotation, so snap-RMSD dedup keeps both; `merge_mirror_images` must
    drop the mirror copy but keep genuinely different conformers."""
    import qcinf

    from mepd.conformers import merge_mirror_images, mirror_image

    confs = generate_conformers(
        _butane_node(smiles="CCCCCC"), ConformerInputs(n_conformers=None, n_embed=50)
    )
    # the most chiral-looking conformer: furthest from its own mirror image
    chiral = max(confs, key=lambda c: qcinf.snap_rmsd(c.structure, mirror_image(c.structure)))
    assert qcinf.snap_rmsd(chiral.structure, mirror_image(chiral.structure)) > 0.5
    mirror = StructureNode(structure=mirror_image(chiral.structure))

    kept = merge_mirror_images(confs + [mirror], rmsd_cutoff=0.5)
    assert len(kept) <= len(confs)
    assert not any(np.allclose(k.structure.geometry, mirror.structure.geometry) for k in kept)
    assert len(merge_mirror_images([chiral, mirror], rmsd_cutoff=0.5)) == 1


def test_auto_n_embed_scales_with_rotatable_bonds():
    from rdkit import Chem

    from mepd.conformers import _auto_n_embed

    assert _auto_n_embed(Chem.MolFromSmiles("CCCC")) == 50
    assert _auto_n_embed(Chem.MolFromSmiles("C" * 12)) == 200
    assert _auto_n_embed(Chem.MolFromSmiles("C" * 18)) == 300


def test_generate_conformers_respects_n_conformers_cap():
    node = _butane_node()
    confs = generate_conformers(
        node, ConformerInputs(n_conformers=1, n_embed=30, random_seed=0)
    )
    assert len(confs) == 1


def test_generate_conformers_preserves_atom_order_and_reuses_graph():
    node = _butane_node()
    confs = generate_conformers(
        node, ConformerInputs(n_conformers=10, n_embed=30, random_seed=0)
    )
    for conf in confs:
        assert list(conf.structure.symbols) == list(node.structure.symbols)
        assert conf.graph is node.graph


def test_generate_conformers_are_pairwise_distinct_by_rmsd():
    node = _butane_node()
    inputs = ConformerInputs(n_conformers=10, n_embed=30, rmsd_cutoff=0.5, random_seed=0)
    confs = generate_conformers(node, inputs)
    for i in range(len(confs)):
        for j in range(i + 1, len(confs)):
            rmsd = RMSD(confs[i].coords, confs[j].coords)[0]
            assert rmsd >= inputs.rmsd_cutoff


def test_subselect_conformers_discards_a_distorted_embedding_instead_of_crashing(monkeypatch):
    """Regression test: a distorted ETKDG embedding can drift far enough
    that qcinf's own geometry-based connectivity perception disagrees with
    an already-kept conformer's, and `snap_rmsd` raises ValueError
    ("Structures not isomorphic. Same connectivity required.") -- that
    candidate must be dropped, not allowed to crash the whole run."""
    import mepd.conformers as conformers_module

    node = _butane_node()
    candidates = [node] + [node.copy() for _ in range(3)]

    def fake_snap_rmsd(a, b, **kwargs):
        fake_snap_rmsd.calls += 1
        if fake_snap_rmsd.calls == 2:
            # Simulates a distorted candidate whose geometry-perceived
            # connectivity disagrees with an already-kept conformer's.
            raise ValueError("Structures not isomorphic. Same connectivity required.")
        return 1.0  # otherwise always "distinct enough" to keep

    fake_snap_rmsd.calls = 0
    monkeypatch.setattr(conformers_module.qcinf, "snap_rmsd", fake_snap_rmsd)

    # Must not raise, and the poisoned candidate must simply be excluded.
    result = conformers_module._subselect_conformers(candidates, n_max=10, rmsd_cutoff=0.5)
    assert len(result) == len(candidates) - 1


def test_generate_conformers_unknown_backend_raises():
    node = _butane_node()
    with pytest.raises(ValueError):
        generate_conformers(node, ConformerInputs(backend="not-a-backend"))


def _write_fake_crest(tmp_path, frames_text: str):
    """A stand-in `crest` executable that ignores its arguments and writes
    `frames_text` as the ensemble into its working directory, recording the
    argv it was called with next to itself."""
    script = tmp_path / "crest"
    (tmp_path / "ensemble.xyz").write_text(frames_text)
    script.write_text(
        "#!/bin/sh\n"
        f'echo "$@" > "{tmp_path}/argv.txt"\n'
        f'echo "$OMP_NUM_THREADS" > "{tmp_path}/omp.txt"\n'
        f'cp "{tmp_path}/ensemble.xyz" crest_conformers.xyz\n'
    )
    script.chmod(0o755)
    return script


def _frames_xyz(nodes, energies, symbols=None) -> str:
    from qcconst.constants import BOHR_TO_ANGSTROM

    blocks = []
    for node, energy in zip(nodes, energies):
        coords = np.asarray(node.structure.geometry) * BOHR_TO_ANGSTROM
        frame_symbols = symbols or list(node.structure.symbols)
        lines = [str(len(frame_symbols)), f"  {energy:.8f}"]
        lines += [f"{s} {x:.6f} {y:.6f} {z:.6f}" for s, (x, y, z) in zip(frame_symbols, coords)]
        blocks.append("\n".join(lines))
    return "\n".join(blocks) + "\n"


def _distinct_pentane_conformers(n: int) -> list:
    confs = generate_conformers(
        _butane_node(smiles="CCCCC"), ConformerInputs(n_conformers=n, n_embed=50, random_seed=0)
    )
    assert len(confs) >= n
    return confs[:n]


def test_parse_multiframe_xyz_reads_energies_and_tolerates_blank_comment():
    from mepd.conformers import _parse_multiframe_xyz

    text = (
        "3\n -5.1\nO 0 0 0.1\nH 0 0.7 -0.4\nH 0 -0.7 -0.4\n"
        "3\n\nO 0 0 0.2\nH 0 0.7 -0.4\nH 0 -0.7 -0.4\n"
    )
    frames = _parse_multiframe_xyz(text)
    assert [f[2] for f in frames] == [-5.1, None]
    assert frames[1][0] == ["O", "H", "H"]
    assert frames[1][1][0] == [0.0, 0.0, 0.2]


def test_parse_multiframe_xyz_rejects_truncated_frame():
    from mepd.conformers import CrestError, _parse_multiframe_xyz

    with pytest.raises(CrestError):
        _parse_multiframe_xyz("3\ncomment\nO 0 0 0\n")


def test_build_crest_argv_passes_charge_and_unpaired_electrons():
    from mepd.conformers import CrestInputs, _build_crest_argv

    argv = _build_crest_argv(CrestInputs(threads=4), "crest", "in.xyz", 0, 1)
    assert argv == ["crest", "in.xyz", "--gfn2", "--ewin", "6.0", "-T", "4"]
    argv = _build_crest_argv(
        CrestInputs(method="--gfnff", extra_args=("--quick",)), "crest", "in.xyz", -1, 3
    )
    assert argv[argv.index("--chrg") + 1] == "-1"
    assert argv[argv.index("--uhf") + 1] == "2"
    assert argv[-1] == "--quick"


def test_generate_conformers_crest_backend_orders_by_energy_and_keeps_input_first(tmp_path):
    from mepd.conformers import CrestInputs

    input_node, a, b = _distinct_pentane_conformers(3)
    # Written out of energy order: b is CREST's lowest.
    fake = _write_fake_crest(tmp_path, _frames_xyz([a, b], energies=[-10.0, -12.0]))

    confs = generate_conformers(
        input_node,
        ConformerInputs(backend="crest", crest=CrestInputs(executable=str(fake), threads=3)),
    )

    assert "-T 3" in (tmp_path / "argv.txt").read_text()
    assert (tmp_path / "omp.txt").read_text().strip() == "3"
    assert len(confs) == 3
    for got, want in zip(confs, [input_node, b, a]):
        assert np.allclose(got.structure.geometry, want.structure.geometry, atol=1e-4)
        assert list(got.structure.symbols) == list(input_node.structure.symbols)
        assert got.graph is input_node.graph


def test_generate_conformers_crest_backend_drops_reordered_frames(tmp_path):
    from mepd.conformers import CrestInputs

    input_node, a, b = _distinct_pentane_conformers(3)
    reversed_symbols = list(input_node.structure.symbols)[::-1]
    ensemble = _frames_xyz([a], [-12.0]) + _frames_xyz([b], [-13.0], symbols=reversed_symbols)
    fake = _write_fake_crest(tmp_path, ensemble)

    confs = generate_conformers(
        input_node, ConformerInputs(backend="crest", crest=CrestInputs(executable=str(fake)))
    )
    assert len(confs) == 2
    assert np.allclose(confs[1].structure.geometry, a.structure.geometry, atol=1e-4)


def test_generate_conformers_crest_backend_missing_executable_raises():
    from mepd.conformers import CrestError, CrestInputs

    node = _butane_node()
    with pytest.raises(CrestError):
        generate_conformers(
            node,
            ConformerInputs(backend="crest", crest=CrestInputs(executable="no-such-crest-binary")),
        )


def test_conformer_inputs_copy_does_not_share_crest_settings():
    inputs = ConformerInputs(backend="crest")
    copied = inputs.copy()
    copied.crest.threads = 8
    assert inputs.crest.threads == 1


@pytest.mark.skipif(shutil.which("crest") is None, reason="needs the crest binary")
def test_generate_conformers_crest_backend_real_binary_finds_butane_conformers():
    from mepd.conformers import CrestInputs

    node = _butane_node()
    confs = generate_conformers(
        node,
        ConformerInputs(
            backend="crest", n_conformers=5,
            crest=CrestInputs(method="--gfnff", threads=2, extra_args=("--quick",)),
        ),
    )
    assert 1 < len(confs) <= 5
    for conf in confs:
        assert list(conf.structure.symbols) == list(node.structure.symbols)


def _pentane_water(shift_bohr: float, energy: float) -> StructureNode:
    """pentane + water, the water moved `shift_bohr` further out; same
    internal geometry of both molecules every time."""
    from qcconst.constants import ANGSTROM_TO_BOHR

    mol = Chem.AddHs(Chem.MolFromSmiles("CCCCC.O"))
    AllChem.EmbedMolecule(mol, randomSeed=3)
    pos = mol.GetConformer().GetPositions() * ANGSTROM_TO_BOHR
    oxygen = next(a for a in mol.GetAtoms() if a.GetSymbol() == "O")
    water = [oxygen.GetIdx()] + [n.GetIdx() for n in oxygen.GetNeighbors()]
    pentane = [i for i in range(len(pos)) if i not in water]
    pos[water] += pos[pentane].mean(0) + np.array([8.0 + shift_bohr, 0, 0]) - pos[water].mean(0)
    node = StructureNode(structure=Structure(
        geometry=pos, symbols=[a.GetSymbol() for a in mol.GetAtoms()], charge=0, multiplicity=1,
    ))
    node._cached_energy = energy
    return node


def test_merge_degenerate_complex_conformers_ignores_how_far_apart_the_molecules_are():
    """The CREST Diels-Alder failure: a loose complex whose molecules just
    drift apart gives endless all-atom-distinct "conformers" at one energy."""
    from mepd.conformers import merge_degenerate_complex_conformers

    near = _pentane_water(0.0, -10.0)
    far = _pentane_water(6.0, -10.0 + 0.1 / 627.5)    # same molecules, 0.1 kcal/mol
    farther = _pentane_water(20.0, -10.0 + 0.2 / 627.5)
    bound = _pentane_water(1.0, -10.0 - 3.0 / 627.5)  # 3 kcal/mol lower: a real arrangement

    kept = merge_degenerate_complex_conformers([near, far, farther, bound], 0.5, 0.5)
    assert kept == [bound, near]
    # tolerance 0 disables it
    assert len(merge_degenerate_complex_conformers([near, far, farther], 0.5, 0.0)) == 3


def test_merge_degenerate_complex_conformers_leaves_single_molecules_alone():
    from mepd.conformers import merge_degenerate_complex_conformers

    confs = generate_conformers(_butane_node(smiles="CCCCCC"), ConformerInputs(n_embed=30))
    for i, c in enumerate(confs):
        c._cached_energy = 0.0
    assert merge_degenerate_complex_conformers(confs, 0.5, 5.0) == confs


def test_crest_backend_adds_nci_mode_only_for_complexes(tmp_path):
    from mepd.conformers import CrestInputs

    single = _butane_node(smiles="CCCCC")
    fake = _write_fake_crest(tmp_path, _frames_xyz([single], [-1.0]))
    stats = {}
    generate_conformers(single, ConformerInputs(backend="crest", crest=CrestInputs(executable=str(fake))), stats)
    assert "--nci" not in (tmp_path / "argv.txt").read_text() and stats["crest_nci"] is False

    complex_ = _pentane_water(0.0, 0.0)
    fake = _write_fake_crest(tmp_path, _frames_xyz([complex_], [-1.0]))
    generate_conformers(complex_, ConformerInputs(backend="crest", crest=CrestInputs(executable=str(fake))), stats)
    assert "--nci" in (tmp_path / "argv.txt").read_text() and stats["crest_nci"] is True


def test_rdkit_backend_samples_s_cis_butadiene_only_without_torsion_prefs():
    """Regression: ETKDG's experimental torsion preferences give butadiene
    s-trans every time, so a Diels-Alder never had an s-cis diene. The
    default "both" must include s-cis conformers; "etkdg" alone must not."""
    from rdkit.Chem import rdMolTransforms

    def s_cis_count(mode):
        confs = generate_conformers(
            _butane_node(smiles="C=CC=C"),
            ConformerInputs(n_conformers=None, n_embed=50, rdkit_torsion_prefs=mode),
        )
        from qcconst.constants import BOHR_TO_ANGSTROM
        n = 0
        for c in confs:
            g = np.asarray(c.structure.geometry) * BOHR_TO_ANGSTROM
            b0, b1, b2 = g[0] - g[1], g[2] - g[1], g[3] - g[2]
            b1n = b1 / np.linalg.norm(b1)
            v = b0 - (b0 @ b1n) * b1n
            w = b2 - (b2 @ b1n) * b1n
            dih = abs(np.degrees(np.arctan2(np.cross(b1n, v) @ w, v @ w)))
            n += dih < 60
        return n

    assert s_cis_count("etkdg") == 0
    assert s_cis_count("both") >= 1


def test_rdkit_backend_keeps_the_input_when_bond_orders_cant_be_perceived():
    """Radicals / odd charge states (e.g. OH radical) make RDKit's bond-order
    perception raise; that used to crash the whole `channels` run."""
    node = StructureNode(structure=Structure(
        symbols=["O", "H"], geometry=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.83]]),
        charge=0, multiplicity=2,
    ))
    stats = {}
    confs = generate_conformers(node, ConformerInputs(), stats)
    assert len(confs) == 1 and confs[0] is node
    assert stats["n_generated"] == 0 and "rdkit_skipped" in stats
