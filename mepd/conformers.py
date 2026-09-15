"""Conformer generation for a single endpoint (reactant or product), feeding
the conformer-driven MEP-sampling workflow (`mepd conformer-network`): every
reactant conformer is paired with every product conformer and each pair gets
its own recursive MSMEP run.

The source neb-dynamics prototype (`NetworkBuilder.create_endpoint_conformers`)
called `StructureNode.sample_all_conformers(...)`, a CREST wrapper that was
never actually implemented (dead code -- the method doesn't exist anywhere in
that codebase). CREST is the eventual production backend here too (see
`backend="crest"` below), but since it isn't always available, RDKit's ETKDG
embedder is the default and only implemented backend for now.
"""
from __future__ import annotations

from dataclasses import dataclass

import qcinf
from qcconst.constants import ANGSTROM_TO_BOHR
from qcdata.models.structure import Structure

from mepd.nodes.node import StructureNode


@dataclass
class ConformerInputs:
    """
    `backend`: conformer-generation backend. Only 'rdkit' is implemented;
        'crest' is a planned follow-up (raises NotImplementedError for now).

    `n_conformers`: maximum number of distinct conformers to keep for this
        endpoint after deduplication (the per-endpoint cap).

    `n_embed`: number of raw ETKDG embeddings to attempt before dedup/cap
        (rdkit backend only) -- should comfortably exceed `n_conformers`
        since many embeddings collapse to the same conformer after MMFF
        relaxation, especially for rigid molecules.

    `rmsd_cutoff`: minimum pairwise RMSD (bohr) for two conformers to count
        as distinct, via `qcinf.snap_rmsd` -- symmetry/permutation-aware
        (e.g. a methyl group's three hydrogens getting relabeled by the
        embedder doesn't look like a different conformer) and Kabsch-aligned.

    `optimize_with_mmff`: whether to relax each embedded conformer with the
        MMFF94 force field before ranking/deduplication. Only ever a cheap
        pre-filter -- these are not QM minima.

    `random_seed`: seed for ETKDG embedding, for reproducibility.
    """

    backend: str = "rdkit"
    n_conformers: int = 10
    n_embed: int = 50
    rmsd_cutoff: float = 0.5
    optimize_with_mmff: bool = True
    random_seed: int = 0

    def copy(self) -> "ConformerInputs":
        return ConformerInputs(**self.__dict__)


def generate_conformers(
    node: StructureNode, inputs: ConformerInputs | None = None
) -> list[StructureNode]:
    """Generate up to `inputs.n_conformers` distinct conformers of `node`'s
    molecule (same atoms/connectivity/atom order, different 3D geometry),
    ranked most-stable-first (by force-field energy where available) and
    deduplicated by RMSD.

    The input `node` itself is always included as a candidate (and is often
    the one kept, for a molecule with little conformational freedom).
    """
    inputs = inputs or ConformerInputs()
    backend = (inputs.backend or "rdkit").lower()

    if backend == "rdkit":
        candidates = _generate_conformers_rdkit(node, inputs)
    elif backend == "crest":
        raise NotImplementedError(
            "backend='crest' is not implemented yet -- CREST conformer "
            "generation requires the external `crest` binary and is planned "
            "as a follow-up. Use backend='rdkit' for now."
        )
    else:
        raise ValueError(
            f"Unknown conformer backend '{inputs.backend}'. "
            "Known: 'rdkit', 'crest' (not yet implemented)."
        )

    return _subselect_conformers(
        candidates, n_max=inputs.n_conformers, rmsd_cutoff=inputs.rmsd_cutoff
    )


def _generate_conformers_rdkit(
    node: StructureNode, inputs: ConformerInputs
) -> list[StructureNode]:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDetermineBonds

    structure = node.structure
    mol = Chem.MolFromXYZBlock(structure.to_xyz())
    if mol is None:
        raise ValueError("RDKit could not parse this structure's xyz block.")

    # Bond perception + stereochemistry from the input 3D geometry (conformer
    # 0) -- this is what lets ETKDG respect the input's existing stereocenters
    # rather than embedding a random/inconsistent one.
    rdDetermineBonds.DetermineBonds(mol, charge=structure.charge)
    Chem.AssignStereochemistryFrom3D(mol, confId=0)

    params = AllChem.ETKDGv3()
    params.randomSeed = int(inputs.random_seed)
    params.useRandomCoords = True
    params.clearConfs = False  # keep the input geometry as conformer 0
    AllChem.EmbedMultipleConfs(mol, numConfs=int(inputs.n_embed), params=params)

    if inputs.optimize_with_mmff and mol.GetNumConformers() > 1:
        try:
            mmff_results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=2000)
            energies = [energy for _, energy in mmff_results]
        except Exception:
            energies = [None] * mol.GetNumConformers()
    else:
        energies = [None] * mol.GetNumConformers()

    symbols = list(structure.symbols)
    charge = structure.charge
    multiplicity = structure.multiplicity

    scored: list[tuple[float, int, StructureNode]] = []
    for rank, conf in enumerate(mol.GetConformers()):
        positions_angstrom = conf.GetPositions()
        new_structure = Structure(
            geometry=positions_angstrom * ANGSTROM_TO_BOHR,
            symbols=symbols,
            charge=charge,
            multiplicity=multiplicity,
        )
        new_node = StructureNode(structure=new_structure, graph=node.graph)
        energy = energies[rank]
        # Conformer 0 (the original input geometry) always sorts first when
        # its energy is unknown/tied, so it's never discarded by the cap
        # ahead of an RDKit-generated conformer of equal or unknown rank.
        sort_key = energy if energy is not None else float("-inf")
        scored.append((sort_key, rank, new_node))

    scored.sort(key=lambda item: (item[0], item[1]))
    return [node for _, _, node in scored]


def _subselect_conformers(
    conformers: list[StructureNode], n_max: int, rmsd_cutoff: float
) -> list[StructureNode]:
    """Greedily keep conformers (in the given order) whose `qcinf.snap_rmsd`
    (symmetry/permutation-aware, Kabsch-aligned; bohr) is at least
    `rmsd_cutoff` from every conformer already kept, up to a maximum of
    `n_max`."""
    if not conformers:
        return []

    selected = [conformers[0]]
    for candidate in conformers[1:]:
        if len(selected) >= n_max:
            break
        if all(
            qcinf.snap_rmsd(candidate.structure, kept.structure) >= rmsd_cutoff
            for kept in selected
        ):
            selected.append(candidate)
    return selected
