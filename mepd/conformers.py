"""Conformer generation for a single endpoint (reactant or product), feeding
`mepd channels`'s default `--method conformers` seeding strategy: every
reactant conformer is paired with every product conformer and each pair gets
its own recursive MSMEP run.

The source neb-dynamics prototype (`NetworkBuilder.create_endpoint_conformers`)
called `StructureNode.sample_all_conformers(...)`, a CREST wrapper that was
never actually implemented (dead code -- the method doesn't exist anywhere in
that codebase). Two backends are implemented here: RDKit's ETKDG embedder
(`backend="rdkit"`, the default -- cheap, no external dependency) and CREST's
iterative metadynamics (`backend="crest"`, needs the external `crest` binary).
Both feed the same `_subselect_conformers` snap-RMSD dedup/cap.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import qcinf
from qcconst.constants import ANGSTROM_TO_BOHR
from qcdata.models.structure import Structure

from mepd.nodes.node import StructureNode


@dataclass
class CrestInputs:
    """CREST-specific settings for `ConformerInputs(backend="crest")`.

    `executable`: the `crest` binary, resolved on PATH unless absolute.

    `method`: sampling Hamiltonian flag passed straight through -- "--gfn2"
        (default), "--gfnff" (force field, much faster), or "--gfn2//gfnff"
        (GFN-FF sampling, GFN2 reranking). Either way this is CREST's own
        level of theory, not mepd's engine; every kept conformer is
        re-minimized with the engine afterwards by `channels --minimize-ends`.

    `ewin_kcal`: CREST's `--ewin` energy window (kcal/mol) for retained
        conformers. Widening it hands `_subselect_conformers` more candidates;
        it doesn't by itself increase sampling effort.

    `threads`: CREST's `-T`, and the OMP/BLAS thread count CREST runs
        under. CREST's parallel scaling is poor, so 1 (the default) with
        parallelism across reactions is usually the better use of cores.

    `timeout_s`: wall-clock ceiling for one CREST call; None disables it.

    `extra_args`: appended verbatim (e.g. "--quick", "--alpb water").

    `nci_for_complexes`: run CREST in its NCI mode (`--nci`: an ellipsoidal
        wall around the complex) whenever the endpoint is more than one
        molecule. Without it the fragments of a loosely bound complex drift
        apart during the metadynamics, and every separation distance comes
        back as a "distinct" conformer -- a butadiene + ethylene endpoint
        gave 764, of which 744 were the two molecules 5-70 A apart.
    """

    executable: str = "crest"
    method: str = "--gfn2"
    ewin_kcal: float = 6.0
    threads: int = 1
    timeout_s: Optional[float] = 3600
    extra_args: tuple[str, ...] = ()
    nci_for_complexes: bool = True

    def copy(self) -> "CrestInputs":
        return CrestInputs(**self.__dict__)


class CrestError(RuntimeError):
    """CREST could not be run, or produced no usable conformer ensemble."""


@dataclass
class ConformerInputs:
    """
    `backend`: conformer-generation backend, 'rdkit' (ETKDG + MMFF) or
        'crest' (CREST metadynamics; settings in `crest`).

    `n_conformers`: maximum number of distinct conformers to keep for this
        endpoint after deduplication (the per-endpoint cap). None keeps every
        distinct conformer the backend's own parameters let through (CREST's
        energy window; RDKit's `n_embed` and `rdkit_ewin_kcal`).

    `n_embed`: number of raw ETKDG embeddings to attempt before dedup/cap
        (rdkit backend only) -- should comfortably exceed `n_conformers`
        since many embeddings collapse to the same conformer after MMFF
        relaxation, especially for rigid molecules. None picks it from the
        molecule's flexibility (`_auto_n_embed`).

    `rdkit_torsion_prefs`: rdkit backend only -- "both" (default) embeds
        `n_embed` conformers with ETKDG's experimental torsion preferences AND
        `n_embed` without them (plain distance geometry) and pools them;
        "etkdg" or "none" uses just one. The preferences alone never give
        e.g. an s-cis diene.

    `rdkit_ewin_kcal`: rdkit backend only -- keep only embeddings within this
        many kcal/mol (MMFF94) of the lowest one, the counterpart of CREST's
        `--ewin`. None keeps every embedding.

    `rmsd_cutoff`: minimum pairwise RMSD (bohr) for two conformers to count
        as distinct, via `qcinf.snap_rmsd` -- symmetry/permutation-aware
        (e.g. a methyl group's three hydrogens getting relabeled by the
        embedder doesn't look like a different conformer) and Kabsch-aligned.

    `optimize_with_mmff`: whether to relax each embedded conformer with the
        MMFF94 force field before ranking/deduplication. Only ever a cheap
        pre-filter -- these are not QM minima.

    `random_seed`: seed for ETKDG embedding, for reproducibility (rdkit
        backend only; CREST's metadynamics isn't seeded through this
        interface).

    `crest`: CREST settings, used only by the crest backend.
    """

    backend: str = "rdkit"
    n_conformers: Optional[int] = None
    n_embed: Optional[int] = None
    rdkit_ewin_kcal: Optional[float] = None
    rdkit_torsion_prefs: str = "both"
    rmsd_cutoff: float = 0.5
    optimize_with_mmff: bool = True
    random_seed: int = 0
    crest: CrestInputs = field(default_factory=CrestInputs)

    def copy(self) -> "ConformerInputs":
        return ConformerInputs(**{**self.__dict__, "crest": self.crest.copy()})


def generate_conformers(
    node: StructureNode, inputs: ConformerInputs | None = None, stats: dict | None = None
) -> list[StructureNode]:
    """Generate up to `inputs.n_conformers` distinct conformers of `node`'s
    molecule (same atoms/connectivity/atom order, different 3D geometry):
    the input `node` itself first, then the backend's conformers
    most-stable-first, deduplicated by snap-RMSD.

    The input is always kept (and is often the only one, for a molecule with
    little conformational freedom).

    If `stats` is given it is filled with what each stage let through --
    `n_generated` (backend output), `n_in_window` (after the backend's
    energy window), `n_kept` (after dedup/cap, input included) -- and the
    wall time in `seconds`, so backends can be compared on cost as well as
    yield.
    """
    import time

    inputs = inputs or ConformerInputs()
    backend = (inputs.backend or "rdkit").lower()
    stats = stats if stats is not None else {}
    stats["backend"] = backend
    started = time.perf_counter()

    if backend == "rdkit":
        candidates = _generate_conformers_rdkit(node, inputs, stats)
    elif backend == "crest":
        candidates = _generate_conformers_crest(node, inputs, stats)
    else:
        raise ValueError(
            f"Unknown conformer backend '{inputs.backend}'. Known: 'rdkit', 'crest'."
        )

    kept = _subselect_conformers(
        candidates, n_max=inputs.n_conformers, rmsd_cutoff=inputs.rmsd_cutoff,
        stats=stats,
    )
    stats["n_kept"] = len(kept)
    stats["seconds"] = round(time.perf_counter() - started, 3)
    return kept


def _auto_n_embed(mol) -> int:
    """Embedding budget from rotatable-bond count, after Ebejer, Morris &
    Deane (J. Chem. Inf. Model. 2012): 50 for <=7 rotatable bonds, 200 for
    8-12, 300 above -- enough for ETKDG to reproduce bioactive conformers
    without spending 300 embeddings on a rigid ring."""
    from rdkit.Chem import rdMolDescriptors

    n_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    if n_rot <= 7:
        return 50
    if n_rot <= 12:
        return 200
    return 300


def _generate_conformers_rdkit(
    node: StructureNode, inputs: ConformerInputs, stats: dict | None = None
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
    # RDKit derives each conformer's embedding seed from `randomSeed`
    # multiplicatively, so a seed of exactly 0 gives every conformer the same
    # seed and EmbedMultipleConfs returns n_embed copies of one geometry.
    # Shift non-negative seeds by one so the default --random-seed 0 still
    # samples; -1 keeps RDKit's own "unseeded" meaning.
    seed = int(inputs.random_seed)
    params.randomSeed = seed + 1 if seed >= 0 else seed
    params.useRandomCoords = True
    params.clearConfs = False  # conformer 0 stays the input geometry
    n_embed = inputs.n_embed if inputs.n_embed is not None else _auto_n_embed(mol)

    # ETKDG's experimental torsion preferences (crystal-structure statistics)
    # all but forbid conformers that are rare in crystals but matter for
    # reactions: butadiene comes out s-trans in 300/300 embeddings, so a
    # Diels-Alder never gets an s-cis diene to start from. Plain distance
    # geometry (preferences off) samples those, at the cost of rougher
    # geometries that MMFF and the engine minimization clean up. "both"
    # embeds n_embed of each and pools them.
    mode = (inputs.rdkit_torsion_prefs or "both").lower()
    if mode not in ("both", "etkdg", "none"):
        raise ValueError(f"rdkit_torsion_prefs must be 'both', 'etkdg' or 'none', not {mode!r}.")
    counts = {}
    for label, use_prefs in (("etkdg", True), ("none", False)):
        if mode not in ("both", label):
            continue
        params.useExpTorsionAnglePrefs = use_prefs
        # a different seed per batch, so "none" doesn't retrace "etkdg"'s starts
        params.randomSeed = (seed + 1 + (0 if use_prefs else 7919)) if seed >= 0 else seed
        before = mol.GetNumConformers()
        AllChem.EmbedMultipleConfs(mol, numConfs=int(n_embed), params=params)
        counts[label] = mol.GetNumConformers() - before

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

    # ETKDG knows nothing about where the fragments of a non-covalent complex
    # (a bimolecular reactant, a multi-product side) sit relative to each
    # other and embeds them overlapping, which snap-RMSD then rightly rejects
    # as a different molecule -- every embedding lost, the pool silently just
    # the input. What ETKDG does sample well is each fragment's internal
    # conformation, so keep that and put every fragment back where it sits
    # in the input complex. (MMFF's default ignoreInterfragInteractions
    # means the overlap never distorted the fragments themselves.)
    fragments = Chem.GetMolFrags(mol)
    if len(fragments) > 1:
        reference = np.asarray(structure.geometry) / ANGSTROM_TO_BOHR
        for conf in list(mol.GetConformers())[1:]:
            positions = conf.GetPositions()
            for frag in fragments:
                idx = list(frag)
                positions[idx] = _kabsch_onto(positions[idx], reference[idx])
            for i, xyz in enumerate(positions):
                conf.SetAtomPosition(i, xyz.tolist())

    # Conformer 0 is the input geometry (MMFF-relaxed along with the rest,
    # which is only needed for its energy as a window reference); the
    # untouched input `node` itself is prepended below instead.
    generated = list(zip(list(mol.GetConformers())[1:], energies[1:]))
    known = [e for e in energies if e is not None]
    in_window = generated
    if inputs.rdkit_ewin_kcal is not None and known:
        ceiling = min(known) + float(inputs.rdkit_ewin_kcal)
        in_window = [(c, e) for c, e in generated if e is None or e <= ceiling]
    if stats is not None:
        stats["n_embed"] = int(n_embed)
        stats["n_embedded_by_torsion_prefs"] = counts
        stats["n_generated"] = len(generated)
        stats["n_in_window"] = len(in_window)

    scored: list[tuple[float, int, StructureNode]] = []
    for rank, (conf, energy) in enumerate(in_window):
        new_structure = Structure(
            geometry=conf.GetPositions() * ANGSTROM_TO_BOHR,
            symbols=symbols,
            charge=charge,
            multiplicity=multiplicity,
        )
        sort_key = energy if energy is not None else float("inf")
        scored.append((sort_key, rank, StructureNode(structure=new_structure, graph=node.graph)))

    scored.sort(key=lambda item: (item[0], item[1]))
    return [node] + [n for _, _, n in scored]


def _kabsch_onto(mobile, target):
    """`mobile` (n x 3) rigidly rotated and translated to best overlay
    `target` (n x 3), without reflection."""
    mobile = np.asarray(mobile, dtype=float)
    target = np.asarray(target, dtype=float)
    mc, tc = mobile.mean(axis=0), target.mean(axis=0)
    if len(mobile) < 2:
        return mobile - mc + tc
    u, _, vt = np.linalg.svd((mobile - mc).T @ (target - tc))
    d = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag([1.0, 1.0, d]) @ vt
    return (mobile - mc) @ rotation + tc


def _parse_multiframe_xyz(text: str) -> list[tuple[list[str], list[list[float]], Optional[float]]]:
    """Parse a multi-frame xyz (CREST's `crest_conformers.xyz`) into
    `(symbols, coords_angstrom, energy)` per frame. CREST writes the frame's
    energy (Hartree) on the comment line; a comment that doesn't contain a
    number yields `energy=None` rather than failing."""
    frames = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        try:
            n_atoms = int(line)
        except ValueError as exc:
            raise CrestError(
                f"malformed xyz: expected an atom count at line {i + 1}, got {line!r}"
            ) from exc
        if n_atoms <= 0 or i + 2 + n_atoms > len(lines):
            raise CrestError(f"truncated or malformed xyz frame starting at line {i + 1}")

        energy = None
        for token in lines[i + 1].split():
            try:
                energy = float(token)
                break
            except ValueError:
                continue

        symbols, coords = [], []
        for j in range(i + 2, i + 2 + n_atoms):
            parts = lines[j].split()
            if len(parts) < 4:
                raise CrestError(f"malformed xyz atom line {j + 1}: {lines[j]!r}")
            symbols.append(parts[0].capitalize())
            coords.append([float(x) for x in parts[1:4]])
        frames.append((symbols, coords, energy))
        i += 2 + n_atoms
    return frames


def _build_crest_argv(
    crest_inputs: CrestInputs, executable: str, xyz_name: str, charge: int, multiplicity: int
) -> list[str]:
    argv = [executable, xyz_name, crest_inputs.method]
    argv += ["--ewin", str(float(crest_inputs.ewin_kcal))]
    argv += ["-T", str(int(crest_inputs.threads))]
    if charge:
        argv += ["--chrg", str(int(charge))]
    # CREST takes the number of unpaired electrons, not the multiplicity.
    if multiplicity and int(multiplicity) > 1:
        argv += ["--uhf", str(int(multiplicity) - 1)]
    argv += list(crest_inputs.extra_args)
    return argv


def _crest_env(threads: int) -> dict:
    """The caller's environment with every OMP/BLAS thread knob pinned to
    CREST's own `-T`, so an inherited OMP_NUM_THREADS can't silently
    oversubscribe (or undersubscribe) the CREST call."""
    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[var] = str(int(threads))
    return env


def _generate_conformers_crest(
    node: StructureNode, inputs: ConformerInputs, stats: dict | None = None
) -> list[StructureNode]:
    """Run CREST on `node` and return its conformer ensemble, input geometry
    first (mirroring the rdkit backend's "the input is always a candidate")
    and then CREST's conformers lowest-energy-first.

    CREST preserves the input atom order, and its default topology check
    already discards any structure whose bonding changed during the
    metadynamics. A frame whose element sequence still differs from the
    input's is dropped: `node.graph`'s indices refer to the input order, and
    pairing a permuted conformer would hand MSMEP two differently-indexed
    endpoints."""
    crest_inputs = inputs.crest
    executable = shutil.which(crest_inputs.executable)
    if executable is None:
        raise CrestError(
            f"CREST executable {crest_inputs.executable!r} not found. Install "
            "CREST and put it on PATH, or pass its absolute path."
        )

    structure = node.structure
    symbols = list(structure.symbols)

    with tempfile.TemporaryDirectory(prefix="mepd_crest_") as workdir:
        workdir = Path(workdir)
        (workdir / "input.xyz").write_text(structure.to_xyz())
        argv = _build_crest_argv(
            crest_inputs, executable, "input.xyz", structure.charge, structure.multiplicity
        )
        nci = (
            crest_inputs.nci_for_complexes
            and len(fragments(structure)) > 1
            and not any(a.lstrip("-") == "nci" for a in argv)
        )
        if nci:
            argv.append("--nci")
        if stats is not None:
            stats["crest_nci"] = bool(nci or any(a.lstrip("-") == "nci" for a in argv))
        try:
            proc = subprocess.run(
                argv, cwd=workdir, capture_output=True, text=True,
                timeout=crest_inputs.timeout_s, env=_crest_env(crest_inputs.threads),
            )
        except subprocess.TimeoutExpired as exc:
            raise CrestError(
                f"CREST timed out after {crest_inputs.timeout_s}s ({' '.join(argv)})."
            ) from exc

        ensemble = workdir / "crest_conformers.xyz"
        if not ensemble.is_file():
            tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-15:])
            raise CrestError(
                f"CREST exited {proc.returncode} without writing crest_conformers.xyz.\n"
                f"command: {' '.join(argv)}\nlast output:\n{tail}"
            )
        frames = _parse_multiframe_xyz(ensemble.read_text())

    scored: list[tuple[float, int, StructureNode]] = []
    for rank, (frame_symbols, coords, energy) in enumerate(frames):
        if frame_symbols != symbols:
            continue
        new_structure = Structure(
            geometry=[[x * ANGSTROM_TO_BOHR for x in xyz] for xyz in coords],
            symbols=symbols,
            charge=structure.charge,
            multiplicity=structure.multiplicity,
        )
        sort_key = energy if energy is not None else float("inf")
        scored.append((sort_key, rank, StructureNode(structure=new_structure, graph=node.graph)))
    scored.sort(key=lambda item: (item[0], item[1]))
    if stats is not None:
        # CREST applies its own --ewin before writing the ensemble.
        stats["n_generated"] = len(frames)
        stats["n_in_window"] = len(scored)
    return [node] + [n for _, _, n in scored]


def mirror_image(structure: Structure) -> Structure:
    """`structure` reflected through the yz plane."""
    geometry = np.asarray(structure.geometry, dtype=float).copy()
    geometry[:, 0] *= -1.0
    return Structure(
        geometry=geometry, symbols=structure.symbols,
        charge=structure.charge, multiplicity=structure.multiplicity,
    )


def same_up_to_mirror(a: Structure, b: Structure, rmsd_cutoff: float) -> bool:
    """True if `b` is within `rmsd_cutoff` (snap-RMSD, bohr) of `a` or of
    `a`'s mirror image. Raises ValueError like `snap_rmsd` if the two aren't
    the same molecule.

    For an achiral molecule, mirror-image conformers (gauche+/gauche-) are
    equal in energy and lead to mirror-image paths, so as far as the search
    is concerned they are the same conformer -- but `snap_rmsd` only
    superimposes by rotation and never sees it. For a molecule with
    stereocentres, a conformer can't match another's mirror image unless it
    is the enantiomer, which doesn't belong in the pool either."""
    if qcinf.snap_rmsd(a, b) < rmsd_cutoff:
        return True
    return qcinf.snap_rmsd(mirror_image(a), b) < rmsd_cutoff


def merge_mirror_images(
    conformers: list[StructureNode], rmsd_cutoff: float
) -> list[StructureNode]:
    """Drop every conformer that is (within `rmsd_cutoff`) the mirror image
    of one kept earlier in the list."""
    kept: list[StructureNode] = []
    for candidate in conformers:
        try:
            duplicate = any(
                same_up_to_mirror(k.structure, candidate.structure, rmsd_cutoff) for k in kept
            )
        except ValueError:
            duplicate = False
        if not duplicate:
            kept.append(candidate)
    return kept


def fragments(structure: Structure) -> list[list[int]]:
    """Atom indices of each separate molecule in `structure` (bonding
    perceived from the geometry); one group for a single molecule, and a
    single all-atom group if the bonding can't be perceived."""
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    try:
        mol = Chem.MolFromXYZBlock(structure.to_xyz())
        rdDetermineBonds.DetermineConnectivity(mol, charge=structure.charge)
        return [sorted(f) for f in Chem.GetMolFrags(mol)]
    except Exception:
        return [list(range(len(structure.symbols)))]


def _fragment_structure(structure: Structure, idx: list[int]) -> Structure:
    return Structure(
        geometry=np.asarray(structure.geometry)[idx],
        symbols=[structure.symbols[i] for i in idx],
        charge=0, multiplicity=1,
    )


def merge_degenerate_complex_conformers(
    conformers: list[StructureNode], rmsd_cutoff: float, energy_tol_kcal: float,
) -> list[StructureNode]:
    """For an endpoint made of several molecules: treat two conformers as the
    same when EVERY fragment's internal conformation matches (snap-RMSD
    below `rmsd_cutoff`, fragment by fragment) and their energies agree to
    `energy_tol_kcal`, keeping the lowest-energy one.

    All-atom snap-RMSD can't do this for a loosely bound complex: moving one
    molecule 0.5 bohr further from the other already makes a "new"
    conformer, although nothing about either molecule changed and the
    energy didn't either. Arrangements that differ in energy by more than
    the tolerance are kept -- the reactive approach geometry is usually one
    of those.

    Single-molecule pools, and conformers without an energy, pass through
    unchanged."""
    if energy_tol_kcal <= 0 or len(conformers) < 2:
        return list(conformers)
    frags = fragments(conformers[0].structure)
    if len(frags) < 2:
        return list(conformers)

    def _energy(node):
        try:
            return float(node.energy)
        except Exception:
            return None

    with_e = [(n, _energy(n)) for n in conformers]
    known = sorted((it for it in with_e if it[1] is not None), key=lambda it: it[1])
    kept: list[tuple[StructureNode, float]] = []
    for node, energy in known:
        duplicate = False
        for other, other_e in kept:
            if abs(energy - other_e) * 627.5095 >= energy_tol_kcal:
                continue
            try:
                duplicate = all(
                    qcinf.snap_rmsd(
                        _fragment_structure(node.structure, idx),
                        _fragment_structure(other.structure, idx),
                    ) < rmsd_cutoff
                    for idx in frags
                )
            except ValueError:
                duplicate = False
            if duplicate:
                break
        if not duplicate:
            kept.append((node, energy))
    return [n for n, _ in kept] + [n for n, e in with_e if e is None]


def _subselect_conformers(
    conformers: list[StructureNode], n_max: Optional[int], rmsd_cutoff: float,
    stats: dict | None = None,
) -> list[StructureNode]:
    """Greedily keep conformers (in the given order) whose `qcinf.snap_rmsd`
    (symmetry/permutation-aware, Kabsch-aligned; bohr) is at least
    `rmsd_cutoff` from every conformer already kept, up to a maximum of
    `n_max` (None: no maximum).

    `snap_rmsd` perceives connectivity from each structure's own 3D geometry
    and raises `ValueError` if the two don't come out isomorphic. A distorted
    ETKDG embedding (e.g. a ring that opened up, or two atoms pushed on top
    of each other) can drift far enough that this disagrees with an
    already-kept, presumably-good conformer of the same molecule -- that
    candidate isn't a real conformer of `node`'s molecule, so it's discarded
    rather than letting one bad embedding crash the whole run.
    """
    if not conformers:
        return []

    selected = [conformers[0]]
    n_not_isomorphic = 0
    for candidate in conformers[1:]:
        if n_max is not None and len(selected) >= n_max:
            break
        try:
            distinct_from_all_kept = all(
                qcinf.snap_rmsd(candidate.structure, kept.structure) >= rmsd_cutoff
                for kept in selected
            )
        except ValueError:
            n_not_isomorphic += 1
            continue
        if distinct_from_all_kept:
            selected.append(candidate)
    if stats is not None:
        # Reported rather than swallowed: a pool that loses most of its
        # candidates here is a generation failure, not a rigid molecule.
        stats["n_rejected_not_isomorphic"] = n_not_isomorphic
    return selected
