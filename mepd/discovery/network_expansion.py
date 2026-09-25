"""Reaction network expansion: propose products of a structure by graph
rules instead of Hessian sampling, optimize them, and repeat from the new
species.

The built-in generator ("bond-rules") applies every combination of up to
`n_break` bond breaks and `n_form` bond formations to the molecular graph
(formations only between atoms within `form_distance` of each other), keeps
graphs whose atoms stay within normal coordination ranges and that have a
valid Lewis structure for the total charge and spin (RDKit's
DetermineBondOrders), and builds each product's 3D guess from the reactant
geometry by a restrained relaxation onto the new bonds -- so every product
keeps the reactant's atom order and can go straight into a path search.

Methods (see REFERENCES):
  * the break/form enumeration over the bond graph is the ZStruct scheme
    (Zimmerman 2013), in the n-break/n-form form YARP uses (Zhao & Savoie
    2021);
  * the Lewis-structure filter is RDKit's DetermineBondOrders, an
    implementation of xyz2mol (Kim & Kim 2015);
  * the restrained relaxation that builds each product guess is mepd's own
    heuristic (springs on the product bonds, repulsion elsewhere, a weak
    tether to the source geometry), not a published method;
  * the live view's animations are geodesic interpolations (Zhu, Thompson &
    Martinez 2019).

Any other generator plugs in by import path: a callable
`generate(structure, **options)` returning product Structures in the same
atom order (or xyz paths); `products_file` imports products an external
tool (autodE, Chemoton, YARP, ...) already wrote, as a multi-frame xyz in
the reactant's atom order.
"""
from __future__ import annotations

import importlib
import itertools
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np
from qcconst.constants import ANGSTROM_TO_BOHR, HARTREE_TO_KCAL_PER_MOL

from mepd.nodes.node import StructureNode
from mepd.nodes.nodehelpers import _connectivity_matches

OnEvent = Optional[Callable[[str, dict], None]]

# What each stage is, and where it comes from (written into summary.json and
# shown by the CLI and the web UI).
REFERENCES = {
    "enumeration": {
        "method": "break up to n bonds and form up to m bonds on the molecular graph (ZStruct; the b2f2 enumeration of YARP)",
        "cite": ["P. M. Zimmerman, J. Comput. Chem. 34, 1385-1392 (2013), doi:10.1002/jcc.23271",
                 "Q. Zhao, B. M. Savoie, Nat. Comput. Sci. 1, 479-490 (2021), doi:10.1038/s43588-021-00101-3"],
    },
    "lewis_filter": {
        "method": "bond orders and formal charges from connectivity: RDKit DetermineBondOrders (xyz2mol)",
        "cite": ["Y. Kim, W. Y. Kim, Bull. Korean Chem. Soc. 36, 1769-1777 (2015), doi:10.1002/bkcs.10334"],
    },
    "guess_geometry": {
        "method": "mepd heuristic: restrained relaxation of the source geometry onto the product bonds (not a published method)",
        "cite": [],
    },
    "animation": {
        "method": "geodesic interpolation between source and product (live view only)",
        "cite": ["X. Zhu, K. C. Thompson, T. J. Martinez, J. Chem. Phys. 150, 164103 (2019), doi:10.1063/1.5090303"],
    },
}
Edge = tuple[int, int]

# (min, max) number of bonded neighbours an atom may have in a proposed
# product (neutral atoms; a charged system widens these by one).
COORDINATION = {
    "H": (1, 1), "B": (2, 4), "C": (1, 4), "N": (1, 4), "O": (1, 2), "F": (0, 1),
    "Si": (2, 4), "P": (1, 5), "S": (1, 4), "Cl": (0, 1), "Se": (1, 2), "Br": (0, 1), "I": (0, 1),
}


def _emit(on_event: OnEvent, event: str, **payload) -> None:
    if on_event is not None:
        on_event(event, payload)


def _edge(i: int, j: int) -> Edge:
    return (i, j) if i < j else (j, i)


def graph_edges(node: StructureNode) -> set[Edge]:
    return {_edge(int(i), int(j)) for i, j in node.graph.edges}


@dataclass
class Proposal:
    """One proposed product of `source`: the bonds it breaks and forms."""
    source: int
    broken: tuple[Edge, ...]
    formed: tuple[Edge, ...]
    smiles: str
    structure: object = None  # guess geometry (qcdata Structure), reactant atom order
    frames: Optional[list] = field(default=None, repr=False)  # live view: source-to-guess animation

    @property
    def label(self) -> str:
        parts = [f"-{i}-{j}" for i, j in self.broken] + [f"+{i}-{j}" for i, j in self.formed]
        return " ".join(parts) or "external"


def lewis_smiles(symbols: Sequence[str], edges: Iterable[Edge], charge: int = 0, multiplicity: int = 1,
                 *, allow_radicals: bool = False, allow_zwitterions: bool = False) -> Optional[str]:
    """Canonical SMILES of the graph if it has a Lewis structure with the
    given total charge (and, unless `allow_radicals`, exactly
    `multiplicity - 1` unpaired electrons); None otherwise."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdDetermineBonds

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.RWMol()
    for s in symbols:
        atom = Chem.Atom(s)
        atom.SetNoImplicit(True)
        mol.AddAtom(atom)
    for i, j in edges:
        mol.AddBond(int(i), int(j), Chem.BondType.SINGLE)
    mol.AddConformer(Chem.Conformer(len(symbols)))
    mol = mol.GetMol()
    try:
        rdDetermineBonds.DetermineBondOrders(mol, charge=int(charge), allowChargedFragments=True)
    except Exception:
        return None
    if sum(a.GetFormalCharge() for a in mol.GetAtoms()) != charge:
        return None
    n_charged = sum(1 for a in mol.GetAtoms() if a.GetFormalCharge() != 0)
    if n_charged > abs(charge) + (2 if allow_zwitterions else 0):
        return None
    radicals = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    if not allow_radicals and radicals != multiplicity - 1:
        return None
    try:
        return Chem.MolToSmiles(Chem.RemoveHs(mol, sanitize=False))
    except Exception:
        return Chem.MolToSmiles(mol)


def _coordination_ok(symbols, degree, charged: bool) -> bool:
    for s, d in zip(symbols, degree):
        lo, hi = COORDINATION.get(s, (0, 6))
        if d < lo - charged or d > hi + charged:
            return False
    return True


def enumerate_bond_changes(
    symbols: Sequence[str], coords_angstrom: np.ndarray, edges: set[Edge], *,
    charge: int = 0, multiplicity: int = 1, n_break: int = 2, n_form: int = 2,
    form_distance: float = 4.0, max_products: int = 50, max_combinations: int = 500_000,
    allow_radicals: bool = False, allow_zwitterions: bool = False, source: int = 0,
    exclude_smiles: Iterable[str] = (),
) -> tuple[list[Proposal], dict]:
    """Every distinct product species reachable by <= n_break breaks and
    <= n_form formations, fewest changes first (then the closest formed
    atoms), at most `max_products`. Also returns counts for reporting."""
    n = len(symbols)
    coords = np.asarray(coords_angstrom, dtype=float)
    dist = np.linalg.norm(coords[:, None] - coords[None], axis=-1)
    bonds = sorted(edges)
    formable = sorted(_edge(i, j) for i in range(n) for j in range(i + 1, n)
                      if (i, j) not in edges and dist[i, j] <= form_distance)
    degree0 = np.zeros(n, dtype=int)
    for i, j in bonds:
        degree0[i] += 1
        degree0[j] += 1
    seen = set(exclude_smiles)
    reactant = lewis_smiles(symbols, bonds, charge, multiplicity, allow_radicals=True, allow_zwitterions=True)
    if reactant:
        seen.add(reactant)
    found: list[tuple[tuple, Proposal]] = []
    stats = {"combinations": 0, "lewis_checked": 0, "clipped": False, "formable_pairs": len(formable)}
    for nb, nf in sorted(((b, f) for b in range(n_break + 1) for f in range(n_form + 1) if b + f),
                         key=lambda bf: (sum(bf), bf)):
        for broken in itertools.combinations(bonds, nb):
            deg_b = degree0.copy()
            for i, j in broken:
                deg_b[i] -= 1
                deg_b[j] -= 1
            for formed in itertools.combinations(formable, nf):
                stats["combinations"] += 1
                if stats["combinations"] > max_combinations:
                    stats["clipped"] = True
                    break
                deg = deg_b.copy()
                for i, j in formed:
                    deg[i] += 1
                    deg[j] += 1
                if not _coordination_ok(symbols, deg, charge != 0):
                    continue
                product = (set(bonds) - set(broken)) | set(formed)
                stats["lewis_checked"] += 1
                smi = lewis_smiles(symbols, product, charge, multiplicity,
                                   allow_radicals=allow_radicals, allow_zwitterions=allow_zwitterions)
                if smi is None or smi in seen:
                    continue
                seen.add(smi)
                key = (nb + nf, sum(dist[i, j] for i, j in formed))
                found.append((key, Proposal(source, tuple(broken), tuple(formed), smi)))
            if stats["clipped"]:
                break
        if stats["clipped"]:
            break
    found.sort(key=lambda kp: kp[0])
    stats["distinct_products"] = len(found)
    return [p for _, p in found[:max_products]], stats


def embed_product(symbols: Sequence[str], coords_angstrom: np.ndarray, product_edges: set[Edge],
                  *, tether: float = 0.02, attempts: int = 6) -> np.ndarray:
    """A guess geometry (Angstrom) for the product graph, starting from the
    reactant's: bonded pairs pulled to covalent length, 1-3 pairs and
    non-bonded pairs pushed out of contact, a weak tether to the start so
    the rest of the molecule stays put."""
    from rdkit import Chem
    from scipy.optimize import minimize

    table = Chem.GetPeriodicTable()
    x0 = np.asarray(coords_angstrom, dtype=float)
    n = len(symbols)
    rcov = np.array([table.GetRcovalent(s) for s in symbols])
    iu, ju = np.triu_indices(n, 1)
    rsum = rcov[iu] + rcov[ju]
    adj = np.zeros((n, n), dtype=bool)
    for i, j in product_edges:
        adj[i, j] = adj[j, i] = True
    one_three = (adj.astype(int) @ adj.astype(int)) > 0
    bonded = adj[iu, ju]
    target = np.where(bonded, rsum, np.where(one_three[iu, ju], 1.5 * rsum, np.minimum(2.2 * rsum, 3.0)))

    def fg(flat):
        x = flat.reshape(n, 3)
        d = x[iu] - x[ju]
        r = np.linalg.norm(d, axis=1) + 1e-12
        dev = r - target
        active = bonded | (dev < 0)  # bonds are springs; everything else only repels
        e = np.sum(dev[active] ** 2) + tether * np.sum((x - x0) ** 2)
        coef = np.where(active, 2 * dev / r, 0.0)[:, None] * d
        g = 2 * tether * (x - x0)
        np.add.at(g, iu, coef)
        np.add.at(g, ju, -coef)
        return e, g.ravel()

    def relax(start):
        res = minimize(fg, start.ravel(), jac=True, method="L-BFGS-B", options={"maxiter": 2000})
        return res.x.reshape(n, 3), float(res.fun)

    # Restarts from randomly kicked starts escape symmetric traps (e.g. an H
    # moving along a linear molecule's axis has to go around the atom in
    # between, not through it); the first whose bonds are the product's wins.
    rng = np.random.default_rng(0)
    best = None
    for attempt in range(attempts):
        x, fun = relax(x0 + (0.0 if attempt == 0 else 0.3 * attempt) * rng.standard_normal(x0.shape))
        if _perceived_edges(symbols, x) == set(product_edges):
            return x
        if best is None or fun < best[1]:
            best = (x, fun)
    return best[0]


def _perceived_edges(symbols, coords_angstrom) -> set[Edge]:
    """Bonds as mepd perceives them (openbabel, as StructureNode.graph)."""
    from qcdata import Structure

    node = StructureNode(structure=Structure(symbols=list(symbols), geometry=np.asarray(coords_angstrom) * ANGSTROM_TO_BOHR))
    return graph_edges(node)


def _structure_with(structure, coords_angstrom):
    return structure.model_copy(update={"geometry": np.asarray(coords_angstrom) * ANGSTROM_TO_BOHR})


def _import_generator(path: str) -> Callable:
    module_name, _, attr = str(path).replace(":", ".").rpartition(".")
    if not module_name:
        raise ValueError(f"generator must be 'bond-rules' or 'package.module:function', got {path!r}.")
    return getattr(importlib.import_module(module_name), attr)


def external_proposals(structures: Iterable, source_structure, source: int) -> list[Proposal]:
    """Proposals from product structures made elsewhere; they must list the
    same elements in the same order as the source."""
    from qcdata import Structure

    out = []
    for k, s in enumerate(structures):
        if not isinstance(s, Structure):
            s = Structure.open(str(s))
        if list(s.symbols) != list(source_structure.symbols):
            raise ValueError(f"Product {k} does not list the source's atoms in the same order "
                             f"({len(s.symbols)} vs {len(source_structure.symbols)} atoms); products must "
                             "be atom-mapped to the reactant.")
        s = s.model_copy(update={"charge": source_structure.charge, "multiplicity": source_structure.multiplicity})
        out.append(Proposal(source, (), (), f"external-{k}", structure=s))
    return out


@dataclass
class Species:
    node: StructureNode
    smiles: str
    round: int
    rel_energy_kcal: float = 0.0
    validation: Optional[dict] = None


@dataclass
class ProposedEdge:
    source: int
    proposal: Proposal
    outcome: str  # "new_species", "known_species", "reverted", "not_minimum", "failed"
    target: Optional[int] = None  # species index it optimized to
    intended: Optional[bool] = None  # landed on the proposed connectivity
    error: str = ""


@dataclass
class ExpansionResult:
    species: List[Species] = field(default_factory=list)
    edges: List[ProposedEdge] = field(default_factory=list)
    rounds: List[dict] = field(default_factory=list)
    rejected: List[StructureNode] = field(default_factory=list)  # not minima after the Hessian check

    def connections(self) -> list[tuple[int, int]]:
        """Distinct (source, product) species pairs to connect by path search."""
        pairs = []
        for e in self.edges:
            if e.target is not None and e.target != e.source:
                pair = (min(e.source, e.target), max(e.source, e.target))
                if pair not in pairs:
                    pairs.append(pair)
        return pairs


def _optimize(engine, nodes: list[StructureNode], maxiter: int, on_event: OnEvent,
              on_start: Callable[[int], None] = lambda k: None) -> list:
    """Optimized node (or an exception) per input, isolating failures.
    `on_start(k)` is called as optimization k (0-based) starts."""
    keywords = {"coordsys": "cart", "maxiter": int(maxiter)}
    batch = getattr(engine, "compute_geometry_optimizations", None)
    if callable(batch) and len(nodes) > 1:
        for k in range(len(nodes)):
            on_start(k)
        try:
            trajs = batch(nodes, keywords=keywords)
            if len(trajs) == len(nodes):
                _emit(on_event, "candidate_done", index=len(nodes), total=len(nodes))
                return [t[-1] if t else ValueError("empty trajectory") for t in trajs]
        except Exception:
            pass
    out = []
    for k, node in enumerate(nodes, start=1):
        on_start(k - 1)
        try:
            try:
                traj = engine.compute_geometry_optimization(node, keywords=keywords)
            except TypeError:
                traj = engine.compute_geometry_optimization(node)
            out.append(traj[-1] if traj else ValueError("empty trajectory"))
        except Exception as exc:
            out.append(exc)
        _emit(on_event, "candidate_done", index=k, total=len(nodes))
    return out


_OUTCOME_LABEL = {"new_species": "new species", "known_species": "known species", "reverted": "back to source",
                  "not_minimum": "not a minimum", "failed": "failed"}


def _morph_frames(job) -> list[str]:
    """xyz frames of a geodesic interpolation (Zhu, Thompson & Martinez,
    J. Chem. Phys. 150, 164103 (2019)) from structure a to b, moved as one
    rigid body so the first frame sits on `reference` (Bohr). A plain
    function of plain data, so it runs in a worker process."""
    from qcdata import Structure

    from mepd.chainhelpers import run_geodesic

    symbols, charge, mult, xa, xb, reference, nimages = job
    a, b = (StructureNode(structure=Structure(symbols=list(symbols), geometry=np.asarray(x), charge=charge,
                                              multiplicity=mult)) for x in (xa, xb))
    try:
        nodes = list(run_geodesic([a, b], nimages=nimages))
    except Exception:
        nodes = [a, b]
    coords = [np.asarray(n.coords, dtype=float) for n in nodes]
    shift = coords[0].mean(axis=0)
    u, _, vt = np.linalg.svd((coords[0] - shift).T @ reference)
    rot = u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt
    return [n.structure.model_copy(update={"geometry": (c - shift) @ rot}).to_xyz() for n, c in zip(nodes, coords)]


class _LiveReactions:
    """Live view (web UI): one stream per proposed reaction, animating the
    source turning into the product by geodesic interpolation -- into the
    proposed guess while it waits and optimizes, then into what it
    optimized to. Costs nothing when nobody is watching."""

    def __init__(self, seed: StructureNode, seed_energy: float, nimages: int = 16, workers: int = 1):
        from mepd import progress

        self._progress = progress
        self.enabled = progress._stream_path("probe") is not None
        self.e0 = seed_energy
        self.nimages = nimages
        self.workers = max(1, int(workers))
        self._streams: dict[int, str] = {}
        self._frames_of: dict[int, list] = {}   # proposal -> seed-to-guess frames
        self._pool = None
        self._pending: list = []
        # Every animation starts from the seed in one fixed pose (centred,
        # principal axes along x, y, z), so they all begin in the same place.
        x = np.asarray(seed.coords, dtype=float)
        x = x - x.mean(axis=0)
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        if np.linalg.det(vt) < 0:
            vt[-1] *= -1
        self.reference = x @ vt.T

    def job(self, a: StructureNode, b: StructureNode):
        st = a.structure
        return (list(st.symbols), int(st.charge), int(st.multiplicity), np.asarray(a.coords, dtype=float),
                np.asarray(b.coords, dtype=float), self.reference, self.nimages)

    def _submit(self, job, done: Callable[[list], None]) -> None:
        """Compute frames off the main loop (forked workers when that is safe,
        else threads) and hand them to `done` when ready."""
        if self._pool is None:
            from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

            from mepd.cli_common import _unsafe_to_fork

            if self.workers > 1 and _unsafe_to_fork() is None:
                import multiprocessing

                self._pool = ProcessPoolExecutor(self.workers, mp_context=multiprocessing.get_context("fork"))
            else:
                self._pool = ThreadPoolExecutor(1)
        fut = self._pool.submit(_morph_frames, job)
        fut.add_done_callback(lambda f: done(f.result() if f.exception() is None else []))
        self._pending.append(fut)

    def close(self) -> None:
        """Wait for the last animations, so every stream ends finished."""
        if self._pool is not None:
            for fut in self._pending:
                try:
                    fut.result()
                except Exception:
                    pass
            self._pool.shutdown(wait=True)
            self._pool = None

    def _write(self, p: Proposal, species, frames: list, *, status: str, outcome=None,
               product_kcal=None, product_smiles=None, note: str = "") -> None:
        stream = self._streams.setdefault(id(p), f"rxn{len(self._streams) + 1:04d}")
        src = species[p.source]
        sym = list(src.node.symbols)
        bonds = [f"−{sym[i]}{i}–{sym[j]}{j}" for i, j in p.broken] + [f"+{sym[i]}{i}–{sym[j]}{j}" for i, j in p.formed]
        self._progress.write_morph(
            stream, frames, label=f"#{int(stream[3:])} from species {p.source}",
            caption=(" ".join(bonds) or "from the external generator") + note,
            status=status, finished=status in ("done", "failed"), outcome=outcome,
            energies_kcal=(src.rel_energy_kcal, product_kcal),
            reactant_smiles=src.smiles, product_smiles=product_smiles or p.smiles,
            extra={"source": p.source, "bonds": " ".join(bonds)})

    def _event(self, edge: ProposedEdge, species) -> None:
        """A new species (or a new link between two known ones) as soon as
        it is accepted, so a viewer can add it to its network right away."""
        src = species[edge.source]
        sym = list(src.node.symbols)
        p = edge.proposal
        caption = " ".join([f"−{sym[i]}{i}–{sym[j]}{j}" for i, j in p.broken]
                           + [f"+{sym[i]}{i}–{sym[j]}{j}" for i, j in p.formed])
        if edge.outcome == "new_species":
            s = species[edge.target]
            self._progress.append_live_event({
                "event": "species", "index": edge.target, "parent": edge.source, "smiles": s.smiles,
                "energy_hartree": float(s.node.energy), "rel_energy_kcal": s.rel_energy_kcal, "round": s.round,
                "validation": s.validation, "xyz": s.node.structure.to_xyz(), "caption": caption})
        elif edge.outcome == "known_species" and edge.target is not None and edge.target != edge.source:
            self._progress.append_live_event({
                "event": "reaction", "source": edge.source, "target": edge.target, "caption": caption})

    def propose(self, p: Proposal, species, frames: Optional[list] = None, status: str = "queued") -> None:
        if not self.enabled:
            return
        if frames is not None:
            self._frames_of[id(p)] = frames
        frames = self._frames_of.get(id(p))
        if frames is None:  # not built with the guess (external generator): make it now
            frames = _morph_frames(self.job(species[p.source].node, StructureNode(structure=p.structure)))
            self._frames_of[id(p)] = frames
        self._write(p, species, frames, status=status)

    def optimizing(self, p: Proposal, species) -> None:
        self.propose(p, species, status="running")

    def finish(self, edge: ProposedEdge, species, product: Optional[StructureNode] = None) -> None:
        if not self.enabled:
            return
        self._event(edge, species)
        p = edge.proposal
        node = product if product is not None else (
            species[edge.target].node if edge.target is not None else StructureNode(structure=p.structure))
        try:
            kcal = (float(node.energy) - self.e0) * HARTREE_TO_KCAL_PER_MOL
        except Exception:
            kcal = None
        note = " · relaxed to a different product than proposed" if edge.intended is False else ""
        if edge.error:
            note += f" · {edge.error}"
        write = lambda frames: self._write(  # noqa: E731
            p, species, frames or self._frames_of.get(id(p)) or [], status="failed" if edge.outcome == "failed" else "done",
            outcome=_OUTCOME_LABEL.get(edge.outcome, edge.outcome), product_kcal=kcal,
            product_smiles=species[edge.target].smiles if edge.target is not None else None, note=note)
        self._submit(self.job(species[p.source].node, node), write)


def _build_guesses(props: list[Proposal], node: StructureNode, coords: np.ndarray, live, workers: int) -> None:
    """Each proposal's 3D guess (and, for a watching live view, its
    seed-to-guess animation), across `workers` forked processes."""
    from mepd.cli_common import _fork_map

    symbols = list(node.symbols)
    edges = graph_edges(node)

    def build(k):
        p = props[k]
        guess = embed_product(symbols, coords, (edges - set(p.broken)) | set(p.formed))
        frames = None
        if live.enabled:
            structure = _structure_with(node.structure, guess)
            frames = _morph_frames(live.job(node, StructureNode(structure=structure)))
        return guess, frames

    for p, (guess, frames) in zip(props, _fork_map(build, list(range(len(props))), workers)):
        p.structure = _structure_with(node.structure, guess)
        p.frames = frames


def expand_network(
    seed: StructureNode, engine, *, rounds: int = 1, energy_window_kcal: float = 60.0,
    generator: str = "bond-rules", generator_options: Optional[dict] = None,
    products_file: Optional[str] = None, maxiter: int = 500, n_break: int = 2, n_form: int = 2,
    form_distance: float = 4.0, max_products: int = 50, allow_radicals: bool = False,
    allow_zwitterions: bool = False, max_species: int = 200, validate_minima: Optional[dict] = None,
    workers: int = 1, on_event: OnEvent = None,
) -> ExpansionResult:
    """Breadth-first network expansion from `seed`. Each round proposes
    products of every species found in the previous round (within
    `energy_window_kcal` of the seed), optimizes them, and adds the ones
    that are new species. Species identity is connectivity + stereo
    (conformers of one species are merged). A proposal whose SMILES is an
    already-known species becomes an edge to it without being optimized
    again. `validate_minima` ({"frequency_cutoff", "rescue_displacement"})
    Hessian-checks every new species before it is accepted."""
    seed = seed.copy()
    if seed._cached_energy is None:
        engine.compute_energies([seed])
    e0 = float(seed.energy)
    symbols = list(seed.symbols)
    charge, mult = int(seed.structure.charge), int(seed.structure.multiplicity)
    result = ExpansionResult(species=[Species(seed, lewis_smiles(symbols, graph_edges(seed), charge, mult,
                                                                 allow_radicals=True, allow_zwitterions=True) or "", 0)])
    live = _LiveReactions(seed, e0, workers=workers)

    def _classify(p, guess, opt, record) -> ProposedEdge:
        intended = _connectivity_matches(opt, guess) if p.broken or p.formed else None
        match = next((k for k, s in enumerate(result.species) if _connectivity_matches(opt, s.node)), None)
        if match is not None:
            if match != p.source and float(opt.energy) < float(result.species[match].node.energy):
                result.species[match].node = opt.copy()  # keep the lowest conformer found
                result.species[match].rel_energy_kcal = (float(opt.energy) - e0) * HARTREE_TO_KCAL_PER_MOL
            return ProposedEdge(p.source, p, "reverted" if match == p.source else "known_species", match, intended)
        if len(result.species) >= max_species:
            return ProposedEdge(p.source, p, "failed", error="max_species reached")
        rel = (float(opt.energy) - e0) * HARTREE_TO_KCAL_PER_MOL
        smi = lewis_smiles(symbols, graph_edges(opt), charge, mult, allow_radicals=True, allow_zwitterions=True) or ""
        result.species.append(Species(opt.copy(), smi, rnd, rel, record))
        idx = len(result.species) - 1
        _emit(on_event, "species_found", index=idx, smiles=smi, rel_energy_kcal=rel, round=rnd)
        return ProposedEdge(p.source, p, "new_species", idx, intended)

    frontier = [0]
    for rnd in range(1, rounds + 1):
        if not frontier:
            break
        proposals, stats_all = [], []
        for src in frontier:
            node = result.species[src].node
            coords = np.asarray(node.coords) / ANGSTROM_TO_BOHR
            if products_file is not None or generator != "bond-rules":
                if products_file is not None:  # an external tool's products of the seed
                    from mepd.qcdata_structure_helpers import read_multiple_structure_from_file
                    items = read_multiple_structure_from_file(products_file, charge, mult) if src == 0 else []
                else:
                    items = _import_generator(generator)(node.structure, **dict(generator_options or {}))
                props = external_proposals(items, node.structure, src)
                stats = {"external": len(props)}
            else:
                props, stats = enumerate_bond_changes(
                    symbols, coords, graph_edges(node), charge=charge, multiplicity=mult,
                    n_break=n_break, n_form=n_form, form_distance=form_distance, max_products=max_products,
                    allow_radicals=allow_radicals, allow_zwitterions=allow_zwitterions, source=src,
                )
                _build_guesses(props, node, coords, live, workers)
            stats_all.append({"source": src, **stats})
            proposals.extend(props)
        known = {s.smiles: k for k, s in enumerate(result.species) if s.smiles}
        for p in [p for p in proposals if p.smiles in known]:
            edge = ProposedEdge(p.source, p, "known_species", known[p.smiles], True)
            result.edges.append(edge)
            live.finish(edge, result.species)
        proposals = [p for p in proposals if p.smiles not in known]
        for p in proposals:
            live.propose(p, result.species, frames=p.frames)
        _emit(on_event, "proposed", round=rnd, total=len(proposals))
        guesses = [StructureNode(structure=p.structure) for p in proposals]
        optimized = _optimize(engine, guesses, maxiter, on_event,
                              on_start=lambda k: live.optimizing(proposals[k], result.species))
        new_frontier = []
        for p, guess, opt in zip(proposals, guesses, optimized):
            edge, record = None, None
            if isinstance(opt, Exception):
                edge = ProposedEdge(p.source, p, "failed", error=f"{type(opt).__name__}: {opt}")
            elif validate_minima is not None:
                from mepd.elementarystep import validate_minimum_with_rescue

                opt, record = validate_minimum_with_rescue(
                    opt, engine, frequency_cutoff=float(validate_minima.get("frequency_cutoff", 0.0)),
                    rescue_displacement=float(validate_minima.get("rescue_displacement", 0.1)),
                    label=f"proposal {p.label}")
                if not record["is_minimum"]:
                    edge = ProposedEdge(p.source, p, "not_minimum", error=str(record.get("validation")))
                    result.rejected.append(opt.copy())
            if edge is None:
                edge = _classify(p, guess, opt, record)
            result.edges.append(edge)
            live.finish(edge, result.species, product=None if isinstance(opt, Exception) else opt)
            if edge.outcome == "new_species" and result.species[edge.target].rel_energy_kcal <= energy_window_kcal:
                new_frontier.append(edge.target)
        found = [e.target for e in result.edges if e.outcome == "new_species" and result.species[e.target].round == rnd]
        result.rounds.append({"round": rnd, "sources": frontier, "proposed": len(proposals),
                              "new_species": found, "expanded_next": new_frontier, "generator_stats": stats_all})
        frontier = new_frontier
    live.close()
    return result
