"""Path searches on QM/MM systems: the solvent has to correspond between the
two ends, and move continuously along the path.

Two ends minimized separately carry unrelated arrangements of the moving
environment (waters up to several Å apart, or swapped). A path search between
them mostly moves solvent, through each other: high, meaningless "barriers",
and TS searches that slide off the reaction. So, before a QM/MM path search:

1. **Matching** (`match_solvent`): identical environment molecules are
   interchangeable, so the end's moving solvent molecules (and identical atoms
   within each, e.g. a water's two H) are renumbered to sit as close as
   possible to the start's (assignment on centroid distances, then on atom
   positions within each pair).
2. **Marching** (`march`): the QM region is interpolated (geodesic, on the QM
   atoms and the MM atoms of cut bonds only), and the moving environment is
   relaxed image by image, each image starting from the previous image's
   environment, with the QM region fixed (the microiteration machinery,
   mepd.engines.microiter). The environment then follows the chemistry
   continuously.
3. The **end** is re-minimized from where the march arrives (its QM species is
   the same; only its solvent configuration now descends from the start's).

The result is the initial path (and the ends) the path search starts from.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from qcconst.constants import ANGSTROM_TO_BOHR

from mepd.qmmm import QMMMRegion, bonds, molecules


def solvent_molecules(region: QMMMRegion, coords_bohr: np.ndarray, movable: Optional[set] = None) -> list[list[int]]:
    """Whole environment molecules whose every atom moves."""
    xyz = np.asarray(coords_bohr, dtype=float) / ANGSTROM_TO_BOHR
    movable = set(region.active_mm_atoms) if movable is None else set(movable)
    partners = {m for _, m in region.links}
    out = []
    for mol in molecules(region.natoms, bonds(region.symbols, xyz)):
        if all(a in movable and a not in partners for a in mol):
            out.append(sorted(mol))
    return out


def match_solvent(region: QMMMRegion, start_bohr: np.ndarray, end_bohr: np.ndarray) -> tuple[np.ndarray, dict]:
    """The end's coordinates with its moving solvent molecules renumbered to
    match the start's. Returns (new end coordinates, report)."""
    from scipy.optimize import linear_sum_assignment

    a = np.asarray(start_bohr, dtype=float)
    b = np.asarray(end_bohr, dtype=float)
    mols = solvent_molecules(region, a)
    groups: dict[tuple, list[list[int]]] = {}
    for m in mols:
        groups.setdefault(tuple(region.symbols[i] for i in m), []).append(m)
    out = b.copy()
    swapped = 0
    for _, ms in groups.items():
        ca = np.array([a[m].mean(axis=0) for m in ms])
        cb = np.array([b[m].mean(axis=0) for m in ms])
        cost = np.linalg.norm(ca[:, None] - cb[None], axis=2) ** 2
        rows, cols = linear_sum_assignment(cost)
        for i, j in zip(rows, cols):
            slots, src = ms[i], ms[j]
            if i != j:
                swapped += 1
            for el in set(region.symbols[k] for k in slots):
                si = [k for k in slots if region.symbols[k] == el]
                sj = [k for k in src if region.symbols[k] == el]
                c = np.linalg.norm(a[si][:, None] - b[sj][None], axis=2) ** 2
                r2, c2 = linear_sum_assignment(c)
                for p, q in zip(r2, c2):
                    out[si[p]] = b[sj[q]]
    moved = lambda y: np.linalg.norm((y - a)[[k for m in mols for k in m]], axis=1) / ANGSTROM_TO_BOHR \
        if mols else np.zeros(1)
    before, after = moved(b), moved(out)
    return out, {"molecules": len(mols), "renumbered": swapped,
                 "max_before": float(before.max()), "max_after": float(after.max()),
                 "rms_before": float(np.sqrt((before ** 2).mean())), "rms_after": float(np.sqrt((after ** 2).mean()))}


def march(engine, start_node, end_bohr: np.ndarray, nimages: int, *, echo=print) -> list[np.ndarray]:
    """Coordinates of `nimages` images from the start to the end's QM region:
    the QM region interpolated, the moving environment relaxed image by
    image from the previous image's (see the module doc). The last image is
    the end's QM region in the start's descended solvent (not minimized)."""
    from qcdata import Structure

    from mepd.chainhelpers import run_geodesic
    from mepd.engines.microiter import MicroIterations
    from mepd.nodes.node import StructureNode

    qmmm = engine.base
    micro = MicroIterations(qmmm, getattr(engine, "frozen", []))
    outer = micro.outer
    x0 = np.asarray(start_node.coords, dtype=float)
    xe = np.asarray(end_bohr, dtype=float)
    structure = start_node.structure
    sub = [str(structure.symbols[i]) for i in outer]
    a = StructureNode(structure=Structure(symbols=sub, geometry=x0[outer]), has_molecular_graph=False)
    b = StructureNode(structure=Structure(symbols=sub, geometry=xe[outer]), has_molecular_graph=False)
    try:
        qm_path = [np.asarray(n.coords) for n in run_geodesic([a, b], nimages=nimages, align=False)]
    except Exception:
        qm_path = [x0[outer] + t * (xe[outer] - x0[outer]) for t in np.linspace(0, 1, nimages)]
    images = [x0.copy()]
    x = x0.copy()
    charge = int(structure.charge)
    for k in range(1, len(qm_path)):
        y = x.copy()
        y[outer] = qm_path[k]
        res = qmmm.evaluate([StructureNode(structure=structure.model_copy(update={"geometry": y}),
                                           has_molecular_graph=False)])[0]
        charges = res.get("qm_charges")
        corr = micro._correction(y, res, charges, charge)
        for _ in range(3):      # a few bounded relaxations: the environment can follow further
            y_new = micro.relax(y, charges, charge, corr)
            done = np.abs(y_new - y).max() < 0.4
            y = y_new
            if done:
                break
        images.append(y)
        x = y
        echo(f"  image {k}/{len(qm_path) - 1}: QM/MM energy {res['energy']:.6f} Eh before the environment relaxed")
    return images


def prepare(run_inputs, start_node, end_node, nimages: int, *, echo=print):
    """Matched, marched and re-minimized: (start node, end node, image
    coordinates incl. both ends) for a QM/MM path search, or None when the
    run is not on a QM/MM system with a moving environment."""
    from mepd.engines.frozen import FrozenAtomsEngine
    from mepd.engines.qmmm import QMMMEngine
    from mepd.nodes.node import StructureNode

    eng = run_inputs.engine
    if not (isinstance(eng, FrozenAtomsEngine) and isinstance(eng.base, QMMMEngine)):
        return None
    region = eng.base.region
    if not region.active_mm_atoms or not getattr(region, "solvent_correspondence", True):
        return None
    x0 = np.asarray(start_node.coords, dtype=float)
    xe, rep = match_solvent(region, x0, np.asarray(end_node.coords, dtype=float))
    echo(f"Solvent correspondence: {rep['molecules']} moving solvent molecules, {rep['renumbered']} renumbered to match "
         f"the start's; largest solvent move {rep['max_before']:.2f} -> {rep['max_after']:.2f} Å "
         f"(rms {rep['rms_before']:.2f} -> {rep['rms_after']:.2f}).")
    echo(f"Solvent march: relaxing the moving environment along the {nimages}-image path, each image from the "
         "previous one...")
    images = march(eng, start_node, xe, nimages, echo=echo)
    echo("Minimizing the end from where the march arrives (its solvent descends from the start's)...")
    traj = eng.compute_geometry_optimization(
        StructureNode(structure=end_node.structure.model_copy(update={"geometry": images[-1]}),
                      has_molecular_graph=False), keywords={"fmax": 0.02, "maxiter": 500})
    end = traj[-1]
    images[-1] = np.asarray(end.coords, dtype=float)
    given = end_node._cached_energy
    if given is not None:
        echo(f"  end energy: {float(end.energy):.6f} Eh (as given, separately solvated: {float(given):.6f} Eh)")
    new_end = StructureNode(structure=end.structure)
    new_end._cached_energy, new_end._cached_gradient = end._cached_energy, end._cached_gradient
    return start_node, new_end, images


def peak_guess(nodes, energies=None):
    """A TS guess from a converged path: the energy maximum of a parabola
    through the highest interior image and its neighbours, interpolated
    between them, and the path tangent there (next image minus previous;
    full system, bohr), which tells a TS search which way is the reaction.
    Returns (structure, tangent) or (None, None) if the path has no interior
    maximum."""
    xs = [np.asarray(n.coords if hasattr(n, "coords") else n.geometry, dtype=float) for n in nodes]
    if energies is None:
        energies = [float(n.energy) for n in nodes]
    e = np.asarray(energies, dtype=float)
    if len(e) < 3:
        return None, None
    k = int(np.argmax(e[1:-1])) + 1
    a, b, c = e[k - 1], e[k], e[k + 1]
    denom = a - 2 * b + c
    t = 0.0 if abs(denom) < 1e-12 else float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))
    x = xs[k] + t * ((xs[k + 1] - xs[k]) if t > 0 else (xs[k] - xs[k - 1]))
    tangent = xs[k + 1] - xs[k - 1]
    structure = (nodes[k].structure if hasattr(nodes[k], "structure") else nodes[k]).model_copy(update={"geometry": x})
    return structure, tangent
