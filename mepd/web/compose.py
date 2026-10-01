"""Compose a reaction from species already in the workspace (Analyze ›
New reaction).

The reactants (shuttles included: a species on both sides) are placed side
by side as one reaction complex. With products given, those are placed the
same way; the TS search maps the atoms itself. Without products, the bond
rules of the network expansion propose what the complex can become (break
and form up to n bonds, valid Lewis structures), each with a 3D guess in the
complex's own atom order; the ones picked become reactions. Either way the
reaction is an ordinary workspace reaction: hidden complex structures joined
by an edge, so "Find TS" runs on it like on a nanoreactor reaction.
"""
from __future__ import annotations

import time
from collections import Counter
from typing import Optional

import numpy as np
from qcconst.constants import ANGSTROM_TO_BOHR

from mepd.web import chem
from mepd.web.workspace import Workspace, WorkspaceError, new_id

_Z = {"H": 1, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Br": 35, "I": 53}


def place(structures: list, gap: float = 2.6):
    """One structure holding every molecule, side by side along x, each
    centred, the closest contact between neighbours `gap` Angstrom."""
    from qcdata import Structure

    symbols, blocks, x_end = [], [], None
    for s in structures:
        xyz = np.asarray(s.geometry, dtype=float).reshape(-1, 3) / ANGSTROM_TO_BOHR
        xyz = xyz - xyz.mean(axis=0)
        if x_end is not None:
            xyz[:, 0] += x_end - xyz[:, 0].min() + gap
        x_end = xyz[:, 0].max()
        symbols += list(s.symbols)
        blocks.append(xyz)
    coords = np.vstack(blocks)
    coords -= coords.mean(axis=0)
    charge = sum(int(s.charge) for s in structures)
    electrons = sum(_Z.get(x, 0) for x in symbols) - charge
    return Structure(symbols=symbols, geometry=coords * ANGSTROM_TO_BOHR, charge=charge,
                     multiplicity=1 if electrons % 2 == 0 else 2)


def _elements(structures) -> Counter:
    c = Counter()
    for s in structures:
        c.update(s.symbols)
    return c


def species_structures(ws: Workspace, sids: list[str]) -> list:
    out = []
    for sid in sids:
        rec = ws.structure(sid)
        if rec.get("role") in ("ts", "complex"):
            raise WorkspaceError(f"{rec['name']} is not a molecule (a {rec['role']})")
        out.append(ws.load_structure(sid))
    return out


def propose(ws: Workspace, reactants: list[str], *, n_break: int = 2, n_form: int = 2,
            max_products: int = 40) -> dict:
    """What the reactants' complex can become by the bond rules: product
    guesses in the complex's atom order, fewest bond changes first."""
    from mepd.discovery.generators import propose as propose_products
    from mepd.discovery.network_expansion import _perceived_edges, embed_product

    if not reactants:
        raise WorkspaceError("pick at least one reactant")
    rc = place(species_structures(ws, reactants))
    symbols = list(rc.symbols)
    coords = np.asarray(rc.geometry).reshape(-1, 3) / ANGSTROM_TO_BOHR
    edges = _perceived_edges(symbols, coords)
    props, _ = propose_products(
        "bond-rules", symbols, coords, edges, charge=int(rc.charge), multiplicity=int(rc.multiplicity),
        n_break=n_break, n_form=n_form, form_distance=4.5, max_products=max_products, allow_radicals=False,
        allow_zwitterions=False, source=0)
    names = Counter(chem.canonical_key(ws.structure(s)["smiles"] or "") for s in reactants)
    out = []
    for p in props:
        frags = sorted(p.smiles.split("."))
        if Counter(chem.canonical_key(f) for f in frags) == names:
            continue   # the same molecules again (a degenerate exchange)
        guess = embed_product(symbols, coords, (edges - set(p.broken)) | set(p.formed))
        s = rc.model_copy(update={"geometry": np.asarray(guess) * ANGSTROM_TO_BOHR})
        out.append({"smiles": p.smiles, "fragments": frags, "broken": [list(b) for b in p.broken],
                    "formed": [list(f) for f in p.formed], "xyz": s.to_xyz()})
    return {"complex_xyz": rc.to_xyz(), "charge": int(rc.charge), "multiplicity": int(rc.multiplicity),
            "proposals": out}


def _fragments(structure) -> list:
    """The molecules of a structure (connected components) as Structures,
    each with the charge and spin of its Lewis structure (summing to the
    structure's charge)."""
    from qcdata import Structure

    from mepd.discovery.nanoreactor import Labeler, components, perceive_bonds

    symbols = list(structure.symbols)
    coords = np.asarray(structure.geometry).reshape(-1, 3) / ANGSTROM_TO_BOHR
    bonds = perceive_bonds(symbols, coords)
    mols = components(len(symbols), bonds)
    labels = Labeler(symbols, max_charge=max(1, abs(int(structure.charge)))).assign(mols, bonds, int(structure.charge))
    return [Structure(symbols=[symbols[a] for a in comp], geometry=coords[list(comp)] * ANGSTROM_TO_BOHR,
                      charge=lab.charge, multiplicity=lab.multiplicity) for comp, lab in zip(mols, labels)]


def compose(ws: Workspace, reactants: list[str], *, products: Optional[list[str]] = None,
            proposal: Optional[dict] = None, level: Optional[dict] = None) -> dict:
    """A new workspace reaction. Returns {"reaction", "optimize": structure
    ids to minimize (the complexes, and product species new to the graph)}."""
    from mepd.web.nanoreactor import _label_of

    rs = species_structures(ws, reactants)
    to_optimize = []
    if proposal is not None:
        (rc,) = chem.structures_from_xyz_text(proposal["complex_xyz"], proposal.get("charge"),
                                             proposal.get("multiplicity"))
        (pc,) = chem.structures_from_xyz_text(proposal["xyz"], rc.charge, rc.multiplicity)
        products = []
        for frag in _fragments(pc):
            res = ws.add_or_merge(frag, optimized=False, origin={"kind": "composed", "label": "proposed product"})
            products.append(res["rec"]["id"])
            if not res["merged"]:
                to_optimize.append(res["rec"]["id"])
    elif products:
        ps = species_structures(ws, products)
        if _elements(rs) != _elements(ps):
            raise WorkspaceError("the atoms do not balance: add the missing species (e.g. a shuttle on both "
                                 "sides, or a second product)")
        if sum(int(s.charge) for s in rs) != sum(int(s.charge) for s in ps):
            raise WorkspaceError("the charges do not balance")
        rc, pc = place(rs), place(ps)
    else:
        raise WorkspaceError("give the products, or pick one of the proposals")
    names = {sid: ws.structure(sid)["name"] for sid in [*reactants, *products]}
    label = _label_of(reactants, products, names)
    key = new_id("c_")
    sids = []
    for side, s in (("reactant", rc), ("product", pc)):
        rec = ws.add_structure(s, name=f"{label} [{side}s]", smiles=chem.perceive_smiles(s), optimized=False,
                               role="complex", merge=False,
                               origin={"kind": "composed", "label": f"composed {side}s"})
        sids.append(rec["id"])
        to_optimize.append(rec["id"])
    edge = ws.add_edge(sids[0], sids[1], origin={"kind": "composed", "proposed": True,
                                                 "headline": "composed reaction: run a TS search on it"})
    rec = ws.put_reaction(reactants=list(reactants), products=list(products),
                          origin={"kind": "composed", "job": None, "index": key}, label=label, count=0,
                          reverse_count=0, delta_e_kcal=None, complexes=sids, edge=edge["id"], events=[],
                          composed=time.time())
    return {"reaction": rec, "optimize": to_optimize}


def reaction_from_endpoints(ws: Workspace, start, end, *, origin: dict, energies=(None, None),
                            level: Optional[dict] = None, label: str = "", ts: Optional[dict] = None) -> Optional[dict]:
    """Two endpoint geometries (same atoms, same order) as a reaction in the
    uniform model -- if either holds more than one molecule: each end split
    into its molecules (species nodes, merged with known ones), the exact
    endpoint geometries kept as the reaction's two complexes, joined by the
    reaction edge (where TS searches run). `ts` ({job, entry, barrier_kcal})
    puts a TS already found on that edge. Returns None for a one-molecule to
    one-molecule reaction (a plain edge between the two species is that
    reaction already)."""
    from mepd.web.nanoreactor import _label_of

    sides = [_fragments(start), _fragments(end)]
    if len(sides[0]) < 2 and len(sides[1]) < 2:
        return None
    ids, added = [], []
    for frags in sides:
        side = []
        for f in frags:
            res = ws.add_or_merge(f, optimized=False, origin={**origin, "label": f"{origin.get('label', 'reaction')} (molecule)"})
            added.append(res)
            side.append(res["rec"]["id"])
        ids.append(side)
    if sorted(ids[0]) == sorted(ids[1]):
        raise WorkspaceError("both sides are the same molecules, so there is no reaction to search")
    names = {sid: ws.structure(sid)["name"] for sid in ids[0] + ids[1]}
    text = label or _label_of(ids[0], ids[1], names)
    cx = []
    for side, s, e in (("reactant", start, energies[0]), ("product", end, energies[1])):
        cx.append(ws.add_structure(s, name=f"{text} [{side}s]", smiles=chem.perceive_smiles(s), energy=e,
                                   optimized=e is not None, level=level if e is not None else None, role="complex",
                                   merge=False, origin={**origin, "label": f"{origin.get('label', 'reaction')} {side}s"})["id"])
    edge_origin = {**origin, "proposed": ts is None, "headline": "run a TS search on this reaction"}
    if ts is not None:
        edge_origin = {"kind": "job", "job": ts.get("job"), "entry": ts.get("entry"), "group": "irc", "has_ts": True,
                       "barrier_kcal": ts.get("barrier_kcal"), "headline": "TS + IRC from Design"}
    edge = ws.add_edge(cx[0], cx[1], origin=edge_origin)
    rec = ws.put_reaction(reactants=ids[0], products=ids[1], origin={**origin, "job": None, "index": new_id("d_")},
                          label=text, count=0, reverse_count=0, delta_e_kcal=None, complexes=cx, edge=edge["id"],
                          events=[])
    return {"reaction": rec, "edge": edge["id"], "species": ids, "added": added}
