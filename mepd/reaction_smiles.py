"""Reaction SMILES ("reactants>>products", or "reactants>agents>products")
to a pair of 3D endpoints with the same atom order.

The atom-to-atom mapping comes from the SMILES itself when every heavy atom
carries a map number ([CH3:1]...), otherwise from SLAPMapper (the same
mapper `mepd run` uses for a SMILES pair). Only heavy atoms are mapped
either way; hydrogens, and any choice between SLAPMapper's tied mappings,
are settled by the smallest endpoint RMSD after aligning the two embedded
structures. That criterion is cheap and meant for building the pair fast
(e.g. live in the web UI's Design tab); a path search with
--atom-mapping still weighs other mappings itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class ReactionSmilesError(ValueError):
    pass


def is_reaction_smiles(text: str) -> bool:
    return isinstance(text, str) and text.count(">") == 2 and "->" not in text


def split_reaction(text: str) -> tuple[str, str, str]:
    """(reactants, agents, products). Agents (between the two '>') are
    returned for reporting; they take no part in the endpoints."""
    parts = text.strip().split(">")
    if len(parts) != 3:
        raise ReactionSmilesError(f"{text!r} is not a reaction SMILES: expected 'reactants>>products' "
                                  "(or 'reactants>agents>products')")
    reactants, agents, products = (p.strip() for p in parts)
    if not reactants or not products:
        raise ReactionSmilesError(f"{text!r}: both the reactant and the product side need a molecule")
    return reactants, agents, products


@dataclass
class ReactionPair:
    reactant: object            # RDKit Mol with explicit H and one 3D conformer
    product: object             # same atoms, same order as `reactant`
    mapped_smiles: str          # the heavy-atom mapping used, as a mapped reaction SMILES
    source: str                 # "given" (map numbers in the input) or "slapmapper"
    rmsd: float                 # endpoint RMSD (Angstrom) of the chosen mapping
    n_candidates: int = 1
    agents: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def charge(self) -> int:
        from rdkit import Chem

        return int(Chem.GetFormalCharge(self.reactant))


def _chem():
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    return Chem


def _side(smiles: str, what: str):
    Chem = _chem()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ReactionSmilesError(f"could not read the {what} SMILES {smiles!r}")
    return mol


def _map_numbers(mol) -> list[int]:
    return [a.GetAtomMapNum() for a in mol.GetAtoms()]


def _fully_mapped(r, p) -> bool:
    rm = [n for a, n in zip(r.GetAtoms(), _map_numbers(r)) if a.GetAtomicNum() > 1]
    pm = [n for a, n in zip(p.GetAtoms(), _map_numbers(p)) if a.GetAtomicNum() > 1]
    return bool(rm) and all(rm) and all(pm) and sorted(rm) == sorted(pm) and len(set(rm)) == len(rm)


def _formula(mol) -> dict:
    Chem = _chem()
    h = Chem.AddHs(mol)
    out: dict = {}
    for a in h.GetAtoms():
        out[a.GetSymbol()] = out.get(a.GetSymbol(), 0) + 1
    return out


def _formula_text(f: dict) -> str:
    return "".join(f"{el}{n if n > 1 else ''}" for el, n in sorted(f.items(), key=lambda kv: (kv[0] != "C", kv[0] != "H", kv[0])))


def _embed(mol, seed: int):
    Chem = _chem()
    from rdkit.Chem import AllChem

    h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(h, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(h, params) != 0:
            raise ReactionSmilesError(f"could not build 3D coordinates for {Chem.MolToSmiles(mol)}")
    try:
        if AllChem.MMFFHasAllMoleculeParams(h):
            AllChem.MMFFOptimizeMolecule(h, maxIters=500)
        else:
            AllChem.UFFOptimizeMolecule(h, maxIters=500)
    except Exception:
        pass
    _pack(h, seed)
    return h


def _pack(mol, seed: int) -> None:
    """Several molecules in one SMILES ('CC=O.O') are embedded on top of
    each other; pack them (mepd.complexes.packed: random orientations in a
    small sphere, 2 Angstrom apart at their closest), so the geometry has the
    molecules' own bonds only and is a usable starting point."""
    from qcconst.constants import ANGSTROM_TO_BOHR
    from qcdata import Structure

    from mepd.complexes import packed

    Chem = _chem()
    frags = [list(f) for f in Chem.GetMolFrags(mol)]
    if len(frags) < 2:
        return
    conf = mol.GetConformer()
    pos = conf.GetPositions()
    parts = [Structure(symbols=[mol.GetAtomWithIdx(i).GetSymbol() for i in f],
                       geometry=pos[f] * ANGSTROM_TO_BOHR, charge=0, multiplicity=1) for f in frags]
    placed = np.asarray(packed(parts, seed=seed).geometry).reshape(-1, 3) / ANGSTROM_TO_BOHR
    for i, xyz in zip([i for f in frags for i in f], placed):
        conf.SetAtomPosition(i, xyz.tolist())


def _kabsch(P: np.ndarray, Q: np.ndarray):
    """Rotation R and centroids so that (P - cp) @ R + cq best fits Q."""
    cp, cq = P.mean(axis=0), Q.mean(axis=0)
    H = (P - cp).T @ (Q - cq)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, 1.0, d])
    return U @ D @ Vt, cp, cq


def _rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    R, cp, cq = _kabsch(P, Q)
    return float(np.sqrt((((P - cp) @ R + cq - Q) ** 2).sum(axis=1).mean()))


def _match(r, p, strict: bool = True) -> tuple[list[int], float]:
    """order such that product atom order[i] corresponds to reactant atom i.
    Atoms with the same map number on both sides correspond; the rest are
    assigned per element (heavy atoms first, hydrogens last), preferring an
    atom whose bonded neighbours already correspond, then the nearest after
    aligning the two sides on the atoms matched so far -- i.e. the smallest
    endpoint RMSD. `strict=False` (the Design tab, mid-edit) leaves atoms
    without a counterpart at -1 instead of raising."""
    from scipy.optimize import linear_sum_assignment

    X = r.GetConformer().GetPositions()
    Y = p.GetConformer().GetPositions()
    pmap = {a.GetAtomMapNum(): a.GetIdx() for a in p.GetAtoms() if a.GetAtomMapNum()}
    order = [-1] * r.GetNumAtoms()
    for a in r.GetAtoms():
        j = pmap.get(a.GetAtomMapNum()) if a.GetAtomMapNum() else None
        if j is not None and p.GetAtomWithIdx(j).GetAtomicNum() == a.GetAtomicNum():
            order[a.GetIdx()] = j

    def aligned():
        fixed = [i for i in range(len(order)) if order[i] >= 0]
        if len(fixed) >= 3:
            R, cp, cq = _kabsch(Y[[order[i] for i in fixed]], X[fixed])
            return (Y - cp) @ R + cq
        return Y - Y.mean(axis=0) + X.mean(axis=0)

    for el in sorted({a.GetAtomicNum() for a in r.GetAtoms()} | {a.GetAtomicNum() for a in p.GetAtoms()},
                     key=lambda z: (z == 1, -z)):
        used = set(order)
        ri = [a.GetIdx() for a in r.GetAtoms() if a.GetAtomicNum() == el and order[a.GetIdx()] < 0]
        pi = [a.GetIdx() for a in p.GetAtoms() if a.GetAtomicNum() == el and a.GetIdx() not in used]
        if len(ri) != len(pi) and strict:
            raise ReactionSmilesError("the reaction is not balanced once hydrogens are added")
        if not ri or not pi:
            continue
        Yal = aligned()
        cost = np.linalg.norm(X[ri][:, None] - Yal[pi][None], axis=-1)
        for a, i in enumerate(ri):
            rn = {order[n.GetIdx()] for n in r.GetAtomWithIdx(i).GetNeighbors()} - {-1}
            if not rn:
                continue
            for b, j in enumerate(pi):
                if rn & {n.GetIdx() for n in p.GetAtomWithIdx(j).GetNeighbors()}:
                    cost[a, b] -= 100.0
        rows, cols = linear_sum_assignment(cost)
        for a, b in zip(rows, cols):
            order[ri[a]] = pi[b]
    matched = [i for i in range(len(order)) if order[i] >= 0]
    rmsd = _rmsd(Y[[order[i] for i in matched]], X[matched]) if len(matched) >= 3 else 0.0
    return order, rmsd


def _stereo_centres(mol) -> dict:
    """{atom index: 'R'/'S'} of the stereocentres `mol` specifies."""
    Chem = _chem()
    return {i: str(lab) for i, lab in Chem.FindMolChiralCenters(mol, includeUnassigned=False,
                                                                 useLegacyImplementation=False)}


def _stereo_of(p, xyz) -> dict:
    """The stereocentres `p` would have in geometry `xyz`."""
    Chem = _chem()
    t = Chem.Mol(p)
    conf = t.GetConformer()
    for i, x in enumerate(xyz):
        conf.SetAtomPosition(i, [float(v) for v in x])
    Chem.AssignStereochemistryFrom3D(t)
    return _stereo_centres(t)


def _invert_centres(p, xyz, wrong) -> "np.ndarray | None":
    """`xyz` with each centre in `wrong` inverted like an umbrella: the
    centre and its hydrogen reflected through the plane of its other three
    neighbours, which stay put (None if a centre has no hydrogen)."""
    import numpy as np

    x = np.array(xyz, dtype=float)
    for i in wrong:
        atom = p.GetAtomWithIdx(i)
        hs = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
        others = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() not in hs[:1]]
        if not hs or len(others) != 3:
            return None
        a, b, c = x[others]
        n = np.cross(b - a, c - a)
        n /= np.linalg.norm(n) or 1.0
        for k in (i, hs[0]):
            x[k] = x[k] - 2 * np.dot(x[k] - a, n) * n
    return x


def _in_reactant_frame(r, p) -> tuple[bool, str]:
    """Rebuild the product's geometry (atoms already in the reactant's
    order) from the reactant's: each atom starts where it is in the
    reactant, the product's bonds are pulled to length
    (network_expansion.embed_product), then a short force-field relaxation.
    With several molecules on a side, two independently packed sides put the
    molecules in unrelated places: a path between them drags whole molecules
    through each other (interpolations that look atomized), however good the
    atom mapping. Returns (whether the rebuilt product has exactly its own
    bonds and stereocentres, why not)."""
    from rdkit.Chem import AllChem

    from mepd.discovery.network_expansion import _perceived_edges, embed_product

    symbols = [a.GetSymbol() for a in r.GetAtoms()]
    want = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in p.GetBonds()}
    stereo = _stereo_centres(p)

    def bonds_ok(x) -> bool:
        return {tuple(sorted(e)) for e in _perceived_edges(symbols, x)} == want

    try:
        xyz = embed_product(symbols, r.GetConformer().GetPositions(), want)
    except Exception:
        return False, "embedding failed"
    # Pulling bonds to length does not keep handedness: a stereocentre that
    # forms (e.g. an aldol's new C-OH) can come out inverted. Flip it back
    # (its H through the other three neighbours' plane) and re-embed.
    wrong = [i for i, lab in stereo.items() if _stereo_of(p, xyz).get(i) != lab]
    if wrong:
        flipped = _invert_centres(p, xyz, wrong)
        if flipped is not None:
            xyz = embed_product(symbols, flipped, want)
        if _stereo_of(p, xyz) != stereo:
            return False, "its stereocentres came out inverted"
    if not bonds_ok(xyz):
        return False, "its bonds changed"
    trial = _chem().Mol(p)
    conf = trial.GetConformer()
    for i, x in enumerate(xyz):
        conf.SetAtomPosition(i, [float(v) for v in x])
    relaxed = _chem().Mol(trial)
    try:
        if AllChem.MMFFHasAllMoleculeParams(relaxed):
            AllChem.MMFFOptimizeMolecule(relaxed, maxIters=200)
        else:
            AllChem.UFFOptimizeMolecule(relaxed, maxIters=200)
    except Exception:
        pass
    # The force field can wreck a strained start (atoms pushed onto each
    # other): then the embedded geometry, whose bonds and handedness are right.
    rx = relaxed.GetConformer().GetPositions()
    if bonds_ok(rx) and _stereo_of(p, rx) == stereo:
        trial = relaxed
    p.RemoveAllConformers()
    p.AddConformer(trial.GetConformer(), assignId=True)
    return True, ""


def _reordered(mol, order: list[int]):
    Chem = _chem()
    return Chem.RenumberAtoms(mol, order)


def _candidates(reactants: str, products: str) -> tuple[list[str], str, list[str]]:
    """Mapped reaction SMILES to choose from, and where they came from."""
    r, p = _side(reactants, "reactant"), _side(products, "product")
    fr, fp = _formula(r), _formula(p)
    if fr != fp:
        raise ReactionSmilesError(f"the reaction is not balanced: reactants are {_formula_text(fr)}, "
                                  f"products {_formula_text(fp)} (every atom must appear on both sides)")
    if _fully_mapped(r, p):
        return [f"{reactants}>>{products}"], "given", []
    notes = []
    if any(_map_numbers(r)) or any(_map_numbers(p)):
        notes.append("Only some atoms carry map numbers; they were ignored and the whole reaction mapped anew.")
        for m in (r, p):
            for a in m.GetAtoms():
                a.SetAtomMapNum(0)
        reactants, products = _chem().MolToSmiles(r), _chem().MolToSmiles(p)
    from mepd.atom_mapping import _require_slapmapper

    try:
        _require_slapmapper()
    except Exception as exc:
        raise ReactionSmilesError(f"this reaction SMILES has no atom map numbers, and mapping it needs SLAPMapper: {exc}")
    from slapmapper.aam import SlapAAM

    mapper = SlapAAM(binary=True)
    mapper.map_smiles(f"{reactants}>>{products}")
    if not mapper.results:
        raise ReactionSmilesError(f"SLAPMapper found no atom mapping for {reactants}>>{products}")
    return [res["smiles"] for res in mapper.results[:8]], "slapmapper", notes


def reaction_pair(text: str, seed: int = 11) -> ReactionPair:
    """Both endpoints of a reaction SMILES, embedded in 3D (ETKDG, then a
    force field) with the product's atoms in the reactant's order."""
    reactants, agents, products = split_reaction(text)
    mapped, source, notes = _candidates(reactants, products)
    if agents:
        notes.append(f"Agents ({agents}) are not part of the endpoints.")
    best = None
    for k, rxn in enumerate(mapped):
        rs, ps = rxn.split(">>")
        r = _embed(_side(rs, "reactant"), seed)
        p = _embed(_side(ps, "product"), seed)
        order, rmsd = _match(r, p)
        if best is None or rmsd < best[0] - 1e-6:
            best = (rmsd, r, _reordered(p, order), rxn)
    rmsd, r, p, rxn = best
    Chem = _chem()
    if len(Chem.GetMolFrags(r)) > 1 or len(Chem.GetMolFrags(p)) > 1:
        ok, why = _in_reactant_frame(r, p)
        if ok:
            rmsd = _rmsd(p.GetConformer().GetPositions(), r.GetConformer().GetPositions())
        else:
            notes.append(f"The product could not be rebuilt in the reactants' arrangement ({why}); its molecules "
                         "are placed independently, so a path search must also move them into place.")
    if Chem.GetFormalCharge(r) != Chem.GetFormalCharge(p):
        raise ReactionSmilesError(f"the two sides have different charges ({Chem.GetFormalCharge(r):+d} and "
                                  f"{Chem.GetFormalCharge(p):+d})")
    for m in (r, p):
        for a in m.GetAtoms():
            a.SetAtomMapNum(0)
    return ReactionPair(reactant=r, product=p, mapped_smiles=rxn, source=source, rmsd=rmsd,
                        n_candidates=len(mapped), agents=agents, notes=notes)


def mol_to_structure(mol, charge=None, multiplicity=None):
    """A qcdata Structure (bohr) of an embedded RDKit Mol."""
    from qcdata import Structure

    from qcconst.constants import ANGSTROM_TO_BOHR

    pos = mol.GetConformer().GetPositions() * ANGSTROM_TO_BOHR
    chem = _chem()
    return Structure(symbols=[a.GetSymbol() for a in mol.GetAtoms()], geometry=pos,
                     charge=int(chem.GetFormalCharge(mol)) if charge is None else int(charge),
                     multiplicity=int(multiplicity or 1))


def reaction_structures(text: str, charge=None, multiplicity=None):
    """(start Structure, end Structure, ReactionPair) for a reaction SMILES."""
    pair = reaction_pair(text)
    return (mol_to_structure(pair.reactant, charge, multiplicity),
            mol_to_structure(pair.product, charge, multiplicity), pair)
