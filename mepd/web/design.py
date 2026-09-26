"""Chemistry behind the Design tab: a 3D molecule the user edits.

The molecule travels as an MDL molblock (explicit hydrogens, 3D
coordinates, bond orders, formal charges); every edit is one call here that
returns the new molblock. RDKit keeps the chemistry honest: valences are
checked on every edit, hydrogens are re-added to the atoms an edit touched
(placed from their neighbours), new atoms and groups are positioned from
the geometry around them, and only the atoms an edit added or moved are
relaxed (MMFF94, else UFF), so the rest of the structure -- a transition
state, say -- is left exactly where it was.

Edits (see `edit`):
  element   {atom, element}           change an atom's element
  add       {atom, element}           bond a new atom to `atom` (taking one of its H's place if it has one)
  delete    {atom}                    remove an atom (and its hydrogens)
  bond      {a, b, order}             set a bond order 1/2/3, 1.5 aromatic, 0 = remove it
  group     {atom, group}             swap a hydrogen, or a terminal group, for a functional group
  charge    {atom, delta}             change an atom's formal charge by +-1
  hydrogens {}                        re-add hydrogens everywhere from valences
  place     {atom, species, count}    put whole molecules/ions (water, Li+, Mg2+, BF3, ...) next to
                                      `atom`, not bonded: count waters make a small solvation shell
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from mepd.web.workspace import WorkspaceError

# Substituents as SMILES with the attachment point [*].
GROUPS = {
    "H": "[*][H]",
    "methyl": "[*]C",
    "ethyl": "[*]CC",
    "isopropyl": "[*]C(C)C",
    "tert-butyl": "[*]C(C)(C)C",
    "vinyl": "[*]C=C",
    "ethynyl": "[*]C#C",
    "phenyl": "[*]c1ccccc1",
    "benzyl": "[*]Cc1ccccc1",
    "hydroxyl": "[*]O",
    "methoxy": "[*]OC",
    "amino": "[*]N",
    "dimethylamino": "[*]N(C)C",
    "fluoro": "[*]F",
    "chloro": "[*]Cl",
    "bromo": "[*]Br",
    "iodo": "[*]I",
    "trifluoromethyl": "[*]C(F)(F)F",
    "cyano": "[*]C#N",
    "nitro": "[*][N+](=O)[O-]",
    "formyl": "[*]C=O",
    "acetyl": "[*]C(C)=O",
    "carboxyl": "[*]C(=O)O",
    "ester (CO2Me)": "[*]C(=O)OC",
    "amide": "[*]C(N)=O",
    "thiol": "[*]S",
    "sulfonyl (SO2Me)": "[*]S(C)(=O)=O",
    "trimethylsilyl": "[*][Si](C)(C)C",
    "boronic acid": "[*]B(O)O",
}

# Species to place next to an atom (catalysts, counter-ions, explicit solvent),
# as SMILES, and the distance (Å) from the clicked atom to the species' contact
# atom (its first atom).
SPECIES = {
    "water": ("O", 2.8),
    "methanol": ("OC", 2.8),
    "ammonia": ("N", 3.0),
    "hydronium": ("[OH3+]", 2.6),
    "hydroxide": ("[OH-]", 2.7),
    "HF": ("F", 2.7),
    "Li+": ("[Li+]", 2.0),
    "Na+": ("[Na+]", 2.35),
    "K+": ("[K+]", 2.75),
    "Mg2+": ("[Mg+2]", 2.05),
    "Ca2+": ("[Ca+2]", 2.4),
    "Zn2+": ("[Zn+2]", 2.05),
    "Cu+": ("[Cu+]", 2.0),
    "Al3+": ("[Al+3]", 1.9),
    "BF3": ("B(F)(F)F", 1.65),
    "AlCl3": ("[Al](Cl)(Cl)Cl", 2.0),
    "H+ (proton)": ("[H+]", 1.05),
}

_BOND = {1: "SINGLE", 2: "DOUBLE", 3: "TRIPLE", 1.5: "AROMATIC"}


# ------------------------------------------------------------------ io

def _chem():
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    return Chem


def read(molblock: str):
    Chem = _chem()
    mol = Chem.MolFromMolBlock(molblock, removeHs=False, sanitize=False)
    if mol is None:
        raise WorkspaceError("the design could not be read (not a valid molblock)")
    mol.UpdatePropertyCache(strict=False)
    return mol


def _sanitized(mol, context: str = ""):
    """A sanitized copy, or a WorkspaceError saying which atom is wrong."""
    Chem = _chem()
    m = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(m)
    except Exception as exc:
        raise WorkspaceError(f"{context + ': ' if context else ''}{_valence_message(m, exc)}") from None
    return m


def _valence_message(mol, exc) -> str:
    text = str(exc)
    for atom in mol.GetAtoms():
        try:
            atom.UpdatePropertyCache(strict=True)
        except Exception:
            return (f"{atom.GetSymbol()}{atom.GetIdx() + 1} would have {int(atom.GetExplicitValence())} bonds "
                    f"(charge {atom.GetFormalCharge():+d}), more than it can take")
    return text.split("\n")[0]


def describe(mol, *, sanitized: bool = True) -> dict:
    """What the UI shows next to the viewer."""
    Chem = _chem()
    from rdkit.Chem import rdMolDescriptors

    mol.UpdatePropertyCache(strict=False)
    try:
        m = _sanitized(mol) if sanitized else mol
        smiles = Chem.MolToSmiles(Chem.RemoveHs(m))
    except Exception:
        smiles = None
    radicals = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    return {
        "molblock": Chem.MolToMolBlock(mol, kekulize=False),
        "smiles": smiles,
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "natoms": mol.GetNumAtoms(),
        "charge": int(sum(a.GetFormalCharge() for a in mol.GetAtoms())),
        "multiplicity": int(radicals) + 1,
    }


def to_xyz(molblock: str, charge: Optional[int] = None, multiplicity: Optional[int] = None) -> str:
    mol = read(molblock)
    info = describe(mol, sanitized=False)
    pos = mol.GetConformer().GetPositions()
    lines = [str(mol.GetNumAtoms()),
             f"charge={charge if charge is not None else info['charge']} "
             f"multiplicity={multiplicity if multiplicity is not None else info['multiplicity']}"]
    lines += [f"{a.GetSymbol()} {x:.6f} {y:.6f} {z:.6f}" for a, (x, y, z) in zip(mol.GetAtoms(), pos)]
    return "\n".join(lines) + "\n"


def from_smiles(smiles: str) -> dict:
    Chem = _chem()
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        raise WorkspaceError(f"could not read SMILES {smiles!r}")
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=0xF00D) != 0:
        AllChem.EmbedMolecule(mol, randomSeed=0xF00D, useRandomCoords=True)
    _relax(mol, None)
    return describe(mol)


def from_xyz(xyz: str, charge: int = 0) -> tuple[dict, list[str]]:
    """A molblock from a graph structure's xyz: bond orders from the
    geometry (xyz2mol) when they can be assigned, otherwise connectivity
    only (e.g. a transition state), with a warning."""
    Chem = _chem()
    from rdkit.Chem import rdDetermineBonds

    mol = Chem.MolFromXYZBlock(xyz)
    if mol is None:
        raise WorkspaceError("could not read the structure's xyz")
    warnings = []
    try:
        trial = Chem.Mol(mol)
        rdDetermineBonds.DetermineBonds(trial, charge=int(charge))
        mol = trial
        Chem.SanitizeMol(mol)
    except Exception:
        mol = Chem.Mol(Chem.MolFromXYZBlock(xyz))
        rdDetermineBonds.DetermineConnectivity(mol)
        mol.UpdatePropertyCache(strict=False)
        warnings.append("Bond orders could not be assigned from this geometry (e.g. a transition state's partial "
                        "bonds): every bond is drawn single. Set bond orders with the Bond tool if you need them; "
                        "edits that re-add hydrogens use them.")
    for a in mol.GetAtoms():
        a.SetNoImplicit(True)
    return describe(mol, sanitized=not warnings), warnings


# ------------------------------------------------------------------ geometry helpers

def _positions(mol) -> np.ndarray:
    return np.array(mol.GetConformer().GetPositions(), dtype=float)


def _set_position(mol, idx: int, xyz) -> None:
    from rdkit.Geometry import Point3D

    mol.GetConformer().SetAtomPosition(int(idx), Point3D(*map(float, xyz)))


def _bond_length(a: str, b: str, order: float = 1.0) -> float:
    from rdkit import Chem

    pt = Chem.GetPeriodicTable()
    r = pt.GetRcovalent(a) + pt.GetRcovalent(b)
    return r * {1.0: 1.0, 1.5: 0.93, 2.0: 0.87, 3.0: 0.78}.get(float(order), 1.0)


def _open_direction(mol, idx: int) -> np.ndarray:
    """A unit vector pointing away from `idx`'s neighbours (where a new bond
    goes)."""
    pos = _positions(mol)
    here = pos[idx]
    nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors()]
    if not nbrs:
        return np.array([1.0, 0.0, 0.0])
    vecs = [(here - pos[n]) / (np.linalg.norm(here - pos[n]) + 1e-9) for n in nbrs]
    d = np.sum(vecs, axis=0)
    if np.linalg.norm(d) < 0.3:     # neighbours cancel (linear / planar): go perpendicular
        ref = vecs[0]
        trial = np.cross(ref, [0.0, 0.0, 1.0])
        if np.linalg.norm(trial) < 0.1:
            trial = np.cross(ref, [0.0, 1.0, 0.0])
        d = trial
    return d / np.linalg.norm(d)


def _rotation(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotation matrix taking unit vector u onto unit vector v."""
    u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)
    c = float(np.dot(u, v))
    if c > 1 - 1e-9:
        return np.eye(3)
    if c < -1 + 1e-9:
        axis = np.cross(u, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(u, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    w = np.cross(u, v)
    k = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    return np.eye(3) + k + k @ k / (1 + c)


def _relax(mol, moving: Optional[set], steps: int = 400) -> Optional[str]:
    """Force-field relaxation of the atoms in `moving` (all atoms if None),
    everything else held fixed. MMFF94, else UFF; returns a warning if
    neither applies."""
    from rdkit.Chem import AllChem

    Chem = _chem()
    try:
        m = _sanitized(mol)
    except WorkspaceError:
        return "no force field could be set up (unassigned bond orders or valences): new atoms were placed but not relaxed"
    ff = None
    try:
        props = AllChem.MMFFGetMoleculeProperties(m)
        if props is not None:
            ff = AllChem.MMFFGetMoleculeForceField(m, props, ignoreInterfragInteractions=False)
    except Exception:
        ff = None
    if ff is None:
        try:
            ff = AllChem.UFFGetMoleculeForceField(m, ignoreInterfragInteractions=False)
        except Exception:
            ff = None
    if ff is None:
        return "no force field covers these atoms: new atoms were placed but not relaxed"
    if moving is not None:
        for i in range(m.GetNumAtoms()):
            if i not in moving:
                ff.AddFixedPoint(i)
    ff.Minimize(maxIts=steps)
    conf = m.GetConformer()
    for i in range(m.GetNumAtoms()):
        if moving is None or i in moving:
            _set_position(mol, i, conf.GetAtomPosition(i))
    return None


def _fix_hydrogens(mol, atoms: set):
    """Remove the hydrogens on `atoms` and re-add as many as their valence
    wants, placed from their neighbours. Returns (new mol, indices of the
    added hydrogens). The rest of the molecule keeps its atom order."""
    Chem = _chem()
    rw = Chem.RWMol(mol)
    heavy = sorted(i for i in atoms if rw.GetAtomWithIdx(i).GetAtomicNum() != 1)
    drop = sorted({n.GetIdx() for i in heavy for n in rw.GetAtomWithIdx(i).GetNeighbors() if n.GetAtomicNum() == 1},
                  reverse=True)
    # Indices shift as hydrogens go: track the heavy atoms by a property.
    for i in heavy:
        rw.GetAtomWithIdx(i).SetIntProp("_touched", 1)
    for h in drop:
        rw.RemoveAtom(h)
    m = rw.GetMol()
    touched = [a.GetIdx() for a in m.GetAtoms() if a.HasProp("_touched")]
    for a in m.GetAtoms():
        a.SetNoImplicit(a.GetIdx() not in touched)
        if a.GetIdx() in touched:
            a.SetNumExplicitHs(0)
    m = _sanitized(m, "after this edit")
    before = m.GetNumAtoms()
    m = Chem.AddHs(m, addCoords=True, onlyOnAtoms=tuple(touched))
    for a in m.GetAtoms():
        a.SetNoImplicit(True)
        if a.HasProp("_touched"):
            a.ClearProp("_touched")
    return m, set(range(before, m.GetNumAtoms()))


def _neighbors(mol, idx: int) -> list[int]:
    return [n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors()]


def _side(mol, root: int, anchor: int) -> set:
    """Atoms reached from `root` without crossing the root-anchor bond."""
    seen, todo = {root}, [root]
    while todo:
        a = todo.pop()
        for n in _neighbors(mol, a):
            if n not in seen and not (a == root and n == anchor):
                seen.add(n)
                todo.append(n)
    if anchor in seen:
        raise WorkspaceError("that atom is in a ring with the rest of the molecule: pick a hydrogen or a terminal group")
    return seen


# ------------------------------------------------------------------ edits

def edit(molblock: str, op: dict) -> dict:
    """Apply one edit; returns describe() of the result plus `warnings` and
    `changed` (atom indices the viewer can highlight)."""
    mol = read(molblock)
    kind = op.get("op")
    fn = _EDITS.get(kind)
    if fn is None:
        raise WorkspaceError(f"unknown design edit {kind!r}")
    for key in ("atom", "a", "b"):
        if key in op and not 0 <= int(op[key]) < mol.GetNumAtoms():
            raise WorkspaceError(f"there is no atom {int(op[key]) + 1}")
    mol, changed, warnings = fn(mol, op)
    out = describe(mol, sanitized=True)
    out.update(warnings=[w for w in warnings if w], changed=sorted(changed))
    return out


def _edit_element(mol, op):
    Chem = _chem()
    idx, sym = int(op["atom"]), str(op["element"]).strip()
    try:
        z = Chem.GetPeriodicTable().GetAtomicNumber(sym)
    except Exception:
        raise WorkspaceError(f"unknown element {sym!r}") from None
    rw = Chem.RWMol(mol)
    atom = rw.GetAtomWithIdx(idx)
    atom.SetAtomicNum(z)
    atom.SetFormalCharge(0)
    atom.SetIsAromatic(atom.GetIsAromatic() and z in (5, 6, 7, 8, 15, 16))
    if z == 1:   # to hydrogen: keep one bond at most
        for n in _neighbors(rw, idx)[1:]:
            rw.RemoveBond(idx, n)
        m = rw.GetMol()
        _sanitized(m, "after this edit")
        return m, {idx}, []
    m, new_h = _fix_hydrogens(rw.GetMol(), {idx})
    moving = new_h | {idx}
    return m, moving, [_relax(m, moving)]


def _edit_add(mol, op):
    Chem = _chem()
    parent, sym = int(op["atom"]), str(op.get("element") or "C")
    rw = Chem.RWMol(mol)
    pos = _positions(rw)
    hs = [n for n in _neighbors(rw, parent) if rw.GetAtomWithIdx(n).GetAtomicNum() == 1]
    if hs:   # take a hydrogen's place: along its bond
        h = hs[0]
        d = pos[h] - pos[parent]
        d /= np.linalg.norm(d)
        rw.RemoveAtom(h)
        if h < parent:
            parent -= 1
    else:
        d = _open_direction(rw, parent)
    new = rw.AddAtom(Chem.Atom(sym))
    rw.AddBond(parent, new, Chem.BondType.SINGLE)
    _set_position(rw, new, _positions(rw)[parent] + d * _bond_length(rw.GetAtomWithIdx(parent).GetSymbol(), sym))
    m, new_h = _fix_hydrogens(rw.GetMol(), {new, parent})
    moving = new_h | {new}
    return m, moving, [_relax(m, moving)]


def _edit_delete(mol, op):
    Chem = _chem()
    idx = int(op["atom"])
    rw = Chem.RWMol(mol)
    nbrs = [n for n in _neighbors(rw, idx) if rw.GetAtomWithIdx(n).GetAtomicNum() != 1]
    hs = sorted([n for n in _neighbors(rw, idx) if rw.GetAtomWithIdx(n).GetAtomicNum() == 1] + [idx], reverse=True)
    for i in nbrs:
        rw.GetAtomWithIdx(i).SetIntProp("_nbr", 1)
    for i in hs:
        rw.RemoveAtom(i)
    if rw.GetNumAtoms() == 0:
        raise WorkspaceError("that would delete the last atom")
    m = rw.GetMol()
    touched = {a.GetIdx() for a in m.GetAtoms() if a.HasProp("_nbr")}
    m, new_h = _fix_hydrogens(m, touched)
    return m, new_h, [_relax(m, new_h) if new_h else None]


def _edit_bond(mol, op):
    Chem = _chem()
    a, b, order = int(op["a"]), int(op["b"]), float(op.get("order", 1))
    if a == b:
        raise WorkspaceError("pick two different atoms")
    rw = Chem.RWMol(mol)
    bond = rw.GetBondBetweenAtoms(a, b)
    if order == 0:
        if bond is None:
            raise WorkspaceError("those atoms are not bonded")
        rw.RemoveBond(a, b)
    else:
        if order not in _BOND:
            raise WorkspaceError("bond order must be 1, 1.5, 2, 3 or 0")
        btype = getattr(Chem.BondType, _BOND[order])
        if bond is None:
            rw.AddBond(a, b, btype)
        else:
            bond.SetBondType(btype)
            bond.SetIsAromatic(order == 1.5)
    touched = {a, b}
    m, new_h = _fix_hydrogens(rw.GetMol(), touched)
    warnings = []
    pos = _positions(m)
    target = _bond_length(m.GetAtomWithIdx(a).GetSymbol(), m.GetAtomWithIdx(b).GetSymbol(), order or 1)
    if order and abs(np.linalg.norm(pos[a] - pos[b]) - target) > 0.35:
        warnings.append(f"atoms {a + 1} and {b + 1} are {np.linalg.norm(pos[a] - pos[b]):.2f} Å apart; "
                        "Clean or Minimize will pull them to bonding distance")
    warnings.append(_relax(m, new_h) if new_h else None)
    return m, new_h | touched, warnings


def _edit_charge(mol, op):
    Chem = _chem()
    idx, delta = int(op["atom"]), int(op.get("delta", 1))
    rw = Chem.RWMol(mol)
    atom = rw.GetAtomWithIdx(idx)
    atom.SetFormalCharge(atom.GetFormalCharge() + delta)
    m, new_h = _fix_hydrogens(rw.GetMol(), {idx})
    return m, new_h | {idx}, [_relax(m, new_h) if new_h else None]


def _edit_hydrogens(mol, op):
    heavy = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 1}
    m, new_h = _fix_hydrogens(mol, heavy)
    return m, new_h, [_relax(m, new_h) if new_h else None]


def _edit_group(mol, op):
    """Replace a hydrogen -- or a terminal group, rooted at the clicked atom
    -- by a functional group, attached along the old bond."""
    Chem = _chem()
    from rdkit.Chem import AllChem

    idx, name = int(op["atom"]), str(op.get("group"))
    if name not in GROUPS:
        raise WorkspaceError(f"unknown group {name!r}")
    nbrs = _neighbors(mol, idx)
    heavy_nbrs = [n for n in nbrs if mol.GetAtomWithIdx(n).GetAtomicNum() != 1]
    if mol.GetAtomWithIdx(idx).GetAtomicNum() == 1:
        if len(nbrs) != 1:
            raise WorkspaceError("that hydrogen is not bonded to one atom")
        anchor = nbrs[0]
    elif len(heavy_nbrs) == 1:
        anchor = heavy_nbrs[0]
    else:
        raise WorkspaceError("click a hydrogen, or the first atom of a terminal group (bonded to one heavy atom)")
    remove = _side(mol, idx, anchor)
    pos = _positions(mol)
    direction = pos[idx] - pos[anchor]
    direction /= np.linalg.norm(direction)

    # The group in 3D, its attachment atom's bond to [*] pointing along -direction.
    frag = Chem.AddHs(Chem.MolFromSmiles(GROUPS[name]))
    if AllChem.EmbedMolecule(frag, randomSeed=7) != 0:
        AllChem.EmbedMolecule(frag, randomSeed=7, useRandomCoords=True)
    try:
        AllChem.MMFFOptimizeMolecule(frag)
    except Exception:
        pass
    dummy = next(a.GetIdx() for a in frag.GetAtoms() if a.GetAtomicNum() == 0)
    attach = _neighbors(frag, dummy)[0]
    fpos = _positions(frag)
    rot = _rotation(fpos[dummy] - fpos[attach], -direction)
    anchor_sym = mol.GetAtomWithIdx(anchor).GetSymbol()
    place = pos[anchor] + direction * _bond_length(anchor_sym, frag.GetAtomWithIdx(attach).GetSymbol())
    fpos = (fpos - fpos[attach]) @ rot.T + place

    # Spin the group about the new bond to the least crowded angle.
    keep = [i for i in range(mol.GetNumAtoms()) if i not in remove]
    others = pos[keep]
    frag_atoms = [i for i in range(frag.GetNumAtoms()) if i != dummy]
    best, best_pos = -1.0, fpos
    for k in range(12):
        ang = 2 * np.pi * k / 12
        axis = direction
        kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        r = np.eye(3) + np.sin(ang) * kx + (1 - np.cos(ang)) * kx @ kx
        trial = (fpos - place) @ r.T + place
        d = np.linalg.norm(trial[frag_atoms][:, None] - others[None], axis=-1).min() if len(others) else 9.9
        if d > best:
            best, best_pos = d, trial

    rw = Chem.RWMol(mol)
    for i in sorted(remove, reverse=True):
        rw.RemoveAtom(i)
    new_anchor = anchor - sum(1 for i in remove if i < anchor)
    offset = rw.GetNumAtoms()
    mapping = {}
    for i in frag_atoms:
        a = frag.GetAtomWithIdx(i)
        new = Chem.Atom(a.GetAtomicNum())
        new.SetFormalCharge(a.GetFormalCharge())
        new.SetIsAromatic(a.GetIsAromatic())
        new.SetNoImplicit(True)
        mapping[i] = rw.AddAtom(new)
    for b in frag.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if dummy in (i, j):
            continue
        rw.AddBond(mapping[i], mapping[j], b.GetBondType())
        if b.GetIsAromatic():
            rw.GetBondBetweenAtoms(mapping[i], mapping[j]).SetIsAromatic(True)
    rw.AddBond(new_anchor, mapping[attach], Chem.BondType.SINGLE)
    m = rw.GetMol()
    for i, j in mapping.items():
        _set_position(m, j, best_pos[i])
    _sanitized(m, f"adding {name} there")
    moving = set(mapping.values())
    return m, moving, [_relax(m, moving)]


def _fibonacci_directions(n: int) -> np.ndarray:
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * k / n)
    theta = np.pi * (1 + 5 ** 0.5) * k
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


def _edit_place(mol, op):
    """Whole molecules or ions next to `atom`, not bonded to it: each at the
    species' contact distance, in the free direction with the most room
    (the atom's open side first), turned to clear the rest; only the placed
    species are then relaxed. `count` > 1 builds a small shell (e.g. of
    waters) around the atom."""
    Chem = _chem()
    from rdkit.Chem import AllChem

    idx, name = int(op["atom"]), str(op.get("species"))
    count = max(1, min(int(op.get("count", 1)), 30))
    if name not in SPECIES:
        raise WorkspaceError(f"unknown species {name!r}")
    smiles, contact = SPECIES[name]
    if mol.GetAtomWithIdx(idx).GetAtomicNum() == 1:
        contact -= 0.9   # clicked a hydrogen: an H-bond / contact to the H itself, not to its heavy atom
    frag = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if frag.GetNumAtoms() > 1:
        if AllChem.EmbedMolecule(frag, randomSeed=11) != 0:
            AllChem.EmbedMolecule(frag, randomSeed=11, useRandomCoords=True)
        try:
            AllChem.UFFOptimizeMolecule(frag)
        except Exception:
            pass
        fpos0 = _positions(frag)
    else:   # a single atom (an ion)
        conf = Chem.Conformer(1)
        frag.AddConformer(conf, assignId=True)
        fpos0 = np.zeros((1, 3))
    fpos0 = fpos0 - fpos0[0]             # the contact atom at the origin
    rng = np.random.default_rng(int(op.get("seed", 0)))
    rw = Chem.RWMol(mol)
    placed = set()
    open_dir = _open_direction(mol, idx)
    dirs = _fibonacci_directions(96)
    for k in range(count):
        pos = _positions(rw)
        centre = pos[idx]
        best, best_xyz = -1.0, None
        # Prefer the atom's open side, then anywhere with room.
        for d in sorted(dirs, key=lambda v: -float(np.dot(v, open_dir)))[:64]:
            for r in (contact, contact + 0.3, contact + 0.6):
                for _ in range(6):
                    q, _r = np.linalg.qr(rng.normal(size=(3, 3)))
                    xyz = fpos0 @ q.T + centre + d * r
                    # The rest of the species points away from the atom.
                    if len(xyz) > 1 and np.linalg.norm(xyz[1:].mean(axis=0) - centre) < r:
                        xyz = (fpos0 @ (-q).T) + centre + d * r
                    clearance = np.linalg.norm(xyz[:, None] - np.delete(pos, [idx], axis=0)[None], axis=-1).min() \
                        if len(pos) > 1 else 9.9
                    score = min(clearance, 2.6) + 0.2 * float(np.dot(d, open_dir)) - 0.05 * (r - contact)
                    if score > best:
                        best, best_xyz = score, xyz
        start = rw.GetNumAtoms()
        for a in frag.GetAtoms():
            new = Chem.Atom(a.GetAtomicNum())
            new.SetFormalCharge(a.GetFormalCharge())
            new.SetNoImplicit(True)
            rw.AddAtom(new)
        for b in frag.GetBonds():
            rw.AddBond(start + b.GetBeginAtomIdx(), start + b.GetEndAtomIdx(), b.GetBondType())
        for i, p in enumerate(best_xyz):
            _set_position(rw, start + i, p)
        placed |= set(range(start, start + frag.GetNumAtoms()))
    m = rw.GetMol()
    m.UpdatePropertyCache(strict=False)
    warnings = []
    if len(placed) > 1:
        warnings.append(_relax(m, placed, steps=300))
    return m, placed, warnings


_EDITS = {"place": _edit_place, "element": _edit_element, "add": _edit_add, "delete": _edit_delete, "bond": _edit_bond,
          "group": _edit_group, "charge": _edit_charge, "hydrogens": _edit_hydrogens}


def clean(molblock: str) -> dict:
    """Force-field relaxation of the whole molecule (MMFF94, else UFF)."""
    mol = read(molblock)
    warning = _relax(mol, None, steps=2000)
    out = describe(mol)
    out["warnings"] = [warning] if warning else []
    return out


def with_coordinates(molblock: str, xyz: str) -> dict:
    """The design with new coordinates (e.g. after minimizing it at a level
    of theory), same atoms and bonds."""
    Chem = _chem()
    mol = read(molblock)
    new = Chem.MolFromXYZBlock(xyz)
    if new is None or new.GetNumAtoms() != mol.GetNumAtoms():
        raise WorkspaceError("the minimized structure does not match the design's atoms")
    for i, p in enumerate(new.GetConformer().GetPositions()):
        _set_position(mol, i, p)
    return describe(mol, sanitized=False)
