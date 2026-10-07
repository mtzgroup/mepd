"""QM/MM partitions: which atoms are quantum, where the QM region is cut
from its environment, and which environment atoms may move.

A `QMMMRegion` belongs to one *system* (a fixed list of atoms in a fixed
order: a solute in a solvent shell, an active site in a protein). Every
structure of a reaction network on that system has the same atoms in the same
order, so one region describes them all.

* **QM atoms** are computed with the QM engine (any mepd engine).
* **Link atoms.** Every covalent bond from a QM atom Q to an MM atom M is
  cut and capped with a hydrogen L on the Q-M line, at L = Q + g (M - Q) with
  g = (r_Q + r_H) / (r_Q + r_M) (covalent radii), so the cap sits at a Q-H
  bond length. Its gradient goes back to Q and M by the chain rule.
* **Active / frozen atoms.** QM/MM optimizations move the QM atoms and an
  active shell of environment atoms (everything within `active_radius` Å
  of the QM region, whole small molecules at a time); the rest is frozen in
  place. Frozen atoms are fixed in every optimization, path, TS search and
  IRC, and are left out of Hessians.

The bonds a region cuts are taken from its *reference* geometry (the system
as it was set up) and never change: a QM/MM boundary must not break.

Link atoms: U. C. Singh, P. A. Kollman, J. Comput. Chem. 7, 718 (1986),
doi:10.1002/jcc.540070604. Subtractive (ONIOM) embedding: M. Svensson,
S. Humbel, R. D. J. Froese, T. Matsubara, S. Sieber, K. Morokuma,
J. Phys. Chem. 100, 19357 (1996), doi:10.1021/jp962071j.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from qcconst.constants import ANGSTROM_TO_BOHR

# Low levels ("MM") a region can be embedded with; see mepd/engines/qmmm.py.
MM_METHODS = {
    "gfnff": "GFN-FF force field (any system, no parameters needed)",
    "gfn2": "GFN2-xTB (semiempirical environment)",
    "gfn1": "GFN1-xTB (semiempirical environment)",
    "amber": "AMBER force field through OpenMM (needs a prmtop, or a PDB of standard residues)",
    "tip3p": "TIP3P water through OpenMM (the environment must be water; for electrostatic embedding)",
    "terachem": "TeraChem's own QM/MM (prmtop + rst7; needs TeraChem or ChemCloud)",
}
EMBEDDINGS = ("mechanical", "electrostatic")

# Covalent radii (Å) for bond perception and link placement: Pyykkö &
# Atsumi, Chem. Eur. J. 15, 186 (2009); others fall back to 1.5 Å.
_RCOV = {
    "H": 0.32, "He": 0.46, "Li": 1.33, "Be": 1.02, "B": 0.85, "C": 0.75, "N": 0.71, "O": 0.63, "F": 0.64,
    "Ne": 0.67, "Na": 1.55, "Mg": 1.39, "Al": 1.26, "Si": 1.16, "P": 1.11, "S": 1.03, "Cl": 0.99, "Ar": 0.96,
    "K": 1.96, "Ca": 1.71, "Sc": 1.48, "Ti": 1.36, "V": 1.34, "Cr": 1.22, "Mn": 1.19, "Fe": 1.16,
    "Co": 1.11, "Ni": 1.10, "Cu": 1.12, "Zn": 1.18, "Ga": 1.24, "Ge": 1.21, "As": 1.21, "Se": 1.16,
    "Br": 1.14, "Kr": 1.17, "Rb": 2.10, "Sr": 1.85, "Mo": 1.38, "Ru": 1.25, "Rh": 1.25, "Pd": 1.20,
    "Ag": 1.28, "Cd": 1.36, "Sn": 1.40, "I": 1.33, "Xe": 1.31, "Cs": 2.32, "Ba": 1.96, "Pt": 1.23,
    "Au": 1.24, "Hg": 1.33, "Pb": 1.44,
}
_BOND_SCALE = 1.25        # bonded if d < 1.25 (r_i + r_j)
_SMALL_MOLECULE = 40      # environment molecules up to this size enter the active shell whole
_ELEMENT_Z = {s: z for z, s in enumerate(
    "X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
    "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb "
    "Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi".split())}


def rcov(symbol: str) -> float:
    return _RCOV.get(str(symbol).capitalize(), 1.5)


def parse_indices(raw) -> list[int]:
    """Atom indices (0-based) from a list, or a string such as "0 3 5-9, 12"."""
    if raw is None:
        return []
    if isinstance(raw, (int, np.integer)):
        return [int(raw)]
    if isinstance(raw, str):
        out: list[int] = []
        for tok in raw.replace(",", " ").split():
            if "-" in tok.strip("-") and not tok.startswith("-"):
                a, b = tok.split("-", 1)
                out.extend(range(int(a), int(b) + 1))
            else:
                out.append(int(tok))
        return sorted(set(out))
    return sorted({int(v) for v in raw})


def format_indices(indices: Iterable[int]) -> str:
    """Compact text for an index list: [0,1,2,5] -> "0-2 5"."""
    idx = sorted(set(int(i) for i in indices))
    parts, k = [], 0
    while k < len(idx):
        j = k
        while j + 1 < len(idx) and idx[j + 1] == idx[j] + 1:
            j += 1
        parts.append(str(idx[k]) if j == k else f"{idx[k]}-{idx[j]}")
        k = j + 1
    return " ".join(parts)


def bonds(symbols: Sequence[str], coords_angstrom: np.ndarray) -> list[tuple[int, int]]:
    """Covalent bonds by distance (scaled covalent radii); neighbor search,
    so it stays fast for thousands of atoms."""
    from scipy.spatial import cKDTree

    xyz = np.asarray(coords_angstrom, dtype=float)
    r = np.array([rcov(s) for s in symbols])
    tree = cKDTree(xyz)
    out = []
    for i, j in sorted(tree.query_pairs(r=2 * _BOND_SCALE * float(r.max()))):
        if np.linalg.norm(xyz[i] - xyz[j]) < _BOND_SCALE * (r[i] + r[j]):
            out.append((i, j))
    return out


def molecules(natoms: int, bond_list: Iterable[tuple[int, int]]) -> list[list[int]]:
    """Connected components (molecules) of the bond graph."""
    parent = list(range(natoms))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in bond_list:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    groups: dict[int, list[int]] = {}
    for a in range(natoms):
        groups.setdefault(find(a), []).append(a)
    return sorted(groups.values(), key=lambda g: g[0])


def link_ratio(q_symbol: str, m_symbol: str) -> float:
    return (rcov(q_symbol) + rcov("H")) / (rcov(q_symbol) + rcov(m_symbol))


# QM engines (profile engine_name) that take point charges, i.e. can do
# electrostatic embedding.
POINT_CHARGE_ENGINES = ("xtb", "psi4")


def embedding_problem(embedding: str, engine_name: str) -> Optional[str]:
    """Why a region with this embedding cannot run at this QM level (with
    how to fix it), or None."""
    if embedding != "electrostatic" or str(engine_name or "gxtb").lower() in POINT_CHARGE_ENGINES:
        return None
    return (f"This QM/MM system uses electrostatic embedding (the environment's charges enter the QM calculation, "
            f"e.g. TIP3P water), which needs a QM level that takes point charges; engine_name = "
            f"\"{engine_name or 'gxtb'}\" does not. Use an xtb (GFN2-xTB) or Psi4 profile (engine_name = \"xtb\" or "
            f"\"psi4\"), or rebuild the "
            f"system with a GFN-FF environment (mechanical embedding).")


@dataclass
class QMMMRegion:
    """One system's QM/MM partition. Indices are 0-based into the system's
    atoms. `links` are the cut (QM atom, MM atom) bonds; `frozen_atoms` is
    the resolved frozen set (explicit ones plus everything outside the
    active shell)."""

    qm_atoms: list[int]
    natoms: int
    symbols: list[str] = field(default_factory=list)
    qm_charge: int = 0
    qm_multiplicity: int = 1
    charge: int = 0                       # the whole system's
    mm: str = "gfnff"
    embedding: str = "mechanical"
    active_radius: Optional[float] = 6.0  # Å; None = every atom may move (unless listed frozen)
    frozen_atoms: list[int] = field(default_factory=list)
    links: list[tuple[int, int]] = field(default_factory=list)
    reference: Optional[str] = None       # xyz text of the system as set up (topology for the low level)
    name: str = ""
    # AMBER inputs (mm = "amber" / "terachem"): paths, or file text keyed by name.
    prmtop: Optional[str] = None
    pdb: Optional[str] = None
    forcefield: list[str] = field(default_factory=lambda: ["amber14-all.xml", "amber14/tip3p.xml"])
    tcin: Optional[str] = None            # mm = "terachem": a tc.in template
    # The molecule the system was built around (system atom indices, in that
    # molecule's own order): where another geometry of it (a product, a TS)
    # is put by mepd.qmmm_build.embed. Empty = the QM atoms.
    solute_atoms: list[int] = field(default_factory=list)
    # Optimizations, TS searches and IRCs relax the environment with cheap
    # force-field steps between QM steps (mepd.engines.microiter); used with
    # OpenMM environments (tip3p, amber), where a force-field step is cheap.
    microiterations: bool = True
    # Path searches renumber identical solvent molecules to match between the
    # ends and march the solvent along the initial path (mepd.qmmm_path).
    solvent_correspondence: bool = True

    def __post_init__(self):
        self.qm_atoms = parse_indices(self.qm_atoms)
        self.solute_atoms = [int(i) for i in (self.solute_atoms or [])]
        self.frozen_atoms = parse_indices(self.frozen_atoms)
        self.links = [(int(q), int(m)) for q, m in (self.links or [])]
        self.symbols = [str(s) for s in (self.symbols or [])]
        self.mm = str(self.mm or "gfnff").lower()
        if self.mm not in MM_METHODS:
            raise ValueError(f"qmmm.mm must be one of {', '.join(MM_METHODS)}, got {self.mm!r}")
        if self.embedding not in EMBEDDINGS:
            raise ValueError(f"qmmm.embedding must be one of {', '.join(EMBEDDINGS)}, got {self.embedding!r}")
        if self.mm == "tip3p" and self.embedding != "electrostatic":
            raise ValueError("mm = \"tip3p\" goes with embedding = \"electrostatic\" (its QM atoms carry no "
                             "force-field charges: mechanically embedded, the water would not see the QM region's "
                             "charges at all)")
        if self.embedding == "electrostatic" and self.mm not in ("tip3p", "amber"):
            raise ValueError("electrostatic embedding needs fixed environment charges: mm = \"tip3p\" (water) or "
                             "\"amber\"; GFN-FF/xTB charges follow the geometry and cannot be used as point charges")
        if not self.qm_atoms:
            raise ValueError("a QM/MM region needs at least one QM atom")
        if self.natoms and max(self.qm_atoms) >= self.natoms:
            raise ValueError(f"QM atom {max(self.qm_atoms)} is out of range for {self.natoms} atoms")
        overlap = set(self.qm_atoms) & set(self.frozen_atoms)
        if overlap:
            raise ValueError(f"QM atoms cannot be frozen: {format_indices(overlap)}")

    def __repr__(self) -> str:
        return f"QMMMRegion({self.summary()})"

    # ------------------------------------------------------------ setup
    @classmethod
    def build(cls, structure, qm_atoms, *, qm_charge: Optional[int] = None, qm_multiplicity: int = 1,
              active_radius: Optional[float] = 6.0, frozen_atoms=(), mm: str = "gfnff", name: str = "",
              **extra) -> "QMMMRegion":
        """A region for `structure` (a qcdata Structure, the system as set
        up): cut bonds, active shell and frozen set worked out from its
        geometry. `qm_charge` defaults to the formal charge the QM atoms'
        molecules carry when the whole system is neutral-in-pieces, i.e. the
        system's charge if the QM region holds all of it."""
        symbols = [str(s) for s in structure.symbols]
        xyz = np.asarray(structure.geometry, dtype=float) / ANGSTROM_TO_BOHR
        qm = parse_indices(qm_atoms)
        if mm == "tip3p":       # fixed water charges are for electrostatic embedding
            extra.setdefault("embedding", "electrostatic")
        bl = bonds(symbols, xyz)
        qmset = set(qm)
        links = sorted((i, j) if i in qmset else (j, i) for i, j in bl if (i in qmset) != (j in qmset))
        frozen = set(parse_indices(frozen_atoms))
        if active_radius is not None:
            active = set(active_shell(symbols, xyz, qm, float(active_radius), bl))
            frozen |= set(range(len(symbols))) - active
        frozen -= qmset
        return cls(qm_atoms=qm, natoms=len(symbols), symbols=symbols,
                   qm_charge=int(structure.charge if qm_charge is None else qm_charge),
                   qm_multiplicity=int(qm_multiplicity), charge=int(structure.charge), mm=mm,
                   active_radius=active_radius, frozen_atoms=sorted(frozen), links=links,
                   reference=structure.to_xyz(), name=name, **extra)

    # ------------------------------------------------------------ views
    @property
    def solute(self) -> list[int]:
        """Atoms another geometry of the solute goes onto (see embed)."""
        return list(self.solute_atoms or self.qm_atoms)

    @property
    def mm_atoms(self) -> list[int]:
        q = set(self.qm_atoms)
        return [i for i in range(self.natoms) if i not in q]

    @property
    def active_atoms(self) -> list[int]:
        f = set(self.frozen_atoms)
        return [i for i in range(self.natoms) if i not in f]

    @property
    def active_mm_atoms(self) -> list[int]:
        f, q = set(self.frozen_atoms), set(self.qm_atoms)
        return [i for i in range(self.natoms) if i not in f and i not in q]

    def link_ratios(self) -> list[float]:
        return [link_ratio(self.symbols[q], self.symbols[m]) for q, m in self.links]

    def link_positions(self, coords: np.ndarray) -> np.ndarray:
        """Link-atom positions (same units as `coords`)."""
        x = np.asarray(coords, dtype=float)
        if not self.links:
            return np.zeros((0, 3))
        q = np.array([a for a, _ in self.links])
        m = np.array([b for _, b in self.links])
        g = np.array(self.link_ratios())[:, None]
        return x[q] + g * (x[m] - x[q])

    def model_coords(self, coords: np.ndarray) -> np.ndarray:
        """QM atoms then link atoms."""
        x = np.asarray(coords, dtype=float)
        return np.vstack([x[self.qm_atoms], self.link_positions(x)])

    @property
    def model_symbols(self) -> list[str]:
        return [self.symbols[i] for i in self.qm_atoms] + ["H"] * len(self.links)

    def model_structure(self, structure, charge: Optional[int] = None, multiplicity: Optional[int] = None):
        """The QM model system (QM atoms + link hydrogens) as a Structure."""
        from qcdata import Structure

        return Structure(symbols=self.model_symbols, geometry=self.model_coords(structure.geometry),
                         charge=int(self.qm_charge if charge is None else charge),
                         multiplicity=int(self.qm_multiplicity if multiplicity is None else multiplicity))

    def model_gradient_to_full(self, g_model: np.ndarray) -> np.ndarray:
        """Spread a model-system gradient (QM atoms + links) onto the full
        system: link L = Q + g (M - Q), so dE/dQ += (1-g) dE/dL and
        dE/dM += g dE/dL."""
        g_model = np.asarray(g_model, dtype=float).reshape(-1, 3)
        out = np.zeros((self.natoms, 3))
        nq = len(self.qm_atoms)
        out[self.qm_atoms] += g_model[:nq]
        for k, ((q, m), g) in enumerate(zip(self.links, self.link_ratios())):
            out[q] += (1.0 - g) * g_model[nq + k]
            out[m] += g * g_model[nq + k]
        return out

    def reference_structure(self):
        from qcdata import Structure

        if not self.reference:
            return None
        s = Structure.from_xyz(self.reference)
        return s.model_copy(update={"charge": self.charge, "multiplicity": self.qm_multiplicity})

    def subset_structure(self, structure, atoms: Sequence[int]):
        from qcdata import Structure

        idx = np.asarray(list(atoms), dtype=int)
        return Structure(symbols=[structure.symbols[i] for i in idx], geometry=np.asarray(structure.geometry)[idx],
                         charge=int(structure.charge), multiplicity=int(structure.multiplicity))

    def qm_structure(self, structure):
        """The QM atoms alone (no caps), with the QM charge and spin: what
        graphs, SMILES and species identity are taken from."""
        s = self.subset_structure(structure, self.qm_atoms)
        return s.model_copy(update={"charge": self.qm_charge, "multiplicity": self.qm_multiplicity})

    # ------------------------------------------------------------ checks
    def check(self, structure=None) -> list[str]:
        """Plain-language problems with this partition (empty = fine)."""
        problems = []
        n_e = sum(_ELEMENT_Z.get(s.capitalize(), 0) for s in self.model_symbols) - int(self.qm_charge)
        if (n_e % 2) == (int(self.qm_multiplicity) % 2):
            problems.append(f"QM region with links has {n_e} electrons: multiplicity {self.qm_multiplicity} "
                            f"is impossible (check the QM charge)")
        qmset = set(self.qm_atoms)
        for q, m in self.links:
            if self.symbols and self.symbols[q] == "H":
                problems.append(f"the boundary cuts the bond of hydrogen {q} (QM) to {m} (MM): put both on one side")
            if self.symbols and self.symbols[m] == "H":
                problems.append(f"the boundary cuts the bond of {q} (QM) to hydrogen {m}: put the H in the QM region")
        per_q: dict[int, int] = {}
        for q, _ in self.links:
            per_q[q] = per_q.get(q, 0) + 1
        for q, k in per_q.items():
            if k > 1:
                problems.append(f"QM atom {q} has {k} bonds cut: move its neighbours into the QM region")
        if structure is not None:
            if len(structure.symbols) != self.natoms:
                problems.append(f"structure has {len(structure.symbols)} atoms, the region {self.natoms}")
            else:
                xyz = np.asarray(structure.geometry) / ANGSTROM_TO_BOHR
                for q, m in self.links:
                    d = float(np.linalg.norm(xyz[q] - xyz[m]))
                    if d > 1.6 * (rcov(self.symbols[q]) + rcov(self.symbols[m])):
                        problems.append(f"boundary bond {q}-{m} is broken ({d:.2f} Å): the QM/MM cut must stay bonded")
        for q, m in self.links:
            if self.symbols and (self.symbols[q] in ("O", "N") or self.symbols[m] in ("O", "N")):
                problems.append(f"boundary bond {q}-{m} is polar ({self.symbols[q]}-{self.symbols[m]}): "
                                f"cut a C-C bond instead if you can")
        if not qmset.isdisjoint(self.frozen_atoms):
            problems.append("some QM atoms are frozen")
        if self.mm == "gfnff" and self.natoms > 1500:
            from mepd.engines.gfnff import memory_gb

            problems.append(f"{self.natoms} atoms is large for GFN-FF (about {memory_gb(self.natoms):.0f} GB and "
                            f"several seconds per energy): for a protein, use AMBER (a prmtop) instead")
        return problems

    # ---------------------------------------------------------------- io
    def signature(self) -> str:
        """Fingerprint of everything that changes energies (for level keys):
        two structures' energies compare only if this matches."""
        keep = {k: v for k, v in self.to_dict().items()
                if k not in ("name", "frozen_atoms", "active_radius", "solute_atoms", "microiterations",
                          "solvent_correspondence")}
        return hashlib.sha1(json.dumps(keep, sort_keys=True).encode()).hexdigest()[:12]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["links"] = [list(x) for x in self.links]
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "QMMMRegion":
        data = dict(data)
        known = set(cls.__dataclass_fields__)
        extra = set(data) - known
        if extra:
            raise ValueError(f"unknown [qmmm] keys: {', '.join(sorted(extra))}")
        return cls(**data)

    def save(self, fp) -> None:
        Path(fp).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def open(cls, fp) -> "QMMMRegion":
        return cls.from_dict(json.loads(Path(fp).read_text()))

    def summary(self) -> str:
        return (f"QM {len(self.qm_atoms)} atoms (charge {self.qm_charge}, multiplicity {self.qm_multiplicity}), "
                f"{len(self.links)} link H, {len(self.active_mm_atoms)} active + {len(self.frozen_atoms)} frozen "
                f"environment atoms, low level {self.mm}")


def active_shell(symbols: Sequence[str], coords_angstrom: np.ndarray, qm: Sequence[int], radius: float,
                 bond_list: Optional[list] = None) -> list[int]:
    """Atoms within `radius` Å of any QM atom (plus the QM atoms). Small
    environment molecules (solvent, ions, small ligands) are taken whole, so
    a water is never half frozen; atoms of large ones (a protein) one by one."""
    from scipy.spatial import cKDTree

    xyz = np.asarray(coords_angstrom, dtype=float)
    n = len(symbols)
    if bond_list is None:
        bond_list = bonds(symbols, xyz)
    tree = cKDTree(xyz)
    near = set(qm)
    for hits in tree.query_ball_point(xyz[list(qm)], r=float(radius)):
        near.update(hits)
    out = set(near)
    for mol in molecules(n, bond_list):
        if len(mol) <= _SMALL_MOLECULE and near.intersection(mol):
            out.update(mol)
    # Hydrogens follow their heavy atom (a frozen H on a moving C is a clash).
    for i, j in bond_list:
        for h, x in ((i, j), (j, i)):
            if symbols[h] == "H" and x in out:
                out.add(h)
    return sorted(out)


def region_from_inputs(data) -> Optional[QMMMRegion]:
    """A region from a profile's [qmmm] table (or a path to a region JSON
    in its `file` key); None if there is none."""
    if not data:
        return None
    data = {k: v for k, v in dict(data).items() if k not in ("base_dir",)}
    if "file" in data:
        fp = Path(data.pop("file"))
        region = QMMMRegion.open(fp)
        if data:
            region = QMMMRegion.from_dict({**region.to_dict(), **data})
        return region
    if not data.get("natoms") and data.get("reference"):
        from qcdata import Structure

        ref = Path(data["reference"])
        text = ref.read_text() if ref.suffix == ".xyz" and ref.exists() else data["reference"]
        s = Structure.from_xyz(text).model_copy(update={"charge": int(data.get("charge", 0))})
        args = {k: v for k, v in data.items() if k not in ("reference", "natoms", "symbols", "links", "charge")}
        return QMMMRegion.build(s, args.pop("qm_atoms"), **args)
    return QMMMRegion.from_dict(data)


# ------------------------------------------------------------- diagnostics
def diagnose(region: QMMMRegion, frames: Sequence, energies: Optional[Sequence[float]] = None) -> dict:
    """Per-frame checks that a QM/MM structure or path is not nonsense:

    * frozen atoms must not move (`frozen_drift`, Å, from frame 0);
    * cut (QM-MM) bonds must stay bonded (`boundary`, Å per link);
    * nothing may react in the MM region (`mm_bond_changes`: bonds made or
      broken among environment atoms, against the reference geometry);
    * QM and MM atoms must not clash (`closest_contact`, Å, non-bonded);
    * how much the QM region and the active shell move (`qm_rmsd`,
      `mm_rmsd`, Å from frame 0), and which QM bonds form/break.

    `frames`: qcdata Structures (or coordinate arrays in Bohr). Returns
    plain JSON-able data with `warnings` in plain language."""
    from scipy.spatial import cKDTree

    xyz = [np.asarray(getattr(f, "geometry", f), dtype=float) / ANGSTROM_TO_BOHR for f in frames]
    if not xyz:
        return {"frames": [], "warnings": []}
    symbols = region.symbols
    qm = np.asarray(region.qm_atoms, dtype=int)
    qmset = set(region.qm_atoms)
    frozen = np.asarray(region.frozen_atoms, dtype=int)
    active_mm = np.asarray(region.active_mm_atoms, dtype=int)
    ref = region.reference_structure()
    ref_xyz = np.asarray(ref.geometry) / ANGSTROM_TO_BOHR if ref is not None else xyz[0]
    ref_bonds = set(bonds(symbols, ref_xyz))
    mm_ref = {b for b in ref_bonds if b[0] not in qmset and b[1] not in qmset}
    qm_ref = {b for b in bonds(symbols, xyz[0]) if b[0] in qmset and b[1] in qmset}
    excluded = set(ref_bonds)      # bonded pairs are not "contacts"
    near_link = {m for _, m in region.links}
    out_frames, warnings = [], []
    worst = {"frozen": 0.0, "boundary": 0.0, "contact": 9e9, "mm_changes": 0}
    for k, x in enumerate(xyz):
        fb = set(bonds(symbols, x)) if len(symbols) < 20000 else set()
        mm_now = {b for b in fb if b[0] not in qmset and b[1] not in qmset}
        qm_now = {b for b in fb if b[0] in qmset and b[1] in qmset}
        mm_changes = sorted(mm_now ^ mm_ref)
        drift = float(np.abs(x[frozen] - xyz[0][frozen]).max()) if len(frozen) else 0.0
        bl = [float(np.linalg.norm(x[q] - x[m])) for q, m in region.links]
        stretch = [b / (rcov(symbols[q]) + rcov(symbols[m])) for b, (q, m) in zip(bl, region.links)]
        contact, pair = 9e9, None
        mm_idx = np.array([i for i in range(len(x)) if i not in qmset], dtype=int)
        if len(mm_idx):
            tree = cKDTree(x[mm_idx])
            for qi in qm:
                for j in tree.query_ball_point(x[qi], r=3.0):
                    m = int(mm_idx[j])
                    if (min(qi, m), max(qi, m)) in excluded or m in near_link:
                        continue
                    d = float(np.linalg.norm(x[qi] - x[m]))
                    if d < contact:
                        contact, pair = d, (int(qi), m)
        rec = {
            "frame": k,
            "frozen_drift": drift,
            "qm_rmsd": float(np.sqrt(np.mean(np.sum((x[qm] - xyz[0][qm]) ** 2, axis=1)))),
            "mm_rmsd": float(np.sqrt(np.mean(np.sum((x[active_mm] - xyz[0][active_mm]) ** 2, axis=1))))
            if len(active_mm) else 0.0,
            "mm_max_move": float(np.linalg.norm(x[active_mm] - xyz[0][active_mm], axis=1).max())
            if len(active_mm) else 0.0,
            "boundary": bl,
            "boundary_stretch": stretch,
            "closest_contact": None if pair is None else contact,
            "closest_pair": pair,
            "mm_bond_changes": [list(b) for b in mm_changes[:20]],
            "qm_formed": [list(b) for b in sorted(qm_now - qm_ref)],
            "qm_broken": [list(b) for b in sorted(qm_ref - qm_now)],
        }
        if energies is not None and k < len(energies) and energies[k] is not None:
            rec["energy"] = float(energies[k])
        out_frames.append(rec)
        worst["frozen"] = max(worst["frozen"], drift)
        worst["boundary"] = max([worst["boundary"], *stretch])
        if pair is not None:
            worst["contact"] = min(worst["contact"], contact)
        worst["mm_changes"] = max(worst["mm_changes"], len(mm_changes))
    if worst["frozen"] > 1e-3:
        warnings.append(f"frozen atoms moved by up to {worst['frozen']:.3f} Å")
    if worst["boundary"] > 1.3:
        warnings.append(f"a QM/MM boundary bond is stretched to {worst['boundary']:.2f}× its covalent length")
    if worst["mm_changes"]:
        warnings.append(f"bonds changed in the MM region ({worst['mm_changes']} in the worst frame): "
                        "the environment reacted, which the low level cannot describe; enlarge the QM region")
    if worst["contact"] < 1.2:
        warnings.append(f"QM and MM atoms come within {worst['contact']:.2f} Å (a clash)")
    return {"natoms": region.natoms, "qm_atoms": list(region.qm_atoms), "frames": out_frames,
            "worst": {**worst, "contact": None if worst["contact"] > 1e8 else worst["contact"]},
            "warnings": warnings}
