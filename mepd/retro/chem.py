"""RDKit helpers shared by the retrosynthesis methods."""
from __future__ import annotations

import functools
from collections import Counter
from typing import Iterable, Optional

import numpy as np

# Small molecules a step may release (forward direction) besides its main
# product: a template keeps its leaving groups on the precursors, so the
# atoms the product lacks are made up from these to balance the step.
BYPRODUCTS = ["O", "Cl", "Br", "I", "F", "CO", "C(=O)=O", "N", "CO", "CCO", "CC(=O)O", "CC(C)(C)O", "N#N",
              "O=S(=O)(O)O", "OB(O)O", "[H][H]", "O=C=O", "C=O", "CS(=O)(=O)O", "Cc1ccc(S(=O)(=O)O)cc1",
              "O=C(O)C(F)(F)F", "CCN(CC)CC", "OCC(F)(F)F", "C[Si](C)(C)O", "CC(C)(C)OC(=O)O", "O=CO", "On1nnc2ccccc21",
              "CN(C)C=O", "C1CCOC1", "c1ccc(P(=O)(c2ccccc2)c2ccccc2)cc1", "[Na]Cl", "[K]Br", "O=[N+]([O-])O"]


def mol(smiles: str):
    from rdkit import Chem

    m = Chem.MolFromSmiles(smiles)
    if m is None or m.GetNumAtoms() == 0:
        return None
    return m


@functools.lru_cache(maxsize=100_000)
def canonical(smiles: str) -> Optional[str]:
    """Canonical SMILES (stereo kept), None if RDKit can't read it."""
    from rdkit import Chem

    m = mol(smiles)
    return Chem.MolToSmiles(m) if m is not None else None


@functools.lru_cache(maxsize=100_000)
def inchikey(smiles: str) -> Optional[str]:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    m = mol(smiles)
    if m is None:
        return None
    try:
        return Chem.MolToInchiKey(m) or None
    except Exception:
        return None


def fingerprint(smiles: str, nbits: int = 2048, radius: int = 2) -> np.ndarray:
    """ECFP4 bits as float32, as AiZynthFinder's networks take them."""
    from rdkit import DataStructs

    gen = _morgan(radius, nbits)
    arr = np.zeros((nbits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol(smiles)), arr)
    return arr


@functools.lru_cache(maxsize=8)
def _morgan(radius: int, nbits: int):
    from rdkit.Chem import rdFingerprintGenerator

    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)


@functools.lru_cache(maxsize=100_000)
def sa_score(smiles: str) -> float:
    """Synthetic accessibility, 1 (easy) to 10 (hard); Ertl & Schuffenhauer."""
    from rdkit.Contrib.SA_Score import sascorer

    m = mol(smiles)
    return float(sascorer.calculateScore(m)) if m is not None else 10.0


def heavy_atoms(smiles: str) -> int:
    m = mol(smiles)
    return m.GetNumHeavyAtoms() if m is not None else 0


def formula(smiles_list: Iterable[str]) -> Counter:
    """Element counts (hydrogens included) and total charge ('+' key)."""
    from rdkit import Chem

    c = Counter()
    for s in smiles_list:
        m = mol(s)
        if m is None:
            raise ValueError(f"not a valid SMILES: {s!r}")
        m = Chem.AddHs(m)
        for a in m.GetAtoms():
            c[a.GetSymbol()] += 1
        c["+"] += Chem.GetFormalCharge(m)
    return Counter({k: v for k, v in c.items() if v})


# Small molecules a step may also consume that its proposal leaves out: a
# reduction's H2, a hydrolysis's water, an oxidation's O2.
CO_REACTANTS = ["[H][H]", "O", "O=O"]


def balance_step(precursors: list[str], product: str, *, max_coreactants: int = 4,
                 max_byproducts: int = 3) -> Optional[tuple[list[str], list[str]]]:
    """(co-reactants, byproducts) that balance precursors + co-reactants ->
    product + byproducts, fewest co-reactants first; None if nothing does.
    A molecule is never both added and released."""
    import itertools

    found = balance(precursors, product, max_byproducts=max_byproducts)
    if found is not None:
        return [], found
    for n in range(1, max_coreactants + 1):
        for co in itertools.combinations_with_replacement(CO_REACTANTS, n):
            found = balance([*precursors, *co], product, max_byproducts=max_byproducts)
            if found is not None and not set(co) & {canonical(b) for b in found}:
                return list(co), found
    return None


def balance(precursors: list[str], product: str, *, max_byproducts: int = 3) -> Optional[list[str]]:
    """The byproducts that balance precursors -> product + byproducts (atoms
    and charge), fewest first; [] if it balances already, None if no
    combination of BYPRODUCTS (up to `max_byproducts`) does."""
    import itertools

    left = formula(precursors)
    left.subtract(formula([product]))
    need = Counter({k: v for k, v in left.items() if v})
    if not need:
        return []
    if any(v < 0 for v in need.values()):
        return None   # the product has atoms the precursors lack
    options = list(dict.fromkeys(canonical(b) for b in BYPRODUCTS))
    forms = {b: formula([b]) for b in options}
    for n in range(1, max_byproducts + 1):
        for combo in itertools.combinations_with_replacement(options, n):
            total = Counter()
            for b in combo:
                total.update(forms[b])
            if Counter({k: v for k, v in total.items() if v}) == need:
                return list(combo)
    return None
