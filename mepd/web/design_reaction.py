"""The Design tab's reaction mode: a reactant and a product edited side by
side, where an edit on one side is repeated on the matching atom of the
other ("linked"), so a reaction can be varied (e.g. a substituent swapped
on both ends) and re-run to compare barriers.

The pair is built from a reaction SMILES (mepd/reaction_smiles.py). The
atom correspondence `amap` (amap[i] = the product atom matching reactant
atom i, -1 for none) is carried through edits by tagging both molecules
with atom map numbers: an edit keeps the numbers of the atoms it does not
touch, and the atoms it adds are matched by element, bonded neighbours and
the smallest endpoint RMSD (`reaction_smiles._match`). That is fast rather
than rigorous, as the Design tab wants; a path search with atom mapping
re-examines the mapping anyway.
"""

from __future__ import annotations

from mepd.web import design
from mepd.web.workspace import WorkspaceError

ATOM_FIELDS = ("atom", "a", "b")


def _strip(mol):
    for a in mol.GetAtoms():
        a.SetAtomMapNum(0)
    return mol


def _info(mol, warnings=None, sanitized=True) -> dict:
    out = design.describe(_strip(mol), sanitized=sanitized)
    out["warnings"] = [w for w in (warnings or []) if w]
    return out


def new_pair(text: str) -> dict:
    """{reactant, product, amap, mapping} from a reaction SMILES."""
    from mepd.reaction_smiles import ReactionSmilesError, reaction_pair

    try:
        pair = reaction_pair(text)
    except ReactionSmilesError as exc:
        raise WorkspaceError(str(exc)) from None
    for m in (pair.reactant, pair.product):
        for a in m.GetAtoms():
            a.SetNoImplicit(True)
    n = pair.reactant.GetNumAtoms()
    return {
        "reactant": _info(pair.reactant, pair.notes), "product": _info(pair.product),
        "amap": list(range(n)),
        "mapping": {"source": pair.source, "mapped_smiles": pair.mapped_smiles, "rmsd": round(pair.rmsd, 3),
                    "n_candidates": pair.n_candidates},
    }


def _tagged(molblock: str, numbers: dict[int, int]):
    from rdkit import Chem

    mol = design.read(molblock)
    for a in mol.GetAtoms():
        a.SetAtomMapNum(numbers.get(a.GetIdx(), 0))
    return Chem.MolToMolBlock(mol)


def _translate(op: dict, idx_map: dict[int, int]) -> dict | None:
    """The op with its atom indices moved to the other side, or None when
    an atom it names has no counterpart there."""
    out = dict(op)
    for f in ATOM_FIELDS:
        if f in op and op[f] is not None:
            j = idx_map.get(int(op[f]), -1)
            if j < 0:
                return None
            out[f] = j
    return out


def linked_edit(reactant_mb: str, product_mb: str, amap: list[int], side: str, op: dict, linked: bool) -> dict:
    """Apply `op` to `side` ("reactant"/"product") and, when `linked`, the
    same edit to the matching atoms of the other side. Returns {reactant,
    product, amap, mirrored, note}."""
    from mepd.reaction_smiles import _match

    if side not in ("reactant", "product"):
        raise WorkspaceError("side is 'reactant' or 'product'")
    # Tag: reactant atom i and its product counterpart both get i+1.
    r_num = {i: i + 1 for i in range(len(amap))}
    p_num = {j: i + 1 for i, j in enumerate(amap) if j is not None and j >= 0}
    tagged = {"reactant": _tagged(reactant_mb, r_num), "product": _tagged(product_mb, p_num)}
    other = "product" if side == "reactant" else "reactant"
    r2p = {i: j for i, j in enumerate(amap) if j is not None and j >= 0}
    p2r = {j: i for i, j in r2p.items()}

    results = {side: design.edit(tagged[side], op)}
    mirrored, note = False, None
    if linked:
        op2 = _translate(op, r2p if side == "reactant" else p2r)
        if op2 is None:
            note = f"Only the {side} was changed: the atom you edited has no matching atom in the {other}."
        else:
            try:
                results[other] = design.edit(tagged[other], op2)
                mirrored = True
            except WorkspaceError as exc:
                raise WorkspaceError(f"The same edit on the {other} is not possible ({exc}). Turn off "
                                     f"'Edit both' to change the {side} alone.") from None
    r = design.read(results["reactant"]["molblock"] if "reactant" in results else tagged["reactant"])
    p = design.read(results["product"]["molblock"] if "product" in results else tagged["product"])
    order, rmsd = _match(r, p, strict=False)
    warn = {k: v.get("warnings", []) for k, v in results.items()}
    out = {
        "reactant": _info(r, warn.get("reactant")), "product": _info(p, warn.get("product")),
        "amap": order, "mirrored": mirrored, "note": note, "rmsd": round(rmsd, 3),
        "changed": {k: v.get("changed", []) for k, v in results.items()},
    }
    out["balanced"] = sorted(a.GetSymbol() for a in r.GetAtoms()) == sorted(a.GetSymbol() for a in p.GetAtoms()) \
        and -1 not in order
    return out


def product_order(amap: list[int]) -> list[int]:
    """The permutation that puts the product's atoms in the reactant's order."""
    if -1 in amap or sorted(amap) != list(range(len(amap))):
        raise WorkspaceError("the reactant and the product do not have the same atoms, so they are not a reaction yet")
    return list(amap)
