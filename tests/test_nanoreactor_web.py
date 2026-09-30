"""A nanoreactor job's network in Explore: species as nodes, reactions as
workspace reactions whose optimized subsystem ends are hidden structures
joined by an edge (so pair operations run on them)."""
import json

import numpy as np
import pytest

pytest.importorskip("fastapi")

from mepd.web.nanoreactor import adopt_nanoreactor  # noqa: E402
from mepd.web.workspace import Workspace  # noqa: E402


def _xyz(smiles_list, energy=None, spacing=4.0):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    lines = []
    for k, smi in enumerate(smiles_list):
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        AllChem.EmbedMolecule(mol, randomSeed=7)
        xyz = mol.GetConformer().GetPositions() + np.array([spacing * k, 0.0, 0.0])
        lines += [f"{a.GetSymbol()} {x:.6f} {y:.6f} {z:.6f}" for a, (x, y, z) in zip(mol.GetAtoms(), xyz)]
    comment = f"energy={energy}" if energy is not None else ""
    return f"{len(lines)}\n{comment}\n" + "\n".join(lines) + "\n"


@pytest.fixture
def network(tmp_path):
    out = tmp_path / "job" / "output"
    (out / "species").mkdir(parents=True)
    (out / "reactions").mkdir()
    species = []
    for k, (smi, e) in enumerate([("CC=O", -20.0), ("O", -5.0), ("C=CO", -19.98)]):
        fp = out / "species" / f"species_{k}.xyz"
        fp.write_text(_xyz([smi], e))
        species.append({"id": k, "smiles": smi, "charge": 0, "multiplicity": 1, "formula": "", "first_fs": 0.0,
                        "count": 1, "initial": 1, "instances": [], "energy": e, "file": str(fp),
                        "md_file": None, "note": ""})

    def complex_of(rid, left, right, e_r, e_p):
        files = {}
        for side, smis, e in (("reactant", left, e_r), ("product", right, e_p)):
            fp = out / "reactions" / f"r{rid}_{side}.xyz"
            fp.write_text(_xyz(smis, e))
            files[side] = str(fp)
        return {"instance": 0, "charge": 0, "multiplicity": 1, "reactant_energy": e_r, "product_energy": e_p,
                "delta_e_kcal": (e_p - e_r) * 627.5, **files}

    reactions = [
        {"id": 0, "reactants": [0], "products": [2], "shuttles": [], "label": "CC=O -> C=CO", "first_fs": 10.0,
         "count": 2, "reverse_count": 0, "instances": [], "delta_e_kcal": 12.5,
         "complex": complex_of(0, ["CC=O"], ["C=CO"], -20.0, -19.98), "ts": {}},
        {"id": 1, "reactants": [0, 1], "products": [1, 2], "shuttles": [1], "label": "CC=O + O -> C=CO + O",
         "first_fs": 30.0, "count": 1, "reverse_count": 0, "instances": [], "delta_e_kcal": 12.5,
         "complex": complex_of(1, ["CC=O", "O"], ["C=CO", "O"], -25.01, -24.99),
         "ts": {"barrier_kcal": 31.0, "label": "pair_0_1_ts0", "files": {}}},
    ]
    (out / "network.json").write_text(json.dumps({"species": species, "reactions": reactions, "events": []}))
    return out


def test_species_and_reactions_join_explore(tmp_path, network):
    ws = Workspace(tmp_path / "ws")
    job = {"id": "j1", "op": "nanoreactor", "output_dir": str(network), "params": {},
           "level": {"key": "L", "label": "g-xTB", "profile": None}}
    assert adopt_nanoreactor(ws, job)
    snap = ws.snapshot()
    visible = [s for s in snap["structures"].values() if s["role"] != "complex"]
    assert sorted(s["smiles"] for s in visible) == ["C=CO", "CC=O", "O"]
    rx = sorted(snap["reactions"].values(), key=lambda r: r["origin"]["index"])
    assert len(rx) == 2
    direct, shuttled = rx
    sid = {s["smiles"]: s["id"] for s in visible}
    assert direct["reactants"] == [sid["CC=O"]] and direct["products"] == [sid["C=CO"]]
    # Two reactions between the same two species: the second needs a water, on both sides.
    assert shuttled["shuttles"] == [sid["O"]]
    for r in rx:   # each has its own subsystem pair, joined by an edge
        a, b = r["complexes"]
        assert snap["structures"][a]["role"] == "complex" and snap["edges"][r["edge"]]["source"] == a
    assert snap["structures"][shuttled["complexes"][0]]["natoms"] == 10     # acetaldehyde + the shuttle water
    assert snap["edges"][direct["edge"]]["origin"]["proposed"]
    assert snap["edges"][shuttled["edge"]]["origin"]["barrier_kcal"] == 31.0   # its TS came with the job
    # Unchanged file: nothing to do; a second import does not duplicate anything.
    assert not adopt_nanoreactor(ws, job)
    assert adopt_nanoreactor(ws, job, final=True) is False
    assert len(ws.snapshot()["reactions"]) == 2


def test_deleting_a_species_or_the_edge_removes_its_reactions(tmp_path, network):
    ws = Workspace(tmp_path / "ws")
    job = {"id": "j1", "op": "nanoreactor", "output_dir": str(network), "params": {}, "level": None}
    adopt_nanoreactor(ws, job)
    snap = ws.snapshot()
    direct = next(r for r in snap["reactions"].values() if r["origin"]["index"] == 0)
    ws.delete_edge(direct["edge"])     # what deleting the selected dot does
    snap = ws.snapshot()
    assert direct["id"] not in snap["reactions"]
    assert not set(direct["complexes"]) & set(snap["structures"])
    water = next(s for s in snap["structures"].values() if s["smiles"] == "O")
    removed = ws.delete_many([water["id"]], [])
    assert len(removed["reactions"]) == 1 and ws.snapshot()["reactions"] == {}
    assert not [s for s in ws.snapshot()["structures"].values() if s["role"] == "complex"]
