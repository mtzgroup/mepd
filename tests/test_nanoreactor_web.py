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


def test_live_and_final_versions_of_a_reaction_are_one_reaction(tmp_path, network):
    """--refine-live writes live_network.json during the MD; the final
    network.json numbers species and reactions differently. They are
    matched by identity, never doubled."""
    data = json.loads((network / "network.json").read_text())
    live = {"live": True, "species": [dict(sp, id=10 + sp["id"]) for sp in data["species"]],
            "reactions": [dict(r, id=5, reactants=[10 + i for i in r["reactants"]],
                               products=[10 + i for i in r["products"]]) for r in data["reactions"][:1]]}
    (network / "network.json").rename(network / "network.final")
    (network / "live_network.json").write_text(json.dumps(live))
    ws = Workspace(tmp_path / "ws")
    job = {"id": "j1", "op": "nanoreactor", "output_dir": str(network), "params": {"refine_live": True}, "level": None}
    assert adopt_nanoreactor(ws, job)
    assert len(ws.snapshot()["reactions"]) == 1
    (network / "network.final").rename(network / "network.json")
    adopt_nanoreactor(ws, job, final=True)
    rx = ws.snapshot()["reactions"]
    assert len(rx) == 2                                   # the live one plus the one only the final run has
    assert sorted(r["label"] for r in rx.values()) == ["CC=O + O -> C=CO + O", "CC=O -> C=CO"]
    assert len([s for s in ws.snapshot()["structures"].values() if s["role"] != "complex"]) == 3


def test_kinetics_uses_reactions_with_barriers_and_ranks_what_controls_the_target(tmp_path, network):
    from mepd.web.kinetics import analyze

    ws = Workspace(tmp_path / "ws")
    job = {"id": "j1", "op": "nanoreactor", "output_dir": str(network), "params": {}, "level": None}
    adopt_nanoreactor(ws, job)
    snap = ws.snapshot()
    rx = {r["label"]: r for r in snap["reactions"].values()}
    direct, shuttled = rx["CC=O -> C=CO"], rx["CC=O + O -> C=CO + O"]
    jobs = {"t1": {"id": "t1", "targets": {"edges": [direct["edge"]]}, "status": "done",
                   "summary": {"barrier_kcal": 40.0, "barrier_verified": True}}}
    sid = {s["smiles"]: s["id"] for s in snap["structures"].values() if s["role"] == "minimum"}
    # Short enough to be under kinetic control (at equilibrium no barrier would matter, X = 0).
    out = analyze(snap, jobs, None, initial={sid["CC=O"]: 1.0, sid["O"]: 1.0}, held=[], temperature=500,
                  time_s=1e-10, target=sid["C=CO"])
    # the shuttled route has its barrier from the run itself (31 kcal/mol, on its edge)
    labels = [s["label"] for s in out["steps"]]
    assert sorted(labels) == ["CC=O + O -> C=CO + O", "CC=O -> C=CO"]
    assert out["target"]["final"] > 0
    ctl = dict(zip(labels, out["control"]["steps"]))
    assert ctl["CC=O + O -> C=CO + O"] > 0.9 > ctl["CC=O -> C=CO"]       # the lower route controls it
    out2 = analyze(snap, {}, None, initial={sid["CC=O"]: 1.0}, held=[], temperature=500, time_s=1.0, target=None)
    assert [s["label"] for s in out2["steps"]] == ["CC=O + O -> C=CO + O"]  # without the TS job: only the run's own
    assert any(e["label"] == "CC=O -> C=CO" and e["reason"] == "no barrier yet" for e in out2["excluded"])


def test_reactor_view_shows_packing_and_relaxation_before_the_md(tmp_path):
    from mepd.web.nanoreactor import reactor_view

    md = tmp_path / "md"
    md.mkdir()
    frame = lambda x, e=None: f"2\n{'' if e is None else f' energy: {e} gnorm: 0.1'}\nO 0 0 0\nO {x} 0 0\n"
    (md / "schedule.json").write_text(json.dumps({"dump_fs": 2.0, "time_ps": 1.0, "segments": [[0.5, 6.0], [0.5, 4.5]]}))
    assert reactor_view(tmp_path)["prep"] is None                     # not packed yet
    (md / "packed.xyz").write_text(frame(1.0))
    prep = reactor_view(tmp_path)["prep"]
    assert prep["stage"] == "packed" and prep["radius"] == 6.0 and len(prep["frames"]) == 1
    # xtb writing its optimization: complete frames only, energies relative to the first step.
    (md / "xtbopt.log").write_text(frame(1.1, -10.0) + frame(1.2, -10.01) + "2\n energy")
    prep = reactor_view(tmp_path)["prep"]
    assert prep["stage"] == "relaxing" and prep["steps"] == 2 and prep["frames"][-1][3] == 1.2
    assert prep["energy_kcal"] == [None, 0.0, pytest.approx(-6.3, abs=0.1)]
    (md / "reactor.xyz").write_text(frame(1.2))
    assert reactor_view(tmp_path)["prep"]["stage"] == "relaxed"
    # The first MD segment running, none finished yet: its frames show right away.
    (md / "xtb.trj").write_text(frame(1.3) + frame(1.4) + "2\n")
    view = reactor_view(tmp_path)
    assert view["prep"] is None and view["n_frames"] == 2 and view["symbols"] == ["O", "O"]
    (md / "xtb.trj").unlink()                                          # the segment ends: moved to segment_000.xyz
    (md / "segment_000.xyz").write_text(frame(1.3) + frame(1.4))
    assert reactor_view(tmp_path)["n_frames"] == 2


def test_engine_relaxation_writes_its_steps(tmp_path):
    from mepd.discovery import nanoreactor as nr

    class Spring:   # two atoms on a spring, wall far away
        def __call__(self, pos, radius):
            d = pos[1] - pos[0]
            r = np.linalg.norm(d)
            g = (r - 2.0) * d / r
            return 0.5 * (r - 2.0) ** 2, np.array([-g, g])

    pos = nr._relax_in_wall(Spring(), np.array([[0.0, 0, 0], [3.0, 0, 0]]), 50.0, np.array([16.0, 16.0]),
                            trajectory=tmp_path / "relax.xyz", symbols=["O", "O"])
    assert abs(np.linalg.norm(pos[1] - pos[0]) - 2.0) < 0.05
    syms, frames, comments = nr.read_xyz_frames(tmp_path / "relax.xyz")
    assert syms == ["O", "O"] and len(frames) > 1 and "energy:" in comments[0]
