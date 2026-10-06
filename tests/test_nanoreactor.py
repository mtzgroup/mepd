"""Nanoreactor: event detection, subsystem extraction and the reaction
network, on hand-built bond histories (no MD needed)."""
import numpy as np
import pytest

from mepd.discovery import nanoreactor as nr

# Two acetaldehydes, a shuttle water and a spectator water.
#   CH3-CHO #1: C0 C1 O2 H3 H4 H5 (on C0) H6 (on C1)
#   CH3-CHO #2: C7 C8 O9 H10 H11 H12 (on C7) H13 (on C8)
#   water (shuttle): O14 H15 H16;  water (spectator): O17 H18 H19
SYMBOLS = ["C", "C", "O", "H", "H", "H", "H"] * 2 + ["O", "H", "H"] * 2
ACETALDEHYDE = [(0, 1), (1, 2), (0, 3), (0, 4), (0, 5), (1, 6)]
INITIAL = ({tuple(b) for b in ACETALDEHYDE} | {(i + 7, j + 7) for i, j in ACETALDEHYDE}
           | {(14, 15), (14, 16), (17, 18), (17, 19)})
DT = 2.0


def _sides(label):
    left, right = label.split(" -> ")
    return set(left.split(" + ")), set(right.split(" + "))


def _history(changes, n_frames=600):
    return nr.BondHistory(n_frames, set(INITIAL), sorted(changes))


def test_denoise_drops_short_blips_and_keeps_real_changes():
    s = [False] * 10 + [True] * 2 + [False] * 10 + [True] * 30
    clean = nr._denoise(s, 5)
    assert clean == [False] * 22 + [True] * 30
    # a short state at the very end is not a new state yet
    assert nr._denoise([True] * 20 + [False] * 3, 5) == [True] * 23


def test_direct_and_shuttled_tautomerization_are_two_reactions():
    changes = [
        (100, 0, 3, False), (100, 2, 3, True),                           # direct: H3 from C0 to O2
        (300, 7, 10, False), (302, 10, 14, True),                        # H10 to the water O14 ...
        (320, 14, 15, False), (321, 9, 15, True),                        # ... and H15 on to O9
    ]
    hist = _history(changes)
    det = nr.DetectSettings()
    events = nr.detect_events(len(SYMBOLS), hist, det, DT)
    assert len(events) == 2
    direct, shuttled = events
    assert direct.atoms == tuple(range(7))                               # only the molecule that reacts
    assert shuttled.atoms == tuple(range(7, 17))                         # acetaldehyde + the shuttle water
    assert not set(range(17, 20)) & set(shuttled.atoms)                  # the spectator is left out
    assert shuttled.reactant_frame < 300 and shuttled.product_frame > 321

    frames = np.zeros((hist.n_frames, len(SYMBOLS), 3))
    species, reactions, records = nr.build_network(SYMBOLS, frames, hist, events, nr.Labeler(SYMBOLS),
                                                   0, DT)
    names = {s.id: s.smiles for s in species}
    assert sorted(names.values()) == sorted(["CC=O", "O", "C=CO"])
    by_label = {r.label: r for r in reactions}
    assert set(by_label) == {"CC=O -> C=CO", "CC=O + O -> C=CO + O"}
    shuttle = by_label["CC=O + O -> C=CO + O"]
    assert [names[i] for i in shuttle.shuttles] == ["O"]
    assert by_label["CC=O -> C=CO"].shuttles == []
    water = next(s for s in species if s.smiles == "O")
    assert water.initial == 2


def test_collision_partner_and_concurrent_reaction_are_split_off():
    # The direct tautomerization of #1 while its H3 briefly touches the
    # spectator water (O17), and #2 loses H13 at the same time after also
    # touching that water: two reactions, the water in neither.
    changes = [(100, 0, 3, False), (101, 3, 17, True), (110, 3, 17, False), (111, 2, 3, True),
               (105, 13, 17, True), (106, 8, 13, False), (120, 13, 17, False)]
    hist = _history(changes)
    events = nr.detect_events(len(SYMBOLS), hist, nr.DetectSettings(), DT)
    assert [e.atoms for e in events] == [tuple(range(7)), tuple(range(7, 14))]
    _, reactions, _ = nr.build_network(SYMBOLS, np.zeros((600, 20, 3)), hist, events, nr.Labeler(SYMBOLS), 0, DT)
    assert [r.shuttles for r in reactions] == [[], []]


def test_reverse_event_counts_on_the_same_reaction():
    changes = [(100, 0, 3, False), (100, 2, 3, True),                    # keto -> enol
               (400, 2, 3, False), (400, 0, 3, True)]                    # and back
    hist = _history(changes)
    events = nr.detect_events(len(SYMBOLS), hist, nr.DetectSettings(), DT)
    _, reactions, _ = nr.build_network(SYMBOLS, np.zeros((600, 20, 3)), hist, events, nr.Labeler(SYMBOLS), 0, DT)
    assert len(reactions) == 1
    assert (reactions[0].count, reactions[0].reverse_count) == (1, 1)


def test_collision_that_undoes_itself_is_not_a_reaction():
    # A water H binds to the carbonyl O and leaves again within the merge window.
    changes = [(200, 2, 18, True), (230, 2, 18, False)]
    events = nr.detect_events(len(SYMBOLS), _history(changes), nr.DetectSettings(), DT)
    assert events == []


def test_degenerate_exchange_is_recorded_but_is_no_reaction():
    # H swaps between the two waters: water + water -> water + water.
    changes = [(200, 14, 15, False), (201, 17, 15, True), (210, 17, 18, False), (211, 14, 18, True)]
    hist = _history(changes)
    events = nr.detect_events(len(SYMBOLS), hist, nr.DetectSettings(), DT)
    assert len(events) == 1
    _, reactions, records = nr.build_network(SYMBOLS, np.zeros((600, 20, 3)), hist, events,
                                             nr.Labeler(SYMBOLS), 0, DT)
    assert reactions == [] and records[0]["reaction"] is None


def test_partial_charges_decide_ions():
    # O14-H15 breaks: H atom + OH radical, unless the partial charges say
    # it left as a proton.
    changes = [(200, 14, 15, False)]
    hist = _history(changes)
    events = nr.detect_events(len(SYMBOLS), hist, nr.DetectSettings(), DT)
    frames = np.zeros((600, 20, 3))
    _, rx, _ = nr.build_network(SYMBOLS, frames, hist, events, nr.Labeler(SYMBOLS), 0, DT)
    assert _sides(rx[0].label) == ({"O"}, {"[H]", "[OH]"})
    q = np.zeros(20)
    q[15], q[[14, 16]] = 0.9, -0.45
    _, rx, _ = nr.build_network(SYMBOLS, frames, hist, events, nr.Labeler(SYMBOLS), 0, DT,
                                charges_at=lambda f: q if f > 200 else np.zeros(20))
    assert _sides(rx[0].label) == ({"O"}, {"[H+]", "[OH-]"})


def test_no_neutral_lewis_structure_gives_an_ion_pair():
    # H15 from one water to the other: neutral H3O has no Lewis structure.
    changes = [(200, 14, 15, False), (200, 17, 15, True)]
    hist = _history(changes)
    events = nr.detect_events(len(SYMBOLS), hist, nr.DetectSettings(), DT)
    _, rx, _ = nr.build_network(SYMBOLS, np.zeros((600, 20, 3)), hist, events, nr.Labeler(SYMBOLS), 0, DT)
    assert rx[0].label == "2 O -> [OH-] + [OH3+]"


def test_piston_schedule_closes_in_steps():
    s = nr.ReactorSettings(radius=10.0, compress=0.5, period_ps=1.0, duty=0.75, time_ps=2.0, ramp_fs=100.0)
    sched = nr.piston_schedule(s)
    assert sum(d for d, _ in sched) == pytest.approx(2.0)
    radii = [r for _, r in sched]
    assert radii[:7] == pytest.approx([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 5.0])


def test_pack_reactor_grows_an_automatic_radius_until_the_molecules_fit():
    """Two copies of a 13-atom molecule (6 C in a row, 5.3 Angstrom long): the
    atom-count radius (5.4 Angstrom) cannot hold both, so it grows; a radius
    given by hand is kept and refused instead."""
    from qcdata import Structure

    chain = [[1.3 * k - 3.25, 0.0, 0.0] for k in range(6)]
    geom = chain + [[x, 1.0, 0.0] for x, _, _ in chain[:5]] + [[-3.25, -1.2, 0.0], [3.25, -1.2, 0.0]]
    mol = Structure(symbols=["C"] * 6 + ["H"] * 5 + ["O"] * 2, geometry=np.array(geom) * nr.ANGSTROM_TO_BOHR)
    symbols, xyz, radius, owner = nr.pack_reactor([mol, mol])
    assert len(symbols) == 26 and radius > nr.auto_radius(26)
    assert np.max(np.linalg.norm(xyz, axis=1)) < radius
    with pytest.raises(RuntimeError, match="larger radius"):
        nr.pack_reactor([mol, mol], nr.auto_radius(26))


def test_pack_reactor_keeps_molecules_apart_and_inside():
    from qcdata import Structure

    water = Structure(symbols=["O", "H", "H"],
                      geometry=np.array([[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]]) * nr.ANGSTROM_TO_BOHR)
    symbols, xyz, radius, owner = nr.pack_reactor([water] * 8, seed=3)
    assert len(symbols) == 24 and radius == pytest.approx(nr.auto_radius(24))
    assert np.max(np.linalg.norm(xyz, axis=1)) < radius
    for a in range(8):
        for b in range(a + 1, 8):
            ia = [k for k, o in enumerate(owner) if o == a]
            ib = [k for k, o in enumerate(owner) if o == b]
            assert np.min(np.linalg.norm(xyz[ia][:, None] - xyz[ib][None], axis=2)) >= 2.0


def test_why_a_reaction_has_no_ts_endpoints():
    # H2 -> 2 H: the product side (two H atoms) bonds back when optimized.
    two_h, h2 = set(), {(0, 1)}
    v = nr._verdict("product", h2, two_h, h2, 2)
    assert v["code"] == "reverts" and not v["ts"]
    v = nr._verdict("reactant", two_h, h2, two_h, 2)
    assert v["code"] == "barrierless"
    # Two radicals (atoms 0-1 and 2-3) that bond: recombines; a molecule that splits: falls apart.
    assert nr._verdict("product", {(0, 1), (2, 3), (1, 2)}, {(0, 1), (2, 3)}, {(0, 2)}, 4)["code"] == "recombines"
    assert nr._verdict("reactant", {(0, 1)}, {(0, 1), (1, 2)}, {(0, 2)}, 3)["code"] == "falls_apart"
    assert nr._verdict("reactant", {(0, 2), (1, 2)}, {(0, 1), (1, 2)}, {(0, 1)}, 3)["code"] == "rearranges"


def test_wall_gradient_matches_its_energy():
    rng = np.random.default_rng(0)
    pos = rng.normal(size=(5, 3)) * 6.0
    e0, g = nr._wall(pos, 5.0, 0.02)
    h = 1e-5
    for a in range(5):
        for k in range(3):
            p = pos.copy(); p[a, k] += h
            assert (nr._wall(p, 5.0, 0.02)[0] - e0) / h == pytest.approx(g[a, k], abs=1e-6)


class _MorseEngine:
    """A toy engine: Morse bonds between every pair of H atoms (all H2 molecules)."""

    def compute_gradients(self, nodes):
        out = []
        for node in nodes:
            x = np.asarray(node.structure.geometry).reshape(-1, 3)
            d = x[:, None] - x[None]
            r = np.linalg.norm(d, axis=2) + np.eye(len(x))
            De, a, re = 0.17, 1.0, 1.4
            ex = np.exp(-a * (r - re))
            e = De * (1 - ex) ** 2 - De
            np.fill_diagonal(e, 0.0)
            dedr = 2 * De * a * ex * (1 - ex)
            np.fill_diagonal(dedr, 0.0)
            g = (dedr / r)[:, :, None] * d
            node._cached_energy = float(e.sum() / 2)
            out.append(g.sum(axis=1))
        return np.array(out)


def test_engine_md_runs_on_any_engine_and_writes_the_usual_files(tmp_path):
    rng = np.random.default_rng(1)
    symbols = ["H"] * 8
    coords = np.vstack([np.array([[0, 0, 0], [0.74, 0, 0]]) + rng.uniform(-2.5, 2.5, 3) for _ in range(4)])
    s = nr.ReactorSettings(temperature=1000.0, time_ps=0.2, radius=5.0, period_ps=0.1, duty=0.5, ramp_fs=20.0,
                           method="level", dump_fs=2.0, step_fs=0.5)
    seen = []
    traj = nr.run_engine_md(symbols, coords, charge=0, multiplicity=1, settings=s, engine=_MorseEngine(),
                            workdir=tmp_path / "md", on_event=lambda e, p: seen.append(p))
    syms, frames, _ = nr.read_xyz_frames(traj)
    assert syms == symbols and len(frames) == 100
    assert np.max(np.linalg.norm(frames, axis=2)) < 5.0 + 1.5          # the wall holds
    temps = [p["temperature"] for p in seen if "temperature" in p]
    assert 300 < np.mean(temps) < 2500                                   # thermostatted near 1000 K
    assert len(list((tmp_path / "md").glob("segment_*.xyz"))) == len(nr.piston_schedule(s))
    # resume: nothing is rerun
    seen.clear()
    nr.run_engine_md(symbols, coords, charge=0, multiplicity=1, settings=s, engine=_MorseEngine(),
                     workdir=tmp_path / "md", on_event=lambda e, p: seen.append(p))
    assert not [p for p in seen if "temperature" in p]


def test_automatic_md_choice_falls_back_to_gxtb_then_to_the_profile(monkeypatch):
    monkeypatch.setattr(nr, "missing_programs", lambda m="gfn2": ["xtb"] if m != "level" else [])
    monkeypatch.delenv("GXTB_EXECUTABLE", raising=False)
    monkeypatch.setattr(nr.shutil, "which", lambda name: None)
    monkeypatch.setattr(nr.Path, "home", lambda: nr.Path("/nonexistent"))

    class GXTBCalculator:   # a profile's g-xTB engine, known by name
        executable = __file__   # an existing path stands in for the program
    assert nr.pick_md_method(GXTBCalculator()) == ("gxtb", __file__)
    assert nr.pick_md_method(object()) == ("level", None)


def test_open_shell_fragments_get_a_structure_not_a_code():
    from rdkit import Chem
    from rdkit.Chem import AllChem

    m = Chem.AddHs(Chem.MolFromSmiles("Cn1cnc2c1c(=O)n(C)c(=O)n2C"))   # caffeine
    AllChem.EmbedMolecule(m, randomSeed=1)
    syms = [a.GetSymbol() for a in m.GetAtoms()]
    bonds = nr.perceive_bonds(syms, m.GetConformer().GetPositions())
    lab = nr.Labeler(syms)
    assert lab.candidates(tuple(range(len(syms))), bonds)[0][2] == "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
    ring = next(r for r in m.GetRingInfo().AtomRings() if len(r) == 5)
    cn = next((i, j) for i, j in bonds if i in ring and j in ring and {syms[i], syms[j]} == {"C", "N"}
              and any(n.GetSymbol() == "H" for n in m.GetAtomWithIdx(i if syms[i] == "C" else j).GetNeighbors()))
    opened = nr.Labeler(syms).candidates(tuple(range(len(syms))), bonds - {cn})
    assert opened and not opened[0][2].startswith("?") and "unusual" not in opened[0][2]
    assert Chem.MolFromSmiles(opened[0][2]) is not None
    assert nr.Labeler(["C", "H", "H"]).candidates((0, 1, 2), {(0, 1), (0, 2)})[0][2] == "[CH2]"   # a carbene


def test_an_event_keeps_only_bond_changes_inside_its_own_atoms():
    # Two concurrent reactions, linked for a moment by a contact between them
    # (C1-H13 forms and breaks): each event's changes stay within its atoms.
    changes = [(100, 0, 3, False), (100, 2, 3, True),        # #1: direct tautomerization
               (102, 1, 13, True), (112, 1, 13, False),     # contact with #2
               (105, 8, 13, False), (130, 9, 13, True)]     # #2: its own H shift
    hist = _history(changes)
    for ev in nr.detect_events(len(SYMBOLS), hist, nr.DetectSettings(), DT):
        assert all(i in ev.atoms and j in ev.atoms for _, i, j, _ in ev.changes)


def test_an_extended_run_reuses_what_it_refined_before(tmp_path):
    """Run longer: species, reaction complexes and TSs from the earlier
    network.json are reused by identity; their files are kept under names of
    their own (the new analysis renumbers and would overwrite sp_0 etc.)."""
    import json
    from types import SimpleNamespace as NS

    for name in ("sp_0.xyz", "sp_1.xyz", "rc.xyz", "pc.xyz"):
        (tmp_path / name).write_text(f"1\n{name}\nH 0 0 0\n")
    net = {"species": [
        {"id": 0, "smiles": "CC=O", "charge": 0, "multiplicity": 1, "energy": -1.0, "file": str(tmp_path / "sp_0.xyz")},
        {"id": 1, "smiles": "C=CO", "charge": 0, "multiplicity": 1, "energy": -0.9, "file": str(tmp_path / "sp_1.xyz")},
        {"id": 2, "smiles": "O", "charge": 0, "multiplicity": 1, "energy": None, "note": ""}],     # never refined
        "reactions": [{"id": 0, "reactants": [0], "products": [1],
                       "complex": {"reactant": str(tmp_path / "rc.xyz"), "product": str(tmp_path / "pc.xyz")},
                       "ts": {"barrier_kcal": 60.0}}]}
    (tmp_path / "network.json").write_text(json.dumps(net))
    prev = nr.PreviousRefinement(tmp_path)
    sp, rx = prev.results()
    assert set(sp) == {"CC=O|0|1", "C=CO|0|1"}
    kept = sp["CC=O|0|1"].file
    assert kept != str(tmp_path / "sp_0.xyz") and "previous" in kept
    (tmp_path / "sp_0.xyz").write_text("1\noverwritten\nH 0 0 0\n")      # the new analysis reusing the name
    assert "sp_0.xyz" in open(kept).read().splitlines()[1]
    (rxn,) = rx.values()
    assert "previous" in rxn.complex["reactant"]
    # the same reaction, renumbered: its TS is reused...
    by_id = {5: NS(smiles="CC=O", charge=0, multiplicity=1), 7: NS(smiles="C=CO", charge=0, multiplicity=1)}
    assert prev.ts(NS(reactants=[5], products=[7]), by_id) == {"barrier_kcal": 60.0}
    # ...but not when written the other way round (its barrier is from the other side)
    assert prev.ts(NS(reactants=[7], products=[5]), by_id) is None


def test_ends_that_change_bonds_on_optimization_but_still_differ_are_kept(tmp_path):
    """refine: when no instance keeps its bonds, the first whose optimized
    ends still differ becomes the reaction's ends, relabelled to what they
    are; ends that optimize into the same molecules give no reaction."""
    def end(name, rows, energy):
        fp = tmp_path / name
        fp.write_text(f"{len(rows)}\nenergy={energy} charge=0 mult=1\n" + "".join(f"{s} {x} {y} {z}\n" for s, x, y, z in rows))
        return {"file": str(fp), "energy": energy, "bonds_kept": False}

    ch4 = [("C", 0, 0, 0), ("H", 0.63, 0.63, 0.63), ("H", -0.63, -0.63, 0.63), ("H", -0.63, 0.63, -0.63),
           ("H", 0.63, -0.63, -0.63)]
    # product: two H's left C and bonded to each other (as two H radicals recombining would)
    ch2_h2 = ch4[:3] + [("H", -3.0, 3.0, 0.0), ("H", -3.0, 3.74, 0.0)]
    inst = {"complex": {"reactant": end("r.xyz", ch4, -4.0), "product": end("p.xyz", ch2_h2, -3.9)}}
    c = nr.relaxed_complex([inst])
    assert c is not None and c["reactant"].endswith("r.xyz")
    assert sorted(c["relaxed"]["products"]) == sorted(["[CH2]", "[H][H]"]) and c["relaxed"]["reactants"] == ["C"]
    assert c["relaxed"]["label"].startswith("C -> ") and abs(c["delta_e_kcal"] - 0.1 * 627.509) < 0.1
    # both ends optimized into methane: no reaction
    same = {"complex": {"reactant": end("r2.xyz", ch4, -4.0), "product": end("p2.xyz", ch4, -4.0)}}
    assert nr.relaxed_complex([same]) is None
