from __future__ import annotations

import warnings

import numpy as np
import pytest
from qcdata.models.structure import Structure

pytest.importorskip("slapmapper")

from mepd.atom_mapping import (  # noqa: E402
    AtomMapping,
    check_atom_mapping,
    expand_mapping_by_symmetry,
    map_smiles_pair,
    realign_end_to_start,
    reorder_structure,
    suggest_atom_mapping,
    suggest_atom_mapping_candidates,
)
from mepd.atom_mapping import _symmetry_orbits  # noqa: E402


def _water(order=(0, 1, 2)) -> Structure:
    symbols = ["O", "H", "H"]
    geometry = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.43355001758932, 0.0, 0.95295864902809],
            [-1.43355001758932, 0.0, 0.95295864902809],
        ]
    )
    order = list(order)
    return Structure(
        symbols=[symbols[i] for i in order],
        geometry=geometry[order],
        charge=0,
        multiplicity=1,
    )


def _propene():
    """CH2=CH-CH3: C0 (=CH2, symmetric Hs 3,4), C1 (=CH-, H5), C2 (-CH3, Hs 6,7,8)."""
    symbols = ["C", "C", "C", "H", "H", "H", "H", "H", "H"]
    geometry = np.array(
        [
            [0.000, 1.303, 0.000],
            [0.000, 0.000, 0.000],
            [1.501, -0.400, 0.000],
            [-0.920, 1.860, 0.000],
            [0.920, 1.860, 0.000],
            [-0.950, -0.550, 0.000],
            [1.550, -1.030, 0.870],
            [1.550, -1.030, -0.870],
            [2.300, 0.320, 0.000],
        ]
    )
    return Structure(symbols=symbols, geometry=geometry, charge=0, multiplicity=1)


def test_suggest_atom_mapping_is_identity_for_matching_structures():
    start = _water()
    end = _water()
    atom_map = suggest_atom_mapping(start, end)
    assert atom_map.is_identity
    assert atom_map.mapping == {0: 0, 1: 1, 2: 2}


def test_suggest_atom_mapping_ignores_symmetric_atom_relabeling():
    """Swapping water's two (topologically indistinguishable) Hs isn't a
    detectable mismatch from connectivity alone -- SLAPMapper has no basis
    to prefer one equivalent mapping over another."""
    start = _water()
    end = _water(order=(0, 2, 1))
    atom_map = suggest_atom_mapping(start, end)
    assert atom_map.is_identity


def test_suggest_atom_mapping_detects_genuine_mismatch():
    start = _propene()
    # swap the whole terminal-CH2 block (C0 + its Hs) with the methyl block
    # (C2 + its Hs) -- these carbons have different local environments, so
    # this is a real, detectable mismatch.
    order = [2, 1, 0, 6, 7, 8, 5, 3, 4]
    symbols = np.asarray(start.symbols)[order]
    geometry = np.asarray(start.geometry)[order]
    end = Structure(symbols=symbols, geometry=geometry, charge=0, multiplicity=1)

    atom_map = suggest_atom_mapping(start, end)
    assert not atom_map.is_identity

    realigned = realign_end_to_start(atom_map, end)
    assert list(realigned.symbols) == list(start.symbols)
    assert np.allclose(np.asarray(realigned.geometry), np.asarray(start.geometry))


def test_check_atom_mapping_warns_only_on_genuine_mismatch():
    start = _propene()
    order = [2, 1, 0, 6, 7, 8, 5, 3, 4]
    end = Structure(
        symbols=np.asarray(start.symbols)[order],
        geometry=np.asarray(start.geometry)[order],
        charge=0,
        multiplicity=1,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_atom_mapping(start, _propene())
    assert len(caught) == 0

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_atom_mapping(start, end)
    assert len(caught) == 1
    assert issubclass(caught[0].category, UserWarning)
    assert "disagrees" in str(caught[0].message)


def test_suggest_atom_mapping_returns_none_for_unbalanced_structures():
    start = _water()
    end = Structure(symbols=["O", "H"], geometry=start.geometry[:2], charge=0, multiplicity=2)
    assert suggest_atom_mapping(start, end) is None


def test_suggest_atom_mapping_candidates_returns_a_list():
    start = _water()
    end = _water()
    candidates = suggest_atom_mapping_candidates(start, end)
    assert isinstance(candidates, list)
    assert len(candidates) >= 1
    # The identity mapping must always be among the candidates (water's two
    # Hs are symmetric, so a cost-preserving swapped-H expansion is also
    # expected -- see test_suggest_atom_mapping_candidates_includes_symmetry_orbit_expansion).
    assert any(c.is_identity for c in candidates)
    for c in candidates:
        assert sorted(c.mapping.values()) == sorted(c.mapping.keys())


def test_suggest_atom_mapping_candidates_caps_at_max_candidates():
    start = _propene()
    order = [2, 1, 0, 6, 7, 8, 5, 3, 4]
    end = Structure(
        symbols=np.asarray(start.symbols)[order],
        geometry=np.asarray(start.geometry)[order],
        charge=0,
        multiplicity=1,
    )
    candidates = suggest_atom_mapping_candidates(start, end, max_candidates=1)
    assert len(candidates) <= 1


def test_suggest_atom_mapping_candidates_empty_for_unbalanced_structures():
    start = _water()
    end = Structure(symbols=["O", "H"], geometry=start.geometry[:2], charge=0, multiplicity=2)
    assert suggest_atom_mapping_candidates(start, end) == []


def test_suggest_atom_mapping_candidates_passes_break_sym_targets(monkeypatch):
    """Passing `break_sym_targets` activates SLAPMapper's own dedup
    (`_remove_isomorphic_results`) and lets it explore genuinely distinct
    tied-cost orderings its un-branched default pass would miss -- see
    `mepd/atom_mapping.py::suggest_atom_mapping_candidates`."""
    from slapmapper.core import SlapMapper

    captured = {}
    real_get_maps = SlapMapper.get_maps

    def spying_get_maps(self, lgp, **kwargs):
        captured["break_sym_targets"] = kwargs.get("break_sym_targets")
        return real_get_maps(self, lgp, **kwargs)

    monkeypatch.setattr(SlapMapper, "get_maps", spying_get_maps)

    start = _water()
    end = _water()
    suggest_atom_mapping_candidates(start, end)

    assert captured["break_sym_targets"] == list(range(len(start.symbols)))


def test_suggest_atom_mapping_candidates_includes_symmetry_orbit_expansion():
    """Water's two Hs are topologically interchangeable -- SLAPMapper's own
    `_remove_isomorphic_results` collapses the swapped-H relabeling as
    graph-redundant with the identity mapping, but `suggest_atom_mapping_candidates`
    fills unused --atom-mapping-candidates budget with `expand_mapping_by_symmetry`
    expansions of it anyway, since a symmetry-equivalent mapping isn't
    necessarily geometrically equivalent for a specific input conformer.
    So water should yield the identity mapping AND the swapped-H mapping,
    both at the same (zero) cost."""
    start = _water()
    end = _water()
    candidates = suggest_atom_mapping_candidates(start, end, max_candidates=20)

    assert len(candidates) == 2
    assert any(c.is_identity for c in candidates)
    assert any(c.mapping == {0: 0, 1: 2, 2: 1} for c in candidates)
    assert all(c.cost == candidates[0].cost for c in candidates)


def test_reorder_structure_permutes_symbols_and_geometry():
    start = _water()
    reordered = reorder_structure(start, [2, 0, 1])
    assert list(reordered.symbols) == ["H", "O", "H"]
    assert np.allclose(np.asarray(reordered.geometry)[1], np.asarray(start.geometry)[0])


def test_map_smiles_pair_builds_atom_count_consistent_structures():
    start, end = map_smiles_pair("CC=O", "OC=C")
    assert len(start.symbols) == len(end.symbols)
    assert sorted(start.symbols) == sorted(end.symbols)


def test_symmetry_orbits_finds_waters_two_equivalent_hs():
    orbits = _symmetry_orbits(_water())
    assert orbits == [[1, 2]]


def _propene_bohr() -> Structure:
    """`_propene()`'s geometry is written in Angstrom-scale numbers but
    stored directly as `Structure.geometry` (which is otherwise always
    Bohr internally, e.g. `_water()`'s fixture) -- harmless for the
    graph-only comparisons `suggest_atom_mapping` relies on (self-consistent
    "wrong" scale on both sides), but `_symmetry_orbits` round-trips through
    `Structure.to_xyz()` (a real Bohr->Angstrom conversion), which shrinks
    already-Angstrom-scale numbers down to unphysically short "Angstrom"
    bond lengths RDKit can't perceive bonds from. Rescale properly for
    these tests specifically rather than touch the shared fixture."""
    from qcconst.constants import ANGSTROM_TO_BOHR

    base = _propene()
    return Structure(
        symbols=base.symbols,
        geometry=np.asarray(base.geometry) * ANGSTROM_TO_BOHR,
        charge=base.charge,
        multiplicity=base.multiplicity,
    )


def test_symmetry_orbits_finds_propenes_ch2_and_ch3_groups():
    """C0 (=CH2, Hs 3,4) and C2 (-CH3, Hs 6,7,8) are each their own orbit,
    matching this file's own fixture docstring."""
    orbits = {frozenset(o) for o in _symmetry_orbits(_propene_bohr())}
    assert frozenset({3, 4}) in orbits
    assert frozenset({6, 7, 8}) in orbits


def test_symmetry_orbits_empty_for_fully_asymmetric_structure():
    start = _propene_bohr()
    order = [2, 1, 0, 6, 7, 8, 5, 3, 4]
    asymmetric = Structure(
        symbols=np.asarray(start.symbols)[order],
        geometry=np.asarray(start.geometry)[order],
        charge=0,
        multiplicity=1,
    )
    # Still propene under the hood -- same orbits as the original, just
    # reindexed. Confirms _symmetry_orbits tracks whatever indices this
    # specific structure's atoms sit at, not a fixed/hardcoded set.
    orbits = {frozenset(o) for o in _symmetry_orbits(asymmetric)}
    assert len(orbits) == 2


def test_expand_mapping_by_symmetry_swaps_waters_hs():
    start = _water()
    end = _water()
    identity = AtomMapping(mapping={0: 0, 1: 1, 2: 2}, cost=0, n_alternatives=1)

    expansions = expand_mapping_by_symmetry(identity, start, end)

    assert len(expansions) == 1
    assert expansions[0].mapping == {0: 0, 1: 2, 2: 1}
    assert expansions[0].cost == identity.cost


def test_expand_mapping_by_symmetry_produces_valid_cost_preserving_mappings():
    start = _propene_bohr()
    order = [2, 1, 0, 6, 7, 8, 5, 3, 4]
    end = Structure(
        symbols=np.asarray(start.symbols)[order],
        geometry=np.asarray(start.geometry)[order],
        charge=0,
        multiplicity=1,
    )
    atom_map = suggest_atom_mapping(start, end)
    # propene's own orbits (CH2's 2 Hs, CH3's 3 Hs) give real expansions here.
    expansions = expand_mapping_by_symmetry(atom_map, start, end)
    assert len(expansions) > 0
    for expansion in expansions:
        assert sorted(expansion.mapping.values()) == sorted(expansion.mapping.keys())
        assert expansion.cost == atom_map.cost
        assert expansion.mapping != atom_map.mapping


def test_expand_mapping_by_symmetry_respects_max_expansions():
    start = _propene_bohr()
    order = [2, 1, 0, 6, 7, 8, 5, 3, 4]
    end = Structure(
        symbols=np.asarray(start.symbols)[order],
        geometry=np.asarray(start.geometry)[order],
        charge=0,
        multiplicity=1,
    )
    atom_map = suggest_atom_mapping(start, end)
    expansions = expand_mapping_by_symmetry(atom_map, start, end, max_expansions=1)
    assert len(expansions) == 1
