"""Filtered GI path (--atom-mapping-metric rmsd-geodesic): conformer pairs by
endpoint RMSD first, GI paths only near each mechanism's lowest RMSD."""
from __future__ import annotations

import numpy as np

from mepd.atom_mapping_metrics import METRICS, OFFERED
from mepd.atom_mapping_selection import near_lowest


def test_three_offered_all_accepted():
    assert OFFERED == ("endpoint-rmsd", "geodesic-distance", "rmsd-geodesic")
    assert set(OFFERED) <= set(METRICS) and {"path-rmsd", "gi-energy"} <= set(METRICS)   # older profiles still run


def test_window_is_lowest_plus_k_sigma():
    v = [1.0, 1.1, 1.2, 3.0, 5.0]
    sd = float(np.std(v))
    assert near_lowest(v, 1.0) == [i for i in np.argsort(v) if v[i] <= 1.0 + sd]
    assert near_lowest(v, 0.0) == [0]
    assert near_lowest(v, 0.0, keep=3) == [0, 1, 2]            # the next-lowest fill up to `keep`
    assert near_lowest([2.0, 2.0, 2.0], 1.0) == [0, 1, 2]       # all tied: all kept
    assert near_lowest(v, 10.0) == [0, 1, 2, 3, 4]


def test_second_stage_scores_only_the_window_and_records_rmsd():
    from mepd.cli_channels import _gi_near_lowest_rmsd

    # One mechanism "m": pair (0, j) with endpoint RMSD j; one unmapped pair.
    rows = [{"i": 0, "j": j, "key": "m", "score": float(j), "n_variants": 2, "structure": None} for j in range(1, 9)]
    rows.append({"i": 1, "j": 9, "key": "unmapped", "score": None, "n_variants": 0, "structure": None})
    asked = []

    def score_pair(pair, metric, only_keys):
        asked.append((pair, metric, frozenset(only_keys)))
        return pair, [("m", 100.0 - pair[1], 4, None, True)], None      # GI prefers higher j here

    from mepd.atom_mapping_selection import near_lowest as nl

    out = _gi_near_lowest_rmsd(rows, ["m", "unmapped"], score_pair, 1, 1.0, 3, nl)
    within = [r for r in rows[:-1] if r["score"] <= 1.0 + np.std([r["score"] for r in rows[:-1]])]
    scored = [r for r in out if r["key"] == "m"]
    assert len(asked) == max(3, len(within)) and all(m == "rmsd-geodesic" for _, m, _ in asked)
    assert {r["j"] for r in scored} == {p[1] for p, _, _ in asked}      # the rest are dropped
    assert all(r["rmsd"] == r["j"] and r["score"] == 100.0 - r["j"] for r in scored)
    assert any(r["key"] == "unmapped" for r in out)                    # unmapped pairs are kept as they were


def test_gi_paths_only_for_the_lowest_rmsd_symmetry_variants(monkeypatch):
    import mepd.atom_mapping_selection as S
    from qcdata import Structure

    start = Structure(symbols=["O", "H", "H"], geometry=np.array([[0, 0, 0], [1.8, 0, 0], [-0.5, 1.7, 0]]),
                      charge=0, multiplicity=1)
    # 30 "variants": one H moved 0.02 * k bohr further out, so the aligned RMSD rises with k.
    def moved(k):
        g = np.asarray(start.geometry, dtype=float).copy()
        g[1, 0] += 0.02 * k
        return start.model_copy(update={"geometry": g})
    cands = [S.MappingCandidate(label=f"v{k}", end_structure=moved(k), atom_map=None) for k in range(30)]
    class RI:
        class atom_mapping_inputs:
            gi_variant_cap = 20

    kept = S.lowest_rmsd_variants(cands, start, RI())
    assert {c.label for c in kept} == {f"v{k}" for k in range(20)}
    assert "v29" in {c.label for c in S.lowest_rmsd_variants(cands, start, RI(), always=("v29",))}

    scored = []
    monkeypatch.setattr(S, "score_candidate", lambda c, m, s, r: (scored.append(c.label) or 1.0, None))
    S.select_best_candidate(cands, "rmsd-geodesic", start, RI())
    assert len(scored) == 20 and "v0" in scored          # the current numbering (first) is always scored
    scored.clear()
    S.select_best_candidate(cands, "geodesic-distance", start, RI())
    assert len(scored) == 30                              # the plain GI path is never capped
