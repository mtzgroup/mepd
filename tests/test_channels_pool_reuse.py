"""Reaction channels reuse an earlier run's conformer pools (`--start-pool` /
`--end-pool`) when the web UI finds one sampled the same molecule with the
same settings at the same level of theory."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from qcdata import Structure

from mepd.chain import Chain
from mepd.inputs import ChainInputs
from mepd.nodes.node import StructureNode

ETHANOL = """9

C -0.9250 0.0000 0.0000
C 0.5750 0.0000 0.0000
O 1.0500 1.3400 0.0000
H -1.2950 1.0300 0.0000
H -1.2950 -0.5150 0.8920
H -1.2950 -0.5150 -0.8920
H 0.9450 -0.5150 0.8920
H 0.9450 -0.5150 -0.8920
H 2.0200 1.3400 0.0000
"""


def _node(xyz: str) -> StructureNode:
    return StructureNode(structure=Structure.from_xyz(xyz))


def _write_pool(fp: Path, xyzs: list[str], energies: list[float]) -> None:
    nodes = [_node(x) for x in xyzs]
    for n, e in zip(nodes, energies):
        n._cached_energy = e
    Chain.model_validate({"nodes": nodes, "parameters": ChainInputs()}).write_to_disk(fp)


def test_saved_pool_is_used_only_when_numbered_like_the_endpoint(tmp_path):
    from mepd.cli_channels import _load_conformer_pool

    # The same molecule with its first two atoms (both C) swapped: same
    # symbols, but the bonds sit on other indices -- it would pair the wrong atoms.
    lines = ETHANOL.splitlines()
    swapped = "\n".join([*lines[:2], lines[3], lines[2], *lines[4:]]) + "\n"
    fp = tmp_path / "start_pool.xyz"
    _write_pool(fp, [ETHANOL, swapped], [-154.1, -154.0])
    kept = _load_conformer_pool(fp, _node(ETHANOL), "start", 0, 1)
    assert len(kept) == 1
    assert kept[0].energy == -154.1           # energies come along: no re-minimization needed


def _channels_job(tmp_path, jid, sids, params, level_key, finished, merged=0, new_style=True):
    out = tmp_path / jid / "output"
    (out / "conformers").mkdir(parents=True)
    for side in ("start", "end"):
        _write_pool(out / "conformers" / (f"{side}_pool.xyz" if new_style else f"{side}.xyz"), [ETHANOL], [-154.1])
    (out / "stats.json").write_text(json.dumps({"conformers": {"start": {"n_mirror_images_merged": merged},
                                                               "end": {"n_mirror_images_merged": 0}}}))
    return {"id": jid, "op": "channels", "status": "done", "targets": {"structures": sids, "edges": []},
            "params": params, "level": {"key": level_key}, "charge": 0, "multiplicity": 1,
            "finished": finished, "output_dir": str(out)}


def test_web_reuses_the_newest_matching_pool_for_each_endpoint(tmp_path):
    from mepd.web.operations import ChannelsParams, _build_channels

    a = {"id": "s_a", "charge": 0, "multiplicity": 1, "name": "A"}
    b = {"id": "s_b", "charge": 0, "multiplicity": 1, "name": "B"}
    c = {"id": "s_c", "charge": 0, "multiplicity": 1, "name": "C"}
    rdkit = {"backend": "rdkit", "n_conformers": 10}
    jobs = {
        # A was the END of an earlier C -> A run: its end pool serves as this run's start pool.
        "j_old": _channels_job(tmp_path, "j_old", ["s_c", "s_a"], rdkit, "L1", finished=1),
        "j_new": _channels_job(tmp_path, "j_new", ["s_c", "s_a"], rdkit, "L1", finished=2),
        # Different sampler setting, other level, or not finished: never reused.
        "j_set": _channels_job(tmp_path, "j_set", ["s_b", "s_c"], {**rdkit, "n_conformers": 3}, "L1", 3),
        "j_lvl": _channels_job(tmp_path, "j_lvl", ["s_b", "s_c"], rdkit, "L2", finished=4),
    }
    jdir = tmp_path / "j_this"
    ctx = SimpleNamespace(structures=[a, b], jobs=jobs, job_dir=jdir, output_dir=jdir / "output",
                          level=lambda: {"key": "L1"},
                          endpoint_flags=lambda mode: ["--start", "A.xyz", "--end", "B.xyz"],
                          common_flags=lambda: [])
    argv = _build_channels(ctx, ChannelsParams(**rdkit))
    assert argv[argv.index("--start-pool") + 1] == str(jdir / "inputs" / "start_pool.xyz")
    assert "--end-pool" not in argv                        # nothing matching sampled B
    copied = jdir / "inputs" / "start_pool.xyz"
    assert copied.read_text() == (tmp_path / "j_new/output/conformers/end_pool.xyz").read_text()
    assert copied.with_suffix(".energies").exists()

    # Turned off: sample again.
    argv = _build_channels(ctx, ChannelsParams(**rdkit, reuse_conformers=False))
    assert "--start-pool" not in argv


def test_older_runs_are_reused_only_if_no_mirror_images_were_merged(tmp_path):
    from mepd.web.operations import _saved_pool

    ok = _channels_job(tmp_path, "j1", ["s_a", "s_b"], {}, "L1", 1, merged=0, new_style=False)
    merged = _channels_job(tmp_path, "j2", ["s_a", "s_b"], {}, "L1", 1, merged=2, new_style=False)
    assert _saved_pool(ok, "start").name == "start.xyz"
    assert _saved_pool(merged, "start") is None
    assert _saved_pool(merged, "end").name == "end.xyz"


def test_channels_result_says_when_pools_were_reused(tmp_path):
    from mepd.web.results import collect_channels

    out = tmp_path / "run"
    (out / "conformers").mkdir(parents=True)
    _write_pool(out / "conformers" / "start.xyz", [ETHANOL], [-154.1])
    _write_pool(out / "conformers" / "end.xyz", [ETHANOL], [-154.0])
    (out / "stats.json").write_text(json.dumps({"conformers": {"start": {"reused_from": "x", "n_final": 1},
                                                               "end": {"n_final": 1}}}))
    rows = {r["label"]: r["value"] for r in collect_channels(out, 0, 1)["summary"]}
    assert rows["Conformer pools"] == "reactant reused from an earlier run"
