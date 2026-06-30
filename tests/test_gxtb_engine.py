from __future__ import annotations

import subprocess

import numpy as np
import pytest
from qcio import Structure

from mepd.engines.gxtb import GXTBCalculator
from mepd.inputs import RunInputs
from mepd.msmep import _clone_run_inputs_for_worker, _run_inputs_payload_for_worker
from mepd.nodes.node import StructureNode


def _water_node() -> StructureNode:
    structure = Structure(
        symbols=["O", "H", "H"],
        geometry=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.43355001758932, 0.0, 0.95295864902809],
                [-1.43355001758932, 0.0, 0.95295864902809],
            ],
            dtype=float,
        ),
        charge=0,
        multiplicity=1,
    )
    return StructureNode(structure=structure)


def test_gxtb_engine_parses_energy_and_gradient(monkeypatch):
    calls = []

    def fake_run(cmd, cwd, env, text, capture_output, check):
        calls.append((cmd, cwd, env, text, capture_output, check))
        (cwd / "energy").write_text(
            "$energy\n"
            "     1   -76.43250214643   -76.43250214643   -76.43250214643\n"
            "$end\n"
        )
        (cwd / "gradient").write_text(
            "$grad\n"
            "  cycle =      0    SCF energy =   -76.43250214643   |dE/dxyz| =  0.097592\n"
            "    0.00000000000000      0.00000000000000      0.00000000000000      O\n"
            "    1.43355001758932      0.00000000000000      0.95295864902809      H\n"
            "   -1.43355001758932      0.00000000000000      0.95295864902809      H\n"
            "   1.7549184260902E-16   3.8204626838825E-18   6.6535518937753E-02\n"
            "  -3.7972299641747E-02  -3.6975474899590E-18  -3.3267759468877E-02\n"
            "   3.7972299641747E-02  -1.2291519392354E-19  -3.3267759468877E-02\n"
            "$end\n"
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="normal termination", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    node = _water_node()
    eng = GXTBCalculator(executable="/opt/xtb/bin/xtb", n_threads=3)

    energies = eng.compute_energies([node])
    gradients = eng.compute_gradients([node])

    assert len(calls) == 1
    assert calls[0][0] == [
        "/opt/xtb/bin/xtb",
        "structure.xyz",
        "--silent",
        "--chrg",
        "0",
        "--grad",
        "--gxtb",
    ]
    assert calls[0][2]["OMP_NUM_THREADS"] == "3"
    assert energies == pytest.approx([-76.43250214643])
    assert gradients.shape == (1, 3, 3)
    assert gradients[0, 1, 0] == pytest.approx(-3.7972299641747e-02)
    assert node._cached_energy == pytest.approx(-76.43250214643)


def test_runinputs_builds_gxtb_engine():
    inputs = RunInputs(
        engine_name="gxtb",
        path_min_method="NEB",
        gxtb_engine_kwds={"executable": "/tmp/gxtb", "n_threads": 2},
    )

    assert isinstance(inputs.engine, GXTBCalculator)
    assert inputs.program == "xtb"
    assert inputs.engine.executable == "/tmp/gxtb"
    assert inputs.engine.n_threads == 2


def test_gxtb_worker_payload_preserves_engine_kwargs():
    inputs = RunInputs(
        engine_name="gxtb",
        path_min_method="NEB",
        gxtb_engine_kwds={"executable": "/tmp/gxtb", "n_threads": 2},
    )

    payload = _run_inputs_payload_for_worker(inputs)
    worker_inputs = _clone_run_inputs_for_worker(inputs)

    assert payload["gxtb_engine_kwds"] == {"executable": "/tmp/gxtb", "n_threads": 2}
    assert isinstance(worker_inputs.engine, GXTBCalculator)
    assert worker_inputs.engine.executable == "/tmp/gxtb"
    assert worker_inputs.engine.n_threads == 2


def test_gxtb_engine_geometry_optimization_parses_trajectory(monkeypatch):
    calls = []

    def fake_run(cmd, cwd, env, text, capture_output, check):
        calls.append(cmd)
        if "--opt" in cmd:
            (cwd / "xtbopt.log").write_text(
                "3\n"
                " energy: -76.432502146434 gnorm: 0.097592232891 xtb: 6.7.1 iter: 1\n"
                "O 0.00000000000000 0.00000000000000 0.00000000000000\n"
                "H 0.75860200000000 0.00000000000000 0.50428400000000\n"
                "H -0.75860200000000 0.00000000000000 0.50428400000000\n"
                "3\n"
                " energy: -76.437641532396 gnorm: 0.000251325564 xtb: 6.7.1 iter: 2\n"
                "O -0.00000000006196 0.00000000000000 -0.04698604392203\n"
                "H 0.77007901222618 0.00000000000000 0.52777702198161\n"
                "H -0.77007901216422 -0.00000000000000 0.52777702194042\n"
            )
        else:
            (cwd / "energy").write_text(
                "$energy\n"
                "     1   -76.43764153239   -76.43764153239   -76.43764153239\n"
                "$end\n"
            )
            (cwd / "gradient").write_text(
                "$grad\n"
                "  cycle =      0    SCF energy =   -76.43764153239   |dE/dxyz| =  0.000251\n"
                "   -1.0E-05   0.0E+00   2.0E-05\n"
                "    1.0E-05   0.0E+00  -1.0E-05\n"
                "    0.0E+00   0.0E+00  -1.0E-05\n"
                "$end\n"
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="normal termination", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    eng = GXTBCalculator(executable="/opt/xtb/bin/xtb")
    trajectory = eng.compute_geometry_optimization(_water_node(), keywords={"maxiter": 5})

    assert len(trajectory) == 2
    assert "--opt" in calls[0]
    assert "--cycles" in calls[0]
    assert calls[0][calls[0].index("--cycles") + 1] == "5"
    assert "--grad" in calls[1]
    assert trajectory[-1]._cached_energy == pytest.approx(-76.43764153239)
    assert np.asarray(trajectory[-1]._cached_gradient).shape == (3, 3)
