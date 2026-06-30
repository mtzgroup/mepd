from types import SimpleNamespace

import numpy as np

from mepd.scripts import progress


class _Graph:
    def __init__(self, smiles: str, same_connectivity: bool = True):
        self.smiles = smiles
        self.same_connectivity = same_connectivity

    def force_smiles(self):
        return self.smiles

    def remove_Hs(self):
        return self

    def is_bond_isomorphic_to(self, _other):
        return self.same_connectivity


class _Node:
    def __init__(
        self,
        has_molecular_graph: bool,
        *,
        graph=None,
        coords=None,
        energy: float | None = None,
    ):
        self.has_molecular_graph = has_molecular_graph
        self.graph = graph if graph is not None else object() if has_molecular_graph else None
        self.structure = object()
        self.coords = np.asarray(coords if coords is not None else [[0.0, 0.0, 0.0]])
        if energy is not None:
            self._cached_energy = energy

    @property
    def energy(self):
        if not hasattr(self, "_cached_energy"):
            raise RuntimeError("missing energy")
        return self._cached_energy


class _Chain:
    def __init__(self, has_molecular_graph: bool, nodes=None):
        self.nodes = nodes or [_Node(has_molecular_graph), _Node(has_molecular_graph)]
        self.energies_kcalmol = [0.0, 1.0]


def test_ascii_profile_skips_smiles_for_graphless_nodes(monkeypatch):
    def _fail(_structure):
        raise AssertionError("structure_to_smiles should not be called")

    monkeypatch.setattr(progress, "structure_to_smiles", _fail)
    out = progress.ascii_profile_for_chain(_Chain(has_molecular_graph=False))

    assert "start SMILES: N/A" in out
    assert "end SMILES:   N/A" in out


def test_ascii_profile_uses_smiles_for_graph_nodes(monkeypatch):
    monkeypatch.setattr(progress, "structure_to_smiles", lambda _structure: "C")
    out = progress.ascii_profile_for_chain(_Chain(has_molecular_graph=True))

    assert "start SMILES: C" in out
    assert "end SMILES:   C" in out


def test_ascii_profile_prefers_current_structure_smiles_over_stale_graph_labels(monkeypatch):
    smiles = iter(["C@H", "C@@H"])
    monkeypatch.setattr(progress, "structure_to_smiles", lambda _structure: next(smiles))
    chain = _Chain(
        has_molecular_graph=True,
        nodes=[
            _Node(True, graph=_Graph("stale-start")),
            _Node(True, graph=_Graph("stale-end")),
        ],
    )

    out = progress.ascii_profile_for_chain(chain)

    assert "start SMILES: C@H" in out
    assert "end SMILES:   C@@H" in out
    assert "stale-start" not in out
    assert "stale-end" not in out


def test_ascii_profile_reports_different_endpoint_connectivity(monkeypatch):
    monkeypatch.setattr(progress, "structure_to_smiles", lambda _structure: "C")
    chain = _Chain(
        has_molecular_graph=True,
        nodes=[
            _Node(True, graph=_Graph("C", same_connectivity=False)),
            _Node(True, graph=_Graph("O")),
        ],
    )

    out = progress.ascii_profile_for_chain(chain)

    assert "endpoints differ in: different connectivity" in out


def test_ascii_profile_uses_structure_smiles_to_override_stale_isomorphic_graphs(monkeypatch):
    smiles = iter(
        [
            "C=P(c1ccccc1)(c1ccccc1)c1ccccc1.CC(C)=O",
            "C=C(C)C.O=P(c1ccccc1)(c1ccccc1)c1ccccc1",
        ]
    )
    monkeypatch.setattr(progress, "structure_to_smiles", lambda _structure: next(smiles))
    chain = _Chain(
        has_molecular_graph=True,
        nodes=[
            _Node(True, graph=_Graph("stale", same_connectivity=True)),
            _Node(True, graph=_Graph("stale", same_connectivity=True)),
        ],
    )

    out = progress.ascii_profile_for_chain(chain)

    assert "endpoints differ in: different connectivity" in out


def test_ascii_profile_reports_different_endpoint_stereoconformers(monkeypatch):
    smiles = iter(["C@H", "C@@H"])
    monkeypatch.setattr(progress, "structure_to_smiles", lambda _structure: next(smiles))
    chain = _Chain(
        has_molecular_graph=True,
        nodes=[
            _Node(True, graph=_Graph("C")),
            _Node(True, graph=_Graph("C")),
        ],
    )

    out = progress.ascii_profile_for_chain(chain)

    assert "endpoints differ in: different stereoconformers" in out


def test_ascii_profile_reports_endpoint_conformer_metrics(monkeypatch):
    monkeypatch.setattr(progress, "structure_to_smiles", lambda _structure: "C")
    chain = _Chain(
        has_molecular_graph=True,
        nodes=[
            _Node(True, graph=_Graph("C"), coords=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], energy=0.0),
            _Node(True, graph=_Graph("C"), coords=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], energy=0.01),
        ],
    )

    out = progress.ascii_profile_for_chain(chain)

    assert "endpoints differ in: different conformers" in out
    assert "\n                      deltaE=6.275 kcal/mol, rmsd=0.5\n" in out
    assert max(len(line) for line in out.splitlines()[:3]) <= 100


def test_ascii_profile_tolerates_non_finite_energies():
    chain = _Chain(has_molecular_graph=False)
    chain.energies_kcalmol = [float("nan"), 1.0]

    out = progress.ascii_profile_for_chain(chain)

    assert "node index" in out


def test_ascii_profile_tolerates_missing_node_energies():
    class _PartialEnergyNode(_Node):
        def __init__(self, energy):
            super().__init__(has_molecular_graph=False)
            self._cached_energy = energy

        @property
        def energy(self):
            if self._cached_energy is None:
                raise RuntimeError("missing energy")
            return self._cached_energy

    chain = SimpleNamespace(
        nodes=[
            _PartialEnergyNode(0.0),
            _PartialEnergyNode(None),
            _PartialEnergyNode(1.0),
        ]
    )

    out = progress.ascii_profile_for_chain(chain)

    assert "node index" in out


def test_ascii_profile_axis_prefix_width_is_stable():
    plot = progress._build_ascii_energy_profile(
        energies=[-1.0e12, 2.5e12],
        labels=["0", "1"],
        width=12,
        height=5,
    )
    data_lines = plot.splitlines()[:5]
    bar_columns = {line.index("|") for line in data_lines}
    assert len(bar_columns) == 1


def test_monitor_label_is_bounded_to_fixed_width():
    printer = progress.ProgressPrinter(use_rich=False)
    printer._monitor_column_width = 20
    label = printer._format_monitor_label(
        monitor_id="branch-001",
        caption="step 125 | TS gperp: 0.0123 | max rms: 0.0456",
    )
    assert len(label) <= 20
    assert label.startswith("branch-001")


def test_neb_caption_includes_timestep_as_step_metric():
    caption = progress.format_neb_caption(
        step=5,
        ts_grad=0.01234,
        max_rms_grad=0.04567,
        ts_triplet_gspring=0.00089,
        timestep=0.25,
    )

    assert caption == (
        "step 5 | TS gperp: 0.0123 | max rms: 0.0457 | "
        "tspring: 0.0009 | dt=0.250"
    )


def test_compact_ascii_for_live_uses_payload_series():
    printer = progress.ProgressPrinter(use_rich=False)
    state = {
        "chain_plot_payload": {"y": [0.0, 1.0, 0.3, 1.2]},
        "ascii_plot": "unused",
    }

    out = printer._compact_ascii_for_live(state)

    assert "node index" in out


def test_visible_monitor_ids_paginates_and_rotates():
    printer = progress.ProgressPrinter(use_rich=False)
    printer._monitor_page_size = 2
    printer._monitor_page_rotate_seconds = 1.0
    monitor_ids = ["branch-0", "branch-1", "branch-2", "branch-3", "branch-4"]

    ids0, meta0 = printer._visible_monitor_ids(monitor_ids, now=100.0)
    ids1, meta1 = printer._visible_monitor_ids(monitor_ids, now=100.3)
    ids2, meta2 = printer._visible_monitor_ids(monitor_ids, now=101.1)
    ids3, meta3 = printer._visible_monitor_ids(monitor_ids, now=102.2)

    assert ids0 == ["branch-0", "branch-1"]
    assert ids1 == ["branch-0", "branch-1"]
    assert ids2 == ["branch-2", "branch-3"]
    assert ids3 == ["branch-4"]
    assert meta0["total_pages"] == 3
    assert meta2["page"] == 2
    assert meta3["page"] == 3


def test_monitor_active_state_toggles_without_dropping_payload_state():
    printer = progress.ProgressPrinter(use_rich=False)
    printer.mark_monitor_active("branch-7")
    state = printer._state_for_monitor("branch-7")
    state["ascii_plot"] = "plot"

    printer.mark_monitor_inactive("branch-7")

    payload = printer._monitors_payload()
    assert "branch-7" in payload
    assert payload["branch-7"]["active"] is False


def test_set_monitor_status_updates_caption_without_activation():
    printer = progress.ProgressPrinter(use_rich=False)
    printer.set_monitor_status("branch-3", "Running in worker process")

    state = printer._state_for_monitor("branch-3")
    assert state["status_message"] == "Running in worker process"
    assert state["caption"] == "Running in worker process"
    assert "branch-3" not in printer._active_monitor_ids


def test_active_ascii_for_live_is_dense_two_line_summary():
    printer = progress.ProgressPrinter(use_rich=False)
    state = {"chain_plot_payload": {"y": [0.0, 1.5, -0.5, 2.0, 1.0]}}

    out = printer._active_ascii_for_live(state)

    lines = out.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("E[")


def test_active_ascii_for_live_uses_status_when_no_plot_data():
    printer = progress.ProgressPrinter(use_rich=False)
    state = {
        "chain_plot_payload": {},
        "ascii_plot": None,
        "status_message": "Running in worker process (attempt 1/2, 8s)",
    }

    out = printer._active_ascii_for_live(state)

    assert out == "Running in worker process (attempt 1/2, 8s)"


def test_build_live_rows_includes_main_monitor_updates():
    printer = progress.ProgressPrinter(use_rich=False)
    main_state = printer._state_for_monitor("main")
    main_state["ascii_plot"] = "main-chain-ascii"
    main_state["caption"] = "step 2 | TS gperp: 0.1200"

    rows, _meta = printer._build_live_rows()

    assert rows
    labels = [label for label, _chain in rows]
    chains = [chain for _label, chain in rows]
    assert any(label.startswith("main") for label in labels)
    assert any("main-chain-ascii" in chain for chain in chains)
