"""Public demo mode (`mepd web DEMO_ROOT --demo`).

Anyone with the shared password can use the server, so everything that
would hand them the machine is switched off or capped here, server-side:

* no filesystem paths from the browser (no "open existing output", no
  opening sessions by path) -- visitors only ever touch their own workspace
  under DEMO_ROOT/visitors/<visitor>/;
* compute profiles are the admin's (DEMO_ROOT/profiles/*.toml), copied into
  each visitor workspace and read-only: a profile can name executables and
  read files, so visitors cannot write or validate one;
* size/quantity caps (atoms, structures, sessions, jobs) and caps on the
  expensive knobs of each operation;
* a run-time limit per job and a concurrency limit across all visitors
  (JobManager.max_runtime / global_slot_free).

Run it inside the container described in deploy/demo/ for a second wall.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

from mepd.web.workspace import WorkspaceError


@dataclass
class DemoPolicy:
    max_atoms: int = 40
    max_structures: int = 60
    max_sessions: int = 5
    max_active_jobs: int = 2           # queued + running, per visitor
    max_jobs: int = 60                 # kept in a workspace, per visitor session
    max_upload_bytes: int = 1_000_000
    max_smiles_length: int = 300
    job_timeout_s: float = 30 * 60
    # A VRI search computes a Hessian at many IRC points: typically 10-25
    # minutes with 3 workers, up to ~40 with the demo's 2.
    op_timeout_s: dict = field(default_factory=lambda: {"vri": 75 * 60, "vri-check": 45 * 60})
    global_concurrency: int = 4        # running jobs across all visitors
    # op -> field -> max allowed value (inclusive). Booleans listed here are
    # forced off (False is the only allowed value).
    caps: dict = field(default_factory=lambda: {
        "channels": {"workers": 2, "pairs_per_mechanism": 3, "n_conformers": 15, "n_embed": 100,
                     "max_pairs": 30, "crest_timeout": 900},
        "ts": {"parallel_workers": 2, "network_max_followups": 5},
        "hessian-sample": {"max_candidates": 60, "maxiter": 500},
        "hessian-global": {"max_rounds": 5, "max_candidates": 40, "maxiter": 500},
        "network-splits": {"max_pairs": 10},
        "vri": {"workers": 2, "trajectories": 50},
        "vri-check": {"workers": 2, "trajectories": 50},
        "vri-surface": {"workers": 2, "grid": 13},
        "graph-enumeration": {"rounds": 3, "max_products": 30, "n_break": 2, "n_form": 2, "max_pairs": 10,
                              "workers": 2, "maxiter": 500},
        "nanoreactor": {"time_ps": 10.0, "max_connect": 5, "workers": 2, "instances": 3},
        "nanoreactor-more": {"more_ps": 10.0},
        # (a complex's atoms are capped by max_atoms, like any structure: /api/complexes)
        "complex": {"keep": 3},
        # A few seconds of one CPU at these values.
        "retrosynthesis": {"iterations": 100, "time_limit": 60, "max_depth": 6, "routes": 5, "width": 10,
                           "workers": 2},
    })
    # op -> field -> the only values allowed (text and choice fields). A
    # method (family card) fixing a value outside them is shown unavailable.
    allowed_values: dict = field(default_factory=lambda: {
        # Only the interactive reactor's Stop & analyze reads a trajectory (it
        # submits directly); a visitor's own nanoreactor job runs its MD.
        "nanoreactor": {"trajectory": ("",), "trajectory_start": ("",)},
        "retrosynthesis": {
            "method": ("templates",),            # the others need GBs of models, or call a URL the user picks
            "verify": ("none",),                 # top/all: a path search + TS + IRC per step
            "stock": ("auto", "paroutes"),       # zinc needs the AiZynthFinder setup
            "stock_file": ("",),                 # a file path on the server
            "llm_url": ("http://localhost:11434/v1",),
        },
    })
    # Interactive reactors (mepd.web.sandbox): each holds a CPU while it runs.
    sandbox: dict = field(default_factory=lambda: {"total": 2, "per_owner": 1, "idle_s": 180.0,
                                                   "max_s": 15 * 60.0})
    # Operations visitors may run at all.
    allowed_ops: tuple = ("ts", "channels", "tsopt", "hessian-sample", "hessian-global", "optimize",
                          "network-splits", "vri", "vri-check", "vri-surface", "graph-enumeration", "nanoreactor",
                          "solvent", "complex", "retrosynthesis", "nanoreactor-more")

    def public(self) -> dict:
        """What the UI shows (and uses to hide admin-only controls)."""
        d = asdict(self)
        d["allowed_ops"] = list(self.allowed_ops)
        return d

    def adapt_operation(self, desc: dict) -> dict:
        """An operation description as the demo serves it: not-allowed ops
        disabled, and capped fields defaulting to (and limited at) their cap,
        so a form submitted as-is never trips a limit."""
        desc = {**desc}
        if desc["key"] not in self.allowed_ops:
            desc.update(available=False, unavailable_reason="Not available in the demo.")
            return desc
        caps = self.caps.get(desc["key"], {})
        if caps and desc.get("schema"):
            schema = {**desc["schema"], "properties": {k: dict(v) for k, v in desc["schema"]["properties"].items()}}
            for name, cap in caps.items():
                prop = schema["properties"].get(name)
                if prop is None or isinstance(cap, bool):
                    continue
                prop["maximum"] = cap
                if isinstance(prop.get("default"), (int, float)) and prop["default"] > cap:
                    prop["default"] = cap
                prop["description"] = (prop.get("description") or "") + f" (demo limit: {cap})"
            desc["schema"] = schema
        allowed = self.allowed_values.get(desc["key"], {})
        if allowed and desc.get("schema"):
            schema = {**desc["schema"], "properties": {k: dict(v) for k, v in desc["schema"]["properties"].items()}}
            for name, values in allowed.items():
                prop = schema["properties"].get(name)
                if prop is None:
                    continue
                if prop.get("enum"):
                    prop["enum"] = [v for v in prop["enum"] if v in values]
                if prop.get("default") not in values:
                    prop["default"] = values[0]
            desc["schema"] = schema
        if allowed and desc.get("methods"):
            desc["methods"] = [m if all(m.get("fixed", {}).get(k, v[0]) in v for k, v in allowed.items())
                               else {**m, "available": False, "reason": "not available in the demo"}
                               for m in desc["methods"]]
        return desc

    # ------------------------------------------------------------ checks
    def check_op(self, op_key: str, params: Optional[dict]) -> None:
        if op_key not in self.allowed_ops:
            raise WorkspaceError(f"'{op_key}' is not available in the demo")
        for name, cap in self.caps.get(op_key, {}).items():
            value = (params or {}).get(name)
            if value is None:
                continue
            if isinstance(cap, bool):
                if value and not cap:
                    raise WorkspaceError(f"{name} is disabled in the demo")
            elif isinstance(value, (int, float)) and value > cap:
                raise WorkspaceError(f"{name} = {value} is above the demo limit of {cap}")
        allowed = self.allowed_values.get(op_key, {})
        if allowed:   # unset fields run at their default: check that too
            from mepd.web.operations import OPERATIONS

            model = OPERATIONS[op_key].params_model if op_key in OPERATIONS else None
            defaults = {k: f.default for k, f in model.model_fields.items()} if model else {}
        for name, values in allowed.items():
            value = (params or {}).get(name, defaults.get(name))
            if value is not None and value not in values:
                raise WorkspaceError(f"{name} = {value!r} is not available in the demo")

    def check_capacity(self, jobs: list[dict]) -> None:
        active = sum(1 for j in jobs if j["status"] in ("queued", "running"))
        if active >= self.max_active_jobs:
            raise WorkspaceError(f"the demo runs at most {self.max_active_jobs} of your calculations at a time; "
                                 "wait for one to finish or cancel it")
        if len(jobs) >= self.max_jobs:
            raise WorkspaceError(f"this demo session holds at most {self.max_jobs} calculations; delete some "
                                 "or start a new session")

    def check_structures(self, existing: int, new_atom_counts: list[int]) -> None:
        if existing + len(new_atom_counts) > self.max_structures:
            raise WorkspaceError(f"the demo allows at most {self.max_structures} structures per session")
        big = [n for n in new_atom_counts if n > self.max_atoms]
        if big:
            raise WorkspaceError(f"the demo is limited to {self.max_atoms} atoms per structure (got {max(big)})")

    def check_text(self, text: str) -> None:
        if len(text.encode()) > self.max_upload_bytes:
            raise WorkspaceError("input too large for the demo")
        if not text.lstrip()[:1].isdigit():  # SMILES lines, not xyz
            for line in text.splitlines():
                smi = line.strip().partition(" ")[0]
                if len(smi) > self.max_smiles_length:
                    raise WorkspaceError(f"SMILES longer than {self.max_smiles_length} characters")
