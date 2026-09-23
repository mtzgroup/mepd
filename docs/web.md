# mepd web

A browser interface over the mepd CLI. It is meant for three kinds of use:

- **One-off questions.** For example, "the TS between these two", with endpoints given as SMILES or XYZ.
- **Exploration** around a structure.
- **Hand-built reaction networks.** Draw the network, then run any calculation on some or all of its edges.

The web UI is an optional part of the package: its Python dependencies are the `web` extra, and
the page itself ships inside the package.

```bash
uv sync --extra web          # dev checkout
pip install "mepd[web] @ git+https://github.com/mtzgroup/mepd.git"   # or from git
mepd web my_workspace        # opens http://127.0.0.1:8765
mepd web my_workspace --max-jobs 4 --port 9000 --no-open
```

On a remote machine, run it there and use an SSH tunnel:
`ssh -L 8765:localhost:8765 cluster`. There is no authentication. `--host 0.0.0.0` exposes job
submission and file reading to anyone who can reach the port. The page's JavaScript libraries are
vendored under `mepd/web/static/vendor/`, so it works without internet access.

## Mental model

| Concept | What it is |
|---|---|
| **Workspace** | A directory. It holds a structure library, a reaction graph, compute profiles and job folders. Everything is plain files; copy or archive the directory to share a project. |
| **Structure** | A library entry and, equally, a graph node. It is added from SMILES (one per line, `SMILES name`), pasted XYZ (multi-frame means several structures), dropped `.xyz`/`.smi` files, or imported from a result. |
| **Edge** | A pair of structures declared as connected, with a direction (start → end). It is drawn with **Connect** (or the `C` key), made by *Connect as edge* on two selected structures, created implicitly by running a pair calculation, or imported from a result. Atom counts must match. |
| **Operation** | A mepd command the UI can run: TS / MEP (`mepd run`), reaction channels (`mepd channels`), TS from a guess (`mepd ts`), Hessian sampling and basin hopping (`mepd discovery …`), and all-pairs network (`mepd network-splits`). Nanoreactor and graph enumeration are listed but disabled, because mepd has no CLI entry point for them yet. |
| **Profile** | A `RunInputs` TOML (the `--inputs` file): engine, level of theory, path minimizer and thresholds. It is edited and validated in the Profiles tab; *Validate* actually builds the engine in a scratch process. |
| **Job** | One `mepd …` subprocess with its own folder. |
| **Level of theory** | Each workspace has one: a chosen profile (Structures panel → *Level of theory*). Its fingerprint is the engine, program, method and basis settings. Every structure records the level its geometry and energy belong to. |
| **Known minimum** | A structure that is a minimum at a stated level: optimized on entry, or imported from a minima or conformer result of a job. A pair job skips endpoint minimization only when both endpoints are minima **at that job's level**; anything else (force-field embeddings, raw XYZ, structures from another level or an external output) is re-minimized first. |
| **TS structure** | A saddle point imported from a result (role `ts`). It is never minimized, because that would destroy it. |

### One level of theory per workspace

- **Structures added from SMILES or XYZ are minimized at the workspace level** with `mepd optimize`, one job per charge/multiplicity group (the add box has a checkbox to skip this). A SMILES embedding is only an RDKit/MMFF94 force-field geometry.
- **Status is shown on the structure.** It appears straight away as *optimizing…*, and its geometry, SMILES and energy are replaced in place when the job finishes. Its chip shows the level (e.g. `gxtb`), or *not optimized*, *other level*, *opt failed* or *not a minimum*.
- **Hessian check (on by default).** *Verify minima (Hessian)* makes every optimized structure pass a Hessian check: no frequency below the cutoff.
  - A structure that stopped on a saddle point is pushed along its unstable mode, both ways, and re-optimized. The push starts at 0.1 bohr and escalates to 0.3 and then 0.5 bohr, stopping at the first push that reaches a minimum. The structure records which push worked.
  - If every push fails, it is kept but flagged *not a minimum* and never trusted as one.
  - Why escalate: eclipsed ethane optimizes straight onto the rotational saddle (−334 cm⁻¹). A 0.1 bohr push falls back onto it; 0.3 bohr reaches staggered.
  - The same escalating rescue is used by mepd everywhere minima are Hessian-checked: recursive path searches, `mepd optimize`, Hessian sampling and basin hopping.
- **Changing the level** marks structures from the old level as off-level. *Re-optimize N off-level* redoes them, skipping TS structures.
- **Jobs check levels too.** A job run with a profile at a different level shows a warning, and pair jobs re-minimize off-level endpoints themselves. Duplicate detection on import never compares energies from different levels.
- **Path searches (TS, channels, all-pairs network).** *Validate minima with Hessian* is a basic option: every intermediate minimum a recursive split proposes must pass the same check. Its cutoff and push size are under Advanced.
- **Hessian sampling and basin hopping.** The same option (on by default in the web) checks every minimum the sampler reports.
  - Rescued minima replace the originals and are de-duplicated again.
  - Structures that are still saddles are dropped from the minima and listed under *Rejected: not minima* (`rejected.xyz`).
  - Basin hopping only accepts, and seeds later rounds from, validated minima.
  - Imported minima carry their check, and only ones that passed count as known minima.
  - On the CLI: `mepd discovery hessian-sample|hessian-global --validate-minima-with-hessian`. It is off by default there, so existing scripts are unchanged.

### Selection drives what you can run

The right-hand panel lists exactly the operations that fit the current selection:

| Selection | Offered |
|---|---|
| One structure | Explore around it: TS optimization from a guess, Hessian sampling, basin hopping |
| Two structures (the first clicked is the start) or one edge | TS / MEP, reaction channels |
| Several edges | The same pair calculation, **one job per edge** (batch) |
| Several structures | Batch exploration, or the all-pairs network |

Each operation's form is generated from its parameter model:
- Basic knobs are shown directly, with the rest under *Advanced*.
- A knob that only applies when another option is on (for example, CREST settings when the backend is RDKit) is hidden until that option is on.
- *Show command* prints the exact `mepd …` command the job will run.

The last-used parameters and profile are remembered in the browser, per operation.

### Results flow back into the graph

A job's **Results** tab shows everything the command produced, in groups. For a channels run, for
example, the groups are direct channels, multi-step channels, off-target exits, conformer pools and
every TS search.

Each entry has:
- a 3D viewer;
- a frame slider;
- an energy profile (click or drag it to scrub).

From any entry you can:
- **Add ends + edge to graph.** The two path ends become structures and are joined by an edge that remembers the result and its barrier. A structure already in the library is reused (same connectivity and energy within 0.05 kcal/mol).
- **Add TS** or **Add this frame.** The structure becomes a node you can explore from.

Edges are coloured by their state: not computed, queued, running, done or failed. A done edge is
labelled with the best barrier any job or import found for it.

**Barriers** are referenced to the lowest reactant-side energy found anywhere in that result:
- for channels, reactant conformers and reactant-side IRC ends;
- for `run`, the path start and IRC ends of the same connectivity.

They are never referenced to the search's own starting geometry.

**Log warnings.** Problems mepd only reports on stdout (for example, an IRC that failed while the TS
was kept, or an NEB that never converged) are shown above the results.

### Sessions, cleanup, downloads

- **Sessions.** A session is a workspace directory.
  - *New session* starts over in a fresh, empty workspace and copies your compute profiles into it. A bare name creates the directory next to the current workspace; a full path puts it anywhere.
  - *Open…* lists recent sessions, or opens any workspace directory by path.
  - Switching never stops anything: the other session's running jobs keep running, and they are there when you switch back.
  - The recent list lives in `~/.config/mepd/web_sessions.json`.
- **Removing elements.** Select several structures and/or edges (shift/⌘-click, shift-drag, or *Select all*), then press *Delete* in the graph toolbar or the Delete key. Removing a structure removes its edges; calculation outputs are never deleted this way.
- **Downloads.**
  - On a result: a TS table (CSV: group, label, barrier, TS energy, what it connects), all TS geometries as one multi-frame XYZ, and the whole output folder with inputs and logs as a zip.
  - On any entry: the path, the TS or the current frame as XYZ. Comment lines carry the label, frame and energies (Eh and relative kcal/mol).
  - Selected structures: one multi-frame XYZ.

### Jobs

- **Queue.** At most `--max-jobs` jobs run at once. The rest wait, first in, first out.
- **Live tab.** Shows the latest status line and the path's energy profile as it optimizes, plus per-branch mini-plots for MSMEP. For `channels` it also shows the stats.json counters.
- **Cancel** kills the job's whole process group, which includes `--workers` children and CREST.
- **Resume / Rerun** runs the same command into the same output folder. `channels`, `ts` and `network-splits` skip what is already on disk.
- **Server restart.** Queued jobs stay queued. Running jobs are marked *interrupted* and can be resumed.
- **Open existing output…** registers a mepd output directory from anywhere on disk as a read-only job. Past CLI runs can then be browsed and imported into the graph. The directory is never modified.

## Architecture

```
browser (Preact + htm, Cytoscape, 3Dmol; no build step)
   │  REST: mutations          SSE /api/events: workspace / job / progress
   ▼
FastAPI app (mepd/web/app.py)
   ├── Workspace (workspace.py)   workspace.json + structures/*.xyz + profiles/*.toml
   ├── JobManager (jobs.py)       queue → `python -m mepd.cli <argv>` per job, own process group
   │                              progress from stdout, MEPD_DRIVE_PROGRESS_LOG,
   │                              MEPD_DRIVE_CHAIN_JSON, stats.json (polled 1 s)
   ├── OPERATIONS (operations.py) params model + argv builder per operation
   └── collectors (results.py)    output dir → {headline, barrier, groups[entries[frames]]},
                                  run in a spawned worker process (never on the server's threads)
```

Jobs are subprocesses rather than in-process calls, for four reasons:
- mepd's helpers raise `typer.Exit`, fork process pools, and set process-global state when `RunInputs` is built.
- A crash or hung QM call stays contained in its job.
- Killing the process group cancels everything the job started.
- Rerunning the CLI into the same output folder resumes it.

A job's folder:

```
jobs/<id>/job.json     record: argv, status, params, profile, targets, summary
jobs/<id>/inputs/      snapshot of the structures and profile the job used
jobs/<id>/output/      the command's --output directory
jobs/<id>/stdout.log   progress.log   chain.json   result.json (cache)
```

Every job snapshots its inputs, so editing or deleting a library structure or profile later never
changes what a past job ran with.

## Adding an operation

1. In `mepd/web/operations.py`, write a `Params` model. Each knob is a `P(default, title, help, cli="--flag", kind=..., group=..., advanced=..., requires=...)`. `kind` is one of:
   - `value`: `--flag v`
   - `switch`: `--flag` when true
   - `toggle`: `--flag` / `--no-flag`
   - `custom`: handled by the builder
2. Add an `Operation(key, title, summary, target, category, Params, build)`. `target` is `structure`, `pair` or `set`. `build(ctx, params)` returns the argv after `mepd`; `JobContext` has helpers for endpoints, charge/multiplicity and the profile.
3. If the command writes a new output layout, add a collector to `COLLECTORS` in `mepd/web/results.py`, and a rule to `detect_operation` so its output folders can be opened. Bump `RESULT_VERSION` when collectors change.

The form, the action panel, batching, the queue, progress and the log need no other change. For
example, a future `mepd discovery nanoreactor` command becomes one `Params` model plus one
`Operation` with `target="structure"`.

## HTTP API (for scripts)

| Method and path | Purpose |
|---|---|
| `GET /api/state` | Workspace, jobs, operations (with JSON schemas), profiles |
| `GET /api/events` | SSE: `workspace`, `job`, `job_deleted`, `progress`, `profiles` |
| `POST /api/structures` `{text, name?, charge?, multiplicity?}` | Add from SMILES or XYZ text |
| `POST /api/structures/upload` (multipart `files`) | Add from files |
| `PATCH` / `DELETE /api/structures/{id}` | Rename, set charge/multiplicity, delete (and its edges) |
| `POST /api/edges` `{source, target}` | Add an edge. `PATCH` takes `{label, reverse}`; `DELETE` removes it |
| `POST /api/jobs` `{op, structures[], edges[], params, profile, dry_run?}` | Submit. One job per edge or structure for batchable ops. `dry_run` returns the commands only |
| `POST /api/jobs/import` `{path, op?, charge, multiplicity}` | Open an existing output directory |
| `GET /api/jobs/{id}/result` | Parsed result (cached once finished) |
| `POST /api/jobs/{id}/import-entry` `{entry, frames: endpoints\|ts\|one\|all, frame?, connect}` | Result → library/graph |
| `POST /api/jobs/{id}/cancel` \| `retry` | Queue control. `DELETE /api/jobs/{id}` removes the job |
| `GET /api/jobs/{id}/log?which=stdout\|progress&offset=` | Incremental log (`X-Log-Size` header) |
| `GET/PUT/DELETE /api/profiles/{name}`, `POST /api/profiles/validate` | Profiles |
| `POST /api/delete` `{structures[], edges[]}` | Bulk removal (one save, one event) |
| `GET /api/structures-export?ids=a,b` | Structures as a multi-frame xyz |
| `GET /api/jobs/{id}/archive` | Output folder, inputs and logs as a zip |
| `GET /api/sessions`, `POST /api/sessions/new` `{name \| path}`, `POST /api/sessions/open` `{path}` | Sessions |

## Performance notes

- **The path search dominates wall time.** The web layer adds milliseconds per action and about 2 s of process start-up per job. Progress reporting costs about 3% of an NEB step.
- **The path method matters most.** On a 14-atom test pair with g-xTB (a sampled allyl vinyl ether minimum to pent-4-enal, run with `--recursive --use-tsopt --irc`):

  | Profile | Wall time |
  |---|---|
  | Default NEB | 308 s (~1,000 steps) |
  | FNEB | 37 s |
  | GSM | 53 s (but it found a different product) |

- **Endpoints should be relaxed.** With one endpoint left unrelaxed, the same NEB search found a spurious intermediate and was still running at 400+ s.
- **Result parsing reads only final chains**, never NEB histories, and runs in a worker process, so the UI stays responsive while a large output is read. After the first read, results come from the cache.

## Limits worth knowing

- Nanoreactor and graph enumeration are not available until mepd has commands for them.
- Barrier floors are per result. Comparing results from different jobs (for example, RDKit vs CREST channels) still needs one shared floor across them, which the UI does not compute.
- Result parsing loads every chain in an output folder. Large `channels` runs take a few seconds the first time; after that they come from the cache.
- Profiles hold engine settings. Endpoint charge and multiplicity come from the structures, and both endpoints of a pair must agree.
