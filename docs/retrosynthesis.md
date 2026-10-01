# Retrosynthesis

`mepd retro plan` finds routes from building blocks to a target molecule. In
the web UI it is the **Retrosynthesis** row of the *Reaction network
expansion* card, for a selected structure. Each step of a reported route
joins Explore as a reaction coming back from the target, so a TS search on
any step is one click away ("Find TS"). The routes are laid out as a
synthesis tree (the graph's **Tree** button, applied automatically when a
retrosynthesis adds them): the target on the right, the steps that make it
to its left, building blocks at the far left, small co-reactants and side
products (H₂, water) just below their step. Tree works on any graph: each
subnetwork (molecules joined by reactions, not counting water and other
tiny molecules) gets its own layout, side by side along x, and molecules in
no reaction share a grid at the end.

```bash
mepd retro plan "CC(=O)Nc1ccc(OCc2ccccc2)cc1"                       # templates (default), seconds
mepd retro plan TARGET --method reactiont5                          # local chemistry language model
mepd retro plan TARGET --method local-llm -o model=qwen3:8b         # LLM on a local server
mepd retro plan TARGET --method aizynthfinder                       # AiZynthFinder's own MCTS
mepd retro plan TARGET -i gxtb.toml --verify top                    # also path-search each step
mepd retro methods                                                  # what can run here
mepd retro setup [--aizynthfinder]                                  # download data up front
```

Output: `routes.json` (every route, cheapest first, as a tree and as a step
list), `summary.json` (settings, statistics, the methods used and their
citations), `network.json` (species and one reaction per distinct step,
which the web UI adopts), `verify/step_k/` (with `--verify`).

## Methods

All models run locally; nothing is sent to an outside service.

| `--method` | What proposes each step | Needs | Speed |
|---|---|---|---|
| `templates` | 42,554 USPTO retro templates applied with rdchiral, ranked by AiZynthFinder's public policy network (ONNX, run here by onnxruntime) and screened by its feasibility filter | `mepd[retro]`; ~112 MB data on first use (CC-BY 4.0) | ~0.1 s per molecule |
| `reactiont5` | ReactionT5v2 (MIT), a T5 model trained on the Open Reaction Database, with beam search; a forward model checks each proposal (round trip) | `mepd[retro-ml]` (transformers, torch); ~1.6 GB of weights on first use; GPU if available | ~1 s per molecule |
| `local-llm` | An open-weight LLM behind a local OpenAI-compatible server (Ollama, llama.cpp, vLLM, LM Studio), asked for disconnections as JSON | a running server, e.g. `ollama serve` + `ollama pull qwen3:8b` | depends on the model |
| `aizynthfinder` | AiZynthFinder 4.4.1's own MCTS, run as a separate program | `mepd retro setup --aizynthfinder` (~2 GB on disk: env + data incl. ZINC) | ~10 s per target |

Every proposal from a learned model goes through RDKit before it enters the
search: valid SMILES, canonical, not the target itself. LLM answers must also
supply the target's heavy atoms. ReactionT5 writes reagents too, often as
ions ([Al+3].[H-]...). Carbon-free ions and metal species are moved to the
step's reagents, so they are not precursors that need a route. A proposal
the forward model doesn't turn back into the target keeps 0.2 of its weight
(`-o roundtrip_penalty=`), because the forward model misses reactions
written without their reagents.

## Search

The search is an AND-OR best-first search in the manner of Retro* (mepd's
reimplementation):

- **Step cost.** A step costs −ln(score), where the scores are normalized
  per molecule.
- **Stock cost.** A molecule in stock costs 0.
- **Unexpanded molecules** are estimated at 0.5·(SA score − 1), or 0 with
  `--value zero` (Retro*-0).
- **Each iteration** expands the open molecule on the cheapest partial
  route. Once that route is solved, it goes on to the next cheapest, so
  alternatives come back too.
- **No cycles:** a step never brings back a molecule on its own path.
- **Budget:** `--iterations` (100), `--time-limit` (120 s), `--max-depth`
  (6), `--width` (10 proposals per molecule).

## Building blocks (`--stock`)

- `paroutes-n1`, `paroutes-n5`: the PaRoutes stocks (CC-BY 4.0), ~20k
  molecules together, including common reagents (SOCl₂, Ac₂O, Et₃N…).
- `zinc`: ZINC in-stock as shipped by AiZynthFinder (17.4 M entries, 10.5 M
  unique skeletons), available after `mepd retro setup --aizynthfinder`.
  It is stored as a sorted uint64 array of InChIKey first blocks (~80 MB in
  memory).
- A file: one SMILES or InChIKey per line, or a CSV with a `smiles` column.

The default is `zinc` once it is set up, else both PaRoutes sets. Molecules
match on the first InChIKey block, so stereo is ignored: a racemate in stock
covers either enantiomer. `--max-heavy` (2) also counts tiny molecules (H₂,
CO, ethylene) as available. It is kept low so that unstable fragments such
as ketene or enols still need a route.

## Verification (`--verify none|top|all`, default none)

Each step is balanced: a template keeps its leaving groups on the
precursors, so an amide from an acyl chloride also releases HCl, and a
proposal often leaves out what it consumes (a ketone reduced to CH₂ uses
2 H₂ and releases water; an ester hydrolysis uses water). Co-reactants come
from H₂, H₂O and O₂; byproducts from a list of common leaving groups. The balanced step then runs as
`mepd run --reaction "A.B>>P.HCl" --recursive --use-tsopt --irc --minimize-ends`
in `verify/step_k`. SLAPMapper maps the atoms, as for any reaction SMILES.

The barrier is read exactly as the web UI reads any TS job: it counts as
verified only when IRCs connect start to end. Routes with every step checked
are ranked by their highest barrier. A step that can't be balanced with a
known byproduct is reported but not verified. Without `--verify`, the steps
are proposed reactions in Explore, ready for "Find TS".

## Data and installs

Data is fetched on first use into `MEPD_RETRO_DATA` (default
`~/.cache/mepd/retro`) and checked against Zenodo's MD5.

AiZynthFinder gets its own environment, `aizynth-env`. It needs Python < 3.13
and numpy < 2, because its RDKit is built against numpy 1 and segfaults
under numpy 2. Setup runs uv with `--no-config`, so mepd's own numpy
override doesn't leak into that environment.

Not included, and why:
- LocalRetro and RetroBridge: non-commercial licenses.
- SynLlama: academic-only license.
- ASKCOS: needs 32 GB RAM and a Docker stack.
- ether0 and RetroAgent: GPU-sized models.
- SynPlanner: LGPL chemistry core; deferred.
- RetroChimera: Python 3.9 + CUDA environment; a server-side option for later.

Citations: `mepd.retro.REFERENCES` (also in summary.json and the web UI's
References tab).
