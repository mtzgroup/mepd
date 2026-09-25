# Machine-learned potentials

A machine-learned potential (MLIP) is a level of theory like any other: set it
in the RunInputs profile and every mepd command (`run`, `channels`,
`optimize`, `discovery ...`) uses it.

```toml
engine_name = "mlip"

[mlip_engine_kwds]
model = "aimnet2-rxn"      # the only line to change to switch models
device = "cuda"            # optional; CUDA when available, else CPU
```

`mepd models` lists every name, whether it needs an account, whether its
package is installed, what charge/spin it handles, and which elements it
covers.

## Models that need no account

| model | family | notes |
|---|---|---|
| `aimnet2-rxn` | AIMNet2 | trained for reactions; H C N O; closed shell |
| `aimnet2`, `aimnet2-2025` | AIMNet2 | general organic and main-group molecules; closed shell |
| `aimnet2-nse` | AIMNet2 | explicit spin: radicals and open-shell species |
| `orb-v3-conservative-omol` | Orb | trained on OMol25 (the UMA training data) |
| `orb-v3-direct-omol` | Orb | faster, non-conservative forces |
| `ani-2x` | ANI | H C N O F S Cl; neutral closed shell only |
| `mace-off` | MACE | organic force field; neutral closed shell only |

The weights download on first use.

## Gated models (UMA, eSEN)

The FAIR-Chem models (`uma-s-1p2p1`, `uma-m-1p1`, `esen-...`) are gated on
Hugging Face. Using one without access stops with a message that says what to
do:

1. Open the model page (e.g. https://huggingface.co/facebook/UMA) while logged
   in, and request access. Approval can take a while.
2. Log in on this machine: `hf auth login`.

Until then, use one of the open models above.

## Installing

```bash
uv sync --inexact --extra mlip      # fairchem-core, AIMNet2, ANI
```

Orb and MACE pin versions that conflict with fairchem-core, so each goes in an
environment of its own:

```bash
UV_PROJECT_ENVIRONMENT=.venv-orb uv sync --extra orb --extra ase
UV_PROJECT_ENVIRONMENT=.venv-mace uv sync --extra mace --extra ase
```

## Your own model

A local checkpoint of a known family:

```toml
[mlip_engine_kwds]
model = "my-model"
family = "aimnet2"            # or "orb", "fairchem", "mace", "ani"
checkpoint = "/path/to/model.pt"
```

Anything with an ASE calculator, by import path (a class or a factory
function):

```toml
[mlip_engine_kwds]
calculator = "my_package.module:MyCalculator"
calculator_kwds = { some_option = 1 }
```

## Things to know

- **Charge and spin.** Each structure's charge and multiplicity reach the
  model through `atoms.info["charge"]` / `atoms.info["spin"]`. Models marked
  "closed shell" or "neutral only" in `mepd models` ignore them, so don't use
  those models for ions or radicals.
- **One process on the GPU.** A process that has put a model on the GPU can't
  fork, so mepd runs pairs serially there instead of across forked workers.
  The FAIR-Chem engine batches every image of a chain into one forward pass
  instead.
- **Hessians.** These are central differences of the model's own gradients.
