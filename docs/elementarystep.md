# Elementary-Step Checks

Use the `elementarystep` extra when you already have a path as a multi-frame XYZ file and only want to determine whether that path is a single elementary step. This install path avoids the full NEB workflow dependencies while keeping the qcop/geomeTRIC pieces needed for geometry optimizations during the report.

## Install

From a local checkout or worktree:

```bash
uv add "/path/to/mepd[elementarystep]"
```

To test the current refactor branch without changing an existing checkout:

```bash
git fetch origin
git clone https://github.com/mtzgroup/mepd.git /tmp/mepd-elementarystep
cd /tmp/mepd-elementarystep
uv add "$PWD[elementarystep]"
```

The extra intentionally installs `qcop`, `geometric`, `qcio`, molecular graph tools, and numerical dependencies. It should not install ChemCloud, FairChem, Typer, or the full `mepd` application stack.

You also need a working external program on `PATH` for qcop. The default is `crest`:

```bash
command -v crest
crest --help
```

## Quick Check

Given a multi-frame chain:

```bash
mepd-elementarystep path.xyz
```

Typical output:

```text
Checking if elementary step
is_elem_step=False
is_concave=True
splitting_criterion=maxima
number_grad_calls=2
new_structures_count=1
new_structures_xyz=path_new_structures.xyz
```

The exit code is:

| Code | Meaning |
|------|---------|
| `0` | The path was classified as elementary |
| `1` | The path was classified as non-elementary |
| `2` | The check could not complete because setup or calculation failed |

## Inputs

The command accepts a standard multi-frame XYZ file:

```bash
mepd-elementarystep /tmp/crest-neb.xyz
```

If sidecar files are present next to the XYZ, they are loaded automatically:

```text
crest-neb.xyz
crest-neb.energies
crest-neb.gradients
crest-neb_grad_shapes.txt
```

If sidecars are missing, the command computes the initial chain energies and gradients with qcop before running the elementary-step check.

Use `--cached-only` when you want to fail rather than launch new calculations:

```bash
mepd-elementarystep /tmp/crest-neb.xyz --cached-only
```

## Backend Options

Defaults:

```bash
mepd-elementarystep path.xyz \
  --program crest \
  --geometry-optimizer geometric \
  --method gfn2 \
  --basis gfn2
```

Use `--program` for the qcop program used for energies/gradients, and `--geometry-optimizer` for the qcop program used to optimize candidate minima or pseudo-IRC endpoints.

## New Structures

When a path is not elementary, `mepd-elementarystep` reports the newly discovered structures explicitly.

By default, new structures are written as a multi-frame XYZ:

```text
new_structures_count=2
new_structures_xyz=crest-neb_new_structures.xyz
```

Override the output path:

```bash
mepd-elementarystep /tmp/crest-neb.xyz \
  --new-structures-out /tmp/discovered_intermediates.xyz
```

Suppress writing:

```bash
mepd-elementarystep /tmp/crest-neb.xyz --no-write-new-structures
```

## Python API

For scripts, call the lightweight entry point directly:

```python
from mepd.elementarystep import check_cached_xyz_elem_step

result = check_cached_xyz_elem_step("/tmp/crest-neb.xyz")

print(result.is_elem_step)
print(result.splitting_criterion)
print(result.number_grad_calls)

for node in result.new_structures:
    print(node.structure.to_xyz())
```

The `new_structures` field contains structures that are not identical to either endpoint. This is the preferred API for downstream workflow integration.

## Workflow Integrations

### Bash Gate

Use the exit code to split a workflow into elementary and non-elementary cases:

```bash
set +e
mepd-elementarystep /tmp/crest-neb.xyz \
  --new-structures-out /tmp/new_structures.xyz
status=$?
set -e

case "$status" in
  0)
    echo "Path is elementary"
    ;;
  1)
    echo "Path is not elementary; see /tmp/new_structures.xyz"
    ;;
  2)
    echo "Elementary-step check failed" >&2
    exit 2
    ;;
esac
```

### Python Driver

```python
from pathlib import Path

from mepd.elementarystep import check_cached_xyz_elem_step

path = Path("/tmp/crest-neb.xyz")
result = check_cached_xyz_elem_step(path)

if result.is_elem_step:
    print(f"{path} is elementary")
else:
    out = path.with_name(f"{path.stem}_new_structures.xyz")
    with out.open("w", encoding="utf-8") as handle:
        for node in result.new_structures:
            handle.write(node.structure.to_xyz().rstrip() + "\n")
    print(f"{path} is not elementary; wrote {out}")
```

### Snakemake

```python
rule elementary_step_check:
    input:
        xyz="paths/{name}.xyz"
    output:
        new="paths/{name}_new_structures.xyz",
        log="paths/{name}_elementarystep.log"
    shell:
        r"""
        set +e
        mepd-elementarystep {input.xyz} \
          --new-structures-out {output.new} > {output.log} 2>&1
        status=$?
        set -e
        if [ "$status" -eq 2 ]; then
          cat {output.log}
          exit 2
        fi
        touch {output.new}
        """
```

### CI/Remote Sanity Test

The branch includes a script that creates a fresh uv project, installs only the `elementarystep` extra, verifies imports, and runs the command:

```bash
XYZ_PATH=/tmp/crest-neb.xyz \
bash scripts/test_elementarystep_extra_install.sh
```

Set `KEEP_WORKDIR=1` to inspect the temporary environment after the run:

```bash
KEEP_WORKDIR=1 XYZ_PATH=/tmp/crest-neb.xyz \
bash scripts/test_elementarystep_extra_install.sh
```
