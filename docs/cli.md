# CLI Reference

The public CLI exposes path minimization, NEB refinement, visualization, and
input-generation utilities.

## `mepd run`

Run a path minimization from endpoints or a prebuilt path.

```bash
mepd run --start start.xyz --end end.xyz --inputs inputs.toml
```

## `mepd run-refine`

Run cheap discovery followed by expensive NEB refinement.

```bash
mepd run-refine examples/oxycope.xyz -i expensive.toml -ci cheap.toml --mode neb
```

## `mepd refine`

Refine an existing NEB/MSMEP/chain/network result with expensive inputs.

```bash
mepd refine previous_result.xyz --inputs expensive.toml --mode neb
```

## `mepd make-default-inputs`

Write a default `RunInputs` TOML file.

```bash
mepd make-default-inputs --name inputs.toml
```

Use `--overwrite` to replace an existing file.

## `mepd version`

Print the installed package version.

```bash
mepd version
```

## `mepd-elementarystep`

Run the elementary-step command-line entry point.

```bash
mepd-elementarystep path.xyz
```
