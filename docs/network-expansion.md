# Reaction network expansion

`mepd discovery expand` proposes the products of a structure by rewriting its
bond graph instead of sampling Hessians, optimizes them, and repeats from each
new species. With `--connect` it then runs a recursive path search for every
proposed reaction and builds the network from the paths it finds. In the web
UI this is the **Reaction network expansion** operation for a selected
structure.

```bash
mepd discovery expand "C=CCOC=C" -i gxtb.toml                 # one round: products of the seed
mepd discovery expand seed.xyz -i gxtb.toml --rounds 2 -H     # products of the products, Hessian-checked
mepd discovery expand seed.xyz -i gxtb.toml --connect         # then run a path search per reaction
```

## How products are proposed (`--generator bond-rules`, built in)

1. **Enumerate.** Every combination of up to `--n-break` (2) bond breaks and
   `--n-form` (2) bond formations is tried. A bond can only form between
   atoms that are at most `--form-distance` (4 Å) apart in the source.
2. **Filter.** Each atom must keep a normal number of bonded neighbours
   (C 1–4, N 1–4, O 1–2, H 1, halogens 0–1, …). The new graph must also have
   a Lewis structure with the system's total charge (RDKit
   `DetermineBondOrders`).
   - By default the product must be closed-shell, with exactly
     `multiplicity − 1` unpaired electrons. `--allow-radicals` also admits
     carbenes and diradicals.
   - By default products with separated formal charges are rejected.
     `--allow-zwitterions` admits them; that is needed for isocyanides
     (HCN → HNC) and CO.
3. **Deduplicate and rank.** Products are deduplicated by canonical SMILES
   and ranked with the fewest bond changes first, then by how close the
   atoms that form bonds are. At most `--max-products` (50) are kept for
   each source species.
4. **Build a guess.** Each product's 3D guess is built from the source
   geometry by a restrained relaxation onto the new bonds. It keeps the
   source's atom order, so it can go straight into a path search. Randomly
   kicked restarts get it out of symmetric traps; for example, an H
   migrating along a linear molecule has to go around the middle atom.
5. **Optimize.** The guess is optimized at the level of theory in `--inputs`.
6. **Classify.** Each optimized guess is recorded as one of:
   - a new species;
   - a known species, merged with it (the lowest conformer is kept);
   - `reverted` to its source;
   - `not_minimum`: it failed the Hessian check (`-H`).

   Species identity is connectivity plus stereochemistry.

Species within `--energy-window` (60 kcal/mol) of the seed are expanded in
the next round. If a later proposal matches a species already found, it is
recorded as an edge to that species without being optimized again.

## Other generators

These options plug in in place of the built-in generator:

- **Your own code:** `--generator package.module:function`. It is called as
  `function(structure, **options)`, with options from repeatable
  `--generator-option key=value` (values parsed as JSON). It must return
  product `qcdata.Structure`s, or xyz paths, **in the source's atom
  order**.
- **Another tool's output:** `--products products.xyz` imports products that
  another tool (autodE, Chemoton, YARP, …) already proposed for the seed. The
  file is a multi-frame xyz, atom-mapped to the seed. mepd then optimizes,
  classifies, and (with `--connect`) connects them as above.

## Output

| file | contents |
|---|---|
| `species.xyz` | the seed first, then every species found (lowest conformer) |
| `species/species_<k>.xyz` | one file per species, e.g. to pass to `mepd network-splits` or `channels` |
| `proposals.xyz` | every product guess, before optimization |
| `rejected.xyz` | species that failed the Hessian check |
| `summary.json` | the species (SMILES, round, energy relative to the seed, Hessian record); one record per proposed reaction (source, target, bonds broken and formed, outcome, whether it landed on the proposed connectivity); rounds; `connections` |
| `pairs/`, `network.json` | with `--connect`: one MSMEP tree per reaction, and the network built from them (as in `mepd network-splits`) |

## Limits

- The enumeration is combinatorial. `--n-break 2 --n-form 2` on a molecule
  with 30 atoms reaches the `max_combinations` cap (500k); `stats.clipped` in
  `summary.json` reports when that happens. Lowering `--form-distance`, or
  using `--n-form 1`, keeps it small.
- The rules propose graph changes, not mechanisms. A proposal that optimizes
  back to its source says nothing about the barrier. `--connect` is what
  checks that a path (TS plus IRC) exists.
