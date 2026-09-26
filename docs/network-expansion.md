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

## Stopping the network from growing forever: flux steering

With `--rounds` > 1 the expansion is steered by kinetics by default
(`--steer auto`; `--steer window` restores the plain energy window). Every
round:

1. **Propose and optimize products** of the species chosen for expansion (as above).
2. **Find a TS for every new reaction.** mepd runs a recursive path search
   (MSMEP, with the profile's path method), then TS optimization and IRC for
   every step it splits into. A step counts only if its IRC runs between two
   different minima. IRC ends that are no known species are minimized and
   join the network as intermediates. Everything goes in `pairs/` and `ts/`.
3. **Simulate the kinetics.**
   - Eyring rates come from each verified TS:
     k = (k<sub>B</sub>T/h) exp(−ΔE<sup>‡</sup>/RT).
   - Barriers are electronic energies, measured from each species' lowest
     conformer found.
   - Reverse rates come from the same TS, so detailed balance holds.
   - The first-order network is integrated from pure seed for `--time`
     seconds at `--temperature`.
4. **Expand by flux.** Only species whose *concentration flux* (the total
   material that flowed into them, as a fraction of the seed) reaches
   `--flux-threshold` are expanded next round. When none does, the network
   has converged and the run stops, even if `--rounds` allows more.

```bash
mepd discovery expand seed.xyz -i gxtb.toml --rounds 5                        # 298 K, 1 h, threshold 0.01
mepd discovery expand seed.xyz -i gxtb.toml --rounds 5 --temperature 600 --time 10
```

`summary.json` records:
- `steering`: the settings;
- `steps`: each verified step, its barriers in both directions, and its TS/IRC files;
- each species' `flux` and its `max_concentration` / `final_concentration`;
- each round's `kinetics`.

Limits:
- Each proposed product is one species, a pair of fragments included, so
  every step is first order. There is no bimolecular recombination.
- Barriers are electronic, with no thermal or entropic corrections.
- Fast equilibria count their back-and-forth traffic as flux, so
  equilibrating species always pass the threshold, as they should.
- The cost is one path search plus TS/IRC per reaction; `--workers`
  parallelizes the TS/IRC stage and the pair searches.

## Methods and references

| stage | method | reference |
|---|---|---|
| Enumeration | break ≤ n and form ≤ m bonds on the molecular graph, filtered by coordination | ZStruct: P. M. Zimmerman, *J. Comput. Chem.* **34**, 1385–1392 (2013), [doi:10.1002/jcc.23271](https://doi.org/10.1002/jcc.23271); the "b2f2" enumeration of YARP: Q. Zhao, B. M. Savoie, *Nat. Comput. Sci.* **1**, 479–490 (2021), [doi:10.1038/s43588-021-00101-3](https://doi.org/10.1038/s43588-021-00101-3) |
| Lewis-structure filter | bond orders and formal charges from connectivity (RDKit `DetermineBondOrders`) | xyz2mol: Y. Kim, W. Y. Kim, *Bull. Korean Chem. Soc.* **36**, 1769–1777 (2015), [doi:10.1002/bkcs.10334](https://doi.org/10.1002/bkcs.10334) |
| Product guess geometry | restrained relaxation of the source geometry onto the product bonds | mepd's own heuristic, not a published method |
| Flux steering | Eyring rates from verified TSs, first-order microkinetics, expand by concentration flux | concentration-flux-steered exploration: M. Bensberg, M. Reiher, *Isr. J. Chem.* **63**, e202200123 (2023), [doi:10.1002/ijch.202200123](https://doi.org/10.1002/ijch.202200123); rate-based model enlargement (as in RMG): R. G. Susnow, A. M. Dean, W. H. Green, P. Peczak, L. J. Broadbelt, *J. Phys. Chem. A* **101**, 3731–3740 (1997), [doi:10.1021/jp9637690](https://doi.org/10.1021/jp9637690) |
| Live-view animation | geodesic interpolation from source to product | X. Zhu, K. C. Thompson, T. J. Martínez, *J. Chem. Phys.* **150**, 164103 (2019), [doi:10.1063/1.5090303](https://doi.org/10.1063/1.5090303) |

The same list is written to `summary.json` (`methods`) and shown with the
results in the web UI.

**What is ours and what isn't.** The expansion is built on these published
methods, and anything that uses or reports it should cite them. mepd
reimplements them; it contains no code from ZStruct, YARP or Chemoton. In
particular, the product enumeration is YARP's b2f2 scheme (Zhao & Savoie,
2021), built on the same break-and-form idea as ZStruct (Zimmerman, 2013). The flux-steered
growth is Bensberg & Reiher's concentration-flux criterion (2023), in the
spirit of RMG's rate-based enlargement (Susnow et al., 1997). What mepd
adds is the glue:
- the atom-mapped 3D guess;
- optimization and species merging;
- TS verification through its own path search, TS optimization and IRC;
- the live view.

**Compared with YARP's own filters.** mepd's enumeration only forms bonds
between atoms within `--form-distance` (4 Å), and it filters with a
coordination check and RDKit's Lewis-structure check instead of YARP's
Lewis-score and formal-charge filters. So the two propose somewhat different
sets. On 6-hydroxyhexanal (`OCCCCC=O`, b2f2), mepd proposes 48 species and
YARP 65, with 39 in common. YARP's extras include ring closures between
atoms that start more than 4 Å apart; a larger `--form-distance` brings those
in. mepd in turn keeps some products that YARP's filters drop.

## Other generators

Generators are registered in `mepd/discovery/generators.py`; `bond-rules` is
the built-in one. To add one (autodE, Chemoton, …), write a
`propose(symbols, coords, edges, **settings)` that returns
`(proposals, stats)` and register it in `GENERATORS`.

These plug in in place of the built-in generator:

- **Your own code:** `--generator package.module:function`. It is called as
  `function(structure, **options)`, with options from repeatable
  `--generator-option key=value` (values parsed as JSON). It must return
  product `qcdata.Structure`s, or xyz paths, **in the source's atom
  order**.
- **Another tool's output:** `--products products.xyz` imports products that
  another tool (autodE, Chemoton, …) already proposed for the seed. The
  file is a multi-frame xyz, atom-mapped to the seed. mepd then optimizes,
  classifies, and (with `--connect`) connects them as above.

## Output

| file | contents |
|---|---|
| `species.xyz` | the seed first, then every species found (lowest conformer) |
| `species/species_<k>.xyz` | one file per species, e.g. to pass to `mepd network-splits` or `channels` |
| `proposals.xyz` | every product guess, before optimization |
| `rejected.xyz` | species that failed the Hessian check |
| `summary.json` | the methods and references used; the species (SMILES, round, energy relative to the seed, Hessian record); one record per proposed reaction (source, target, bonds broken and formed, outcome, whether it landed on the proposed connectivity); rounds; `connections` |
| `pairs/`, `network.json` | with `--connect`: one MSMEP tree per reaction, and the network built from them (as in `mepd network-splits`) |

## Limits

- The enumeration is combinatorial. `--n-break 2 --n-form 2` on a molecule
  with 30 atoms reaches the `max_combinations` cap (500k); `stats.clipped` in
  `summary.json` reports when that happens. Lowering `--form-distance`, or
  using `--n-form 1`, keeps it small.
- The rules propose graph changes, not mechanisms. A proposal that optimizes
  back to its source says nothing about the barrier. `--connect` is what
  checks that a path (TS plus IRC) exists.
