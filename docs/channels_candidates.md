# How `mepd channels` generates path-search candidates

`mepd channels` turns one reactant/product pair into a set of path searches
(MSMEP), then optimizes and classifies every TS those searches find. This
page describes exactly how that set is built, with the defaults, and states
the assumption each step rests on, so it's clear where the method should and
shouldn't be trusted.

Default command:

```
mepd channels --start <R> --end <P> -i inputs.toml --workers N
```

The defaults are: `--atom-mapping` on, `--backend rdkit`, `--n-conformers 0`
(no cap), `--n-embed 0` (automatic), `--max-pairs 0` (no cap),
`--pairs-per-mechanism 3`, `--rmsd-cutoff 0.5` bohr. Every run writes
`stats.json` (how many candidates survive each stage, and the time spent in
each) and `pair_mechanisms.json` (what each path search is).

## 1. Endpoints

A SMILES endpoint is embedded (RDKit, with openbabel as a fallback for
multi-fragment sides) and minimized with the engine (`--minimize-ends`). The
product's atom order is then set from SLAPMapper by a single geodesic-scored
choice (`_check_endpoint_atom_mapping`). **That choice only fixes a
consistent atom ordering to start from; it doesn't restrict the
mechanisms** (step 6 re-maps every pair).

## 2. Conformer generation (per endpoint)

The input geometry is always the first conformer.

**RDKit** (`--backend rdkit`)
- ETKDGv3 embeddings. The number is set from the rotatable-bond count
  (Ebejer et al. 2012): 50 for up to 7, 200 for 8–12, 300 above.
- **Two batches of that size, pooled** (`--rdkit-torsion-prefs both`): one
  with ETKDG's experimental torsion preferences and one without (plain
  distance geometry). The preferences are crystal-structure statistics and
  all but forbid conformers that are rare in crystals but matter for
  reactions. Butadiene comes out s-trans in 300 of 300 embeddings with them,
  so a Diels–Alder never got an s-cis diene. Without them, the same 300
  include 58 s-cis. The rougher geometries are cleaned up by MMFF and the
  engine minimization.
- RDKit's seed is `--random-seed + 1`. With a seed of exactly 0, RDKit gives
  every conformer the same geometry.
- Each embedding is relaxed with MMFF94. An optional MMFF energy window is
  available (`--rdkit-ewin`, off by default).
- **Multi-fragment sides (complexes): each fragment's sampled conformation
  is Kabsch-aligned back onto that fragment's position in the input.**
  ETKDG otherwise stacks the fragments on top of each other.
- *Limitations:*
  - RDKit samples each fragment's internal conformation only, **never how the
    fragments are arranged relative to each other.** The input arrangement is
    the only one tried.
  - Where MMFF has no parameters (radicals, unusual valences), embeddings stay
    unrelaxed.

**CREST** (`--backend crest`)
- iterative metadynamics at GFN2-xTB (`--crest-method`), with a
  6 kcal/mol window (`--crest-ewin`);
- single-threaded: `-T` and `OMP_NUM_THREADS` are pinned together, since
  CREST scales poorly;
- CREST's own topology check drops frames whose bonding changed, and frames
  with a different atom order are dropped;
- CREST also samples the arrangement of fragments in a complex.
- **For an endpoint made of more than one molecule, CREST runs in NCI mode**
  (`--nci`, an ellipsoidal wall around the complex; `--no-crest-nci` turns
  it off). Without it, a loosely bound complex comes apart during the
  metadynamics. Butadiene + ethylene gave 764 "conformers", 744 of them the
  two molecules drifted 5–72 Å apart at the same energy, and so 1,610 pairs.
  With it: 60 frames, all in contact.
- *Limitations:*
  - It costs minutes per endpoint, against well under a second for RDKit.
  - It isn't deterministic, so pools change from run to run.
  - It samples at GFN2-xTB, not at the engine's level; step 3 re-minimizes
    everything.

## 3. Deduplication, twice

1. **snap-RMSD** (`qcinf.snap_rmsd`): permutation-aware and rotationally
   aligned. A conformer is kept only if it is at least `--rmsd-cutoff`
   (0.5 bohr) from every conformer already kept, walking from lowest
   energy up. A candidate that doesn't perceive as the same molecule is
   rejected, and the count goes to `n_rejected_not_isomorphic` in
   `stats.json`.
2. **Engine minimization, then snap-RMSD again**, because distinct
   starting geometries often relax into the same minimum.

3. **Complexes only, compared molecule by molecule** (`--complex-energy-tol`,
   default 0.5 kcal/mol; 0 turns it off). Two conformers of a multi-molecule
   endpoint count as the same when every molecule's own conformation matches
   (snap-RMSD per fragment) and the engine energies agree within the
   tolerance. The lowest-energy one is kept. All-atom snap-RMSD can't do
   this: moving one molecule 0.5 bohr further away already makes a "new"
   conformer, though neither molecule changed and the energy barely did.
   Arrangements that differ in energy by more than the tolerance are kept,
   because the reactive approach geometry is usually one of those. With both
   this and NCI mode, butadiene + ethylene gives 8 reactant conformers (16
   pairs), all in contact, including 3 with s-cis butadiene.

*Limitations:*
- snap-RMSD counts hydrogens and only removes near-exact duplicates.
  Conformers that differ only in how an end group is rotated survive as
  distinct.
- The complex rule deliberately collapses arrangements that are
  energy-degenerate. If two arrangements with the same energy lead to
  different chemistry, only one of them is searched.

## 4. Mirror images: merged in one pool only

For an achiral molecule, mirror-image conformers (gauche+/gauche−) have
equal energies and give mirror-image paths, but snap-RMSD (rotation only)
doesn't see them as the same.

They are merged in **exactly one** of the two pools: whichever leaves fewer
pairs. With reactant conformers {R, R\*} and product conformers {P, P\*},
(R, P) mirrors (R\*, P\*), and (R, P\*) mirrors (R\*, P). Those are two
genuinely different combinations, and merging in both pools would keep only
one of them.

For a molecule with stereocentres, a conformer can't match another's mirror
image unless it is the enantiomer, so this step can't merge a genuine
conformer away.

## 5. Pairing

Every reactant conformer is paired with every product conformer
(`--max-pairs 0`). Pairs aren't searched yet; step 7 decides which are.

## 6. Mechanisms per pair (SLAPMapper)

For each conformer pair (`select_per_mechanism`, parallel over `--workers`):

1. **SLAPMapper with `binary=True`** gives the correspondences tied at the
   minimum cost. Bonds are perceived from the geometry by openbabel, and **bond
   orders are ignored.**
2. The results are grouped into **mechanisms** (`mechanism_key`): the bonds
   broken and formed, with each atom named by its element and its graph
   symmetry class. Relabeled copies, and the same mechanism in another pair,
   therefore get the same key.
3. Each mechanism is **fully expanded** into every relabeling of its
   symmetry-equivalent atoms: the cross product over the equivalent-hydrogen
   groups (`expand_mapping_fully`), capped at `--atom-mapping-candidates`
   (200) per mechanism.
4. Every variant is scored against **this pair's geometry** with
   `--atom-mapping-metric` (default: geodesic path length). The best variant
   **within each mechanism** is kept. A bad hydrogen assignment can force a
   CH₂ to rotate along the path, so this is decided per pair. **Scores never
   choose between mechanisms.**
5. The pair's current ordering is scored as a variant of whichever mechanism
   it implies. If it isn't one of SLAPMapper's mechanisms, it becomes a
   mechanism of its own.

*Assumptions and limits:*
- Only **minimal-cost mappings under adjacency** are proposed.
  - A mechanism that needs more bond edits than the minimum is **never
    tried**, e.g. an extra proton shuttle, or a route that only wins once
    bond orders count.
  - The [3,3] Claisen is found only because ignoring bond orders makes it tie
    with the [1,3] shift.
- The cap of 200 variants truncates the cross product for systems with many
  equivalent groups (three methyls already give 6³ = 216).
- **Cost: scoring every variant of every mechanism for every pair can
  dominate a run.** Reaction-QM `RXN_0000104508` (19 atoms, 9 × 21 conformers)
  spent **3,709 s mapping 189 pairs** (3 workers) to launch 3 path searches,
  about 60 s per pair, because its hydrogen groups multiply to the 200-variant
  cap and each variant is a full geodesic interpolation. This is an open
  problem, and that reaction is a test case for it.
  - *Rejected:* pre-ranking variants by aligned endpoint RMSD and
    interpolating only the best few. It was about 30× faster, but it discards
    variants before they are scored, and so can discard the labeling that
    leads to the right path. A replacement has to keep every variant in play,
    or prove the ones it drops can't win.
- Mechanisms are told apart by openbabel's bonds between the two endpoint
  minima. Mechanisms that break and form the same bonds but differ in
  stereochemistry (e.g. conrotatory versus disrotatory) get the **same key**.

## 7. Pair subselection

For each mechanism, only the `--pairs-per-mechanism` pairs (default **3**)
with the best step-6 scores get a path search. Every mechanism is tried,
from the pairs whose geometry suits it best.

*Limitation:* geodesic length rewards least motion, and the shortest-path
pairs **aren't necessarily the ones leading to the lowest TS**. On the
Claisen, the 3 best [3,3] pairs all found the boat TS (37.7 kcal/mol). The
chair (32.8) was reached, but only by one of the [1,3]-mapped searches. The
value of K has only been checked against the every-pair run on that one
system. Raise K, or use 0 (every pair × every mechanism), when a missed
low channel would matter.

## 8–9. Path search, TS optimization, classification

- One MSMEP per selected (pair, mechanism), run `--workers` at a time.
- A pair whose root NEB fails is skipped instead of stopping the run.
- Every leaf is TS-optimized and followed by an IRC, also in parallel. A
  TS's classification comes from the **connectivity of its IRC endpoints,
  not from the mapping that seeded it**:
  - `channels/`: a single step from reactant to product;
  - `alternate-channels/`: multistep routes;
  - `offtarget-exit-channels/`: steps from which the product is never
    reached.
- **The same TS found more than once** is merged when its energy agrees
  within `node_ene_thre` (1 kcal/mol) and its snap-RMSD is below
  `node_rms_thre` (1 bohr), also checked against its mirror image. If
  snap-RMSD can't perceive the TS's half-formed bonds, the fallback is the
  sorted interatomic-distance list (within 0.1 bohr).
- Species are compared by **connectivity and stereochemistry**
  (`_connectivity_matches`), so cis/trans and E/Z isomers are different
  species. A TS from cis-3,4-dimethylcyclobutene to the (E,Z) diene is a
  channel; a TS from the *trans* isomer to the (E,E) diene, found while
  aiming for (E,E), is an off-target exit. Only the step-6 mechanism key is
  stereo-blind.
- Stereo SMILES can't express E/Z for a double bond inside a ring of 3–7
  atoms, so species comparison also requires the same number of **trans or
  twisted small-ring alkenes**: ring C–C=C–C dihedral above 60°, measured
  from each structure's own geometry. Without that check, an antarafacial
  Diels–Alder TS leading to trans-cyclohexene (ring dihedral about 90°,
  54 kcal/mol above cyclohexene) was classified as a channel to ordinary
  cyclohexene.

## Level of theory: closed-shell only

Every run uses one fixed charge and multiplicity (singlet, restricted, for
all the examples). Routes through open-shell intermediates don't exist on
that surface, so no amount of candidate generation will find them. An
example is the stepwise diradical Diels–Alder, through hex-1-ene-3,6-diyl.
What does get found, the concerted [4+2], and a stepwise [2+2] to
vinylcyclobutane followed by a rearrangement, is only the closed-shell part
of the picture.

## 10. Barriers

Every barrier is measured from **the lowest reactant-side structure found
anywhere**: sampled conformers, IRC endpoints, and separated fragments. When
comparing runs (for example different backends), use the lowest over all of
them. A barrier measured from the conformer a search happened to start from
can be made to look small by starting from a strained conformer.

## Validation so far (g-xTB)

| system | channels found | cost |
|---|---|---|
| Claisen, allyl vinyl ether → pent-4-enal (RDKit, K=3) | chair 32.8, boat 37.7, [1,3] 69.8 kcal/mol | 6 path searches, 153 s on 12 workers |
| Methyl acetate + water → acetic acid + methanol | lowest 51.3 kcal/mol, same for both backends | — |
