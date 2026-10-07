# QM/MM in mepd

A QM/MM system is a fixed list of atoms, such as a molecule in a shell of
explicit solvent or an active site in a protein. A few of its atoms (the
**QM region**) are computed at the profile's level of theory, with any mepd
engine (g-xTB, an MLIP, an ASE calculator...). The rest (the **environment**)
is computed with a force field. Every mepd command runs on a QM/MM system:
endpoint minimization, NEB/FNEB/GSM, TS optimization, IRC, Hessians and
reaction network expansion.

This replaces neb-dynamics' experimental `QMMMEngine`. That engine needed
TeraChem for both levels; this one does not. TeraChem's own QM/MM is kept as
an option for when TeraChem is available (see below).

## The model

* **Embedding.** Two low-level choices:
  * `mm = "gfnff"` (default), `"gfn2"` or `"gfn1"` use the subtractive
    (ONIOM) scheme, `E = E_QM(model) + E_low(real) - E_low(model)`, with the
    low level run by the xtb program. GFN-FF needs no parameters, so any
    structure can be embedded as it is.
  * `mm = "amber"` uses the additive scheme,
    `E = E_QM(model) + E_MM(everything except QM-QM terms)`, through OpenMM,
    from a prmtop (or a PDB of standard residues plus force-field XMLs).

  Both are *mechanical* embeddings. The QM region feels the environment's
  sterics, dispersion and the low level's charges, but its electrons are not
  polarized by them.
* **Link atoms.** A covalent bond cut by the boundary is capped with a
  hydrogen on the bond, at the QM atom's C–H distance. Its force goes back
  to the two atoms of the cut bond. Prefer to cut C–C bonds: polar cuts,
  cut X–H bonds and several cuts on one atom are reported as problems.
* **Frozen atoms.** QM/MM optimizations move the QM atoms and a shell of
  the environment (`active_radius`, Å; small molecules such as waters are
  taken whole). The rest stays exactly where it is. Frozen atoms are held
  in every optimization (ASE `FixAtoms`), in every path (their gradient is
  zero, geodesic interpolation keeps them fixed, and the ends are never
  rotated onto each other), and they are left out of Hessians. Hessians are
  partial: they are taken over the QM atoms only (partial Hessian
  vibrational analysis).
* **Species identity.** Molecular graphs, elementary-step checks and SMILES
  use the QM region only (capped with its link hydrogens). A water that
  moves does not count as a reaction.
* **One surface for the whole network.** GFN-FF's bonded terms are taken
  from the system's *reference* geometry (the system as set up), for every
  structure. Bonds breaking in the QM region therefore never switch
  force-field terms mid-path, and every species is on the same surface.

Frozen atoms also work without QM/MM: `chain_inputs.frozen_atom_indices`
wraps any engine in the same `FrozenAtomsEngine`.

## Command line

```bash
# A solute in explicit water; the solute is the QM region.
mepd qmmm build "NCC(=O)O" --solvent water --shell 6 --active-radius 5 -o gly_water

# A region on your own structure (xyz or PDB).
mepd qmmm region system.pdb --qm "0-23" --qm-charge -1 --active-radius 8 -o site

# A gas-phase reaction in solvent: the reactant is solvated, the product
# (and a TS) are put into that same solvent shell (same atom order).
mepd qmmm reaction --start reactant.xyz --end product.xyz --ts ts.xyz --solvent water -o rx
mepd optimize rx/system.xyz rx/product.xyz -i rx/qmmm_profile.toml -H
mepd ts --guess rx/ts.xyz -i rx/qmmm_profile.toml --irc

# Another geometry of the solute (a product, a TS) into an existing system.
mepd qmmm embed product.xyz --system gly_water -o emb      # --into: a minimized system structure

# An old neb-dynamics / TeraChem input (tc.in + prmtop + rst7 + qmindices).
# By default it runs now, with AMBER through OpenMM and the QM region at the
# profile's level; --mm terachem keeps TeraChem's own QM/MM.
mepd qmmm from-tc path/to/tc.in -o converted

# Any command, through the profile written next to the region:
mepd optimize gly_water/system.xyz -i gly_water/qmmm_profile.toml
mepd run --start a.xyz --end b.xyz -i gly_water/qmmm_profile.toml
mepd discovery expand gly_water/system.xyz -i gly_water/qmmm_profile.toml --allow-zwitterions

# Check a structure or path, and split its energy into QM and environment.
mepd qmmm inspect gly_water/region.json path.xyz -i gly_water/qmmm_profile.toml -o report.json
```

The profile carries the region in a `[qmmm]` table. Files named in it are
relative to the profile:

```toml
engine_name = "gxtb"        # the QM level
[qmmm]
file = "region.json"        # or: reference = "system.xyz", qm_atoms = "0-11", active_radius = 6.0, mm = "gfnff"
```

`[qmmm]` and `[solvation]` do not combine: put explicit solvent in the
environment instead.

## Taking a gas-phase reaction into solvent

Both ends must be one QM/MM system: the same solvent molecules, in the
same order, only the solute's geometry differing. Solvating the two ends
separately would give two unrelated solvent shells, and a path between them
would mostly move solvent. So the start is solvated once and every other
geometry of the solute (the end, a TS) is *embedded*: aligned onto the
solute where it sits in the system and moved out of contact with the
solvent, keeping its own geometry (a TS keeps its partial bonds), the
solvent left exactly where it is (`mepd.qmmm_build.embed`). The region
records which atoms are the solute (`solute_atoms`), so this works after
the QM region was grown to include solvent molecules too.

For a known reaction, re-optimizing the embedded gas-phase TS (with its
IRC) is the most direct route; a path search between the solvated ends
works as well but, like in the gas phase, FNEB can overshoot the barrier:
optimize the TS from its result. Checked on acetaldehyde → vinyl alcohol
(g-xTB QM, GFN-FF water): gas-phase barrier 74.9 kcal/mol; the embedded
TS re-optimized in water, IRC connecting both solvated ends, 77.1 kcal/mol.

### From a converged path to a TS

`mepd run` on a QM/MM system first matches the end's solvent molecules to
the start's and marches the solvent along the interpolated QM path
(`mepd.qmmm_path`). `mepd ts --guess path.xyz` (a path's frames, with its
`.energies`) starts from the interpolated energy maximum. With
microiterations (an OpenMM environment), the TS search then computes an exact
Hessian of the QM region there, by finite differences of the QM/MM gradient
with the environment held (6 gradients per QM atom), and starts Sella on it.
Of its negative modes it climbs the one along the path's tangent; any others
are made positive. Sella's model Hessian with only the tangent as a first
guess lost the reaction mode within a few steps on the environment-relaxed
surface and slid into a minimum. Pass `exact_hessian: false` in the TS
keywords to skip the Hessian.

Checked on Menshutkin NMe3 + CH3Cl in 56 moving TIP3P waters: B3LYP/def2-SVP
(Psi4) TS C–N 2.13 / C–Cl 2.23 Å, 15.0 kcal/mol above the IRC's reactant end
(gas phase 33.6), 33 min for TS and IRC; GFN2-xTB (`engine_name = "xtb"`)
the whole `mepd run --use-tsopt --irc` in 3 min, TS C–N 2.04 / C–Cl 2.12 Å,
9.4 kcal/mol. Both IRCs connect the path's reactant and product.

When a path search's output goes to a file instead of a terminal, every step
is logged as one line: step, gradients, the peak (kcal/mol, image) and the
end energy relative to the start, and the time the step took.

## Checking that a result is not nonsense

`mepd qmmm inspect` (and the web UI's *QM/MM checks*) report, frame by frame:

| check | should be |
|---|---|
| frozen atoms moved | 0 Å |
| bonds made or broken in the MM region | none (the force field cannot describe a reaction) |
| cut-bond stretch | near 1× the covalent length |
| closest non-bonded QM–MM contact | not a clash (≳ 1.5 Å) |
| QM / environment RMSD from frame 0 | where the motion is |

With `-i`, each frame's energy is also split into the QM region and its
environment. This shows whether a barrier comes from the reaction itself or
from the solvent or protein around it.

## Web UI

* **Put in explicit solvent (QM/MM)**: an operation on any molecule. It
  builds the solvent shell, adds the system to Explore as a QM/MM node, and
  minimizes it embedded.
* **Model in solvent (QM/MM)**, on a gas-phase edge or two gas-phase ends
  (same atoms, same order): the start is solvated, the end put into that
  same solvent shell, both minimized embedded and joined by a QM/MM edge.
  If the edge has an IRC-verified TS, it is put in too and re-optimized as a
  TS in the solvent, with its IRC.
* **Put into this QM/MM system**, on a QM/MM structure plus a gas-phase
  structure of its solute (e.g. a product or a TS from the gas phase): the
  structure goes into that system's solvent in place of the solute; a
  minimum is minimized embedded and joined to the QM/MM structure by an
  edge, a TS is re-optimized as a TS with its IRC.
* **Add › QM/MM system**: upload a whole system (XYZ/PDB) with its QM atoms,
  or point to a TeraChem input on the server.
* **Define a QM region (QM/MM)** (Inspector, any structure) and **Edit the QM
  region** (QM/MM structures): click atoms in 3D to add or remove them, and
  set the QM charge, spin, moving shell and environment level. The cut
  bonds, moving and frozen counts and any problems update as you click.
* A QM/MM node is named by its QM region ("NCC(=O)O · in water"). Only the
  operations that run embedded are offered (optimize, TS search, TS
  optimization, Hessian sampling, network expansion, paths between
  isomers). Its energies are at their own level ("… / QM/MM", keyed by the
  region), never compared with gas-phase ones or with another region's.
* **3D views** of a QM/MM system, wherever one is shown (Explore, results,
  live paths, optimization trees), draw the QM atoms as balls and sticks,
  cut bonds as dashed orange bonds with their link hydrogens, the moving
  environment as sticks and the frozen environment as grey lines. You can
  show all, the moving part only, or the QM atoms only. *Motion* colours the
  environment by how far it moved from the first frame; a frozen atom that
  moved turns magenta.
* **QM/MM checks** under every result path: the table above, per-frame
  curves, and *Split energies: QM vs environment*.

## Limits

* Electrostatic embedding (the environment's charges polarizing the QM
  region) needs fixed environment charges (TIP3P water, AMBER) and a QM
  level that takes point charges: `engine_name = "xtb"` (GFN2-/GFN1-xTB,
  xtb's own embedding, ~0.05 s per gradient for an 18-atom QM region in
  ~320 water charges) or `"psi4"` (any Psi4 method; B3LYP/def2-SVP ~8 s).
  g-xTB and the GFN-FF environment are mechanical embedding only, so
  charged or zwitterionic QM regions in polar environments are described
  less well with them.
* The boundary is fixed: atoms do not move between the QM and MM regions
  (no adaptive QM/MM).
* Implicit solvent cannot be combined with QM/MM.
* Network expansion's whole-molecule generators (CREST msreact, the
  nanoreactor, an imported generator, a products file of the QM model) run
  on the QM model:
  the QM atoms capped with their link hydrogens, at the QM charge and spin.
  Each product is aligned back onto the QM atoms, relaxed briefly against
  the nearby environment and put into the system, which the QM/MM
  optimization then relaxes. A product that breaks a cut bond (its link
  hydrogen no longer caps its QM atom) is dropped, and the run says so.
* Not offered on QM/MM systems (they need whole molecules): reaction
  channels (conformers, atom mappings), conformers, complexes,
  retrosynthesis, VRI, solvent/substituent follow-ups. The Nanoreactor
  operation (several molecules, reactions cut out with only the molecules
  they need) is replaced on a QM/MM system by the `nanoreactor` generator of
  network expansion: hot piston MD of the capped QM region in vacuum; its
  state after each reaction event goes back into the environment. The
  environment is not present during that MD unless `embedded=true` (web:
  "MD with QM/MM forces"), which runs the hot MD with QM/MM forces and the
  solvent held near room temperature; see network-expansion.md.
* TeraChem's QM/MM (`mm = "terachem"`) is ported but untested here, since
  no TeraChem is available. Its optimizations, TS searches and IRCs run
  through ASE/Sella on TeraChem gradients.
