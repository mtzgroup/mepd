# Reaction conditions: solvents, temperature, force, substituents, light

mepd finds transition states. This part turns them into statements about
experiments: *would this run faster in DMSO?*, *what temperature does it
need?*, *what would the flask hold after two hours at 60 °C?*

Three pieces, from cheap to complete:

| What | Where | Cost |
|---|---|---|
| **Solvent effects** on a finished TS search: barriers in several implicit solvents, half-lives, the temperature each needs, plain-language insights | `mepd solvent`; web: a TS job's page › *Reaction conditions* | seconds (single points) to minutes per solvent (re-optimized) |
| **Setups in Explore**: solvent + temperature + reaction time + starting structures; every edge shows whether it runs; *Predict outcome* runs the graph forward in time | web: Explore › *Conditions* | instant (uses what is computed) |
| **A solvated level of theory**: every calculation of a profile in solvent | profile `[solvation]` table; web: Settings › *Solvent* | as the gas-phase level, plus two GFN2-xTB calls per gradient |

## The solvent model

GFN2-xTB carries the implicit solvent models: ALPB (default), GBSA and
CPCM-X. For any other engine (g-xTB, MLIPs, ASE calculators) mepd adds
GFN2-xTB's solvation free energy to that engine's energy, and the same
for the gradient:

    E_solv(x) = E_engine(x) + [E_GFN2+ALPB(x) − E_GFN2(x)]

This is a composite model, and the results say so. Because the correction
has a gradient, TS optimizations, IRCs and minimizations all run on the
solvated surface (`mepd.solvation.SolvatedEngine`). When the engine *is*
GFN2-xTB, its own solvent flag is used instead, and gives the same numbers.

g-xTB itself has no ALPB/GBSA parameters: xtb exits with *No ALPB/GBSA
parameters found*. Worse, with `--cpcmx` it silently returns the gas-phase
energy. `SolvationCorrection` therefore refuses to continue when the
solvent model leaves the energy unchanged.

A profile runs in solvent with:

```toml
[solvation]
solvent = "water"      # see mepd.solvation.SOLVENTS
model = "alpb"         # alpb | gbsa | cpcmx
method = "auto"        # auto (native for GFN2-xTB, else correction) | native | correction
```

The solvent is part of the level of theory (`workspace.effective_level`),
so solvated and gas-phase energies are never compared by accident.

Energies in solvent are E + ΔG_solv, taken in ALPB's reference state
(1 M gas to 1 M solution). They are not free energies: no thermal or
entropic terms are added, and rates from them are orders of magnitude.

## `mepd solvent`

```
mepd solvent RUN_OR_TS_OUTPUT --solvent water --solvent dmso --solvent hexane \
    [--mode single-point|reoptimize] [--model alpb] [--temperature 298.15] [-i profile.toml]
```

It reads a gas-phase `mepd run` or `mepd ts` output folder and writes
`<output>/<solvent>/`, with the same files as the source (MEP, TSs, IRCs)
and energies in that solvent. `summary.json` holds the barriers, their
shifts, half-lives, the temperature for a 1 h half-life, and insights.

* **single-point** (default, fast): every frame keeps its gas-phase
  geometry. The barrier is the highest *solvated* point along the
  gas-phase IRC, which need not be the gas-phase TS. The run always carries
  the warning that geometries were not re-optimized. It adds a *caution*
  when the shift exceeds 8 kcal/mol, or when the solvated peak sits away
  from the gas-phase TS; both mean the TS moves in solvent.
* **reoptimize**: the TS is searched again in each solvent. The search
  starts from the solvated peak along the gas-phase IRC (to first order,
  that is where the TS has moved), then from the gas-phase TS. The TS
  counts only if its IRC connects minima with the same bonds as the
  gas-phase IRC's ends. The path's ends are re-minimized too. The single
  points are kept in `single-point/` for comparison.

Barriers follow the barrier floor rule: one floor per phase, taken as the
lowest reactant-side point. A negative barrier is reported and explained,
never clamped.

**Insights** (`mepd.conditions.solvent_insights`) cover:
- solvents that accelerate the reaction, with the rate factor and how the
  half-life changes;
- solvents that slow it down;
- a polarity trend (rank correlation of the shift with the Kirkwood
  function (ε−1)/(2ε+1)), read as "the TS is more polar than the reactant";
- the temperature a 1 h half-life needs, checked against the solvent's
  boiling and melting points, with a higher-boiling solvent of the same
  kind suggested;
- reaction energies that change sign;
- a reminder that protic solvents can take part in a way implicit solvent
  cannot model: test that with an explicit molecule in Design.

### Check: Menshutkin reaction NH₃ + CH₃Cl → CH₃NH₃⁺ Cl⁻ (g-xTB)

Charges separate on the way to the TS, so this is the textbook
solvent-accelerated reaction.

| Medium | single points on the gas path | re-optimized in solvent |
|---|---|---|
| gas phase | 42.5 | 42.5 |
| water | 21.6 | 21.2 |
| DMSO | 24.8 | 25.1 |
| n-hexane | 30.3 | 30.3 |

All values are ΔE‡ in kcal/mol. The order follows polarity, as observed
for Menshutkin reactions. Taking the *peak* of the solvated single points
matters. The single-point energy at the gas-phase TS itself is 3.2
kcal/mol *below* the reactant in water. The late gas-phase TS is almost an
ion pair, which water over-stabilizes. mepd flags that case and does not
report it as a barrier. A naive re-optimization from the gas-phase TS
slid into the product basin and found a rotation saddle between two
ion-pair conformers. The connectivity check exists because of this.

## Setups in Explore (prototype)

A setup is a named set of conditions: medium (gas phase or a solvent),
temperature, reaction time, and the structures the experiment starts
from. It is stored in `workspace.json` (`setups`, `active_setup`). With a
setup shown:

* Every edge is labelled with its barrier under the setup and its
  half-life, and coloured green when it runs within the reaction time,
  amber within 100× that, grey otherwise. The barrier comes from a
  *Solvent effects* job when one exists (re-optimized results win over
  single points), else the gas-phase value, marked "(gas)". The phases
  are never mixed silently.
* *Compute (fast)* / *Re-optimize* queues *Solvent effects* on every edge
  that has a gas-phase TS but no value in the setup's solvent.
* *Predict outcome* integrates first-order kinetics over the graph
  (`mepd.conditions.simulate_first_order`), starting from the chosen
  structures. Rates are Eyring's, forward and reverse. The reverse barrier
  comes from the solvent job's reaction energy, or from the two nodes'
  energies when both are minima at one level. The output is what the flask
  holds at the end. Notes point out equilibria that lie on the reactant
  side; warnings list gas-phase fallbacks and one-way steps.

Every node is one species or one complex, as the path searches treat
them, so all steps are first order and concentrations do not enter the
rates.

## Mechanical force (`mepd force`; command line only for now)

Switched off in the web UI (2026-09-29) until its results are trusted: it is not offered, runs and resumes are refused, and its results and citations are not shown. `mepd force` still works on the command line.

A constant force F pulling atoms i and j apart adds −F·d_ij to the energy.
To first order the stationary points stay put, so each barrier shifts by
−F·Δq‡, where Δq‡ = d_ij(TS) − d_ij(reactant) (Bell). 1 nN·Å is
14.39 kcal/mol.

For every heavy-atom pair and every channel of the source (a `run`/`ts`
result, or all direct and off-target channels of a `channels` run), the
summary lists:
- **Levers:** the pairs that speed each channel up (Δq‡ > 0) and hold it
  back (Δq‡ < 0), with the barrier at 1 nN and the force for a 1 h
  half-life. Forces beyond 2.5 nN are called out of reach: beyond Bell's
  range and near bond rupture.
- **Selectivity switches:** the pair and the smallest force at which a
  slower channel overtakes the fastest one, with a 0.5 kcal/mol margin.
  When the fastest channel is off-target and a direct one can overtake it,
  that is reported as steering toward the intended product. The report
  also says when a switch works mainly by holding the leader back.

`--mode reoptimize` checks the main levers on the force-modified surface
itself, E − F·d (EFEI, `mepd.mechanochem.ForcedEngine`). At each force, the
reactant is re-minimized and the TS and its IRC re-optimized, and the IRC
must still connect the same minima.

It stacks with the solvent wrapper: both are `mepd.engines.modified.
ModifiedEngine` subclasses.

Checks at g-xTB:
- **Menshutkin, pulling C–Cl apart.** Bell gives −11 kcal/mol per nN.
  Re-optimized under force: 37.6 / 32.6 / 27.9 kcal/mol at 0.5 / 1 / 1.5 nN,
  against Bell's 37.0 / 31.4 / 25.9. The gap grows with force as the
  geometries relax, as extended Bell theory predicts.
- **A demo channels run** (ethene + methanol + water). The intended direct
  channel is 20.5 kcal/mol above an off-target exit without force. Bell
  says pulling C2–O4 apart with about 1.5 nN would make it win.

Pairs are ranked over all heavy atoms. In an experiment the force enters
through handles (polymer chains, a tether, a stiff-stilbene) at the pulled
atoms, so which pairs are practical is the chemist's call. Not built yet:
- the second-order (compliance) correction from Hessians;
- COGEF-style rupture forces;
- force as part of an Explore setup.

## Substituent effects (`mepd substituents`; web: *Substituent effects*)

Hydrogens of the existing reaction are swapped for functional groups one at
a time, and each variant's barrier is compared with the parent's.

- **Sites:** every hydrogen that stays on the same heavy atom at both IRC
  ends (a transferred H is never a site). Symmetry-equivalent H's count
  once. Sites are ordered by distance to the atoms whose bonds change.
- **Building a variant:** the group (from Design's list, embedded with
  RDKit) is attached along the old X–H bond at the covalent bond length. It
  is turned about that bond to stay clear of the rest, identically in the
  reactant end, the TS and the product end.
- **Fast mode (default):** only the new group is relaxed, then single points.
- **Re-optimize mode:** TS search and IRC, which must connect the same
  bonds, plus a minimized reactant. The parent goes through the same
  protocol, so the shift compares like with like.
- **Guards:**
  - a group that forms or breaks bonds while relaxing is left out as
    "reacted" (on a channels job, 17 of 54 variants: an F migrating onto
    the core gave a fake −73 kcal/mol);
  - clashes and negative barriers are left out and reported;
  - fast shifts of 8+ kcal/mol are marked "(check)".
- **Output:** a site × group table of shifts, a Hammett slope against σp
  per site, and, across channels, the variants that change which channel
  wins.

Checks at g-xTB, NH₃ + CH₃Cl. Fast and re-optimized shifts, where the
re-optimized TS is still the same reaction:

| | Me on N | NH₂ on N | OH on N | Me on C | F on C | NO₂ on C | NH₂ on C |
|---|---|---|---|---|---|---|---|
| fast | −2.7 | +0.4 | +3.2 | +4.9 | +6.4 | +7.0 | −6.4 |
| re-optimized | −3.1 | −1.3 | +1.7 | +1.5 | +5.0 | +4.4 | −6.5 |

N-methyl speeds the nucleophile up; any group on the attacked carbon
slows it (steric hindrance at the α-carbon). OH on C looked like −11.9
fast. Re-optimized, its TS connected other minima: a different mechanism,
which the fast mode cannot see. Hence the "(check)" mark.

## Network properties (Explore › Conditions › *Analyze network*, *Compare*)

`mepd/network_model.py` turns the Explore graph under a setup into a model:
species and steps with one energy scale, and the provenance of every
number (job, level of theory, phase or solvent, single-point or
re-optimized, substituent variant). `mepd/web/setups.build_model` builds
it.

- **Which barrier each edge uses:**
  - its lowest verified barrier (only from jobs at the setup's level of
    theory, if one is set);
  - replaced by the setup solvent's result when an edge has one;
  - plus the substituent shift for the setup's variant, from that edge's
    *Substituent effects* result (for a channels edge, that channel's own
    row).
- **Species energies:** placed from one start per connected part, through
  each step's reaction energy. Cycles whose energies disagree by more than
  1 kcal/mol are reported, as are gas-phase fallbacks, missing variants and
  mixed levels of theory.

Properties of the whole network:

| Property | What it says |
|---|---|
| amounts after the reaction time, conversion | the outcome (exact first order; symmetrized eigen-decomposition, stable at any time) |
| equilibrium populations, distance from equilibrium | kinetic vs thermodynamic control |
| relaxation times | when the network has equilibrated |
| formation time of each product | when it first reaches half of its peak (the mean first-passage time, also available, is dominated by rare detours into deep traps) |
| bottleneck route, effective barrier | the route whose highest TS is lowest, and the step that sets it |
| degree of control of every TS and species | X = d ln P / d(−G/RT) for a product's yield and formation rate: which atomic-scale numbers the network-level result depends on |

**Compare** gives, for each product's yield and rate, both values, the
factor between them, and a first-order split of the change over the
energies that moved (Σ X·(−ΔG/RT)). When that split explains less than
half, or more than double, of the real change, it says "nonlinear"
instead. An example: in water the Menshutkin product is saturated at
equilibrium, and in DMSO it is 16 kcal/mol less stabilized. Its yield falls
from 99.7% to 20% through thermodynamics that the base's near-zero
sensitivities cannot see. The formation rate (×0.0016) is split cleanly:
100% from the TS, which DMSO raises by 3.8 kcal/mol.

Every property refers to one level of theory, solvent, temperature and
variant. So the same network can be compared across levels of theory,
solvents, temperatures and substituents.

## Toward the whole network (plan)

The long-term aim: generate a network once, then ask how it *as a whole*
(its kinetics and sampled products) responds to a change of conditions or
of functional groups. The pieces fit as follows.

- **Perturbations are transforms of stationary points.** Solvent,
  force and substituent all take (reactant, TS, product) of a verified
  step and return new energies, fast or re-optimized. Applying one to
  every edge of a network gives the perturbed network.
- **Atom indices are shared.** A network grown from one seed (network
  expansion, channels) keeps the seed's atom order in every species, so a
  site defined on the seed (an H on atom 4) means the same place in every
  step. The only exceptions are steps where that H is transferred, and
  those are reported, not silently skipped.
- **Built:** setups with a variant, level of theory, solvent and
  temperature; network properties; and setup comparison (see above).
  Still to do: "compute this variant on every edge" (as the solvent fill
  already does), force as a setup dimension, and flux through each channel.
- **Sampling products:** the flux-steered expansion (`--steer flux`)
  already decides which species to expand from the kinetics. Running it
  under a perturbed setup gives a network that grows differently under
  different conditions, which is the "sampled products" part.

## Photochemistry and photocatalysis (plan, not built)

Light changes which surface a reaction runs on. Two things follow for
mepd: new channels opened by light, and spectroscopic signals that
experiments can watch.

**What this machine can do today.** There is no excited-state method
here: no sTDA-xTB, TD-DFT, PySCF or Psi4. The local TeraChem image stops
right after its setup on this GPU; its CUDA 11.8 build probably predates
it. mepd already reads TeraChem excited-state energies (`cistarget`,
hh-TDA parsing in `nodes/nodehelpers.py`), so ChemCloud's TeraChem is the
likely route for true excited states.

**Stage 1: build on ground-state machinery (feasible now).** The lowest
triplet and the radical ions are *ground states* of their own spin or
charge sector. mepd's whole machinery (paths, TSs, IRCs, channels, solvent,
force) therefore works on them unchanged, with multiplicity 3 or charge ±1.
That covers the two main modes of photocatalysis:

- **Energy transfer (triplet sensitization).** Compute each species'
  adiabatic triplet energy E_T by ΔSCF (a T1 minimum at UHF GFN2/g-xTB,
  minus the S0 minimum), then search the reaction on the T1 surface. Report
  it as: "in the dark 45 kcal/mol; on T1 8 kcal/mol; a sensitizer with
  E_T ≥ 58 kcal/mol can reach it". The chemist supplies the sensitizer's E_T
  (or mepd computes it the same way).
- **Photoredox (single-electron transfer).** Compute the substrate's
  oxidation and reduction potentials in solvent: ΔSCF with ALPB, plus a
  calibration offset fitted to a small set of experimental potentials. Rough:
  semiempirical potentials are good to perhaps a few tenths of a volt. Then:
  - check the catalyst's excited-state potentials, E*red = E_red + E00 and
    E*ox = E_ox − E00, with Rehm–Weller ΔG_ET for each quenching route;
  - if electron transfer is downhill, add the radical ion as a new node and
    search its channels.
- **Crossing back.** A triplet or radical-ion path must return to the
  ground state. Minimum-energy crossing points between S0 and T1 need only
  the two ground-state-type gradients (Harvey's MECP algorithm), so they
  are also within reach.

The insight has the same shape as the solvent one: "this channel opens
under visible light if the catalyst's E_T or E* passes X", next to the
thermal barrier.

**Stage 2: true excited states (TD-DFT, or sTDA-xTB if installed).**
- **Photochemical signals.** Vertical absorption of every node:
  intermediates, catalyst resting state and active species. The graph can
  then say which species should absorb where, i.e. what a UV-vis or
  transient-absorption experiment would see if the predicted cycle is
  right. This is a direct experimental check on a computed mechanism.
- **Direct excitation.** S1 minima and conical intersections (MECI), for
  reactions that start on S1 rather than via a sensitizer. This needs
  TD-DFT gradients at minimum; branching ratios need nonadiabatic dynamics
  (e.g. TeraChem's AIMS), which is beyond mepd's scope.

Open question for the next step: which signal matters first, light as a
driving condition (photoredox / energy transfer) or spectra as an
experimental readout of intermediates? Stage 1 serves the first; stage 2's
absorption part serves the second.

## How other network tools handle conditions

* **RMG**: a reactor is specified by temperature, pressure, initial
  composition and termination criteria, and the mechanism is grown by
  rate-based enlargement at those conditions. Liquid-phase reactors add a
  solvent, with solvation corrections to thermochemistry and kinetics
  (Jalan, West, Green, J. Phys. Chem. B 117, 2955 (2013); RMG 3.0:
  Liu et al., J. Chem. Inf. Model. 61, 2686 (2021)).
* **Chemoton / SCINE**: the electronic-structure model, including implicit
  solvent, is part of each calculation's settings. Exploration can be
  steered by microkinetic concentration flux from given starting
  concentrations (Unsleber, Grimmel, Reiher, JCTC 18, 5393 (2022);
  Bensberg, Reiher, Isr. J. Chem. 63, e202200123 (2023)). mepd's network
  expansion already uses the flux criterion.
* **autodE**: a reaction profile is computed in a named implicit solvent
  and at a temperature (Young et al., Angew. Chem. Int. Ed. 60, 4266
  (2021)).
* **Retrosynthesis (ASKCOS)**: conditions are *recommended* from reaction
  data, as a catalyst, solvents, reagents and a temperature per reaction
  (Gao et al., ACS Cent. Sci. 4, 1465 (2018)). This is a prior over which
  setups are worth computing, not a computed effect.

mepd is closest to RMG and Chemoton: conditions enter through the
energies (solvent) and the kinetics (temperature, time, starting
materials). Unlike RMG's group-additivity corrections, the solvent
effects here are computed per TS.

## Next steps (not built)

1. **Free energies.** Quasi-RRHO thermochemistry from the Hessians mepd
   already computes, so that temperature acts on ΔG‡ and not only on the
   Eyring prefactor. Bimolecular steps and entropy-driven equilibria need
   this.
2. **Reagents and catalysts as part of a setup.** Additives (an acid, a
   base, water, a metal ion) listed in the setup become partners for
   network expansion, as bimolecular reactive complexes (Chemoton-style),
   and their concentrations enter second-order rates.
3. **Microsolvation.** For protic solvents, place 1–3 explicit solvent
   molecules at the TS (Design already places species), then compare with
   the implicit result. The insights already point there.
4. **Condition recommendation.** A learned prior (Gao et al. 2018) to
   propose which solvents and temperatures to compute first.
5. **Acid/base conditions.** Protonation states per pH, as separate
   species with pKa-derived populations.
