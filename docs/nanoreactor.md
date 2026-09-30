# Nanoreactor

`mepd discovery nanoreactor` (web: Explore › Reaction network expansion › Nanoreactor) discovers
reactions in a box of many molecules, then keeps, for each reaction, **only the molecules it needs**.

Why: a TS search or network expansion on a full system (reactant + solvent + partners) drags every
spectator along. The nanoreactor (Wang, Titov, McGibbon, Liu, Pande, Martínez, *Nat. Chem.* 2014)
runs the exploration on the big system and extracts each reaction event as a small subsystem.

## Pipeline

1. **Pack** the molecules (`CC=O*3 O*4`, or one xyz holding a whole reactor) at random orientations
   inside a sphere, then relax the packing inside the wall (xtb `--opt crude`). An unrelaxed packing
   releases hundreds of kcal/mol as heat and blows atoms out of the box.
2. **Piston MD** with xtb (GFN2-xTB by default, `--md-method gfn1|gxtb`). A spherical logfermi wall
   switches between a wide radius and `--compress` × that radius. It closes in 5 steps over 100 fs so
   nothing gets kicked. Each piston segment is one xtb run, restarted from `mdrestart`.
   - Wall: β = 1/bohr, with the wall "temperature" set for a push of `--wall-force` (20 kcal/mol/Å).
     xtb's default wall (300 K, β = 6) is too soft for hot molecules; a steep wall blew the reactor apart.
   - `--etemp` 3000 K (Fermi smearing): without it, the closed-shell SCC stops converging once a bond
     breaks homolytically, the forces are wrong, and atoms fly off.
   - Resumable: finished segments are kept.
3. **Bond history.** Per frame, a bond forms below 1.15 × (r_i + r_j) and breaks above 1.45 × (Cordero
   radii). A hydrogen caught between two partners stays on the nearer one. A bond state shorter than
   `--min-lifetime` (20 fs) is vibration and is removed.
4. **Events.** Bond changes within `--merge-window` (100 fs) that touch the same molecules form one
   event. Its subsystem is the closure of the molecules its changed atoms belong to, before and after.
   The subsystem is then split into groups of molecules that share atoms between before and after:
   - a molecule that only collided (same atoms, same bonds) drops out;
   - independent reactions that happened at the same time separate;
   - a water that hands over one H and takes another stays linked: it is a **shuttle**.
5. **Species identity.** Each molecule's charge is the rounded sum of its xtb partial charges on that
   frame (as in the original nanoreactor). Its Lewis structure comes from RDKit DetermineBondOrders,
   plus doublet radicals built from the closed-shell ion one electron away (·OH from OH⁻, ·CH₃ from
   CH₃⁺), kept only if the valences sanitize. Species = SMILES + charge + multiplicity.
6. **Reactions** are multisets of reactant species → product species; the reverse direction counts
   on the same reaction. A species on both sides is a shuttle, so `CC=O → C=CO` and
   `CC=O + O → C=CO + O` are **two reactions between the same two species**. A reaction whose two
   sides are the same (H exchange between waters) is recorded as an event but is no reaction.
7. **Refinement** at the `--inputs` level of theory:
   - every species is optimized on its own (lowest of `--instances` occurrences);
   - every reaction's subsystem is cut at a frame before and after the event (same atoms, same order:
     atom-mapped for free) and optimized. A side whose bonds change while it is optimized (e.g. two
     radicals that recombine) is reported and not used.
8. **TS** (`--connect`): MSMEP between the optimized subsystem ends, then TS optimization and IRC. The
   lowest TS whose IRC connects the same bonds wins.

## Live view and TS searches

The job page's **Reactor** tab (the default while a nanoreactor job runs):
- the MD animates as it runs, inside the piston wall (a faint sphere);
- atoms taking part in an event light up while it happens (the rest fade), with the reaction written
  over them;
- a timeline marks every event, and the event list sits beside it.

Clicking an event replays it at full time resolution with only its molecules. Bonds that form are
dashed green and bonds that break dashed red; the reaction's optimized reactant and product
subsystems are shown underneath.
- The view also reads the MD segment xtb is still running (`md/xtb.trj`), so a long segment does
  not hold it at the live edge; there it says "waiting for the MD…". Events still unfolding at the
  end of the trajectory are listed as "analyzing…".
- Live events come from `live_events.json`, which the CLI rewrites about once per simulated ps. It
  takes at most a fifth of the wall time, and molecules are neutral where they can be (no partial
  charges yet). An event is shown only once it is over (merge window + lag + minimum lifetime), so the
  list only grows. The finished run's events (with partial charges) replace them.
- API: `GET /api/jobs/{id}/reactor?start=N` returns frames from raw frame N on, subsampled to about
  10 fs (coarser beyond 40 ps, at most ~4000 frames), 2 decimals, with the wall radius and the events.
  `GET /api/jobs/{id}/reactor/events/{k}` returns one event's frames, atoms and bond changes.

**Find TS:** the Results tab lists the reactions (ΔE, times seen, TS status), each with a **Find TS**
button. The same button is in the event replay and in the Inspector of a selected reaction dot. It
queues an ordinary TS search (path search, TS optimization, IRC) on the reaction's subsystem. A
reaction whose subsystem changed bonds when optimized shows "no subsystem" instead.

Check (g-xTB): direct keto → enol, found by the nanoreactor and searched from the table:
ΔE‡ = 75.4 kcal/mol, IRC-verified. The same run also found the water-shuttled version.

## Analyze

Explore is the map (species and reaction dots). The **Analyze** tab is where reactions are compared
and TS searches asked for:
- every reaction, from every run and composed by hand, grouped by what it does (net reactants → net
  products, shuttles taken out), so parallel routes (direct, via water, ...) sit together, lowest
  barrier first;
- filters (species, no TS yet, with TS, shuttled) and "find TS for all without one";
- the reaction card of the selected route.

**New reaction:** pick reactants from the graph (a species on both sides is a shuttle).
- With products given, mepd places each side's molecules together as a complex; the TS search maps
  the atoms itself.
- Without products, the bond rules (up to 2 bonds broken and 2 formed, valid Lewis structures) propose
  what the complex can become, drawn as structures. The picked ones become reactions, their product
  guesses in the complex's own atom order.
- Complexes, and product species new to the graph, are minimized at the workspace level. The card's
  energy ladder then comes from the graph's own energies.

From Explore, selecting species offers "N reactions of these →" and "New reaction with these →", and
a reaction's card offers "open in Analyze".

**No TS endpoints:** when a reaction's optimized reactant or product complex changes bonds, the reason
is recorded and shown instead of a Find TS button: barrierless, recombines (radicals), unstable
product (reverts), falls apart, or rearranges (another occurrence may work).

## Energies

Different reactions contain different atoms, so absolute energies are never compared. Every number
is a difference within one reaction:

- ΔE: products minus reactants, each optimized alone;
- complex ΔE: optimized subsystem product minus reactant;
- barrier: TS minus the reactant subsystem (also given against the separated reactants).

## Output

- `md/`: packed.xyz, reactor.xyz (relaxed), segment files, trajectory.xyz; `partial_charges/`: xtb single points.
- `species/`: species_k.xyz (optimized) and species_k_md.xyz (as cut from the MD).
- `reactions/reaction_k/`: instance frames and optimized subsystem ends.
- `network.json`: settings, species, reactions (reactants, products, shuttles, counts, energies,
  complex, ts), events (atoms, frames, bond changes) and methods.

## Explore

- Species become ordinary nodes, merged with a molecule already in the graph.
- Each reaction is a **dot** joined to its reactants and products (arrows toward the products).
  Shuttles hang off it dashed. The dot's label is the barrier once known, else ΔE.
- The reaction's optimized subsystem ends are hidden structures (role `complex`) joined by an edge.
  Clicking the dot selects that edge, so *Transition state* / *Reaction channels* run on exactly the
  subsystem, and its barrier shows on the dot.
- Deleting the dot, or a species it involves, removes the reaction and its hidden subsystem.
- The import re-runs whenever network.json changes (after analysis, after refinement, after each TS)
  and once more when the job ends.

## Checks (2026-09-30, laptop, one core)

- 36-atom reactor (formaldehyde, ammonia, water), GFN2 at 1500 K: stable (the wall compresses to about
  4.2 Å), no SCC failures, no reactions in 6 ps. Speed: about 18 s per ps.
- Acetaldehyde ×2 + water ×6, 2000 K, 20 ps: no reactions. An apparent water autoionization was an
  artifact of the shared proton (fixed by the nearest-partner rule).
- Acetaldehyde ×3 + water ×4, 10 ps:
  - 2500 K: radical chemistry (H loss to the vinoxy radical, ketene, H₂);
  - 3000 K: 18 reactions, including direct keto → enol and water-involved steps. Refinement at g-xTB:
    keto → enol ΔE = +10.5 kcal/mol (experiment ≈ +10–11); 7 of 18 reactions kept both subsystem ends.

## Limits and next steps

- A catalyst that binds and leaves with the same atoms (no atom exchange) is split off as a collision
  partner. Keeping it needs a criterion on contacts at the moment the bonds change.
- At 3000 K, H atoms hop between several molecules within 100 fs, and some events grow large. Use a
  shorter `--merge-window`, or run cooler and longer.
- Kinetics over bimolecular reactions (network_model is first-order only), and a live view of the
  reactor trajectory in the job page, are not built yet.
- Discovery could also run on any mepd engine (e.g. MLIPs) through a Python MD driver; only xtb's MD
  is wired now.
