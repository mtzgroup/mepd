# Kinetics (Analyze › Kinetics)

The workspace's network as a microkinetic model (`mepd/microkinetics.py`; adapter `mepd/web/kinetics.py`;
`POST /api/kinetics`; UI `Kinetics.js`).

## The model

- **Species:** every minimum with an energy at the workspace level.
- **Steps:** every reaction (nanoreactor or composed) and every plain edge whose TS search gave a
  barrier. The TS is placed at the reactant side's energy plus that barrier, so all TSs and species
  sit on one energy scale.
  - Unverified barriers (path maxima) are included unless switched off.
  - The same transformation found by several runs is one channel, kept at its lowest TS.
- **Rates:** mass action for any order, from transition-state theory at a 1 M standard state.
  Forward and reverse rates go through the same TS, so detailed balance holds; a TS below one of its
  sides counts as level with it. A shuttle (a species on both sides) enters the rate but is not
  consumed.
- **Batch or constant feed:** a closed flask, or some species held fixed (a chemostat), whose
  long-time limit is a true steady state. The integrator is stiff BDF with an analytic Jacobian.

## What it reports

- **Amounts over time,** and the target's amount and formation rate.
- **Degree of rate control** of every TS for the target (its amount at the end, or its formation
  rate): X = d ln Q / d(−G_TS/RT), by central differences of ±0.2 kcal/mol.
  - X > 0: lowering that barrier gives more of the target. X < 0: that step diverts material away.
  - On a single pathway at steady state the X sum to 1 (tested).
  - Hover: the factor for 1 kcal/mol, exp(X/RT).
- **Thermodynamic degree of control** of intermediates and other products: − means a trap or a
  competing product.
- **Where the material went:** the integrated net conversion through each step.
- **Temperature sweep:** final amounts, the formation rate, and the apparent activation energy
  (Arrhenius, from d ln r / dT).

## Caveats

Electronic energies stand in for free energies. Bimolecular rates lack the entropy cost of
association, so absolute rates are orders of magnitude off; read ratios, controlling steps and trends.
Next steps: quasi-RRHO free energies from Hessians, and an entropy correction for associations.

References: C. T. Campbell, J. Catal. 204, 520 (2001) and ACS Catal. 7, 2770 (2017) (degree of rate
control); C. Stegelmann, A. Andreasen, C. T. Campbell, J. Am. Chem. Soc. 131, 8077 (2009)
(thermodynamic degree of rate control).
