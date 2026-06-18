# API Reference

This section provides detailed API documentation for the main classes and functions.

## Core Classes

### NEB

Main class for Nudged Elastic Band calculations.

```python
from mepd.neb import NEB
```

::: mepd.neb.NEB

### MSMEP

Multi-Step Minimum Energy Path calculator for handling complex reactions.

```python
from mepd import MSMEP
```

::: mepd.msmep.MSMEP

### Chain

Container for a pathway consisting of multiple images.

```python
from mepd import Chain
```

::: mepd.chain.Chain

### StructureNode

A node containing a molecular structure.

```python
from mepd import StructureNode
```

::: mepd.nodes.node.StructureNode

## Input Classes

### NEBInputs

Configuration for NEB optimization.

```python
from mepd.inputs import NEBInputs
```

::: mepd.inputs.NEBInputs

### ChainInputs

Configuration for chain behavior.

```python
from mepd.inputs import ChainInputs
```

::: mepd.inputs.ChainInputs

### GIInputs

Configuration for geodesic interpolation.

```python
from mepd.inputs import GIInputs
```

::: mepd.inputs.GIInputs

### RunInputs

Complete configuration for MSMEP calculations.

```python
from mepd.inputs import RunInputs
```

::: mepd.inputs.RunInputs

## Engines

### Engine (Abstract Base)

Base class for all engines.

```python
from mepd.engines import Engine
```

::: mepd.engines.engine.Engine

### QCOPEngine

Engine using QCOP for electronic structure calculations.

```python
from mepd.engines import QCOPEngine
```

::: mepd.engines.qcop.QCOPEngine

### ASEEngine

Engine using ASE calculators.

```python
from mepd.engines import ASEEngine
```

::: mepd.engines.ase.ASEEngine

## Optimizers

### Optimizer (Abstract Base)

Base class for optimizers.

```python
from mepd.optimizers import Optimizer
```

### VelocityProjectedOptimizer

VPO optimizer with velocity projection.

```python
from mepd.optimizers.vpo import VelocityProjectedOptimizer
```

### ConjugateGradient

Conjugate gradient optimizer.

```python
from mepd.optimizers.cg import ConjugateGradient
```

### LBFGS

Limited-memory BFGS optimizer.

```python
from mepd.optimizers.lbfgs import LBFGS

## Helper Functions

### chainhelpers

Utility functions for chain manipulation and visualization.

```python
import mepd.chainhelpers as ch
```

**Common functions:**

- `ch.run_geodesic()` - Create chain using geodesic interpolation
- `ch.visualize_chain()` - Visualize chain in 3D
- `ch.compute_NEB_gradient()` - Calculate NEB gradient
- `ch.get_g_perps()` - Get perpendicular gradients
- `ch._get_ind_minima()` - Find indices of minima in chain
