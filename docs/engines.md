# Electronic Structure Engines

MEPD supports multiple electronic structure engines through a common interface. Engines handle energy and gradient calculations for molecular structures.

## Available Engines

| Engine | Description | Use Case |
|--------|-------------|----------|
| `QCOPEngine` | Uses QCOP to interface with quantum chemistry codes | Production calculations with ChemCloud, XTB, ORCA, etc. |
| `ASEEngine` | Interfaces with ASE calculators | ML potentials, custom methods |
| `LEPSEngine` | Simple LEPS potential | Testing, model systems |
| `ThreeWellEngine` | Three-well potential | Testing, model systems |

## ChemCloud Setup

ChemCloud is the recommended way to run electronic structure calculations. You'll need:

1. Sign up at https://chemcloud.mtzlab.com/signup
2. Configure authentication (choose one option):

```bash
# Option 1: Run setup_profile() - writes credentials to ~/.chemcloud/credentials
python -c "from chemcloud import setup_profile; setup_profile()"

# Option 2: Use environment variables (for memory-only auth)
export CHEMCLOUD_USERNAME=your_email@chemcloud.com
export CHEMCLOUD_PASSWORD=your_password

# Option 3: Custom server (if using a different domain)
export CHEMCLOUD_DOMAIN="https://your-server-url.com"
```

## QCOPEngine

The QCOPEngine interfaces with various quantum chemistry programs through QCOP.

### Basic Usage

```python
from mepd.engines.qcop import QCOPEngine

# Using ChemCloud (recommended)
eng = QCOPEngine(compute_program="chemcloud")

# Default: uses XTB
eng = QCOPEngine()

# Or specify program arguments
from qcio import ProgramArgs

args = ProgramArgs(
    model={"method": "GFN2xTB", "basis": "GFN2xTB"},
    keywords={"threads": 4}
)
eng = QCOPEngine(program_args=args, program="xtb")

# Optional: also write cached ProgramOutput objects when saving results
eng = QCOPEngine(write_qcio=True)
```

### Features

- **Geometry Optimization**: `eng.compute_geometry_optimization(node)`
- **Energy Calculation**: `eng.compute_energies(chain)`
- **Gradient Calculation**: `eng.compute_gradients(chain)`
- **Supports external programs**: XTB, ORCA, TeraChem, Psi4, etc.
- **Optional `.qcio` output writing**: set `write_qcio=True` to emit cached `qcio.ProgramOutput` objects when chain/history results are written to disk

When `write_qcio=True`, `QCOPEngine` emits a warning because writing every cached `ProgramOutput`
can consume substantial disk space, especially for ChemCloud runs and saved optimization histories.

### Supported Programs

```python
# ChemCloud (recommended)
eng = QCOPEngine(compute_program="chemcloud")

# XTB (default, requires local installation)
eng = QCOPEngine(program="xtb")

# ORCA
eng = QCOPEngine(program="orca")

# TeraChem
eng = QCOPEngine(program="terachem")
```

## ASEEngine

The ASEEngine interfaces with ASE (Atomic Simulation Environment) calculators, enabling use of machine learning potentials and other methods.

### Basic Usage

```python
from mepd.engines.ase import ASEEngine
from mace.calculators import MACECalculator

# Load MACE potential
calc = MACECalculator(model="mace-medium", device="cuda")
eng = ASEEngine(calculator=calc)

# Now run NEB as usual
n = NEB(initial_chain=initial_chain, parameters=nbi, optimizer=opt, engine=eng)
```

### ASE Optimizers

```python
from mepd.engines.ase import ASEEngine

eng = ASEEngine(
    calculator=calc,
    geometry_optimizer="LBFGSLineSearch"  # Default
)
```

Available optimizers: `LBFGS`, `BFGS`, `FIRE`, `LBFGSLineSearch`, `MDMin`

### Configure ASE optimizer from TOML

When running through `RunInputs`/CLI, set ASE optimizer selection in `inputs.toml`:

```toml
engine_name = "ase"
program = "omol25"

[ase_engine_kwds]
geometry_optimizer = "FIRE"
transition_state_optimizer = "SELLA"
```

Note: `[optimizer_kwds]` controls the NEB/path-minimizer optimizer, not the
ASE geometry optimizer class.

### Configure geomeTRIC optimizer keywords from TOML

For `QCOPEngine` through `engine_name = "qcop"` or `"chemcloud"`, set geomeTRIC
optimization keywords in `inputs.toml`:

```toml
[geometry_optimizer_kwds]
coordsys = "tric"
maxit = 1000
convergence_energy = 1e-6
```

Note: `[program_kwds.keywords]` controls the underlying electronic-structure
program, and `[optimizer_kwds]` controls the NEB/path-minimizer optimizer.

## Engine Interface

All engines implement the following interface:

```python
class Engine:
    def compute_gradients(self, chain: Union[Chain, List]) -> NDArray:
        """Compute gradients for each node in the chain"""
        ...

    def compute_energies(self, chain: Union[Chain, List]) -> NDArray:
        """Compute energies for each node in the chain"""
        ...

    def compute_geometry_optimization(self, node: Node) -> List[Node]:
        """Optimize a single node geometry"""
        ...
```

## Using with Chains

Engines work with Chain objects to compute properties:

```python
# Create engine
eng = QCOPEngine(compute_program="chemcloud")

# Compute energies (also computes gradients internally)
energies = eng.compute_energies(chain)

# Compute gradients explicitly
gradients = eng.compute_gradients(chain)

# Optimize a single structure
optimized_node = eng.compute_geometry_optimization(start_node)
trajectory = eng.compute_geometry_optimization(start_node)  # Returns full trajectory
final_structure = trajectory[-1]
```

## Choosing an Engine

### Use QCOPEngine with ChemCloud when:
- Running production calculations
- Don't want to install local quantum chemistry software
- Need reliable cloud computing

### Use ASEEngine when:
- Using machine learning potentials (MACE, NequIP, etc.)
- Need custom ASE calculators
- Have GPU access for ML potentials
