# rt-TDDFT: minimal 1D TDSE solver

A small, reusable solver for the single-particle one-dimensional time-dependent
Schrödinger equation

```text
i dψ/dt = [-1/2 d²/dx² + V(x,t)] ψ
```

in atomic units (`hbar = m = 1`). It uses a centered finite-difference
Hamiltonian with homogeneous Dirichlet boundaries and Crank-Nicolson time
propagation. Crank-Nicolson is unitary for the real midpoint potential used at
each step, so the discretized norm is conserved without renormalizing during
propagation.

## Install

Python 3.9 or newer is required. Runtime dependencies are NumPy and SciPy;
pytest is used only for tests.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
```

## Usage

Both the initial state and potential can be callables. The initial state may
also be an array, and the potential may be a scalar or grid-sized array.
`TDSESolver` normalizes the initial wavefunction after enforcing zero boundary
values. Choose `total_time` as an integer multiple of `dt`.

```python
import numpy as np
from tdse import Grid1D, TDSESolver

solver = TDSESolver(
    grid=Grid1D(-10.0, 10.0, 501),
    dt=0.005,
    total_time=1.0,
    initial_wavefunction=lambda x: np.exp(-0.5 * x**2),
    potential=lambda x, t: 0.5 * x**2,
)
result = solver.run()

print(result.times)          # saved time points
print(result.wavefunctions)  # complex psi(t, x)
print(result.densities)      # |psi(t, x)|^2
print(result.norms)          # integral |psi|^2 dx
```

`V(x,t)` is evaluated at each time-step midpoint, permitting explicitly
time-dependent real potentials while retaining the second-order
Crank-Nicolson update. A complex absorbing potential is intentionally rejected
because it is not norm-preserving.

## Harmonic-oscillator example

The example starts from the analytic harmonic-oscillator ground state,
propagates it in `V(x)=x²/2`, reports norm and density drift, and writes a
compressed archive suitable for plotting:

```bash
python examples/harmonic_oscillator.py
```

The generated `harmonic_oscillator_densities.npz` contains `x`, `times`,
`densities`, and `norms`. For example, an optional Matplotlib plot can be made
with:

```python
import matplotlib.pyplot as plt
import numpy as np

data = np.load("harmonic_oscillator_densities.npz")
plt.plot(data["x"], data["densities"][0], label="initial")
plt.plot(data["x"], data["densities"][-1], "--", label="final")
plt.xlabel("x")
plt.ylabel("|psi|^2")
plt.legend()
plt.show()
```

The ground state is stationary up to spatial discretization error: the test
configuration (`[-8,8]`, 801 points, `dt=0.005`, `T=0.25`) requires maximum
initial-to-final density change below `1e-5`. Tests additionally require norm
drift no greater than `1e-6`.

## Tests

```bash
python -m pytest -q
```
