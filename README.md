# rt-TDDFT

Minimal reference code for one-dimensional real-time quantum propagation.

This repository currently includes a small time-dependent Schrodinger equation
(TDSE) solver in atomic units:

```text
i dpsi(x,t)/dt = [-1/2 d^2/dx^2 + V(x,t)] psi(x,t)
```

The implementation is intentionally compact and NumPy-only so it can be used as
a numerical baseline before adding larger real-time TDDFT pieces.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
```

## Run the harmonic oscillator example

```bash
python examples/harmonic_oscillator.py
```

The example propagates the ground state in the static potential
`V(x) = x^2 / 2`. It prints the initial/final probability norm and the maximum
density drift. It can also save density snapshots:

```bash
python examples/harmonic_oscillator.py --output ho_snapshots.npz
```

## Run tests

```bash
python -m pytest
```

The tests check wavefunction normalization, norm conservation, and that the
harmonic-oscillator ground-state probability density stays stationary within a
documented numerical tolerance.
