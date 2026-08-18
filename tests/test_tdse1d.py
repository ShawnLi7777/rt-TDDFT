import numpy as np
import pytest

from tdse1d import (
    Grid1D,
    SplitOperatorSolver,
    density,
    harmonic_ground_state,
    harmonic_potential,
    norm,
    normalize,
)


def test_normalize_scales_wavefunction_to_unit_norm():
    grid = Grid1D(x_min=-5.0, x_max=5.0, points=512)
    psi = np.exp(-grid.x**2).astype(np.complex128)

    normalized = normalize(psi, grid.dx)

    assert norm(normalized, grid.dx) == pytest.approx(1.0, abs=1e-12)


def test_split_operator_conserves_probability_norm():
    grid = Grid1D(x_min=-10.0, x_max=10.0, points=1024)
    solver = SplitOperatorSolver(grid, harmonic_potential)

    result = solver.propagate(
        initial=harmonic_ground_state(grid.x),
        dt=0.002,
        steps=500,
        snapshot_steps=[0, 500],
    )

    assert np.max(np.abs(result.norms - 1.0)) <= 1e-10


def test_harmonic_ground_state_density_is_stationary():
    grid = Grid1D(x_min=-10.0, x_max=10.0, points=2048)
    solver = SplitOperatorSolver(grid, harmonic_potential)

    result = solver.propagate(
        initial=harmonic_ground_state(grid.x),
        dt=0.002,
        steps=1000,
        snapshot_steps=[0, 250, 500, 750, 1000],
    )

    reference = density(result.snapshots[0])
    max_drift = max(
        float(np.max(np.abs(density(snapshot) - reference)))
        for snapshot in result.snapshots[1:]
    )

    assert max_drift <= 5e-6
