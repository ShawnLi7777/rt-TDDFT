"""Minimal one-dimensional time-dependent Schrödinger equation solver.

Atomic units are used throughout (hbar = particle mass = 1).  The solver uses
second-order finite differences in space, homogeneous Dirichlet boundaries,
and unitary Crank--Nicolson propagation in time.
"""

from dataclasses import dataclass
from os import PathLike
from typing import Callable, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import solve_banded

Wavefunction = Union[ArrayLike, Callable[[NDArray[np.float64]], ArrayLike]]
Potential = Union[
    ArrayLike, Callable[[NDArray[np.float64], float], ArrayLike]
]


@dataclass(frozen=True)
class Grid1D:
    """Uniform one-dimensional spatial grid including both endpoints."""

    x_min: float
    x_max: float
    num_points: int

    def __post_init__(self) -> None:
        if not np.isfinite((self.x_min, self.x_max)).all():
            raise ValueError("grid bounds must be finite")
        if self.x_max <= self.x_min:
            raise ValueError("x_max must be greater than x_min")
        if not isinstance(self.num_points, (int, np.integer)) or self.num_points < 3:
            raise ValueError("num_points must be an integer of at least 3")

    @property
    def x(self) -> NDArray[np.float64]:
        return np.linspace(self.x_min, self.x_max, self.num_points)

    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / (self.num_points - 1)


@dataclass(frozen=True)
class SimulationResult:
    """Sampled wavefunctions and diagnostics from a propagation."""

    x: NDArray[np.float64]
    times: NDArray[np.float64]
    wavefunctions: NDArray[np.complex128]
    norms: NDArray[np.float64]

    @property
    def densities(self) -> NDArray[np.float64]:
        """Probability densities ``|psi(x, t)|**2`` for every stored time."""

        return np.abs(self.wavefunctions) ** 2


class TDSESolver:
    """Propagate a single particle under ``H = -1/2 d2/dx2 + V(x,t)``."""

    def __init__(
        self,
        grid: Grid1D,
        dt: float,
        total_time: float,
        initial_wavefunction: Wavefunction,
        potential: Potential,
    ) -> None:
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if not np.isfinite(total_time) or total_time < 0.0:
            raise ValueError("total_time must be finite and non-negative")

        step_count = int(round(total_time / dt))
        if not np.isclose(step_count * dt, total_time, rtol=1e-12, atol=1e-15):
            raise ValueError("total_time must contain an integer number of steps")

        self.grid = grid
        self.dt = float(dt)
        self.total_time = float(total_time)
        self.initial_wavefunction = initial_wavefunction
        self.potential = potential
        self._step_count = step_count

    def _initial_state(self, x: NDArray[np.float64]) -> NDArray[np.complex128]:
        values = (
            self.initial_wavefunction(x)
            if callable(self.initial_wavefunction)
            else self.initial_wavefunction
        )
        psi = np.asarray(values, dtype=np.complex128)
        if psi.shape != x.shape:
            raise ValueError("initial_wavefunction must have one value per grid point")
        if not np.isfinite(psi).all():
            raise ValueError("initial_wavefunction must contain only finite values")

        psi = psi.copy()
        psi[[0, -1]] = 0.0
        norm = self._norm(psi)
        if norm <= 0.0:
            raise ValueError("initial_wavefunction must have a non-zero norm")
        return psi / np.sqrt(norm)

    def _potential_at(
        self, x: NDArray[np.float64], time: float
    ) -> NDArray[np.float64]:
        values = self.potential(x, time) if callable(self.potential) else self.potential
        potential = np.asarray(values)
        try:
            potential = np.broadcast_to(potential, x.shape)
        except ValueError as error:
            raise ValueError("potential must be scalar or have one value per grid point") from error
        if np.iscomplexobj(potential) and np.any(np.imag(potential) != 0.0):
            raise ValueError("potential must be real for norm-preserving propagation")
        potential = np.asarray(np.real(potential), dtype=np.float64)
        if not np.isfinite(potential).all():
            raise ValueError("potential must contain only finite values")
        return potential

    def _norm(self, psi: NDArray[np.complex128]) -> float:
        # With zero endpoint values, the trapezoidal integral equals this sum.
        return float(self.grid.dx * np.sum(np.abs(psi) ** 2))

    def run(self) -> SimulationResult:
        """Run the propagation and return the state at every time step."""

        x = self.grid.x
        psi = self._initial_state(x)
        times = np.arange(self._step_count + 1, dtype=np.float64) * self.dt
        states = np.empty((self._step_count + 1, x.size), dtype=np.complex128)
        norms = np.empty(self._step_count + 1, dtype=np.float64)
        states[0] = psi
        norms[0] = self._norm(psi)

        kinetic_diagonal = 1.0 / self.grid.dx**2
        hamiltonian_off_diagonal = -0.5 / self.grid.dx**2
        a_off = 0.5j * self.dt * hamiltonian_off_diagonal
        b_off = -0.5j * self.dt * hamiltonian_off_diagonal
        interior_size = x.size - 2

        for step in range(self._step_count):
            midpoint = times[step] + 0.5 * self.dt
            potential = self._potential_at(x, midpoint)[1:-1]
            h_diagonal = kinetic_diagonal + potential

            rhs = (1.0 - 0.5j * self.dt * h_diagonal) * psi[1:-1]
            rhs[1:] += b_off * psi[1:-2]
            rhs[:-1] += b_off * psi[2:-1]

            banded = np.zeros((3, interior_size), dtype=np.complex128)
            banded[0, 1:] = a_off
            banded[1] = 1.0 + 0.5j * self.dt * h_diagonal
            banded[2, :-1] = a_off
            psi = np.zeros_like(psi)
            psi[1:-1] = solve_banded((1, 1), banded, rhs)

            states[step + 1] = psi
            norms[step + 1] = self._norm(psi)

        return SimulationResult(x=x, times=times, wavefunctions=states, norms=norms)


def harmonic_ground_state(x: ArrayLike) -> NDArray[np.float64]:
    """Analytic ground state for ``V(x) = x**2 / 2`` in atomic units."""

    coordinates = np.asarray(x, dtype=np.float64)
    return np.pi ** (-0.25) * np.exp(-0.5 * coordinates**2)


def save_densities(path: Union[str, PathLike[str]], result: SimulationResult) -> None:
    """Save grid, times, densities, and norms to a compressed NumPy archive."""

    np.savez_compressed(
        path,
        x=result.x,
        times=result.times,
        densities=result.densities,
        norms=result.norms,
    )
