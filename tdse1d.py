"""Small one-dimensional TDSE solver using split-operator propagation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.complex128]
RealArray = NDArray[np.float64]
Potential = Callable[[RealArray, float], RealArray]


@dataclass(frozen=True)
class Grid1D:
    """Uniform one-dimensional spatial grid."""

    x_min: float = -8.0
    x_max: float = 8.0
    points: int = 1024

    def __post_init__(self) -> None:
        if self.points < 8:
            raise ValueError("points must be at least 8")
        if self.x_max <= self.x_min:
            raise ValueError("x_max must be greater than x_min")

    @property
    def x(self) -> RealArray:
        return np.linspace(self.x_min, self.x_max, self.points, endpoint=False)

    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / self.points

    @property
    def k(self) -> RealArray:
        return 2.0 * np.pi * np.fft.fftfreq(self.points, d=self.dx)


def harmonic_potential(x: RealArray, _t: float = 0.0) -> RealArray:
    """Return the one-dimensional harmonic oscillator potential x^2 / 2."""

    return 0.5 * x**2


def harmonic_ground_state(x: RealArray) -> Array:
    """Return the normalized analytic ground state for V(x)=x^2/2."""

    psi = np.pi ** (-0.25) * np.exp(-0.5 * x**2)
    return psi.astype(np.complex128)


def norm(psi: Array, dx: float) -> float:
    """Return the integrated probability norm."""

    return float(np.sum(np.abs(psi) ** 2) * dx)


def normalize(psi: Array, dx: float) -> Array:
    """Return a copy of psi scaled to unit probability norm."""

    value = norm(psi, dx)
    if value <= 0.0:
        raise ValueError("cannot normalize a wavefunction with zero norm")
    return psi / np.sqrt(value)


def density(psi: Array) -> RealArray:
    """Return probability density |psi|^2."""

    return np.abs(psi) ** 2


@dataclass
class PropagationResult:
    times: RealArray
    snapshots: list[Array]
    norms: RealArray


class SplitOperatorSolver:
    """Second-order split-operator FFT propagator in atomic units."""

    def __init__(self, grid: Grid1D, potential: Potential):
        self.grid = grid
        self.potential = potential
        self._kinetic_phase_cache: dict[float, Array] = {}

    def _kinetic_phase(self, dt: float) -> Array:
        if dt not in self._kinetic_phase_cache:
            self._kinetic_phase_cache[dt] = np.exp(-0.5j * self.grid.k**2 * dt)
        return self._kinetic_phase_cache[dt]

    def step(self, psi: Array, t: float, dt: float) -> Array:
        """Advance psi by one time step from time t to t + dt."""

        x = self.grid.x
        half_v_start = np.exp(-0.5j * self.potential(x, t) * dt)
        half_v_end = np.exp(-0.5j * self.potential(x, t + dt) * dt)

        updated = half_v_start * psi
        updated = np.fft.ifft(self._kinetic_phase(dt) * np.fft.fft(updated))
        updated = half_v_end * updated
        return updated.astype(np.complex128, copy=False)

    def propagate(
        self,
        initial: Array,
        dt: float,
        steps: int,
        snapshot_steps: Iterable[int] | None = None,
    ) -> PropagationResult:
        """Propagate and record requested snapshots and norm history."""

        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if steps < 0:
            raise ValueError("steps must be non-negative")

        requested = set(snapshot_steps or [])
        invalid = [step for step in requested if step < 0 or step > steps]
        if invalid:
            raise ValueError(f"snapshot steps outside propagation range: {invalid}")

        psi = normalize(np.asarray(initial, dtype=np.complex128), self.grid.dx)
        snapshots: list[Array] = []
        times: list[float] = []
        norms = [norm(psi, self.grid.dx)]

        if 0 in requested:
            snapshots.append(psi.copy())
            times.append(0.0)

        for step_index in range(1, steps + 1):
            t = (step_index - 1) * dt
            psi = self.step(psi, t, dt)
            norms.append(norm(psi, self.grid.dx))

            if step_index in requested:
                snapshots.append(psi.copy())
                times.append(step_index * dt)

        return PropagationResult(
            times=np.asarray(times, dtype=np.float64),
            snapshots=snapshots,
            norms=np.asarray(norms, dtype=np.float64),
        )
