"""Propagate the harmonic-oscillator ground state and save its densities."""

import numpy as np

from tdse import Grid1D, TDSESolver, harmonic_ground_state, save_densities


def harmonic_potential(x: np.ndarray, time: float) -> np.ndarray:
    """Time-independent V(x) = x^2/2; ``time`` supports the common V(x,t) API."""

    del time
    return 0.5 * x**2


def main() -> None:
    solver = TDSESolver(
        grid=Grid1D(-8.0, 8.0, 801),
        dt=0.005,
        total_time=1.0,
        initial_wavefunction=harmonic_ground_state,
        potential=harmonic_potential,
    )
    result = solver.run()
    output = "harmonic_oscillator_densities.npz"
    save_densities(output, result)
    norm_drift = np.max(np.abs(result.norms - 1.0))
    density_drift = np.max(np.abs(result.densities[-1] - result.densities[0]))
    print(f"saved {output}")
    print(f"maximum norm drift: {norm_drift:.3e}")
    print(f"maximum final density change: {density_drift:.3e}")


if __name__ == "__main__":
    main()
