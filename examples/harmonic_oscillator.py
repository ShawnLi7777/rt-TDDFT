"""Run a stationary harmonic-oscillator TDSE propagation example."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tdse1d import (
    Grid1D,
    SplitOperatorSolver,
    density,
    harmonic_ground_state,
    harmonic_potential,
)


def run_example(output: Path | None = None) -> None:
    grid = Grid1D(x_min=-10.0, x_max=10.0, points=2048)
    solver = SplitOperatorSolver(grid, harmonic_potential)
    initial = harmonic_ground_state(grid.x)

    result = solver.propagate(
        initial=initial,
        dt=0.002,
        steps=1000,
        snapshot_steps=[0, 250, 500, 750, 1000],
    )

    initial_density = density(result.snapshots[0])
    max_density_drift = max(
        float(np.max(np.abs(density(snapshot) - initial_density)))
        for snapshot in result.snapshots[1:]
    )
    norm_drift = float(np.max(np.abs(result.norms - 1.0)))

    print(f"initial norm: {result.norms[0]:.12f}")
    print(f"final norm:   {result.norms[-1]:.12f}")
    print(f"max norm drift: {norm_drift:.3e}")
    print(f"max density drift: {max_density_drift:.3e}")

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output,
            x=grid.x,
            times=result.times,
            densities=np.asarray([density(snapshot) for snapshot in result.snapshots]),
            norms=result.norms,
        )
        print(f"saved snapshots to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, help="Optional .npz snapshot path")
    args = parser.parse_args()
    run_example(args.output)


if __name__ == "__main__":
    main()
