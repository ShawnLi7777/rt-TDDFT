import numpy as np
import pytest

from tdse import Grid1D, TDSESolver, harmonic_ground_state, save_densities


def harmonic_potential(x, time):
    del time
    return 0.5 * x**2


def test_initial_state_is_normalized_and_configuration_is_respected():
    grid = Grid1D(-8.0, 8.0, 257)
    solver = TDSESolver(
        grid=grid,
        dt=0.01,
        total_time=0.03,
        initial_wavefunction=lambda x: 3.0 * harmonic_ground_state(x),
        potential=harmonic_potential,
    )

    result = solver.run()

    assert result.wavefunctions.shape == (4, 257)
    assert np.allclose(result.times, [0.0, 0.01, 0.02, 0.03])
    assert result.norms[0] == pytest.approx(1.0, abs=1e-12)
    assert np.all(result.wavefunctions[:, [0, -1]] == 0.0)


def test_crank_nicolson_preserves_norm_to_required_tolerance():
    solver = TDSESolver(
        grid=Grid1D(-8.0, 8.0, 321),
        dt=0.005,
        total_time=0.25,
        initial_wavefunction=harmonic_ground_state,
        potential=harmonic_potential,
    )

    result = solver.run()

    assert np.max(np.abs(result.norms - 1.0)) <= 1e-6


def test_harmonic_ground_state_density_is_stationary():
    solver = TDSESolver(
        grid=Grid1D(-8.0, 8.0, 801),
        dt=0.005,
        total_time=0.25,
        initial_wavefunction=harmonic_ground_state,
        potential=harmonic_potential,
    )

    result = solver.run()

    density_change = np.max(np.abs(result.densities[-1] - result.densities[0]))
    assert density_change < 1e-5


def test_time_dependent_potential_is_evaluated_at_step_midpoints():
    evaluation_times = []

    def potential(x, time):
        evaluation_times.append(time)
        return np.zeros_like(x)

    TDSESolver(
        grid=Grid1D(-5.0, 5.0, 101),
        dt=0.1,
        total_time=0.2,
        initial_wavefunction=lambda x: np.exp(-(x**2)),
        potential=potential,
    ).run()

    assert evaluation_times == pytest.approx([0.05, 0.15])


def test_densities_can_be_saved_for_later_plotting(tmp_path):
    result = TDSESolver(
        grid=Grid1D(-5.0, 5.0, 101),
        dt=0.01,
        total_time=0.02,
        initial_wavefunction=lambda x: np.exp(-(x**2)),
        potential=lambda x, time: np.zeros_like(x),
    ).run()
    output = tmp_path / "densities.npz"

    save_densities(output, result)

    with np.load(output) as data:
        assert np.array_equal(data["x"], result.x)
        assert np.array_equal(data["times"], result.times)
        assert np.array_equal(data["densities"], result.densities)
        assert np.array_equal(data["norms"], result.norms)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"dt": 0.0}, "dt"),
        ({"total_time": -1.0}, "total_time"),
        ({"dt": 0.3, "total_time": 1.0}, "integer number of steps"),
    ],
)
def test_invalid_time_configuration_is_rejected(kwargs, message):
    defaults = dict(
        grid=Grid1D(-5.0, 5.0, 101),
        dt=0.1,
        total_time=0.2,
        initial_wavefunction=lambda x: np.exp(-(x**2)),
        potential=lambda x, time: np.zeros_like(x),
    )
    defaults.update(kwargs)

    with pytest.raises(ValueError, match=message):
        TDSESolver(**defaults)
