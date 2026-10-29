import numpy as np
import pytest

from virtual_accelerator.tests.dependency_profiles import HAS_TWOBUNCH_SURROGATE_DEPS

pytestmark = [
    pytest.mark.requires_surrogate,
]

if HAS_TWOBUNCH_SURROGATE_DEPS:
    from twobunch_cnf import load_model

    from virtual_accelerator.registry import get_model
    from virtual_accelerator.surrogates.twobunch_surrogate import TwoBunchSurrogate
else:
    pytest.skip(
        "requires surrogate optional dependencies: pip install virtual-accelerator[surrogate]",
        allow_module_level=True,
    )

N_PARTICLES = 1000


@pytest.fixture(scope="module")
def cnf():
    return load_model("c_b3_ema")


@pytest.fixture
def surrogate(cnf):
    return TwoBunchSurrogate(cnf, n_particles=N_PARTICLES)


def assert_same_beam(beam, expected):
    for key in ("x", "px", "y", "py", "z", "pz", "t", "weight"):
        np.testing.assert_array_equal(beam[key], expected[key])


def test_variables(surrogate, cnf):
    variables = surrogate.supported_variables
    assert list(variables) == [*cnf.knob_names, "seed", "XTCAVF_beam"]
    phase = variables["L1PhaseSet"]
    assert (phase.unit, phase.value_range, phase.default_value) == (
        "deg",
        (-27.5, -18.5),
        -20.2889213421,
    )
    assert variables["XTCAVF_beam"].read_only
    surrogate.get(list(variables))


def test_beam_is_the_flow_sample(surrogate, cnf):
    expected = cnf.particle_group(n_particles=N_PARTICLES, seed=0)
    assert_same_beam(surrogate.final_particles, expected)
    assert_same_beam(surrogate.get_value("XTCAVF_beam"), expected)


def test_set_keeps_earlier_inputs(surrogate, cnf):
    # -45 deg is outside the training range, which is not enforced.
    surrogate.set({"L2PhaseSet": -45.0})
    surrogate.set({"seed": 3})
    assert surrogate.get_value("L2PhaseSet") == -45.0

    expected = cnf.particle_group(
        {"L2PhaseSet": -45.0}, n_particles=N_PARTICLES, seed=3
    )
    assert_same_beam(surrogate.final_particles, expected)


def test_reset(surrogate):
    initial = surrogate.final_particles
    surrogate.set({"L2PhaseSet": -35.0, "seed": 3})
    surrogate.reset()
    assert_same_beam(surrogate.final_particles, initial)


def test_final_particles_is_a_copy(surrogate):
    surrogate.final_particles.x[:] = 0.0
    assert surrogate.final_particles.x.any()


def test_registry_builds_surrogate(cnf):
    surrogate = get_model(
        "surrogate_f2e_twobunch", n_particles=N_PARTICLES, seed=1, latent="mean"
    )
    expected = cnf.particle_group(n_particles=N_PARTICLES, seed=1, latent="mean")
    assert_same_beam(surrogate.final_particles, expected)
