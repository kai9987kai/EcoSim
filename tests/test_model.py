import numpy as np
import pytest

from ecosim.model import EcoSimulation, SimulationConfig, Traits, toroidal_delta


def small_config(**changes):
    defaults = {
        "grid_size": 20,
        "initial_herbivores": 18,
        "initial_predators": 4,
        "seed": 123,
    }
    defaults.update(changes)
    return SimulationConfig(**defaults)


def test_same_seed_reproduces_complete_timeseries():
    first = EcoSimulation(small_config())
    second = EcoSimulation(small_config())

    assert first.run(35) == second.run(35)


def test_resources_and_positions_remain_within_bounds():
    simulation = EcoSimulation(small_config())
    simulation.run(80)

    assert np.all(simulation.resources >= 0)
    assert np.all(simulation.resources <= simulation.carrying_capacity)
    for agent in simulation.agents:
        assert np.all(agent.position >= 0)
        assert np.all(agent.position < simulation.config.world_size)
        assert np.isfinite(agent.energy)


def test_empty_populations_are_supported():
    simulation = EcoSimulation(
        small_config(initial_herbivores=0, initial_predators=0)
    )

    record = simulation.step()

    assert record["herbivores"] == 0
    assert record["predators"] == 0
    assert record["mean_herbivore_energy"] == 0
    assert record["mean_predator_energy"] == 0


def test_toroidal_delta_uses_shortest_route():
    delta = toroidal_delta(
        np.array([98.0, 3.0]),
        np.array([2.0, 97.0]),
        world_size=100.0,
    )

    assert delta == pytest.approx(np.array([4.0, -6.0]))


def test_mutated_traits_stay_in_biological_bounds():
    rng = np.random.default_rng(4)
    traits = Traits.founder("herbivore", rng)

    for _ in range(500):
        traits = traits.mutate("herbivore", rng, sigma=0.3)
        assert 0.55 <= traits.speed <= 4.0
        assert 4.0 <= traits.perception <= 36.0
        assert 0.20 <= traits.metabolism <= 2.8
        assert 1.5 <= traits.max_intake <= 11.0
        assert 0.35 <= traits.assimilation <= 0.94
        assert 0.02 <= traits.boldness <= 0.98


def test_invalid_configuration_fails_early():
    with pytest.raises(ValueError, match="world_size"):
        SimulationConfig(world_size=0)
