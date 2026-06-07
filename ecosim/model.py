"""Core spatial predator-prey-resource model.

The model is intentionally mechanistic rather than pretending that an
uncalibrated regressor is an animal learning policy. Individuals balance food
intake, maintenance, movement, risk, and reproduction in a heterogeneous
landscape. See MODEL.md for the full ODD-style specification.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import tau
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray

Species = Literal["herbivore", "predator"]
FloatArray = NDArray[np.float64]


def toroidal_delta(source: FloatArray, target: FloatArray, world_size: float) -> FloatArray:
    """Return the shortest signed displacement from source to target."""
    delta = np.asarray(target, dtype=float) - np.asarray(source, dtype=float)
    return (delta + world_size / 2.0) % world_size - world_size / 2.0


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    world_size: float = 100.0
    grid_size: int = 64
    initial_herbivores: int = 90
    initial_predators: int = 18
    seed: int = 7
    resource_capacity: float = 32.0
    resource_growth_rate: float = 0.075
    seasonal_amplitude: float = 0.28
    seasonal_period: int = 180
    half_saturation: float = 9.0
    movement_cost: float = 0.075
    capture_radius: float = 2.4
    capture_probability: float = 0.64
    background_mortality: float = 0.0006
    herbivore_reproduction_rate: float = 0.016
    predator_reproduction_rate: float = 0.0065
    herbivore_reproduction_energy: float = 100.0
    predator_reproduction_energy: float = 128.0
    herbivore_reproduction_cost: float = 52.0
    predator_reproduction_cost: float = 62.0
    herbivore_birth_energy: float = 26.0
    predator_birth_energy: float = 36.0
    herbivore_max_energy: float = 155.0
    predator_max_energy: float = 190.0
    mutation_sigma: float = 0.045
    carcass_recycling: float = 0.18
    max_herbivores: int = 450
    max_predators: int = 140
    max_age: int = 900

    def __post_init__(self) -> None:
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")
        if self.grid_size < 4:
            raise ValueError("grid_size must be at least 4")
        if self.initial_herbivores < 0 or self.initial_predators < 0:
            raise ValueError("initial populations cannot be negative")
        if self.resource_capacity <= 0 or self.resource_growth_rate < 0:
            raise ValueError("resource parameters must be non-negative")
        if not 0 <= self.seasonal_amplitude < 1:
            raise ValueError("seasonal_amplitude must be in [0, 1)")
        if self.seasonal_period <= 0:
            raise ValueError("seasonal_period must be positive")


@dataclass(frozen=True, slots=True)
class Traits:
    speed: float
    perception: float
    metabolism: float
    max_intake: float
    assimilation: float
    boldness: float

    @staticmethod
    def founder(species: Species, rng: np.random.Generator) -> "Traits":
        if species == "herbivore":
            means = np.array([1.75, 15.0, 0.85, 4.0, 0.63, 0.48])
        else:
            means = np.array([2.05, 19.0, 1.02, 0.0, 0.78, 0.58])
        variation = rng.lognormal(mean=0.0, sigma=0.07, size=6)
        values = means * variation
        values[5] = np.clip(means[5] + rng.normal(0.0, 0.06), 0.05, 0.95)
        return Traits(*values)

    def mutate(
        self,
        species: Species,
        rng: np.random.Generator,
        sigma: float,
    ) -> "Traits":
        values = np.array(
            [
                self.speed,
                self.perception,
                self.metabolism,
                self.max_intake,
                self.assimilation,
            ]
        )
        values *= rng.lognormal(mean=0.0, sigma=sigma, size=5)
        bounds = (
            (0.55, 4.0),
            (4.0, 36.0),
            (0.20, 2.8),
            (1.5, 11.0) if species == "herbivore" else (0.0, 0.0),
            (0.35, 0.94),
        )
        for index, (lower, upper) in enumerate(bounds):
            values[index] = np.clip(values[index], lower, upper)
        boldness = float(np.clip(self.boldness + rng.normal(0.0, sigma), 0.02, 0.98))
        return Traits(*values, boldness)


@dataclass(slots=True)
class Agent:
    identifier: int
    species: Species
    position: FloatArray
    energy: float
    age: int
    traits: Traits
    distance_moved: float = 0.0
    alive: bool = True


class EcoSimulation:
    """Spatial individual-based model with renewable patch resources."""

    def __init__(self, config: SimulationConfig | None = None):
        self.config = config or SimulationConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.time = 0
        self._next_identifier = 0
        self.carrying_capacity = self._create_carrying_capacity()
        initial_fraction = self.rng.uniform(0.45, 0.82, self.carrying_capacity.shape)
        self.resources = self.carrying_capacity * initial_fraction
        self.agents: list[Agent] = []
        self.records: list[dict[str, float | int]] = []
        self.last_events = {
            "herbivore_births": 0,
            "predator_births": 0,
            "herbivore_deaths": 0,
            "predator_deaths": 0,
            "kills": 0,
        }

        for _ in range(self.config.initial_herbivores):
            self.agents.append(self._make_founder("herbivore"))
        for _ in range(self.config.initial_predators):
            self.agents.append(self._make_founder("predator"))
        self.records.append(self.summary())

    @property
    def herbivores(self) -> list[Agent]:
        return [agent for agent in self.agents if agent.alive and agent.species == "herbivore"]

    @property
    def predators(self) -> list[Agent]:
        return [agent for agent in self.agents if agent.alive and agent.species == "predator"]

    def _create_carrying_capacity(self) -> FloatArray:
        size = self.config.grid_size
        axis = np.linspace(0.0, self.config.world_size, size, endpoint=False)
        x_grid, y_grid = np.meshgrid(axis, axis, indexing="xy")
        productivity = np.zeros((size, size), dtype=float)
        for _ in range(7):
            center = self.rng.uniform(0.0, self.config.world_size, 2)
            width = self.rng.uniform(8.0, 24.0)
            dx = (x_grid - center[0] + self.config.world_size / 2) % self.config.world_size
            dy = (y_grid - center[1] + self.config.world_size / 2) % self.config.world_size
            dx -= self.config.world_size / 2
            dy -= self.config.world_size / 2
            productivity += np.exp(-(dx * dx + dy * dy) / (2.0 * width * width))
        productivity -= productivity.min()
        productivity /= max(float(productivity.max()), np.finfo(float).eps)
        return self.config.resource_capacity * (0.42 + 1.08 * productivity)

    def _make_founder(self, species: Species) -> Agent:
        energy = 54.0 if species == "herbivore" else 72.0
        return self._make_agent(
            species=species,
            position=self.rng.uniform(0.0, self.config.world_size, 2),
            energy=energy,
            traits=Traits.founder(species, self.rng),
        )

    def _make_agent(
        self,
        species: Species,
        position: FloatArray,
        energy: float,
        traits: Traits,
    ) -> Agent:
        agent = Agent(
            identifier=self._next_identifier,
            species=species,
            position=np.asarray(position, dtype=float) % self.config.world_size,
            energy=float(energy),
            age=0,
            traits=traits,
        )
        self._next_identifier += 1
        return agent

    def _cell(self, position: FloatArray) -> tuple[int, int]:
        scaled = np.floor(position / self.config.world_size * self.config.grid_size).astype(int)
        x_index, y_index = scaled % self.config.grid_size
        return int(y_index), int(x_index)

    def _grow_resources(self) -> None:
        seasonal_factor = 1.0 + self.config.seasonal_amplitude * np.sin(
            tau * self.time / self.config.seasonal_period
        )
        growth = (
            self.config.resource_growth_rate
            * seasonal_factor
            * self.resources
            * (1.0 - self.resources / self.carrying_capacity)
        )
        self.resources = np.clip(
            self.resources + growth,
            0.0,
            self.carrying_capacity,
        )

    def _candidate_positions(self, agent: Agent) -> tuple[FloatArray, FloatArray]:
        phase = self.rng.uniform(0.0, tau)
        angles = phase + np.linspace(0.0, tau, 12, endpoint=False)
        directions = np.column_stack((np.cos(angles), np.sin(angles)))
        displacements = directions * agent.traits.speed
        displacements = np.vstack((np.zeros(2), displacements))
        positions = (agent.position + displacements) % self.config.world_size
        return positions, displacements

    def _distances_to(self, position: FloatArray, others: Sequence[Agent]) -> FloatArray:
        if not others:
            return np.empty(0, dtype=float)
        positions = np.vstack([other.position for other in others])
        deltas = toroidal_delta(position, positions, self.config.world_size)
        return np.linalg.norm(deltas, axis=1)

    def _choose_position(
        self,
        agent: Agent,
        herbivore_positions: FloatArray,
        predator_positions: FloatArray,
    ) -> tuple[FloatArray, float]:
        candidates, displacements = self._candidate_positions(agent)
        travel = np.linalg.norm(displacements, axis=1)
        energy_deficit = np.clip(1.0 - agent.energy / 100.0, 0.0, 1.0)

        if agent.species == "herbivore":
            scaled = np.floor(
                candidates / self.config.world_size * self.config.grid_size
            ).astype(int)
            x_indices = scaled[:, 0] % self.config.grid_size
            y_indices = scaled[:, 1] % self.config.grid_size
            food = self.resources[y_indices, x_indices] / self.carrying_capacity[
                y_indices, x_indices
            ]
            if predator_positions.size:
                deltas = toroidal_delta(
                    candidates[:, np.newaxis, :],
                    predator_positions[np.newaxis, :, :],
                    self.config.world_size,
                )
                nearest_predator = np.linalg.norm(deltas, axis=2).min(axis=1)
                risk = np.exp(-nearest_predator / agent.traits.perception)
            else:
                risk = np.zeros(len(candidates))
            risk_aversion = 2.2 * (1.15 - agent.traits.boldness)
            scores = (2.0 + 2.2 * energy_deficit) * food - risk_aversion * risk
        else:
            if herbivore_positions.size:
                deltas = toroidal_delta(
                    candidates[:, np.newaxis, :],
                    herbivore_positions[np.newaxis, :, :],
                    self.config.world_size,
                )
                nearest_prey = np.linalg.norm(deltas, axis=2).min(axis=1)
                prey_signal = np.exp(-nearest_prey / agent.traits.perception)
            else:
                prey_signal = np.zeros(len(candidates))
            scores = (2.5 + 2.0 * energy_deficit) * prey_signal

        scores -= 0.045 * travel * agent.traits.metabolism
        scores += self.rng.normal(
            0.0,
            0.055 + 0.05 * agent.traits.boldness,
            len(candidates),
        )

        selected = int(np.argmax(scores))
        return candidates[selected], float(travel[selected])

    def _move_agents(self) -> None:
        herbivores = self.herbivores
        predators = self.predators
        herbivore_positions = (
            np.vstack([agent.position for agent in herbivores])
            if herbivores
            else np.empty((0, 2))
        )
        predator_positions = (
            np.vstack([agent.position for agent in predators])
            if predators
            else np.empty((0, 2))
        )
        decisions = [
            (
                agent,
                *self._choose_position(
                    agent,
                    herbivore_positions,
                    predator_positions,
                ),
            )
            for agent in self.agents
            if agent.alive
        ]
        for agent, position, distance in decisions:
            agent.position = position
            agent.distance_moved = distance

    def _feed_herbivores(self) -> None:
        herbivores = self.herbivores
        if not herbivores:
            return
        for index in self.rng.permutation(len(herbivores)):
            agent = herbivores[int(index)]
            cell = self._cell(agent.position)
            biomass = self.resources[cell]
            intake = agent.traits.max_intake * biomass / (
                self.config.half_saturation + biomass
            )
            intake = min(float(intake), float(biomass))
            self.resources[cell] -= intake
            agent.energy = min(
                self.config.herbivore_max_energy,
                agent.energy + intake * agent.traits.assimilation,
            )

    def _predation(self) -> None:
        predators = self.predators
        if not predators:
            return
        for index in self.rng.permutation(len(predators)):
            predator = predators[int(index)]
            prey = self.herbivores
            if not prey:
                return
            distances = self._distances_to(predator.position, prey)
            target_index = int(np.argmin(distances))
            if distances[target_index] > self.config.capture_radius:
                continue
            target = prey[target_index]
            speed_advantage = predator.traits.speed / (
                predator.traits.speed + target.traits.speed
            )
            success_probability = (
                self.config.capture_probability
                * speed_advantage
                * (1.18 - 0.28 * target.traits.boldness)
            )
            if self.rng.random() < success_probability:
                target.alive = False
                predator.energy = min(
                    self.config.predator_max_energy,
                    predator.energy
                    + predator.traits.assimilation
                    * min(
                        55.0,
                        31.0 + max(target.energy, 0.0),
                    ),
                )
                self.last_events["kills"] += 1
                self.last_events["herbivore_deaths"] += 1

    def _recycle_carcass(self, agent: Agent) -> None:
        cell = self._cell(agent.position)
        recycled = self.config.carcass_recycling * (7.0 + max(agent.energy, 0.0))
        self.resources[cell] = min(
            self.carrying_capacity[cell],
            self.resources[cell] + recycled,
        )

    def _metabolism_and_mortality(self) -> None:
        for agent in self.agents:
            if not agent.alive:
                continue
            locomotion = (
                self.config.movement_cost
                * agent.distance_moved
                * (1.0 + 0.12 * agent.traits.speed**2)
            )
            agent.energy -= agent.traits.metabolism + locomotion
            agent.age += 1
            mortality_probability = self.config.background_mortality * (
                1.25 if agent.species == "predator" else 1.0
            )
            dies = (
                agent.energy <= 0.0
                or agent.age >= self.config.max_age
                or self.rng.random() < mortality_probability
            )
            if dies:
                agent.alive = False
                self.last_events[f"{agent.species}_deaths"] += 1
                self._recycle_carcass(agent)

    def _reproduce(self) -> None:
        offspring: list[Agent] = []
        population_counts = {
            "herbivore": len(self.herbivores),
            "predator": len(self.predators),
        }
        for agent in list(self.agents):
            if not agent.alive or agent.age < 18:
                continue
            if agent.species == "herbivore":
                threshold = self.config.herbivore_reproduction_energy
                cost = self.config.herbivore_reproduction_cost
                birth_energy = self.config.herbivore_birth_energy
                rate = self.config.herbivore_reproduction_rate
                population_limit = self.config.max_herbivores
            else:
                threshold = self.config.predator_reproduction_energy
                cost = self.config.predator_reproduction_cost
                birth_energy = self.config.predator_birth_energy
                rate = self.config.predator_reproduction_rate
                population_limit = self.config.max_predators

            if population_counts[agent.species] >= population_limit:
                continue
            energy_factor = np.clip(agent.energy / threshold - 0.75, 0.0, 1.5)
            if agent.energy >= threshold and self.rng.random() < rate * energy_factor:
                jitter = self.rng.normal(0.0, 0.8, 2)
                child = self._make_agent(
                    species=agent.species,
                    position=(agent.position + jitter) % self.config.world_size,
                    energy=birth_energy,
                    traits=agent.traits.mutate(
                        agent.species,
                        self.rng,
                        self.config.mutation_sigma,
                    ),
                )
                agent.energy -= cost
                offspring.append(child)
                population_counts[agent.species] += 1
                self.last_events[f"{agent.species}_births"] += 1
        self.agents.extend(offspring)

    def step(self) -> dict[str, float | int]:
        self.last_events = {key: 0 for key in self.last_events}
        self._grow_resources()
        self._move_agents()
        self._feed_herbivores()
        self._predation()
        self._metabolism_and_mortality()
        self._reproduce()
        self.agents = [agent for agent in self.agents if agent.alive]
        self.time += 1
        record = self.summary()
        self.records.append(record)
        return record

    def run(self, steps: int) -> list[dict[str, float | int]]:
        if steps < 0:
            raise ValueError("steps cannot be negative")
        for _ in range(steps):
            self.step()
        return self.records

    def summary(self) -> dict[str, float | int]:
        herbivores = self.herbivores
        predators = self.predators

        def mean_energy(population: Sequence[Agent]) -> float:
            return float(np.mean([agent.energy for agent in population])) if population else 0.0

        def mean_trait(population: Sequence[Agent], name: str) -> float:
            if not population:
                return 0.0
            return float(np.mean([getattr(agent.traits, name) for agent in population]))

        return {
            "step": self.time,
            "herbivores": len(herbivores),
            "predators": len(predators),
            "resource_biomass": float(self.resources.sum()),
            "resource_fraction": float(
                self.resources.sum() / self.carrying_capacity.sum()
            ),
            "mean_herbivore_energy": mean_energy(herbivores),
            "mean_predator_energy": mean_energy(predators),
            "mean_herbivore_speed": mean_trait(herbivores, "speed"),
            "mean_predator_speed": mean_trait(predators, "speed"),
            **self.last_events,
        }

    def with_config(self, **changes: object) -> "EcoSimulation":
        """Create a fresh simulation with selected configuration changes."""
        return EcoSimulation(replace(self.config, **changes))
