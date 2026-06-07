"""Output helpers for simulation experiments."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .model import EcoSimulation


def write_csv(
    records: Sequence[Mapping[str, float | int]],
    path: str | Path,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        raise ValueError("records cannot be empty")
    with destination.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return destination


def plot_simulation(
    simulation: EcoSimulation,
    path: str | Path,
    *,
    show: bool = False,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    records = simulation.records
    steps = np.array([record["step"] for record in records])

    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    spatial, populations, energetics = axes

    spatial.imshow(
        simulation.resources,
        origin="lower",
        extent=(0, simulation.config.world_size, 0, simulation.config.world_size),
        cmap="YlGn",
        alpha=0.88,
        aspect="equal",
    )
    herbivores = simulation.herbivores
    predators = simulation.predators
    if herbivores:
        positions = np.vstack([agent.position for agent in herbivores])
        spatial.scatter(
            positions[:, 0],
            positions[:, 1],
            s=14,
            c="#2463eb",
            alpha=0.8,
            label="Herbivores",
        )
    if predators:
        positions = np.vstack([agent.position for agent in predators])
        spatial.scatter(
            positions[:, 0],
            positions[:, 1],
            s=28,
            marker="^",
            c="#dc2626",
            alpha=0.9,
            label="Predators",
        )
    spatial.set(title=f"Landscape at step {simulation.time}", xlabel="x", ylabel="y")
    if herbivores or predators:
        spatial.legend(loc="upper right")

    populations.plot(
        steps,
        [record["herbivores"] for record in records],
        label="Herbivores",
        color="#2463eb",
    )
    populations.plot(
        steps,
        [record["predators"] for record in records],
        label="Predators",
        color="#dc2626",
    )
    populations.set(title="Population dynamics", xlabel="Step", ylabel="Individuals")
    populations.legend()
    populations.grid(alpha=0.2)

    energetics.plot(
        steps,
        [record["mean_herbivore_energy"] for record in records],
        label="Herbivore energy",
        color="#2563eb",
    )
    energetics.plot(
        steps,
        [record["mean_predator_energy"] for record in records],
        label="Predator energy",
        color="#b91c1c",
    )
    energetics.plot(
        steps,
        [100 * record["resource_fraction"] for record in records],
        label="Resource capacity (%)",
        color="#15803d",
        linestyle="--",
    )
    energetics.set(title="Energy and resources", xlabel="Step", ylabel="Relative units")
    energetics.legend()
    energetics.grid(alpha=0.2)

    figure.suptitle(
        f"EcoSim | seed={simulation.config.seed} | "
        f"H={len(herbivores)} P={len(predators)}"
    )
    figure.savefig(destination, dpi=180)
    if show:
        plt.show()
    plt.close(figure)
    return destination
