"""Command-line interface for reproducible EcoSim runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .model import EcoSimulation, SimulationConfig
from .reporting import plot_simulation, write_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a spatial eco-evolutionary predator-prey simulation."
    )
    parser.add_argument("--steps", type=int, default=500, help="Simulation steps.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--herbivores", type=int, default=90, help="Initial herbivores.")
    parser.add_argument("--predators", type=int, default=18, help="Initial predators.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for CSV and plot outputs.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip PNG generation.")
    parser.add_argument("--show", action="store_true", help="Open the plot window.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.steps < 0:
        raise SystemExit("--steps must be non-negative")
    config = SimulationConfig(
        seed=args.seed,
        initial_herbivores=args.herbivores,
        initial_predators=args.predators,
    )
    simulation = EcoSimulation(config)
    simulation.run(args.steps)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = write_csv(simulation.records, args.output_dir / "timeseries.csv")
    plot_path = None
    if not args.no_plot:
        plot_path = plot_simulation(
            simulation,
            args.output_dir / "ecosim.png",
            show=args.show,
        )

    summary = simulation.summary()
    print(
        f"Completed {args.steps} steps with seed {args.seed}: "
        f"{summary['herbivores']} herbivores, {summary['predators']} predators, "
        f"{100 * summary['resource_fraction']:.1f}% resource capacity."
    )
    print(f"Time series: {csv_path}")
    if plot_path:
        print(f"Plot: {plot_path}")
    return 0
