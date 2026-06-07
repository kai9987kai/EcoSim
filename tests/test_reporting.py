import csv

from ecosim.model import EcoSimulation, SimulationConfig
from ecosim.reporting import plot_simulation, write_csv


def test_reports_are_created(tmp_path):
    simulation = EcoSimulation(
        SimulationConfig(
            grid_size=16,
            initial_herbivores=8,
            initial_predators=2,
            seed=11,
        )
    )
    simulation.run(4)

    csv_path = write_csv(simulation.records, tmp_path / "run.csv")
    image_path = plot_simulation(simulation, tmp_path / "run.png")

    assert csv_path.stat().st_size > 0
    assert image_path.stat().st_size > 0
    with csv_path.open(encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 5
