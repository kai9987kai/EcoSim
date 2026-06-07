# EcoSim

EcoSim is a reproducible, spatially explicit individual-based ecosystem model.
Herbivores forage across a heterogeneous renewable landscape, avoid predators,
maintain energy reserves, reproduce, and pass mutated traits to offspring.
Predators search for prey under the same energetic constraints. Death also
returns a fraction of biomass to the landscape.

The supported model replaces the old `SGDRegressor` experiment. That code
trained a scalar environmental reward as though it were a movement action and
therefore did not implement reinforcement learning or a valid learned policy.

## Run

```powershell
python -m pip install -e ".[dev]"
python main.py --steps 500 --seed 7
python -m pytest
```

Each run writes:

- `results/timeseries.csv`: populations, energy, resources, births, deaths, kills,
  and evolving mean speed.
- `results/ecosim.png`: the final landscape plus population and energy time series.

Useful options:

```powershell
python main.py --steps 1000 --seed 42 --herbivores 120 --predators 20
python main.py --steps 200 --no-plot --output-dir experiment-01
```

Runs with the same configuration and seed are deterministic. `Beta.py` and
`3dbeta.py` remain only as compatibility launchers. `CanvasBeta.html` is an
older, independent browser prototype and does not share the Python model.

## Model advances

- Dynamic energy budget: intake is stored, while maintenance and locomotion
  consume reserves before reproduction.
- Holling type-II herbivore intake prevents unlimited feeding at dense patches.
- State-dependent movement balances resource gain, predation risk, travel cost,
  perception, and exploration.
- Spatial heterogeneity and seasonal logistic resource renewal create shifting
  opportunities rather than static random points.
- Eco-evolutionary dynamics inherit speed, perception, metabolism, intake,
  assimilation, and boldness with bounded mutation.
- Toroidal geometry avoids artificial edge crowding.
- Randomized feeding and hunting order reduces fixed update-order advantage.
- Seeded experiments, CSV outputs, tests, and an ODD-style specification improve
  transparency and repeatability.

This is a hypothesis-generating simulation, not a calibrated forecast. Parameter
values are dimensionless and should be fitted to empirical data before making
claims about a real ecosystem.

## Research basis

The implementation is a simplified synthesis, not a reproduction of any one
paper:

- Bradley et al. (2025), [dynamic energy budgets inside a spatial
  agent-based model](https://doi.org/10.3389/fevo.2025.1505145): motivates
  reserves, maintenance-first allocation, resource feedback, and heterogeneous
  individual state.
- Szangolies et al. (2024), [individual energetics, movement, and community
  coexistence](https://doi.org/10.1111/1365-2656.14134): motivates explicit
  energy balance and spatial movement.
- Liu et al. (2024), [MetaIBM](https://doi.org/10.1016/j.ecolmodel.2024.110730):
  motivates inherited individual variation in spatial eco-evolutionary models.
- Grimm et al. (2025), [ODD-assisted model
  replication](https://doi.org/10.1016/j.ecolmodel.2024.110967): motivates the
  structured specification in [MODEL.md](MODEL.md).

The model deliberately avoids deep reinforcement learning: without behavioral
observations, a validated reward, and out-of-sample evaluation, adding a neural
policy would increase opacity without increasing ecological credibility.
