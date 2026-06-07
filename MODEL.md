# EcoSim model specification

This is an abbreviated ODD (Overview, Design concepts, Details) description.

## Purpose

EcoSim explores how individual energy balance, spatial resource heterogeneity,
predation, and heritable trait variation can generate population-level dynamics.
It is intended for education and hypothesis generation, not forecasting.

## Entities, state variables, and scales

The square world is continuous and toroidal. A raster of patches stores resource
biomass and heterogeneous carrying capacity. One step is an abstract ecological
time unit.

Individuals belong to the herbivore or predator species and store:

- two-dimensional position;
- energy reserve and age;
- speed, perception, maintenance metabolism, assimilation efficiency, and
  boldness;
- herbivores additionally have a maximum intake trait.

## Process overview and scheduling

At every step:

1. Patch resources grow logistically under a sinusoidal seasonal multiplier.
2. All living individuals choose among twelve movement directions and staying.
3. Movement decisions are applied synchronously.
4. Herbivores consume local biomass with a type-II saturating response.
5. Predators within capture distance make one stochastic hunting attempt.
6. Maintenance and movement costs are deducted; mortality and carcass recycling
   are applied.
7. Eligible high-energy individuals reproduce and traits mutate.
8. Dead individuals are removed and summary metrics are recorded.

Feeding and hunting order are randomized each step to reduce systematic
first-mover advantage.

## Design concepts

**Basic principles.** The model uses a simplified one-reserve dynamic energy
budget. Assimilated intake enters energy reserve; maintenance and locomotion are
paid before reproduction.

**Adaptation.** Movement is a bounded state-dependent decision. Herbivores trade
resource biomass against perceived predator risk. Predators use perceived prey
proximity. Low reserve increases food-seeking pressure.

**Objectives.** The movement score approximates short-horizon energetic value.
It is a transparent heuristic, not a claim of global optimality or cognition.

**Learning.** Individuals do not learn during life. Population traits can change
through differential survival and reproduction.

**Sensing.** Individuals use their internal reserve, local resource biomass, and
distances to the relevant species. Perception controls the distance decay.

**Interaction.** Herbivores compete by depleting shared patches. Predators kill
nearby herbivores probabilistically. Non-predation deaths return some biomass to
the local patch.

**Stochasticity.** Initialization, movement exploration, update order, capture,
mortality, reproduction, and mutation use one seeded NumPy generator.

**Observation.** Population sizes, resource biomass, mean energy, births, deaths,
kills, and mean speed are recorded every step.

## Initialization

Patch carrying capacity is generated from seven periodic Gaussian productivity
hotspots. Initial biomass is 45-82% of local capacity. Founders receive
species-specific mean traits with lognormal variation and random positions.

## Input data

No external data are currently used. All default values are dimensionless.
Calibration requires an explicit target species/system, units, empirical priors,
and sensitivity analysis.

## Submodels and limitations

- Resource growth uses discrete logistic growth and is clipped to local capacity.
- Herbivore intake follows a Holling type-II response.
- Capture probability depends on configured baseline capture, relative speed,
  and prey boldness.
- Reproduction is asexual and energy-gated; offspring receive bounded,
  multiplicatively mutated traits.
- Genetics, sex, disease, nutrient stoichiometry, body size, handling time,
  social behavior, and multiple resource types are omitted.
- Carcass recycling converts abstract animal energy to resource biomass, so it is
  a qualitative feedback rather than a physical mass balance.

These simplifications should be changed only in response to a defined research
question and accompanied by calibration, uncertainty analysis, and validation.
