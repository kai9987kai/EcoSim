"""A reproducible, spatially explicit eco-evolutionary simulation."""

from .model import Agent, EcoSimulation, SimulationConfig, Traits, toroidal_delta

__all__ = [
    "Agent",
    "EcoSimulation",
    "SimulationConfig",
    "Traits",
    "toroidal_delta",
]

__version__ = "1.0.0"
