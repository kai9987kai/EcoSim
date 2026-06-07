"""Compatibility entry point for the retired beta simulation."""

import warnings

from ecosim.cli import main


if __name__ == "__main__":
    warnings.warn(
        "Beta.py now launches the supported EcoSim model. Prefer `python main.py`.",
        FutureWarning,
        stacklevel=1,
    )
    raise SystemExit(main())
