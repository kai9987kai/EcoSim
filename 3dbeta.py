"""Compatibility entry point for the retired experimental 3D simulation."""

import warnings

from ecosim.cli import main


if __name__ == "__main__":
    warnings.warn(
        "The incomplete 3D beta was retired; this launches the supported 2D model.",
        FutureWarning,
        stacklevel=1,
    )
    raise SystemExit(main())
