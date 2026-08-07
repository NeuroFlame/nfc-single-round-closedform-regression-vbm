"""Expose the configured computation through the generic controller."""

from computation.spec import SPEC
from framework.controller import ComputationController


class RuntimeController(ComputationController):
    """Coordinate the configured computation workflow."""

    SPEC = SPEC
