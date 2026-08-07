"""Expose the configured computation through the generic executor."""

from computation.spec import SPEC
from framework.executor import ComputationExecutor


class RuntimeExecutor(ComputationExecutor):
    """Execute configured computation steps at a participating site."""

    SPEC = SPEC
