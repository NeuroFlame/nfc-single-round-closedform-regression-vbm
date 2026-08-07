"""Expose the configured computation through the generic aggregator."""

from computation.spec import SPEC
from framework.aggregator import ComputationAggregator


class RuntimeAggregator(ComputationAggregator):
    """Run server-side aggregation for the configured computation."""

    SPEC = SPEC
