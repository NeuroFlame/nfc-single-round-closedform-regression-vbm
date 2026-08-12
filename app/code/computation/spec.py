"""Declare the four-stage VBM regression workflow."""

from framework import (
    ComputationSpec,
    local_step,
    remote_step,
    site_output_step,
    stepped_workflow,
)

from .steps import (
    aggregate_global_metrics,
    aggregate_global_model,
    aggregate_mask,
    compute_local_metrics,
    fit_local_model,
    load_vbm_data,
    prepare_local_mask,
    write_vbm_outputs,
)

SPEC = ComputationSpec(
    workflow=stepped_workflow(
        local_step(fn=prepare_local_mask, input_fn=load_vbm_data),
        remote_step(fn=aggregate_mask),
        local_step(fn=fit_local_model),
        remote_step(fn=aggregate_global_model),
        local_step(fn=compute_local_metrics),
        remote_step(fn=aggregate_global_metrics),
        site_output_step(fn=write_vbm_outputs),
    ),
)
