"""Coordinate stepped and iterative computations through NVFlare tasks."""

from typing import Callable

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import ClientTask, Task, TaskCompletionStatus
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal

from .artifact_transfer import ArtifactTransferError
from .errors import (
    ERROR_ENVELOPE_KEY,
    ERROR_ORIGIN_CENTRAL,
    ERROR_ORIGIN_SITE,
    clear_terminal_error,
    parse_error_envelope,
    record_terminal_error,
)
from .shared import load_computation_parameters, resolve_output_directory
from .types import (
    ITERATION_INDEX_KEY,
    ITERATION_STOP_KEY,
    ComputationSpec,
    IterativeWorkflow,
    SteppedWorkflow,
)

_AGGREGATOR_COMPONENT_ID = "aggregator"


class RelayedSiteFailure(RuntimeError):
    """Represent a site failure relayed through the central controller."""


class ComputationController(Controller):
    """Dispatch computation steps and route results through the aggregator."""

    SPEC: ComputationSpec = None

    def __init__(
        self,
        min_clients: int = 2,
        wait_time_after_min_received: int = 10,
        task_timeout: int = 0,
    ):
        """Configure participant thresholds and task timing."""
        super().__init__()
        if self.SPEC is None:
            raise ValueError("Controller SPEC must be defined")
        self._task_timeout = task_timeout
        self._min_clients = min_clients
        self._wait_time_after_min_received = wait_time_after_min_received

    def start_controller(self, fl_ctx: FLContext) -> None:
        """Load run parameters and resolve the configured aggregator."""
        output_dir = resolve_output_directory(fl_ctx)
        try:
            clear_terminal_error(output_dir)
            self.aggregator = self._engine.get_component(_AGGREGATOR_COMPONENT_ID)
            fl_ctx.set_prop(
                key="COMPUTATION_PARAMETERS",
                value=load_computation_parameters(fl_ctx),
                private=False,
                sticky=True,
            )
        except Exception as error:
            record_terminal_error(
                output_dir,
                "controller startup",
                error,
                origin=ERROR_ORIGIN_CENTRAL,
                stage="controller_startup",
            )
            raise

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext) -> None:
        """Execute the configured stepped or iterative workflow."""
        try:
            workflow = self.SPEC.workflow
            if isinstance(workflow, SteppedWorkflow):
                self._run_stepped_workflow(workflow, abort_signal, fl_ctx)
                return
            if isinstance(workflow, IterativeWorkflow):
                self._run_iterative_workflow(workflow, abort_signal, fl_ctx)
                return
            raise ValueError(f"Unsupported workflow type: {type(workflow)!r}")
        except Exception as error:
            is_site_failure = isinstance(error, RelayedSiteFailure)
            record_terminal_error(
                resolve_output_directory(fl_ctx),
                "controller",
                error,
                origin=(ERROR_ORIGIN_SITE if is_site_failure else ERROR_ORIGIN_CENTRAL),
                stage="site_result" if is_site_failure else "controller_execution",
            )
            raise

    def _run_stepped_workflow(
        self,
        workflow: SteppedWorkflow,
        abort_signal: Signal,
        fl_ctx: FLContext,
    ) -> None:
        incoming_shareable = Shareable()

        for current_round, step_definition in enumerate(workflow.steps):
            fl_ctx.set_prop(key="CURRENT_ROUND", value=current_round)
            expects_remote_result = step_definition.remote_fn is not None

            self._broadcast_task(
                task_name=step_definition.name,
                data=incoming_shareable,
                result_cb=(
                    self._accept_site_result
                    if expects_remote_result
                    else self._validate_site_result
                ),
                fl_ctx=fl_ctx,
                abort_signal=abort_signal,
            )

            if expects_remote_result:
                incoming_shareable = self.aggregator.aggregate(fl_ctx)
            else:
                incoming_shareable = Shareable()

    def _run_iterative_workflow(
        self,
        workflow: IterativeWorkflow,
        abort_signal: Signal,
        fl_ctx: FLContext,
    ) -> None:
        incoming_shareable = Shareable()

        for current_round in range(workflow.max_iterations):
            fl_ctx.set_prop(key="CURRENT_ROUND", value=current_round)
            incoming_shareable[ITERATION_INDEX_KEY] = current_round
            self._broadcast_task(
                task_name=workflow.iteration_step.name,
                data=incoming_shareable,
                result_cb=self._accept_site_result,
                fl_ctx=fl_ctx,
                abort_signal=abort_signal,
            )
            incoming_shareable = self.aggregator.aggregate(fl_ctx)
            if incoming_shareable.get(ITERATION_STOP_KEY, False):
                break

        fl_ctx.set_prop(key="CURRENT_ROUND", value=current_round + 1)
        incoming_shareable[ITERATION_INDEX_KEY] = current_round + 1
        self._broadcast_task(
            task_name=workflow.output_step.name,
            data=incoming_shareable,
            result_cb=self._validate_site_result,
            fl_ctx=fl_ctx,
            abort_signal=abort_signal,
        )

    def _accept_site_result(self, client_task: ClientTask, fl_ctx: FLContext) -> bool:
        self._validate_site_result(client_task, fl_ctx)
        try:
            accepted = self.aggregator.accept(client_task.result, fl_ctx)
        except ArtifactTransferError as error:
            raise RelayedSiteFailure(
                f"Artifact transfer for task '{client_task.task.name}' failed from "
                f"site '{client_task.client.name}'"
            ) from error
        if not accepted:
            raise RuntimeError(
                f"Task '{client_task.task.name}' returned an invalid result from "
                f"site '{client_task.client.name}'"
            )
        return True

    def _validate_site_result(self, client_task: ClientTask, fl_ctx: FLContext) -> bool:
        return_code = client_task.result.get_return_code()
        if return_code != ReturnCode.OK:
            error_envelope = parse_error_envelope(
                client_task.result.get(ERROR_ENVELOPE_KEY)
            )
            error_type = (
                RelayedSiteFailure
                if error_envelope and error_envelope["origin"] == ERROR_ORIGIN_SITE
                else RuntimeError
            )
            raise error_type(
                f"Task '{client_task.task.name}' failed on site "
                f"'{client_task.client.name}' with return code '{return_code}'"
            )
        return True

    def _broadcast_task(
        self,
        task_name: str,
        data: Shareable,
        result_cb: Callable[[ClientTask, FLContext], bool],
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> None:
        task = Task(
            name=task_name,
            data=data,
            props={},
            timeout=self._task_timeout,
            result_received_cb=result_cb,
        )
        self.broadcast_and_wait(
            task=task,
            min_responses=self._min_clients,
            wait_time_after_min_received=self._wait_time_after_min_received,
            fl_ctx=fl_ctx,
            abort_signal=abort_signal,
        )
        if task.completion_status != TaskCompletionStatus.OK:
            task_exception = getattr(task, "exception", None)
            if isinstance(task_exception, RelayedSiteFailure):
                raise task_exception
            detail = f": {task_exception}" if task_exception else ""
            raise RuntimeError(
                f"Task '{task_name}' ended with status "
                f"'{task.completion_status}'{detail}"
            )

    def process_result_of_unknown_task(
        self,
        client: Client,
        task_name: str,
        client_task_id: str,
        result: Shareable,
        fl_ctx: FLContext,
    ) -> None:
        """Ignore results for tasks no longer owned by this controller."""
        pass

    def stop_controller(self, fl_ctx: FLContext) -> None:
        """Finish controller shutdown without additional cleanup."""
        pass


MultiRoundTabularController = ComputationController
