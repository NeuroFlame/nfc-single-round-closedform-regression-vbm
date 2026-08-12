"""Run computation-defined aggregation on the NVFlare server."""

import threading
from typing import Any, Dict

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.aggregator import Aggregator

from .artifact_transfer import (
    ARTIFACT_MANIFEST_KEY,
    ArtifactTransferError,
    contains_artifacts,
    get_artifact_transfer,
    materialize_incoming_artifacts,
    prepare_outgoing_artifacts,
    validate_artifact_manifests,
)
from .errors import ERROR_ORIGIN_CENTRAL, record_terminal_error
from .logger import close_computation_logger
from .serialization import deserialize_value, serialize_value
from .shared import (
    build_runtime_context,
    load_computation_parameters,
    resolve_site_name,
)
from .types import (
    ITERATION_STOP_KEY,
    ComputationSpec,
    IterativeWorkflow,
    SteppedWorkflow,
)


class ComputationAggregator(Aggregator):
    """Collect site results and invoke the current remote computation step."""

    SPEC: ComputationSpec = None

    def __init__(self):
        """Initialize aggregation state for the configured computation."""
        super().__init__()
        if self.SPEC is None:
            raise ValueError("Aggregator SPEC must be defined")
        self.site_results: Dict[int, Dict[str, Any]] = {}
        self.artifact_reservations: Dict[int, Dict[tuple[str, str], int]] = {}
        self._accept_lock = threading.RLock()
        self.remote_state: Any = None

    def accept(self, site_result: Shareable, fl_ctx: FLContext) -> bool:
        """Store one named site's result for the current workflow round."""
        site_id = site_result.get_peer_prop(key=ReservedKey.IDENTITY_NAME, default=None)
        current_round = fl_ctx.get_prop(key="CURRENT_ROUND", default=None)

        if current_round is None or site_id is None:
            return False

        parameters = fl_ctx.get_prop(key="COMPUTATION_PARAMETERS", default={})
        site_name = resolve_site_name(site_id, parameters)
        site_payload = site_result["result"]
        manifests = site_result.get(ARTIFACT_MANIFEST_KEY)
        new_reservations: list[tuple[str, str]] = []
        if manifests:
            by_token, _ = validate_artifact_manifests(
                manifests,
                expected_stage=self._current_step_name(current_round),
                expected_direction="site_to_central",
                max_file_bytes=self.SPEC.max_artifact_bytes,
            )
            new_reservations = self._reserve_artifact_bytes(
                current_round, site_id, by_token
            )
            try:
                site_payload = materialize_incoming_artifacts(
                    site_payload,
                    manifests,
                    transfer=get_artifact_transfer(fl_ctx),
                    from_site=site_id,
                    fl_ctx=fl_ctx,
                    expected_stage=self._current_step_name(current_round),
                    expected_direction="site_to_central",
                    timeout=self.SPEC.artifact_timeout,
                    retries=self.SPEC.artifact_retries,
                    max_file_bytes=self.SPEC.max_artifact_bytes,
                    max_total_bytes=self.SPEC.max_artifact_total_bytes,
                )
            except Exception as error:
                if not (
                    isinstance(error, ArtifactTransferError) and error.indeterminate
                ):
                    self._roll_back_artifact_reservations(
                        current_round, new_reservations
                    )
                raise
        with self._accept_lock:
            self.site_results.setdefault(current_round, {})
            self.site_results[current_round][site_name] = site_payload
        return True

    def _reserve_artifact_bytes(
        self, current_round: int, site_id: str, by_token: Dict[str, Dict[str, Any]]
    ) -> list[tuple[str, str]]:
        """Atomically reserve cumulative round quota for new authenticated tokens."""
        with self._accept_lock:
            reservations = self.artifact_reservations.setdefault(current_round, {})
            new_keys = [
                (site_id, token)
                for token in by_token
                if (site_id, token) not in reservations
            ]
            additional_bytes = sum(by_token[token]["size"] for _, token in new_keys)
            if (
                sum(reservations.values()) + additional_bytes
                > self.SPEC.max_artifact_total_bytes
            ):
                raise ArtifactTransferError(
                    "Aggregate artifact transfer size exceeds the configured limit"
                )
            for key in new_keys:
                reservations[key] = by_token[key[1]]["size"]
            return new_keys

    def _roll_back_artifact_reservations(
        self, current_round: int, reservation_keys: list[tuple[str, str]]
    ) -> None:
        """Release only reservations created by a failed retrieval attempt."""
        with self._accept_lock:
            reservations = self.artifact_reservations.get(current_round)
            if reservations is None:
                return
            for key in reservation_keys:
                reservations.pop(key, None)
            if not reservations:
                self.artifact_reservations.pop(current_round, None)

    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        """Aggregate current-round site results into the next site payload."""
        current_round = fl_ctx.get_prop(key="CURRENT_ROUND", default=None)
        if current_round is None:
            return Shareable()

        workflow = self.SPEC.workflow
        remote_site_result_type = None
        if isinstance(workflow, SteppedWorkflow):
            step_definition = workflow.steps[current_round]
            remote_fn = step_definition.remote_fn
            remote_site_result_type = step_definition.remote_site_result_type
            if remote_fn is None:
                return Shareable()
        elif isinstance(workflow, IterativeWorkflow):
            remote_fn = workflow.iteration_step.remote_fn
            remote_site_result_type = workflow.iteration_step.remote_site_result_type
        else:
            raise ValueError(f"Unsupported workflow type: {type(workflow)!r}")

        parameters = load_computation_parameters(fl_ctx)
        runtime = build_runtime_context(
            self.SPEC,
            fl_ctx,
            current_round,
            parameters,
            logger_suffix=".remote.log",
        )

        try:
            site_results = {
                site_name: deserialize_value(
                    site_payload,
                    remote_site_result_type,
                    self.SPEC.codecs,
                    max_inline_array_bytes=self.SPEC.max_inline_array_bytes,
                )
                for site_name, site_payload in self.site_results.get(
                    current_round, {}
                ).items()
            }
            step_result = remote_fn(
                site_results,
                parameters,
                self.remote_state,
                runtime,
            )
            should_stop = False
            if isinstance(workflow, IterativeWorkflow) and workflow.stop_fn is not None:
                stop_state = (
                    step_result.remote_state
                    if step_result.remote_state is not None
                    else self.remote_state
                )
                should_stop = workflow.stop_fn(
                    step_result.payload,
                    parameters,
                    stop_state,
                    runtime,
                )

            if step_result.remote_state is not None:
                self.remote_state = step_result.remote_state

            outgoing_shareable = Shareable()
            outgoing_payload = step_result.payload
            outgoing_manifests = []
            if contains_artifacts(outgoing_payload):
                destinations = [
                    client.name for client in fl_ctx.get_engine().get_clients()
                ]
                outgoing_payload, outgoing_manifests = prepare_outgoing_artifacts(
                    outgoing_payload,
                    transfer=get_artifact_transfer(fl_ctx),
                    source_root=runtime.artifact_dir,
                    allowed_requesters=destinations,
                    stage=self._next_step_name(current_round, should_stop),
                    direction="central_to_site",
                    max_file_bytes=self.SPEC.max_artifact_bytes,
                    max_total_bytes=self.SPEC.max_artifact_total_bytes,
                )
            outgoing_shareable["result"] = serialize_value(
                outgoing_payload,
                self.SPEC.codecs,
                max_inline_array_bytes=self.SPEC.max_inline_array_bytes,
            )
            if outgoing_manifests:
                outgoing_shareable[ARTIFACT_MANIFEST_KEY] = outgoing_manifests
            if isinstance(workflow, IterativeWorkflow):
                outgoing_shareable[ITERATION_STOP_KEY] = should_stop
            return outgoing_shareable
        except Exception as error:
            record_terminal_error(
                runtime.output_dir,
                f"remote round {current_round}",
                error,
                origin=ERROR_ORIGIN_CENTRAL,
                stage="aggregation",
            )
            if runtime.logger:
                runtime.logger.critical(
                    "Remote computation failed in round %s",
                    current_round,
                    exc_info=True,
                )
            raise
        finally:
            if runtime.logger:
                close_computation_logger(runtime.logger)

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        """Clear persistent server state when the run ends."""
        if event_type == EventType.END_RUN:
            with self._accept_lock:
                self.site_results.clear()
                self.artifact_reservations.clear()
                self.remote_state = None

    def _current_step_name(self, current_round: int) -> str:
        workflow = self.SPEC.workflow
        if isinstance(workflow, SteppedWorkflow):
            return workflow.steps[current_round].name
        return workflow.iteration_step.name

    def _next_step_name(self, current_round: int, should_stop: bool) -> str:
        workflow = self.SPEC.workflow
        if isinstance(workflow, SteppedWorkflow):
            next_index = current_round + 1
            if next_index >= len(workflow.steps):
                raise RuntimeError(
                    "An artifact cannot be emitted after the final workflow step"
                )
            return workflow.steps[next_index].name
        if should_stop or current_round + 1 >= workflow.max_iterations:
            return workflow.output_step.name
        return workflow.iteration_step.name


MultiRoundTabularAggregator = ComputationAggregator
