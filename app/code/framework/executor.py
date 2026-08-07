"""Execute computation-defined functions at participating sites."""

from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode, SiteType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal

from .artifact_transfer import (
    ARTIFACT_MANIFEST_KEY,
    contains_artifacts,
    get_artifact_transfer,
    materialize_incoming_artifacts,
    prepare_outgoing_artifacts,
)
from .cache import JsonStateStore
from .errors import (
    ERROR_ENVELOPE_KEY,
    ERROR_ORIGIN_SITE,
    build_error_envelope,
    clear_terminal_error,
    record_terminal_error,
)
from .logger import close_computation_logger
from .serialization import deserialize_value, serialize_value
from .shared import (
    build_runtime_context,
    load_computation_parameters,
    resolve_output_directory,
)
from .types import (
    ITERATION_INDEX_KEY,
    ComputationSpec,
    IterativeWorkflow,
    SteppedWorkflow,
)
from .writers import write_standard_outputs


class ComputationExecutor(Executor):
    """Resolve and run the local function associated with an NVFlare task."""

    SPEC: ComputationSpec = None

    def __init__(self):
        """Initialize the per-output-directory state-store registry."""
        super().__init__()
        self._state_stores = {}

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        """Execute one local or output step and return its transport payload."""
        output_dir = resolve_output_directory(fl_ctx)
        runtime = None

        try:
            if self.SPEC is None:
                raise ValueError("Executor SPEC must be defined")

            if output_dir not in self._state_stores:
                clear_terminal_error(output_dir)
            parameters = load_computation_parameters(fl_ctx)
            state_store = JsonStateStore(
                output_dir,
                self.SPEC.codecs,
                self.SPEC.max_inline_array_bytes,
            )
            self._state_stores[output_dir] = state_store
            current_round = fl_ctx.get_prop("CURRENT_ROUND", default=0)
            workflow = self.SPEC.workflow

            local_fn = None
            local_input_type = None
            expects_remote_result = False
            clear_state_after_step = False

            if isinstance(workflow, SteppedWorkflow):
                step_definition = next(
                    (step for step in workflow.steps if step.name == task_name),
                    None,
                )
                if step_definition is None:
                    raise ValueError(f"Unknown task name: {task_name}")

                local_fn = step_definition.local_fn
                local_input_type = step_definition.local_input_type
                expects_remote_result = step_definition.remote_fn is not None
                clear_state_after_step = (
                    step_definition.is_site_output
                    and step_definition is workflow.steps[-1]
                )
            elif isinstance(workflow, IterativeWorkflow):
                if task_name == workflow.iteration_step.name:
                    local_fn = workflow.iteration_step.local_fn
                    local_input_type = workflow.iteration_step.local_input_type
                    expects_remote_result = True
                elif task_name == workflow.output_step.name:
                    local_fn = workflow.output_step.local_fn
                    local_input_type = workflow.output_step.local_input_type
                    clear_state_after_step = True
                else:
                    raise ValueError(f"Unknown task name: {task_name}")
            else:
                raise ValueError(f"Unsupported workflow type: {type(workflow)!r}")

            local_state_type = workflow.local_state_type
            if isinstance(workflow, IterativeWorkflow):
                current_round = shareable.get(ITERATION_INDEX_KEY, current_round)
            local_state = state_store.load_state(local_state_type)
            incoming_value = shareable.get("result")
            incoming_manifests = shareable.get(ARTIFACT_MANIFEST_KEY)
            if incoming_manifests:
                incoming_value = materialize_incoming_artifacts(
                    incoming_value,
                    incoming_manifests,
                    transfer=get_artifact_transfer(fl_ctx),
                    from_site=SiteType.SERVER,
                    fl_ctx=fl_ctx,
                    expected_stage=task_name,
                    expected_direction="central_to_site",
                    timeout=self.SPEC.artifact_timeout,
                    retries=self.SPEC.artifact_retries,
                    max_file_bytes=self.SPEC.max_artifact_bytes,
                    max_total_bytes=self.SPEC.max_artifact_total_bytes,
                )
            incoming_payload = deserialize_value(
                incoming_value,
                local_input_type,
                self.SPEC.codecs,
                max_inline_array_bytes=self.SPEC.max_inline_array_bytes,
            )

            runtime = build_runtime_context(
                self.SPEC,
                fl_ctx,
                current_round,
                parameters,
                logger_suffix=".log",
            )

            step_result = local_fn(
                incoming_payload,
                parameters,
                local_state,
                runtime,
            )

            if step_result.local_state is not None:
                state_store.save_state(step_result.local_state)

            if step_result.outputs:
                write_standard_outputs(step_result.outputs, runtime)

            if clear_state_after_step:
                self._remove_state(output_dir)

            if not expects_remote_result:
                return Shareable()

            outgoing_shareable = Shareable()
            outgoing_payload = step_result.payload
            outgoing_manifests = []
            if contains_artifacts(outgoing_payload):
                peer_ctx = fl_ctx.get_peer_context()
                peer_identity = peer_ctx.get_identity_name() if peer_ctx else None
                requesters = [SiteType.SERVER]
                if peer_identity and peer_identity not in requesters:
                    requesters.append(peer_identity)
                outgoing_payload, outgoing_manifests = prepare_outgoing_artifacts(
                    outgoing_payload,
                    transfer=get_artifact_transfer(fl_ctx),
                    source_root=runtime.artifact_dir,
                    allowed_requesters=requesters,
                    stage=task_name,
                    direction="site_to_central",
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
            return outgoing_shareable
        except Exception as error:
            scope = f"site task {task_name}"
            record_terminal_error(
                output_dir,
                scope,
                error,
                origin=ERROR_ORIGIN_SITE,
                stage="task_execution",
            )
            if runtime and runtime.logger:
                runtime.logger.critical(
                    "Computation task '%s' failed",
                    task_name,
                    exc_info=True,
                )
            failure = Shareable()
            failure.set_return_code(ReturnCode.EXECUTION_EXCEPTION)
            failure[ERROR_ENVELOPE_KEY] = build_error_envelope(
                ERROR_ORIGIN_SITE,
                "task_execution",
                scope,
            )
            return failure
        finally:
            if runtime and runtime.logger:
                close_computation_logger(runtime.logger)

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        """Remove all persistent site state when the run ends."""
        if event_type == EventType.END_RUN:
            for output_dir in tuple(self._state_stores):
                self._remove_state(output_dir)

    def _remove_state(self, output_dir: str) -> None:
        state_store = self._state_stores.pop(output_dir, None)
        if state_store is not None:
            state_store.remove_state()


MultiRoundTabularExecutor = ComputationExecutor
