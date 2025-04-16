import json
from nvflare.apis.impl.controller import Controller, Task, ClientTask
from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal
from nvflare.apis.shareable import Shareable
from utils.utils import get_parameters_file_path
from utils.task_constants import *
from typing import Callable, Dict, Any

class SRCFRController(Controller):
    """
    SrrController handles the flow of tasks for site regression and aggregation
    in the federated learning process. Developers can define and control the
    flow of computations across multiple sites using this class.
    """

    ### Framework-Specific Setup: No modification needed ###
    def __init__(
            self,
            min_clients: int = 2,
            wait_time_after_min_received: int = 10,
            task_timeout: int = 0,
    ):
        """
        Initializes the SrrController with specific parameters for task broadcasting.

        :param min_clients: Minimum number of client responses required.
        :param wait_time_after_min_received: Time to wait after receiving minimum responses.
        :param task_timeout: Timeout for task completion.
        """
        super().__init__()
        self._task_timeout = task_timeout
        self._min_clients = min_clients
        self._wait_time_after_min_received = wait_time_after_min_received

    #### Computation Author Defined Section ####
    ### This is where computation authors will define the control flow logic ###

    def start_controller(self, fl_ctx: FLContext) -> None:
        """
        Called when the controller starts. It assigns the SRR aggregator component
        and loads computation parameters into the shared context.

        This is a Framework-Specific Required Method.

        :param fl_ctx: Federated learning context for this run.
        """
        # Assign the aggregator to the controller
        self.srr_aggregator = self._engine.get_component(SRCFR_AGGREGATOR_ID)
        # Load and set computation parameters for the sites
        self._load_and_set_computation_parameters(fl_ctx)

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext) -> None:
        """
        Main method for defining the control flow of computations. This is where
        developers implement the main workflow. Broadcasts tasks to all sites,
        aggregates results, and broadcasts the final global result.

        This is the primary method that computation authors should focus on.

        :param abort_signal: Signal for aborting the flow if needed.
        :param fl_ctx: Federated learning context for this run.
        """
        #Keeps track of the iteration number
        fl_ctx.set_prop(key="CURRENT_ROUND", value=0)

        # Broadcast the regression task and send site results to the aggregator
        self._broadcast_task(
            task_name=TASK_NAME_LOCAL_CLIENT_STEP1,
            data=Shareable(),
            result_cb=self._accept_site_regression_result,
            fl_ctx=fl_ctx,
            abort_signal=abort_signal,
        )

        # Aggregate results from all sites
        aggregate_result = self.srr_aggregator.aggregate(fl_ctx)

        #Increment iteration number after every aggregation
        fl_ctx.set_prop(key="CURRENT_ROUND", value=1)

        self._broadcast_task(
            task_name=TASK_NAME_LOCAL_CLIENT_STEP2,
            data=aggregate_result,
            result_cb=self._accept_site_regression_result,
            fl_ctx=fl_ctx,
            abort_signal=abort_signal,
        )

        # Aggregate results from all sites
        aggregate_result = self.srr_aggregator.aggregate(fl_ctx)

        # Increment iteration number after every aggregation
        fl_ctx.set_prop(key="CURRENT_ROUND", value=2)

        self._broadcast_task(
            task_name=TASK_NAME_LOCAL_CLIENT_STEP3,
            data=aggregate_result,
            result_cb=self._accept_site_regression_result,
            fl_ctx=fl_ctx,
            abort_signal=abort_signal,
        )

        # Aggregate results from all sites
        aggregate_result = self.srr_aggregator.aggregate(fl_ctx)
        #Increment iteration number after every aggregation
        fl_ctx.set_prop(key="CURRENT_ROUND", value=3)

        # Broadcast the global aggregated results to all sites
        self._broadcast_task(
            task_name=TASK_NAME_LOCAL_CLIENT_STEP4,
            data=aggregate_result,
            result_cb=None,
            fl_ctx=fl_ctx,
            abort_signal=abort_signal,
        )

    def _accept_site_regression_result(self, client_task: ClientTask, fl_ctx: FLContext) -> bool:
        """
        Callback method that processes each site's regression result and sends it
        to the aggregator for aggregation.

        Computation authors can override this if they need to modify how results
        are processed before aggregation, but this is optional.

        :param client_task: The task result received from a client site.
        :param fl_ctx: Federated learning context for this run.
        :return: Boolean indicating whether the result was successfully accepted.
        """
        return self.srr_aggregator.accept(client_task.result, fl_ctx)

    #### End of Computation Author Defined Section ####

    #### Framework Helper Methods: No modification necessary ####

    def _broadcast_task(self, task_name: str, data: Shareable, result_cb: Callable[[ClientTask, FLContext], bool],
                        fl_ctx: FLContext, abort_signal: Signal) -> None:
        """
        Broadcasts a task to all client sites and waits for responses.

        Computation authors can use this method to simplify task broadcasting.
        Typically, this method does not need to be modified.

        :param task_name: Name of the task to broadcast.
        :param data: Shareable object containing the data to send.
        :param result_cb: Callback for handling results from each client site.
        :param fl_ctx: Federated learning context for this run.
        :param abort_signal: Signal used to abort the task if needed.
        """
        self.broadcast_and_wait(
            task=Task(
                name=task_name,
                data=data,
                props={},
                timeout=self._task_timeout,
                result_received_cb=result_cb,
            ),
            min_responses=self._min_clients,
            wait_time_after_min_received=self._wait_time_after_min_received,
            fl_ctx=fl_ctx,
            abort_signal=abort_signal,
        )

    def _load_and_set_computation_parameters(self, fl_ctx: FLContext) -> None:
        """
        Loads computation parameters from a file and sets them in the shared context
        for all sites to access.

        This is a utility method that computation authors typically do not need to modify.

        :param fl_ctx: Federated learning context for this run.
        """
        with open(get_parameters_file_path(fl_ctx), 'r') as f:
            fl_ctx.set_prop(
                key="COMPUTATION_PARAMETERS",
                value=json.load(f),
                private=False,
                sticky=True
            )

    #### Framework-Specific Required Methods: No modification necessary ####

    def process_result_of_unknown_task(self, task: Task, fl_ctx: FLContext) -> None:
        """
        Handles results for tasks that are not explicitly recognized.
        This method can be overridden by developers for custom handling.

        This is a Framework-Specific Required Method.

        :param task: The task whose result is being processed.
        :param fl_ctx: Federated learning context for this run.
        """
        pass

    def stop_controller(self, fl_ctx: FLContext) -> None:
        """
        Called when the controller stops. Developers can override this method
        for cleanup or custom logic.

        This is a Framework-Specific Required Method.

        :param fl_ctx: Federated learning context for this run.
        """
        pass
