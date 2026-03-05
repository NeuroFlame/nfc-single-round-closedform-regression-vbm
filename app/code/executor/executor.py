import json
import logging
import os
from typing import Dict, Optional

from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal

from utils import data_transfer as dt
from utils.logger import NFCLogger
from utils import task_constants as tc
from utils.utils import get_data_directory_path, get_output_directory_path

from . import client_cache_store as ccs
from . import client_executor_methods as cem


class SRCFRExecutor(Executor):
    """
    SRCFRExecutor routes tasks to specific methods based on task name.

    This executor is role-aware:
      - Contributors perform compute steps (1–3).
      - Observers skip compute steps and only participate in the final lifecycle task.

    NOTE:
      - Clients do NOT persist results anymore.
      - Server/aggregator is responsible for writing results.zip in the server output.
    """

    def __init__(self):
        """
        Initialize the SRCFRExecutor. This constructor sets up the logger.
        """
        logging.info("SRCFRExecutor initialized")
        self.logger = None

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        """
        Main execution entry point. Routes tasks to specific methods based on the task name.

        Parameters:
            task_name: Name of the task to perform.
            shareable: Shareable object containing data for the task.
            fl_ctx: Federated learning context.
            abort_signal: Signal object to handle task abortion.

        Returns:
            A Shareable object containing results of the task.
        """
        cache_store = ccs.CacheSerialStore(get_output_directory_path(fl_ctx))

        # Setup per-client log file in output directory
        self.logger = NFCLogger(
            fl_ctx.get_prop(FLContextKey.CLIENT_NAME) + ".log",
            get_output_directory_path(fl_ctx),
            fl_ctx.get_peer_context().get_prop("COMPUTATION_PARAMETERS").get("log_level", "info"),
        )

        # Determine participant role (contributor/observer) for this runKit
        participant_role = self._get_participant_role(fl_ctx)

        # Prepare the Shareable object to send results to other components
        outgoing_shareable = Shareable()

        # ---------------------------------------------------------
        # OBSERVER BEHAVIOR:
        #   - skip compute tasks
        #   - participate in final receive_results lifecycle task
        # ---------------------------------------------------------
        if participant_role == "observer" and task_name != tc.TASK_CLIENT_RECEIVE_RESULTS:
            self.logger.info(f"[ROLE={participant_role}] Skipping task: {task_name}")
            outgoing_shareable.set_return_code(ReturnCode.OK)
            self.logger.close()
            return outgoing_shareable

        # ---------------------------------------------------------
        # CONTRIBUTOR TASKS (compute steps)
        # ---------------------------------------------------------
        if task_name == tc.TASK_CLIENT_STEP1_VALIDATE_INPUTS_GENERATE_MASK:
            client_result = self._step1_validate_inputs_and_compute_mask(
                shareable,
                fl_ctx,
                abort_signal,
                cache_store.get_cache_dict(),
                cache_store.get_cache_dir(),
            )
            cache_store.update_cache_dict(client_result["cache"])
            outgoing_shareable["result"] = client_result["output"]
            outgoing_shareable["transfer_average_nifti"] = dt.file_to_bytes(
                client_result["transfer_files"]["average_nifti"]
            )

        elif task_name == tc.TASK_CLIENT_STEP2_COMPUTE_LOCAL_MODEL:
            client_result = self._step2_compute_local_model(
                shareable,
                fl_ctx,
                abort_signal,
                cache_store.get_cache_dict(),
                cache_store.get_cache_dir(),
            )
            cache_store.update_cache_dict(client_result["cache"])
            outgoing_shareable["result"] = client_result["output"]
            outgoing_shareable["transfer_xTx"] = dt.file_to_bytes(
                client_result["transfer_files"]["x_transpose_x_file_name"]
            )
            outgoing_shareable["transfer_xTy"] = dt.file_to_bytes(
                client_result["transfer_files"]["x_transpose_y_file_name"]
            )

        elif task_name == tc.TASK_CLIENT_STEP3_COMPUTE_METRICS:
            client_result = self._step3_compute_metrics(
                shareable,
                fl_ctx,
                abort_signal,
                cache_store.get_cache_dict(),
                cache_store.get_cache_dir(),
            )
            cache_store.update_cache_dict(client_result["cache"])
            outgoing_shareable["result"] = client_result["output"]

        # ---------------------------------------------------------
        # FINAL LIFECYCLE TASK (everyone)
        # ---------------------------------------------------------
        elif task_name == tc.TASK_CLIENT_RECEIVE_RESULTS:
            # IMPORTANT:
            #   - No client-side persistence anymore.
            #   - Server/aggregator writes results.zip to server output for fileServer.
            self.logger.info(f"[ROLE={participant_role}] Received final results broadcast (ACK + cleanup).")
            cache_store.remove_cache()
            outgoing_shareable.set_return_code(ReturnCode.OK)

        else:
            # Raise an error if the task name is unknown
            raise ValueError(f"Unknown task name: {task_name}")

        # close logger
        self.logger.close()

        return outgoing_shareable

    def _get_participant_role(self, fl_ctx: FLContext) -> str:
        """
        Determines the participant role for this client.

        Priority:
        1) startup/participant_role.json (if you ever write it)
        2) config/participant_roles.json (ridge standard, uses user_roles mapping)
        3) controller-published PARTICIPANT_ROLES mapping in peer context
        4) default contributor
        """
        # 1) per-site runKit marker (optional)
        try:
            with open("startup/participant_role.json", "r") as f:
                role_doc = json.load(f)
                role = str(role_doc.get("participant_role", "")).strip().lower()
                if role in ("contributor", "observer"):
                    return role
        except Exception:
            pass

        # 2) ridge standard config in app package
        try:
            with open("config/participant_roles.json", "r") as f:
                raw = json.load(f)

            # ridge format:
            # { "contributors": [...], "observers": [...], "user_roles": { "<site>": "contributor|observer" } }
            if isinstance(raw, dict) and isinstance(raw.get("user_roles"), dict):
                site = fl_ctx.get_prop(FLContextKey.CLIENT_NAME)
                role = str(raw["user_roles"].get(site, "")).strip().lower()
                if role in ("contributor", "observer"):
                    return role
        except Exception:
            pass

        # 3) controller-published mapping
        roles_map = fl_ctx.get_peer_context().get_prop("PARTICIPANT_ROLES")
        if isinstance(roles_map, dict):
            site = fl_ctx.get_prop(FLContextKey.CLIENT_NAME)
            role = str(roles_map.get(site, "")).strip().lower()
            if role in ("contributor", "observer"):
                return role

        # 4) default
        return "contributor"


    def _step1_validate_inputs_and_compute_mask(
        self,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
        cache_dict: Dict,
        cache_dir: str,
    ) -> Dict:
        """
        Validates inputs and computes mask for local site.

        Returns:
            Dict with:
              - 'output': local output payload
              - 'cache': updated cache dict
              - 'transfer_files': files to transfer to server (bytes packed above)
        """
        data_directory = get_data_directory_path(fl_ctx)
        data_path = data_directory
        covariates_path = os.path.join(data_directory, "covariates.csv")

        computation_parameters = fl_ctx.get_peer_context().get_prop("COMPUTATION_PARAMETERS")

        result = cem.perform_client_step1_validate_inputs_and_compute_mask(
            covariates_path,
            data_path,
            computation_parameters,
            self.logger,
            cache_dict,
            cache_dir,
        )
        return result

    def _step2_compute_local_model(
        self,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
        cache_dict: Dict,
        cache_dir: str,
    ) -> Dict:
        """
        Computes local model parameters needed for global aggregation.

        Receives aggregated result from server in shareable["result"].
        Returns:
            Dict with local outputs and transfer files (xTx, xTy) as paths.
        """
        agg_result = shareable.get("result")
        data_path = get_data_directory_path(fl_ctx)

        result = cem.perform_local_step2_compute_local_model(
            agg_result,
            data_path,
            self.logger,
            cache_dict,
            cache_dir,
        )
        return result

    def _step3_compute_metrics(
        self,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
        cache_dict: Dict,
        cache_dir: str,
    ) -> Dict:
        """
        Computes local metrics using global params, sends metrics back to server.

        Receives aggregated result from server in shareable["result"].
        Returns:
            Dict with local metric outputs and updated cache.
        """
        agg_result = shareable.get("result")

        result = cem.perform_local_step3_compute_metrics_with_global_params(
            agg_result,
            self.logger,
            cache_dict,
            cache_dir,
        )
        return result
