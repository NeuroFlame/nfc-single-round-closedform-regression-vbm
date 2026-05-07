import logging
import os
import traceback
from typing import Dict

from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey
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
    def __init__(self):
        """
        Initialize the SrrExecutor. This constructor sets up the logger.
        """
        logging.info("SrrExecutor initialized")
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
        self.logger = NFCLogger(fl_ctx.get_prop(FLContextKey.CLIENT_NAME) + '.log', get_output_directory_path(fl_ctx),
                                fl_ctx.get_peer_context().get_prop("COMPUTATION_PARAMETERS").get('log_level', "info"))

        # Prepare the Shareable object to send the result to other components
        outgoing_shareable = Shareable()

        try:
            if task_name == tc.TASK_CLIENT_STEP1_VALIDATE_INPUTS_GENERATE_MASK:
                client_result = self._step1_validate_inputs_and_compute_mask(shareable, fl_ctx, abort_signal,
                                                                             cache_store.get_cache_dict(),
                                                                             cache_store.get_cache_dir())
                cache_store.update_cache_dict(client_result['cache'])
                outgoing_shareable['result'] = client_result['output']
                outgoing_shareable["transfer_average_nifti"] = dt.file_to_bytes(
                    client_result['transfer_files']['average_nifti'])

            elif task_name == tc.TASK_CLIENT_STEP2_COMPUTE_LOCAL_MODEL:
                client_result = self._step2_compute_local_model(shareable, fl_ctx, abort_signal,
                                                                cache_store.get_cache_dict(),
                                                                cache_store.get_cache_dir())
                cache_store.update_cache_dict(client_result['cache'])
                outgoing_shareable['result'] = client_result['output']
                outgoing_shareable["transfer_xTx"] = dt.file_to_bytes(
                    client_result['transfer_files']['x_transpose_x_file_name'])
                outgoing_shareable["transfer_xTy"] = dt.file_to_bytes(
                    client_result['transfer_files']['x_transpose_y_file_name'])

            elif task_name == tc.TASK_CLIENT_STEP3_COMPUTE_METRICS:
                client_result = self._step3_compute_metrics(shareable, fl_ctx, abort_signal,
                                                            cache_store.get_cache_dict(),
                                                            cache_store.get_cache_dir())
                cache_store.update_cache_dict(client_result['cache'])
                outgoing_shareable['result'] = client_result['output']

            elif task_name == tc.TASK_CLIENT_STEP4_PERSIST_RESULTS:
                client_result = self._step4_persist_results(shareable, fl_ctx, abort_signal,
                                                            cache_store.get_cache_dict(),
                                                            cache_store.get_cache_dir())
                cache_store.remove_cache()
                #self.logger.format_log()
                # Sending empty sharable object as result

            else:
                raise ValueError(f"Unknown task name: {task_name}")

        except Exception as e:
            self.logger.error(f"Exception in task '{task_name}': {traceback.format_exc()}")
            self.logger.close()
            raise

        # close logger
        self.logger.close()

        return outgoing_shareable

    def _step1_validate_inputs_and_compute_mask(
            self,
            shareable: Shareable,
            fl_ctx: FLContext,
            abort_signal: Signal,
            cache_dict: Dict,
            cache_dir: str
    ) -> Dict:
        """
        Perform the ridge regression on the merged site data.

        This method assumes that data has been validated and is ready for regression analysis.
        It reads the covariates and dependent data, runs the local regression, and saves the results.

        Returns:
            A Shareable object with the regression results.
        """
        # Paths to data directories and logs
        data_directory = get_data_directory_path(fl_ctx)
        data_path = data_directory
        covariates_path = os.path.join(data_directory, "covariates.csv")

        computation_parameters = fl_ctx.get_peer_context().get_prop("COMPUTATION_PARAMETERS")

        # Perform ridge regression using the specified covariates and dependent variables
        result = cem.perform_client_step1_validate_inputs_and_compute_mask(covariates_path, data_path, computation_parameters, self.logger, cache_dict,
                                          cache_dir)

        return result

    def _step2_compute_local_model(
            self,
            shareable: Shareable,
            fl_ctx: FLContext,
            abort_signal: Signal,
            cache_dict: Dict,
            cache_dir: str
    ) -> Dict:
        """
        Receives the global regression parameters, uses these global regression model on local data
        to perform local metrics and sends it to remote to compute global metrics.

        This method retrieves the global regression results from the Shareable object,
         and returns a Shareable object with regression metrics with global regression params.
        """
        # Retrieve the global regression result from the Shareable object
        agg_result = shareable.get("result")

        # Paths to data directories and logs
        data_path = get_data_directory_path(fl_ctx)

        # Perform ridge regression using the specified covariates and dependent variables
        result = cem.perform_local_step2_compute_local_model(agg_result, data_path, self.logger, cache_dict, cache_dir)

        return result

    def _step3_compute_metrics(
            self,
            shareable: Shareable,
            fl_ctx: FLContext,
            abort_signal: Signal,
            cache_dict: Dict,
            cache_dir: str
    ) -> Dict:
        """
        Receives the global regression parameters, uses these global regression model on local data
        to perform local metrics and sends it to remote to compute global metrics.

        This method retrieves the global regression results from the Shareable object,
         and returns a Shareable object with regression metrics with global regression params.
        """
        # Retrieve the global regression result from the Shareable object
        agg_result = shareable.get("result")

        # Perform ridge regression using the specified covariates and dependent variables
        result = cem.perform_local_step3_compute_metrics_with_global_params(agg_result, self.logger, cache_dict, cache_dir)

        return result

    def _step4_persist_results(
            self,
            shareable: Shareable,
            fl_ctx: FLContext,
            abort_signal: Signal,
            cache_dict: Dict,
            cache_dir: str
    ) -> Dict:
        """
        Save the global regression results to a file.

        This method retrieves the global regression results from the Shareable object,
        saves them in JSON and HTML format, and returns a Shareable object.
        """
        # Retrieve the global regression result from the Shareable object
        agg_result = shareable.get("result")

        if agg_result is None:
            raise Exception("Empty aggregation result")

        # Save the global regression results
        print(f'\n\n\ntype(agg_result): {type(agg_result)}')
        output_dir = get_output_directory_path(fl_ctx)
        result = cem.perform_local_step4_persist_results(agg_result, self.logger, cache_dict, cache_dir, output_dir)

        return result
