import os
from typing import Dict, Any

from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.aggregator import Aggregator
from utils import data_transfer as dt
from utils.logger import NFCLogger
from utils.utils import get_output_directory_path

from . import aggregator_methods as am


class SRCFRAggregator(Aggregator):
    """
    SrrAggregator handles the aggregation of results from multiple client sites.
    It stores individual site results and computes a global result based on the aggregation logic.

    This class can be customized if specific aggregation logic is needed.
    """

    def __init__(self):
        """
        Initializes the SrrAggregator with a dictionary to store results from multiple sites.
        """
        super().__init__()
        # Holds results from each client per round; Structure: {round_number: {contributor_name: data}}
        self.site_results: Dict[int, Dict[str, Any]] = {}  # Store results as a dictionary
        self.logger = None
        self.agg_cache: Dict[str, Any] = {}
        self.agg_cache_dir: str = "./temp_agg_cache"
        self.transfer_reader_dir: str = "./temp_agg_transfer_dir"
        os.makedirs(self.agg_cache_dir, exist_ok=True)  # succeeds even if directory exists.
        os.makedirs(self.transfer_reader_dir, exist_ok=True)  # succeeds even if directory exists.

    def accept(self, site_result: Shareable, fl_ctx: FLContext) -> bool:
        """
        Accepts a result from a site and stores it for later aggregation.

        This method is called when a client site sends a result. Developers can override this 
        if they need to handle or validate the results differently before storing them.

        :param site_result: The result received from the client site.
        :param fl_ctx: The federated learning context for this run.
        :return: Boolean indicating if the result was successfully accepted.
        """
        site_name = site_result.get_peer_prop(
            key=ReservedKey.IDENTITY_NAME, default=None)
        contribution_round = fl_ctx.get_prop(key="CURRENT_ROUND", default=None)
        if self.logger == None:
            self.logger = NFCLogger('aggregator.log', get_output_directory_path(fl_ctx),
                                    fl_ctx.get_prop(key="log_level", default="info"))

        self.logger.info(f"Aggregator received contribution from {site_name} for round {contribution_round}")
        if contribution_round is None or site_name is None:
            return False  # Could log a warning/error here as well

        if contribution_round not in self.site_results:
            self.site_results[contribution_round] = {}

        # Store the result for the site using its identity name as the key
        self.site_results[contribution_round][site_name] = site_result["result"]

        # To clean this logic
        if contribution_round == 0:
            # Define the image file path inside temp storage
            image_filename = f"{site_name}_average_nifti.nii"
            image_file_path = os.path.join(self.transfer_reader_dir, image_filename)
            # with open(image_file_path, "wb") as f:
            #    f.write(site_result["transfer_average_nifti"])
            dt.bytes_to_file(image_file_path, site_result["transfer_average_nifti"])
            self.site_results[contribution_round][site_name]["transfer_average_nifti"] = image_file_path

        # To clean this logic
        if contribution_round == 1:
            # Define the image file path inside temp storage
            xtx_file_name = os.path.join(self.transfer_reader_dir, f"{site_name}_XTX.npy")
            dt.bytes_to_file(xtx_file_name, site_result["transfer_xTx"])
            self.site_results[contribution_round][site_name]["x_transpose_x_file_name"] = xtx_file_name

            xty_file_name = os.path.join(self.transfer_reader_dir, f"{site_name}_XTy.npy")
            dt.bytes_to_file(xty_file_name, site_result["transfer_xTy"])
            self.site_results[contribution_round][site_name]["x_transpose_y_file_name"] = xty_file_name

        return True

    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        """
        Aggregates the results from all accepted client sites and produces a global result.

        This is where the global aggregation logic happens. Developers can override this
        if they need to change how the results from each site are combined.

        :param fl_ctx: The federated learning context for this run.
        :return: A Shareable object containing the aggregated global result.
        """
        contribution_round = fl_ctx.get_prop(key="CURRENT_ROUND", default=None)

        if (contribution_round == 0):
            agg_result = am.perform_remote_step1_generate_average_mask(self.site_results[contribution_round], self.agg_cache,
                                                                       self.agg_cache_dir)
            self.agg_cache.update(agg_result['cache'])
            outgoing_shareable = Shareable()
            outgoing_shareable['result'] = agg_result['output']
            outgoing_shareable['result']['mask_file'] = dt.file_to_bytes(agg_result['transfer_files']['mask_file'])
            outgoing_shareable['result']['downsampled_mni'] = dt.file_to_bytes(
                agg_result['transfer_files']['downsampled_mni'])
            return outgoing_shareable

        elif (contribution_round == 1):
            agg_result = am.perform_remote_step2_compute_global_model(self.site_results[contribution_round], self.agg_cache,
                                                                      self.agg_cache_dir)
            self.agg_cache.update(agg_result['cache'])
            outgoing_shareable = Shareable()
            outgoing_shareable['result'] = agg_result['output']
            outgoing_shareable['result']['avg_beta_vector'] = dt.file_to_bytes(
                agg_result['transfer_files']['avg_beta_vector'])
            outgoing_shareable['result']['mean_y_global'] = dt.file_to_bytes(agg_result['transfer_files']['mean_y_global'])

            return outgoing_shareable

        elif (contribution_round == 2):
            agg_result = am.perform_remote_step3_aggregate_metrics(self.site_results[contribution_round], self.agg_cache,
                                                                   self.agg_cache_dir)
            self.agg_cache.update(agg_result.get('cache', {}))
            outgoing_shareable = Shareable()
            outgoing_shareable['result'] = agg_result['output']
            # Formats and closes logs in the end
            self.logger.close()
            self.logger.format_log()
            return outgoing_shareable

        else:
            return Shareable()  # Return an empty Shareable if no data to aggregate
