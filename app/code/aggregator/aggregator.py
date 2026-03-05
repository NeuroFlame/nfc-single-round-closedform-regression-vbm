# aggregator.py

import os
from typing import Dict, Any, Optional

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
    SRCFRAggregator handles the aggregation of results from multiple client sites.
    It stores individual site results and computes a global result based on the aggregation logic.

    This version writes results on the SERVER (aggregator) side:
      - creates output/results/ (results.json, index.html, artifacts/*)
      - creates output/results.zip
    Clients do NOT persist results anymore.
    """

    def __init__(self):
        super().__init__()

        # Holds results from each client per round; Structure: {round_number: {site_name: data}}
        self.site_results: Dict[int, Dict[str, Any]] = {}

        self.logger: Optional[NFCLogger] = None
        self.agg_cache: Dict[str, Any] = {}

        # Local temp directories inside the run container
        self.agg_cache_dir: str = "./temp_agg_cache"
        self.transfer_reader_dir: str = "./temp_agg_transfer_dir"
        os.makedirs(self.agg_cache_dir, exist_ok=True)
        os.makedirs(self.transfer_reader_dir, exist_ok=True)

        # Track whether we already wrote results.zip (defensive)
        self._results_written: bool = False

        for key, value in os.environ.items():
            print(f'{key}: {value}')

    def _ensure_logger(self, fl_ctx: FLContext) -> None:
        if self.logger is not None:
            return
        self.logger = NFCLogger(
            "aggregator.log",
            get_output_directory_path(fl_ctx),
            fl_ctx.get_prop(key="log_level", default="info"),
        )

    def accept(self, site_result: Shareable, fl_ctx: FLContext) -> bool:
        """
        Accepts a result from a site and stores it for later aggregation.
        """
        site_name = site_result.get_peer_prop(key=ReservedKey.IDENTITY_NAME, default=None)
        contribution_round = fl_ctx.get_prop(key="CURRENT_ROUND", default=None)

        self._ensure_logger(fl_ctx)
        self.logger.info(f"Aggregator received contribution from {site_name} for round {contribution_round}")

        if contribution_round is None or site_name is None:
            self.logger.info("Rejecting contribution: missing round or site_name")
            return False

        if contribution_round not in self.site_results:
            self.site_results[contribution_round] = {}

        # Store the (already JSON-serializable) result dict under that site's name
        try:
            self.site_results[contribution_round][site_name] = site_result["result"]
        except Exception:
            self.logger.info("Rejecting contribution: missing site_result['result']")
            return False

        # Round 0 transfer: average nifti -> file on aggregator temp dir
        if contribution_round == 0 and "transfer_average_nifti" in site_result:
            image_filename = f"{site_name}_average_nifti.nii"
            image_file_path = os.path.join(self.transfer_reader_dir, image_filename)
            dt.bytes_to_file(image_file_path, site_result["transfer_average_nifti"])
            self.site_results[contribution_round][site_name]["transfer_average_nifti"] = image_file_path

        # Round 1 transfer: xTx / xTy -> files on aggregator temp dir
        if contribution_round == 1:
            if "transfer_xTx" in site_result:
                xtx_file_name = os.path.join(self.transfer_reader_dir, f"{site_name}_XTX.npy")
                dt.bytes_to_file(xtx_file_name, site_result["transfer_xTx"])
                self.site_results[contribution_round][site_name]["x_transpose_x_file_name"] = xtx_file_name

            if "transfer_xTy" in site_result:
                xty_file_name = os.path.join(self.transfer_reader_dir, f"{site_name}_XTy.npy")
                dt.bytes_to_file(xty_file_name, site_result["transfer_xTy"])
                self.site_results[contribution_round][site_name]["x_transpose_y_file_name"] = xty_file_name

        return True

    def create_results_artifacts(self, fl_ctx: FLContext, global_results_payload: dict) -> dict:
        """
        Server-only: create results artifacts + zip.
        Returns a small metadata dict for clients.
        """
        self._ensure_logger(fl_ctx)

        if self._results_written:
            return {
                "results_zip_name": "results.zip",
                "results_dir_name": "results",
            }

        # IMPORTANT: use the cleaned writer that resolves output dir correctly.
        # aggregator_methods.write_results_artifacts_from_aggregator should be the cleaned function
        # (the one that writes output/results/* and output/results.zip and DOES NOT recurse).
        output = am.write_results_artifacts_from_aggregator(
            global_results=global_results_payload,
            agg_cache_dir=self.agg_cache_dir,
            fl_ctx=fl_ctx,        # ensures it writes to NVFlare output dir
            output_dir=None,      # let it derive from fl_ctx
            clean_first=True,
        )

        self._results_written = True
        self.logger.info(f"Server wrote results bundle: {output.get('results_zip')}")

        return {
            "results_zip_name": "results.zip",
            "results_dir_name": "results",
            # optional extras that can help debugging
            "index_html": os.path.basename(output.get("index_html", "index.html")),
            "results_json": os.path.basename(output.get("results_json", "results.json")),
        }

    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        """
        Aggregates the results from all accepted client sites and produces a global result.
        """
        contribution_round = fl_ctx.get_prop(key="CURRENT_ROUND", default=None)

        self._ensure_logger(fl_ctx)

        if contribution_round is None:
            self.logger.info("aggregate() called with missing CURRENT_ROUND")
            return Shareable()

        # -----------------------------
        # ROUND 0: generate mask + categorical covariate keys
        # -----------------------------
        if contribution_round == 0:
            agg_result = am.perform_remote_step1_generate_average_mask(
                self.site_results.get(contribution_round, {}),
                self.agg_cache,
                self.agg_cache_dir,
            )
            self.agg_cache.update(agg_result.get("cache", {}))

            outgoing_shareable = Shareable()
            outgoing_shareable["result"] = agg_result["output"]

            # Transfer server-generated mask files back to contributor clients (bytes)
            outgoing_shareable["result"]["mask_file"] = dt.file_to_bytes(agg_result["transfer_files"]["mask_file"])
            outgoing_shareable["result"]["downsampled_mni"] = dt.file_to_bytes(
                agg_result["transfer_files"]["downsampled_mni"]
            )

            return outgoing_shareable

        # -----------------------------
        # ROUND 1: compute global model params (avg_beta_vector / mean_y_global)
        # -----------------------------
        elif contribution_round == 1:
            agg_result = am.perform_remote_step2_compute_global_model(
                self.site_results.get(contribution_round, {}),
                self.agg_cache,
                self.agg_cache_dir,
            )
            self.agg_cache.update(agg_result.get("cache", {}))

            outgoing_shareable = Shareable()
            outgoing_shareable["result"] = agg_result["output"]

            # Transfer server-generated arrays back to contributor clients (bytes)
            outgoing_shareable["result"]["avg_beta_vector"] = dt.file_to_bytes(
                agg_result["transfer_files"]["avg_beta_vector"]
            )
            outgoing_shareable["result"]["mean_y_global"] = dt.file_to_bytes(
                agg_result["transfer_files"]["mean_y_global"]
            )

            return outgoing_shareable

        # -----------------------------
        # ROUND 2: aggregate metrics, then write results.zip on the SERVER
        # -----------------------------
        elif contribution_round == 2:
            agg_result = am.perform_remote_step3_aggregate_metrics(
                self.site_results.get(contribution_round, {}),
                self.agg_cache,
                self.agg_cache_dir,
            )
            self.agg_cache.update(agg_result.get("cache", {}))

            # Final global output dict (JSON-serializable)
            global_output = agg_result.get("output", {}) or {}

            # Build a stable payload for results.json / index.html
            global_results_payload = am.build_global_results_payload(
                global_output,
                self.agg_cache,
                self.site_results,
            )

            self.logger.info(f"Global results payload keys: {list(global_results_payload.keys())}")

            # Server-only: write results artifacts
            results_meta = self.create_results_artifacts(fl_ctx, global_results_payload)

            outgoing_shareable = Shareable()
            # Broadcast this small metadata dict to clients for the final lifecycle task.
            outgoing_shareable["result"] = results_meta

            # Close logger at end of run (defensive)
            try:
                self.logger.close()
            except Exception:
                pass

            return outgoing_shareable

        # -----------------------------
        # Any other round: nothing to aggregate
        # -----------------------------
        else:
            self.logger.info(f"No aggregation step for round {contribution_round}")
            return Shareable()
