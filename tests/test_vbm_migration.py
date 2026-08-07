import ast
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app", "code"))
sys.path.insert(0, CODE_DIR)

from computation import binary_io, local_math, nifti, remote_math
from computation.spec import SPEC
from computation.steps import fit_local_model
from framework import ArtifactRef
from framework.types import SteppedWorkflow


class VbmMigrationTests(unittest.TestCase):
    @staticmethod
    def _reference():
        path = Path(__file__).parent / "fixtures" / "vbm_numerical_reference.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def test_binary_format_and_values_match_historical_layout(self):
        value = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "matrix.npy")
            binary_io.save_binary(path, value)
            encoded = Path(path).read_bytes()

            header, payload = encoded.split(b"\n", 1)
            self.assertEqual(header, b"float64 2 3 4")
            self.assertEqual(len(payload), value.nbytes)
            np.testing.assert_array_equal(binary_io.load_binary(path), value)

    def test_workflow_preserves_three_federated_rounds_and_site_output(self):
        self.assertIsInstance(SPEC.workflow, SteppedWorkflow)
        self.assertEqual(
            [step.name for step in SPEC.workflow.steps],
            [
                "prepare_local_mask",
                "fit_local_model",
                "compute_local_metrics",
                "write_vbm_outputs",
            ],
        )
        self.assertTrue(all(step.remote_fn for step in SPEC.workflow.steps[:-1]))

    def test_local_model_exposes_large_outputs_only_as_artifacts(self):
        fake_result = {
            "output": {
                "mean_y_local": [1.0, 2.0],
                "count_local": [3, 4],
                "X_labels": ["age"],
                "lambda": 1,
                "share_local_stats": True,
                "local_beta_vector": [[1.0]],
                "local_pvalues": [[0.5]],
                "local_tvalues": [[2.0]],
                "local_rsquared": [0.25],
                "local_X_labels": ["age"],
            },
            "cache": {
                "covariates": "X.npy",
                "dependents": "y.npy",
                "local_beta_vector": [[1.0]],
                "local_pvalues": [[0.5]],
                "local_tvalues": [[2.0]],
                "local_rsquared": [0.25],
            },
            "transfer_files": {
                "x_transpose_x_file_name": "",
                "x_transpose_y_file_name": "",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            fake_result["transfer_files"]["x_transpose_x_file_name"] = self._write(
                directory, "XTX.npy"
            )
            fake_result["transfer_files"]["x_transpose_y_file_name"] = self._write(
                directory, "XTy.npy"
            )
            with patch(
                "computation.steps.local_math.perform_local_step2_compute_local_model",
                return_value=fake_result,
            ):
                result = fit_local_model(
                    {},
                    state={},
                    data_dir=directory,
                    artifact_dir=directory,
                    logger=None,
                )

            for key in (
                "mean_y_local",
                "count_local",
                "x_transpose_x",
                "x_transpose_y",
                "local_beta_vector",
                "local_pvalues",
                "local_tvalues",
                "local_rsquared",
            ):
                self.assertIsInstance(result.payload[key], ArtifactRef)

    def test_computation_code_has_no_nvflare_imports(self):
        computation_dir = Path(CODE_DIR, "computation")
        for source_path in computation_dir.glob("*.py"):
            tree = ast.parse(source_path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                self.assertFalse(
                    any(
                        name == "nvflare" or name.startswith("nvflare.")
                        for name in names
                    ),
                    source_path,
                )

    def test_historical_central_numerical_reference_and_artifact_hashes(self):
        reference = self._reference()
        expected = reference["expected"]
        with tempfile.TemporaryDirectory() as directory:
            site_results = {}
            for site_name, values in reference["sites"].items():
                xtx_path = os.path.join(directory, f"{site_name}-xtx.npy")
                xty_path = os.path.join(directory, f"{site_name}-xty.npy")
                binary_io.save_binary(xtx_path, np.asarray(values["xtx"]))
                binary_io.save_binary(xty_path, np.asarray(values["xty"]))
                site_results[site_name] = {
                    "x_transpose_x_file_name": xtx_path,
                    "x_transpose_y_file_name": xty_path,
                    "mean_y_local": values["mean"],
                    "count_local": values["count"],
                    "lambda": 0.5,
                    "X_labels": ["const", "age"],
                    "share_local_stats": False,
                }

            model = remote_math.perform_remote_step2_compute_global_model(
                site_results, {}, directory
            )
            beta_path = model["transfer_files"]["avg_beta_vector"]
            mean_path = model["transfer_files"]["mean_y_global"]
            np.testing.assert_allclose(
                binary_io.load_binary(beta_path),
                expected["avg_beta_vector"],
                rtol=reference["rtol"],
                atol=reference["atol"],
            )
            np.testing.assert_allclose(
                binary_io.load_binary(mean_path),
                expected["mean_y_global"],
                rtol=reference["rtol"],
                atol=reference["atol"],
            )
            for path in (beta_path, mean_path):
                self.assertEqual(
                    hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                    expected["artifact_sha256"][Path(path).name],
                )

            metrics_input = {
                site_name: {
                    "SSE_local": values["sse"],
                    "SST_local": values["sst"],
                    "varX_matrix_local": values["var_x"],
                }
                for site_name, values in reference["sites"].items()
            }
            metrics = remote_math.perform_remote_step3_aggregate_metrics(
                metrics_input, model["cache"], directory
            )["output"]
            for key in (
                "r_squared_global",
                "MSE",
                "ts_global",
                "ps_global",
            ):
                np.testing.assert_allclose(
                    metrics[key],
                    expected[key],
                    rtol=reference["rtol"],
                    atol=reference["atol"],
                )

    def test_global_output_inventory_matches_historical_names(self):
        reference = self._reference()

        class Display:
            def savefig(self, path, **_kwargs):
                Path(f"{path}.png").write_bytes(b"png")

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as directory:
            cache_dir = os.path.join(directory, "cache")
            output_dir = os.path.join(directory, "global_stats")
            os.makedirs(cache_dir)
            os.makedirs(output_dir)
            mask = np.zeros((3, 3, 3))
            mask[1, 1, 1] = 1
            nib.save(
                nib.Nifti1Image(mask, np.eye(4)), os.path.join(cache_dir, "mask.nii")
            )
            cache = {"mask_file_name": "mask.nii"}
            with (
                patch(
                    "computation.nifti.plotting.plot_stat_map",
                    return_value=Display(),
                ),
                patch("computation.nifti._relabel_dpi"),
            ):
                nifti.print_beta_images(
                    cache, cache_dir, output_dir, [[1.0, 2.0]], ["const", "age"]
                )
                nifti.print_r2_image(cache, cache_dir, output_dir, [0.5])
                nifti.print_pvals(
                    cache,
                    cache_dir,
                    output_dir,
                    [[0.1, 0.2]],
                    [[2.0, -1.0]],
                    ["const", "age"],
                )
            Path(directory, "index.html").write_text("report", encoding="utf-8")
            actual = sorted(
                str(path.relative_to(directory))
                for path in Path(directory).rglob("*")
                if path.is_file() and "cache" not in path.parts
            )
            self.assertEqual(
                actual, sorted(reference["expected"]["global_output_names"])
            )

    def test_report_rendering_failure_is_terminal_and_does_not_publish_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            report_path = os.path.join(directory, "index.html")
            with patch.object(
                local_math,
                "_persist_html_from_results",
                side_effect=RuntimeError("<sensitive>"),
            ):
                with self.assertRaisesRegex(RuntimeError, "sensitive"):
                    local_math._persist_html_safe_wrapper({}, report_path)
            self.assertFalse(os.path.exists(report_path))

    @staticmethod
    def _write(directory, name):
        path = os.path.join(directory, name)
        binary_io.save_binary(path, np.eye(1))
        return path


if __name__ == "__main__":
    unittest.main()
