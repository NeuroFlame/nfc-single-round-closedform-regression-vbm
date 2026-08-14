"""Regression tests for VBM CSV header compatibility."""

import logging
import os
import tempfile
import unittest
from unittest.mock import Mock

import nibabel as nib
import numpy as np
import pandas as pd
from computation.input_validation import validate_and_get_inputs


class InputHeaderCompatibilityTests(unittest.TestCase):
    """Verify case-insensitive and whitespace-tolerant header matching."""

    def setUp(self):
        """Create standard computation parameters and a test logger."""
        self.parameters = {"Covariates": {"age": "float"}}
        self.logger = Mock(spec=logging.Logger)

    def _validate(self, covariate_header: str, index_header: str = "niftifilename"):
        """Write a one-subject VBM input and return its validation result."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nifti_name = "subject.nii.gz"
            nifti_path = os.path.join(temp_dir, nifti_name)
            nib.save(nib.Nifti1Image(np.ones((2, 2, 2)), np.eye(4)), nifti_path)
            covariates_path = os.path.join(temp_dir, "covariates.csv")
            pd.DataFrame({index_header: [nifti_name], covariate_header: [42]}).to_csv(
                covariates_path, index=False
            )
            return validate_and_get_inputs(
                covariates_path,
                temp_dir,
                self.parameters,
                self.logger,
            )

    def test_headers_are_matched_case_insensitively(self):
        """Accept differently cased headers and restore configured names."""
        is_valid, covariates, omissions = self._validate("AGE", "NIFTIFILENAME")

        self.assertTrue(is_valid)
        self.assertEqual(["age"], covariates.columns.tolist())
        self.assertEqual(0, sum(len(subjects) for subjects in omissions.values()))

    def test_headers_ignore_surrounding_whitespace(self):
        """Accept surrounding header whitespace and restore configured names."""
        is_valid, covariates, omissions = self._validate(" age ", " niftifilename ")

        self.assertTrue(is_valid)
        self.assertEqual(["age"], covariates.columns.tolist())
        self.assertEqual(0, sum(len(subjects) for subjects in omissions.values()))

    def test_missing_header_fails_validation(self):
        """Reject an input that lacks a configured header."""
        is_valid, covariates, omissions = self._validate("height")

        self.assertFalse(is_valid)
        self.assertIsNone(covariates)
        self.assertEqual({}, omissions)
        self.assertTrue(
            any(
                "headers do not contain all expected headers" in str(call)
                for call in self.logger.error.call_args_list
            )
        )

    def test_duplicate_normalized_headers_fail_validation(self):
        """Reject distinct CSV headers that normalize to the same name."""
        with tempfile.TemporaryDirectory() as temp_dir:
            covariates_path = os.path.join(temp_dir, "covariates.csv")
            pd.DataFrame(
                [["subject.nii.gz", 42, 43]],
                columns=["niftifilename", "age", " AGE "],
            ).to_csv(covariates_path, index=False)

            is_valid, covariates, omissions = validate_and_get_inputs(
                covariates_path,
                temp_dir,
                self.parameters,
                self.logger,
            )

        self.assertFalse(is_valid)
        self.assertIsNone(covariates)
        self.assertEqual({}, omissions)
        self.assertTrue(
            any(
                "duplicate headers after normalization" in str(call)
                for call in self.logger.error.call_args_list
            )
        )


if __name__ == "__main__":
    unittest.main()
