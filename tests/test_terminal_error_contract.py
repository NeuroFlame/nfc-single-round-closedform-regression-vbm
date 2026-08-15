"""Verify the version-one terminal-error producer contract."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app", "code"))
sys.path.insert(0, CODE_DIR)

from framework.errors import (
    ERROR_ORIGIN_SITE,
    TERMINAL_ERROR_FILE_NAME,
    TERMINAL_ERROR_MAX_BYTES,
    TERMINAL_ERROR_MAX_MESSAGE_LENGTH,
    TERMINAL_ERROR_MAX_TRACEBACK_LENGTH,
    record_terminal_error,
)


class TerminalErrorContractTests(unittest.TestCase):
    @staticmethod
    def _fixture():
        fixture_path = (
            Path(__file__).parent / "fixtures" / "neuroflame_terminal_error_v1.json"
        )
        return json.loads(fixture_path.read_text(encoding="utf-8"))

    def test_writer_matches_canonical_cross_repository_fixture(self):
        fixture = self._fixture()
        with tempfile.TemporaryDirectory() as output_dir:
            with patch(
                "framework.errors.traceback.format_exc",
                return_value=fixture["traceback"],
            ):
                record_terminal_error(
                    output_dir,
                    fixture["scope"],
                    ValueError(fixture["message"]),
                    origin=ERROR_ORIGIN_SITE,
                    stage=fixture["stage"],
                )

            marker_path = Path(output_dir, TERMINAL_ERROR_FILE_NAME)
            self.assertEqual(
                json.loads(marker_path.read_text(encoding="utf-8")),
                fixture,
            )

    def test_writer_bounds_verbose_unicode_errors_to_consumer_limits(self):
        with tempfile.TemporaryDirectory() as output_dir:
            with patch(
                "framework.errors.traceback.format_exc",
                return_value="🔥" * (TERMINAL_ERROR_MAX_TRACEBACK_LENGTH * 2),
            ):
                record_terminal_error(
                    output_dir,
                    "stage" * 100,
                    ValueError("🔥" * (TERMINAL_ERROR_MAX_MESSAGE_LENGTH * 2)),
                    origin=ERROR_ORIGIN_SITE,
                    stage="stage" * 100,
                )

            marker_path = Path(output_dir, TERMINAL_ERROR_FILE_NAME)
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            self.assertLessEqual(marker_path.stat().st_size, TERMINAL_ERROR_MAX_BYTES)
            self.assertLessEqual(
                len(marker["message"]), TERMINAL_ERROR_MAX_MESSAGE_LENGTH
            )
            self.assertLessEqual(
                len(marker["traceback"]), TERMINAL_ERROR_MAX_TRACEBACK_LENGTH
            )


if __name__ == "__main__":
    unittest.main()
