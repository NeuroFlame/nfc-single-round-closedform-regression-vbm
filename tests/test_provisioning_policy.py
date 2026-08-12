import json
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from system.provision.code.create_run_kits import disable_site_log_streaming


class ProvisioningPolicyTests(unittest.TestCase):
    def test_generated_site_policy_explicitly_disables_log_streaming(self):
        with tempfile.TemporaryDirectory() as directory:
            local = Path(directory, "local")
            local.mkdir()
            resources = local / "resources.json.default"
            resources.write_text(
                json.dumps({"allow_log_streaming": True}), encoding="utf-8"
            )

            disable_site_log_streaming(directory)

            self.assertIs(
                json.loads(resources.read_text(encoding="utf-8"))[
                    "allow_log_streaming"
                ],
                False,
            )

    def test_checked_in_provisioning_sample_uses_current_single_port_schema(self):
        sample = json.loads(
            Path(
                REPOSITORY,
                "system",
                "provision",
                "test_input",
                "provision_input.json",
            ).read_text(encoding="utf-8")
        )
        self.assertTrue(sample["users"])
        self.assertNotIn("user_ids", sample)
        self.assertNotIn("active_participants", sample)
        self.assertNotIn("admin_port", sample)
        self.assertIsInstance(sample["fed_learn_port"], int)


if __name__ == "__main__":
    unittest.main()
