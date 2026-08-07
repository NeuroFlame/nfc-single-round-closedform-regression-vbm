"""Run a provisioned NVFlare edge node and report terminal failures."""

import os
import signal
import subprocess
from types import FrameType
from typing import Optional

from framework.errors import emit_shared_error_summary, raise_for_terminal_errors

STARTUP_SCRIPT_PATH = "/workspace/runKit/startup/sub_start.sh"
FORWARDED_SIGNALS = (signal.SIGTERM, signal.SIGINT)


def _run_client_once() -> subprocess.CompletedProcess:
    """Run one NVFlare client attempt and forward shutdown signals to it."""
    command = ["/bin/bash", STARTUP_SCRIPT_PATH, "--once"]
    child = subprocess.Popen(command, start_new_session=True)
    previous_handlers = {}

    def forward_signal(signum: int, _frame: Optional[FrameType]) -> None:
        """Forward a container signal while the NVFlare child is active."""
        if child.poll() is None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    try:
        for signum in FORWARDED_SIGNALS:
            previous_handlers[signum] = signal.signal(signum, forward_signal)
        return subprocess.CompletedProcess(command, child.wait())
    finally:
        for signum, previous_handler in previous_handlers.items():
            signal.signal(signum, previous_handler)


def main():
    """Run the edge startup process and preserve its exit status."""
    completed_process = _run_client_once()
    output_dir = os.getenv("OUTPUT_DIR", "/workspace/output")
    try:
        raise_for_terminal_errors(output_dir)
    except Exception:
        emit_shared_error_summary(
            output_dir,
            fallback_origin="site",
            fallback_stage="execution",
        )
        raise
    if completed_process.returncode:
        emit_shared_error_summary(
            output_dir,
            fallback_origin="site",
            fallback_stage="execution",
        )
    completed_process.check_returncode()


if __name__ == "__main__":
    main()
