"""Start the central NVFlare node and run its packaged computation job."""

import os
import subprocess

from framework.errors import emit_shared_error_summary, raise_for_terminal_errors
from nvflare.apis.job_def import JobMetaKey, RunStatus
from nvflare.fuel.flare_api.api_spec import TargetType
from nvflare.fuel.flare_api.flare_api import new_secure_session

# Path Constants
STARTUP_SCRIPT_DIRECTORY = "/workspace/runKit/server/startup"
STARTUP_SCRIPT_PATH = "/workspace/runKit/server/startup/start.sh"
ADMIN_DIRECTORY_PATH = "/workspace/runKit/admin"
JOB_DIRECTORY_PATH = "/workspace/runKit/job/"
ADMIN_USER_EMAIL = "admin@admin.com"


def start_server():
    """Start the provisioned NVFlare server process."""
    subprocess.run(
        ["/bin/bash", STARTUP_SCRIPT_PATH],
        cwd=STARTUP_SCRIPT_DIRECTORY,
        check=True,
    )


def main():
    """Submit, monitor, validate, and shut down one computation job."""
    start_server()
    session = new_secure_session(
        ADMIN_USER_EMAIL,
        ADMIN_DIRECTORY_PATH,
    )

    active_error = None
    try:
        job_id = session.submit_job(JOB_DIRECTORY_PATH)
        job_meta = session.wait_for_job(
            job_id,
            timeout=3600,
            poll_interval=10,
        )
        job_status = job_meta[JobMetaKey.STATUS.value]
        print(f"Terminal job status: {job_status}")
        if job_status != RunStatus.FINISHED_COMPLETED.value:
            output_dir = os.getenv("OUTPUT_DIR", "/workspace/output")
            emit_shared_error_summary(
                output_dir,
                fallback_origin="central",
                fallback_stage="execution",
            )
            raise_for_terminal_errors(output_dir)
            raise RuntimeError(f"Job {job_id} ended with status {job_status}")
    except BaseException as error:
        active_error = error
        raise
    finally:
        try:
            session.shutdown(TargetType.ALL)
        except Exception as shutdown_error:
            if active_error is None:
                raise
            active_error.add_note(f"NVFlare shutdown also failed: {shutdown_error}")


if __name__ == "__main__":
    main()
