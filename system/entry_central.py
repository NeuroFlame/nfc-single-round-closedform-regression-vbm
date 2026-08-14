"""Start the central NVFlare node and run its packaged computation job."""

import os
import re
import subprocess

from framework.errors import (
    ERROR_ORIGIN_CENTRAL,
    emit_shared_error_summary,
    find_terminal_errors,
    raise_for_terminal_errors,
    record_terminal_error,
)
from nvflare.apis.job_def import JobMetaKey, RunStatus
from nvflare.fuel.flare_api.api_spec import TargetType
from nvflare.fuel.flare_api.flare_api import new_secure_session

# Path Constants
STARTUP_SCRIPT_DIRECTORY = "/workspace/runKit/server/startup"
STARTUP_SCRIPT_PATH = "/workspace/runKit/server/startup/start.sh"
ADMIN_DIRECTORY_PATH = "/workspace/runKit/admin"
JOB_DIRECTORY_PATH = "/workspace/runKit/job/"
ADMIN_USER_EMAIL = "admin@admin.com"
_DEPLOYMENT_TIMEOUT_SUFFIX = ": no reply (deployment timeout)"
_SAFE_PARTICIPANT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.@-]{0,127}")


def start_server():
    """Start the provisioned NVFlare server process."""
    subprocess.run(
        ["/bin/bash", STARTUP_SCRIPT_PATH],
        cwd=STARTUP_SCRIPT_DIRECTORY,
        check=True,
    )


def _report_runtime_failure(output_dir, error, *, stage, public_stage):
    """Record central runtime failures without replacing computation errors."""
    if not find_terminal_errors(output_dir):
        record_terminal_error(
            output_dir,
            "central runtime",
            error,
            origin=ERROR_ORIGIN_CENTRAL,
            stage=stage,
        )
    emit_shared_error_summary(
        output_dir,
        fallback_origin=ERROR_ORIGIN_CENTRAL,
        fallback_stage=public_stage,
    )


def _job_failure(job_id, job_status, job_meta):
    """Create a useful failure without exposing arbitrary deployment replies."""
    if job_status == RunStatus.FAILED_TO_RUN.value:
        deploy_detail = job_meta.get(JobMetaKey.JOB_DEPLOY_DETAIL.value)
        if isinstance(deploy_detail, list):
            timed_out_participants = []
            for detail in deploy_detail:
                if not isinstance(detail, str) or not detail.endswith(
                    _DEPLOYMENT_TIMEOUT_SUFFIX
                ):
                    continue
                participant = detail[: -len(_DEPLOYMENT_TIMEOUT_SUFFIX)]
                if _SAFE_PARTICIPANT_NAME.fullmatch(participant):
                    timed_out_participants.append(participant)
            if timed_out_participants:
                participants = ", ".join(sorted(set(timed_out_participants)))
                return RuntimeError(
                    f"Job deployment timed out waiting for participants: {participants}"
                )
    return RuntimeError(f"Job {job_id} ended with status {job_status}")


def main():
    """Submit, monitor, validate, and shut down one computation job."""
    output_dir = os.getenv("OUTPUT_DIR", "/workspace/output")
    session = None
    active_error = None
    failure_stage = "controller_startup"
    public_stage = "startup"
    try:
        start_server()
        session = new_secure_session(
            ADMIN_USER_EMAIL,
            ADMIN_DIRECTORY_PATH,
        )
        failure_stage = "controller_execution"
        public_stage = "execution"
        job_id = session.submit_job(JOB_DIRECTORY_PATH)
        job_meta = session.wait_for_job(
            job_id,
            timeout=3600,
            poll_interval=10,
        )
        job_status = job_meta[JobMetaKey.STATUS.value]
        print(f"Terminal job status: {job_status}")
        if job_status != RunStatus.FINISHED_COMPLETED.value:
            raise_for_terminal_errors(output_dir)
            raise _job_failure(job_id, job_status, job_meta)
    except BaseException as error:
        active_error = error
        _report_runtime_failure(
            output_dir,
            error,
            stage=failure_stage,
            public_stage=public_stage,
        )
        raise
    finally:
        if session is not None:
            try:
                session.shutdown(TargetType.ALL)
            except Exception as shutdown_error:
                if active_error is None:
                    _report_runtime_failure(
                        output_dir,
                        shutdown_error,
                        stage="controller_execution",
                        public_stage="execution",
                    )
                    raise
                active_error.add_note(f"NVFlare shutdown also failed: {shutdown_error}")


if __name__ == "__main__":
    main()
