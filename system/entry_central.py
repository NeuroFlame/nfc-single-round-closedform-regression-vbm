import subprocess
import os
from nvflare.fuel.flare_api.flare_api import new_secure_session, Session
from nvflare.apis.job_def import RunStatus, JobMetaKey

# Path Constants
STARTUP_SCRIPT_DIRECTORY = "/workspace/runKit/server/startup"
STARTUP_SCRIPT_PATH = "/workspace/runKit/server/startup/start.sh"
ADMIN_DIRECTORY_PATH = "/workspace/runKit/admin"
JOB_DIRECTORY_PATH = "/workspace/runKit/job/"
ADMIN_USER_EMAIL = "admin@admin.com"

def start_server():
    subprocess.run(["/bin/bash", STARTUP_SCRIPT_PATH], cwd=STARTUP_SCRIPT_DIRECTORY)

# Start the server
start_server()

# Start a session and submit a job
session = new_secure_session(
    ADMIN_USER_EMAIL,
    ADMIN_DIRECTORY_PATH
)

job_id = session.submit_job(JOB_DIRECTORY_PATH)


def job_status_callback(session: Session, job_id: str, job_meta, *cb_args, **cb_kwargs) -> bool:
    job_status = job_meta[JobMetaKey.STATUS]
    print(f"Job status: {job_status}")

    if 'FINISHED' in job_status:
        print(f"Job {job_id} finished, shutting down system")
        session.shutdown("all")
        return False
    else:
        return True


# Monitor the job with the callback
session.monitor_job(job_id, timeout=3600, poll_interval=10, cb=job_status_callback)
