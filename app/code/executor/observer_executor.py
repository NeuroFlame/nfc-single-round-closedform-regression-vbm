# app/code/executor/observer_executor.py
#
# Observer-side executor that downloads results.zip from the NeuroFLAME fileServer.
#
# Key behavior (NEW):
#   - Optionally WAIT until the end-run signal is observed (abort_signal.triggered)
#     before starting any download attempts. This avoids trying to fetch results
#     before the server has finished writing/uploading them.
#
# Task-name plumbing (IMPORTANT):
#   - We accept ONLY utils.task_constants.TASK_CLIENT_RECEIVE_RESULTS as the alias
#     for this task (i.e. "receive_results"). No other aliases are accepted.

import io
import logging
import os
import time
import zipfile
from pathlib import Path
from typing import Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from nvflare.apis.executor import Executor
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal

from utils import task_constants as tc


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)) or default)
    except Exception:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    v = (_env(name, "") or "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "y", "on")


def _resolve_output_paths() -> Tuple[Path, Path]:
    # Desired output layout:
    #   .../<consortiumId>/<runId>/results.zip
    #   .../<consortiumId>/<runId>/results/<unzipped contents>
    run_dir = _env("NEUROFLAME_OBSERVER_RUN_DIR") or str(Path.cwd())
    run_dir_p = Path(run_dir)

    zip_path = Path(_env("NEUROFLAME_RESULTS_ZIP_PATH") or (run_dir_p / "results.zip"))
    extract_dir = Path(_env("NEUROFLAME_RESULTS_EXTRACT_DIR") or (run_dir_p / "results"))
    return zip_path, extract_dir


def _download_results_zip_once(url: str, token: str, timeout_s: int = 120) -> bytes:
    req = Request(url, headers={"x-access-token": token})
    with urlopen(req, timeout=timeout_s) as resp:
        data = resp.read()
        if not data:
            raise RuntimeError("Downloaded results.zip is empty")
        return data


def _wait_for_end_run_signal(abort_signal: Signal) -> None:
    """
    (NEW) Optionally block until we see the end-run signal (abort_signal.triggered).
    This helps ensure results have been written/uploaded before we start downloading.

    IMPORTANT: we cap the wait to avoid deadlocks if the controller expects this task
    to complete BEFORE it ends the run.
    """
    wait_enabled = _env_bool("NEUROFLAME_RESULTS_WAIT_FOR_END_RUN", True)
    if not wait_enabled:
        return

    max_wait_s = _env_int("NEUROFLAME_RESULTS_WAIT_FOR_END_RUN_MAX_SECONDS", 30)
    poll_ms = _env_int("NEUROFLAME_RESULTS_WAIT_FOR_END_RUN_POLL_MS", 250)

    logging.info(
        "ObserverExecutor: wait-for-end-run enabled. "
        f"Waiting up to {max_wait_s}s for abort_signal.triggered before downloading results."
    )

    start = time.monotonic()
    while True:
        if abort_signal and abort_signal.triggered:
            logging.info("ObserverExecutor: abort_signal.triggered detected; starting results download.")
            return

        elapsed = time.monotonic() - start
        if elapsed >= max_wait_s:
            logging.warning(
                "ObserverExecutor: did not observe abort_signal.triggered within "
                f"{max_wait_s}s. Proceeding with download attempts anyway (avoid deadlock)."
            )
            return

        time.sleep(max(0.01, poll_ms / 1000.0))


def _download_results_zip_with_retry(abort_signal: Signal) -> Optional[bytes]:
    file_server = _env("NEUROFLAME_FILESERVER_URL")
    consortium_id = _env("NEUROFLAME_CONSORTIUM_ID")
    run_id = _env("NEUROFLAME_RUN_ID")
    token = _env("NEUROFLAME_ACCESS_TOKEN")

    if not file_server or not consortium_id or not run_id:
        logging.error("ObserverExecutor: missing NEUROFLAME_FILESERVER_URL/CONSORTIUM_ID/RUN_ID")
        return None
    if not token:
        logging.error("ObserverExecutor: missing NEUROFLAME_ACCESS_TOKEN (fileServer uses x-access-token)")
        return None

    file_server = file_server.rstrip("/")
    url = f"{file_server}/download_results/{consortium_id}/{run_id}"

    max_attempts = _env_int("NEUROFLAME_RESULTS_MAX_ATTEMPTS", 5)
    delay_ms = _env_int("NEUROFLAME_RESULTS_RETRY_DELAY_MS", 5000)  # default 5s
    debug = _env_bool("NEUROFLAME_RESULTS_DEBUG", True)

    # If the server ends the run while we’re waiting for results to land,
    # keep trying for a bit so observers can still fetch results.
    abort_grace_s = _env_int("NEUROFLAME_RESULTS_ABORT_GRACE_SECONDS", 120)

    logging.info(
        f"ObserverExecutor: will download results with retry: url={url} "
        f"attempts={max_attempts} delay_ms={delay_ms} abort_grace_s={abort_grace_s}"
    )

    abort_seen_at: Optional[float] = None
    last_err: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        # Detect end-run signal but DO NOT stop immediately.
        if abort_signal and abort_signal.triggered and abort_seen_at is None:
            abort_seen_at = time.monotonic()
            logging.warning(
                "ObserverExecutor: abort_signal triggered (server requested end-run). "
                f"Continuing results download retries for up to {abort_grace_s}s."
            )

        # If abort was seen, enforce grace window.
        if abort_seen_at is not None:
            elapsed = time.monotonic() - abort_seen_at
            if elapsed > abort_grace_s:
                logging.error(
                    f"ObserverExecutor: abort grace window exceeded ({elapsed:.1f}s); giving up results download."
                )
                return None

        try:
            logging.info(f"[{tc.TASK_CLIENT_RECEIVE_RESULTS}] downloading results.zip (attempt {attempt}/{max_attempts})")
            return _download_results_zip_once(url=url, token=token)
        except HTTPError as e:
            code = getattr(e, "code", None)
            body = ""
            try:
                body = e.read().decode("utf-8", errors="ignore")
            except Exception:
                pass

            if debug:
                logging.error(f"[{tc.TASK_CLIENT_RECEIVE_RESULTS}] HTTP error attempt={attempt} code={code} body={body}")

            # Retry on expected/transient errors
            if code in (404, 409, 425, 429, 500, 502, 503, 504):
                last_err = e
            else:
                logging.error(f"[{tc.TASK_CLIENT_RECEIVE_RESULTS}] non-retryable HTTP error {code}: {body or str(e)}")
                return None

        except URLError as e:
            if debug:
                logging.error(f"[{tc.TASK_CLIENT_RECEIVE_RESULTS}] URL error attempt={attempt} err={e}")
            last_err = e

        except Exception as e:
            if debug:
                logging.error(f"[{tc.TASK_CLIENT_RECEIVE_RESULTS}] unexpected error attempt={attempt} err={e}")
            last_err = e

        # Sleep between attempts. If we're in abort-grace mode, prefer shorter sleeps so we
        # can squeeze more tries into the grace window.
        sleep_ms = delay_ms
        if abort_seen_at is not None:
            sleep_ms = min(delay_ms, 1000)  # 1s max once abort is triggered

        time.sleep(max(0.0, sleep_ms / 1000.0))

    logging.error(f"[{tc.TASK_CLIENT_RECEIVE_RESULTS}] exhausted retries; last error: {last_err}")
    return None


class ObserverExecutor(Executor):
    def __init__(self):
        # Keep this log line: it's very useful to confirm which task names are accepted.
        logging.info(f"ObserverExecutor initialized; accepted_tasks=['{tc.TASK_CLIENT_RECEIVE_RESULTS}']")

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        # Accept ONLY the single canonical alias.
        if task_name != tc.TASK_CLIENT_RECEIVE_RESULTS:
            return shareable

        # (NEW) Optionally wait until the end-run signal is observed before attempting downloads.
        _wait_for_end_run_signal(abort_signal)

        zip_bytes = _download_results_zip_with_retry(abort_signal)
        if zip_bytes is None:
            logging.warning("ObserverExecutor.receive_results: download failed; no results were saved")
            return shareable

        results_zip_path, results_extract_dir = _resolve_output_paths()

        try:
            results_extract_dir.mkdir(parents=True, exist_ok=True)
            results_zip_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logging.error(f"ObserverExecutor.receive_results: failed to create output dirs: {e}")
            return shareable

        try:
            results_zip_path.write_bytes(zip_bytes)
            logging.info(f"ObserverExecutor.receive_results: wrote {results_zip_path}")
        except Exception as e:
            logging.error(f"ObserverExecutor.receive_results: failed to write results.zip: {e}")
            return shareable

        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                zf.extractall(results_extract_dir)
            logging.info(f"ObserverExecutor.receive_results: extracted results to {results_extract_dir}")
        except Exception as e:
            logging.error(f"ObserverExecutor.receive_results: failed to extract results.zip: {e}")

        return shareable
