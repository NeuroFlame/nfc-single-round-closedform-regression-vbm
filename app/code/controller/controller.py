# controller.py

import json
import os
import time
from typing import Callable, Dict, List, Tuple, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.impl.controller import Controller, Task, ClientTask
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal

from utils import task_constants as tc
from utils.utils import get_parameters_file_path


class SRCFRController(Controller):
    """
    SRCFRController handles the flow of tasks for site regression and aggregation
    in the federated learning process.

    Role-aware flow:
      - Contributors perform compute tasks (steps 1–3).
      - Observers do not run compute tasks, but they DO participate in the final
        broadcast-only lifecycle task to receive results metadata (and ACK).

    Results are generated on the aggregator/server side (not on clients) and
    distributed via fileServer (e.g., results.zip in the server output).
    """

    def __init__(
        self,
        min_clients: int = 0,
        # Should be large enough that broadcast waits for all targeted clients
        wait_time_after_min_received: int = 1000,
        task_timeout: int = 0,
        # Optional: gate final broadcast until results.zip is downloadable
        gate_results_download: bool = True,
        # Optional: URL to probe (if not present in parameters, will skip gating)
        # If you use the NeuroFLAME fileServer, the client typically downloads from a route like:
        #   <file_server_url>/downloadResults?consortiumId=...&runId=...
        # We try to build it from parameters if possible.
        gate_poll_interval_sec: int = 3,
        gate_poll_timeout_sec: int = 180,
    ):
        super().__init__()
        self._task_timeout = task_timeout
        self._min_clients = int(min_clients) if min_clients is not None else 0
        self._wait_time_after_min_received = wait_time_after_min_received

        self._gate_results_download = bool(gate_results_download)
        self._gate_poll_interval_sec = int(gate_poll_interval_sec)
        self._gate_poll_timeout_sec = int(gate_poll_timeout_sec)

        self.srr_aggregator = None
        self._contributors: List[str] = []
        self._observers: List[str] = []
        self._all_clients: List[str] = []

    # ---------------------------------------------------------------------
    # Framework-specific required methods
    # ---------------------------------------------------------------------
    def start_controller(self, fl_ctx: FLContext) -> None:
        """
        Called when the controller starts. Assign aggregator and load parameters/roles.
        """
        engine = fl_ctx.get_engine()
        all_clients = engine.get_clients()

        client_names = [c.name for c in all_clients]

        print(client_names)

        self.srr_aggregator = self._engine.get_component(tc.SRCFR_AGGREGATOR_ID)
        if self.srr_aggregator is None:
            raise RuntimeError(f"Aggregator component '{tc.SRCFR_AGGREGATOR_ID}' not found in engine components")

        # Load computation parameters into server run context
        self._load_and_set_computation_parameters(fl_ctx)

        # Load participant roles from provisioning output
        roles = self._load_participant_roles(fl_ctx)

        # Split into contributor/observer targets
        self._contributors, self._observers, self._all_clients = self._split_clients_by_role(roles)
        if not self._contributors:
            raise ValueError("No contributors found after loading participant_roles.json (cannot run steps 1–3)")

        # Expose roles in run context (sticky)
        fl_ctx.set_prop(key="PARTICIPANT_ROLES", value=roles, private=False, sticky=True)

        # Best-effort: also mirror to peer context if available
        peer_ctx = fl_ctx.get_peer_context()
        if peer_ctx is not None:
            peer_ctx.set_prop(key="PARTICIPANT_ROLES", value=roles, private=False, sticky=True)

        self.log_info(
            fl_ctx,
            f"Loaded roles: contributors={self._contributors} observers={self._observers} all_clients={self._all_clients}",
        )

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext) -> None:
        """
        Role-aware flow:
          - Steps 1–3 broadcast ONLY to contributors and send results to the aggregator.
          - Aggregator aggregates after each step.
          - Final step broadcasts to ALL clients (contributors + observers) as a lifecycle task
            to receive results metadata and ACK.

        IMPORTANT:
          - The aggregator writes results.zip during its ROUND 2 aggregate().
          - The final broadcast should send the *metadata Shareable* returned by ROUND 2 aggregate().
          - Optional: gate final broadcast until results are downloadable (server-side probe).
        """
        # Task-level response requirements must match target lists
        min_resp_contrib = len(self._contributors)
        min_resp_all = len(self._all_clients)

        # -----------------------------
        # ROUND 0 / STEP 1 – contributors only
        # -----------------------------
        fl_ctx.set_prop(key="CURRENT_ROUND", value=0)

        self._broadcast_task(
            task_name=tc.TASK_CLIENT_STEP1_VALIDATE_INPUTS_GENERATE_MASK,
            data=Shareable(),
            result_cb=self._accept_site_result,
            targets=self._contributors,
            min_responses=min_resp_contrib,
            fl_ctx=fl_ctx,
            abort_signal=abort_signal,
        )
        if abort_signal.triggered:
            self.log_warning(fl_ctx, "Abort signal triggered after STEP 1 broadcast; stopping control flow")
            return

        aggregate_result = self._aggregate_or_fail(fl_ctx, step_name="STEP 1")

        # -----------------------------
        # ROUND 1 / STEP 2 – contributors only
        # -----------------------------
        fl_ctx.set_prop(key="CURRENT_ROUND", value=1)

        self._broadcast_task(
            task_name=tc.TASK_CLIENT_STEP2_COMPUTE_LOCAL_MODEL,
            data=aggregate_result,
            result_cb=self._accept_site_result,
            targets=self._contributors,
            min_responses=min_resp_contrib,
            fl_ctx=fl_ctx,
            abort_signal=abort_signal,
        )
        if abort_signal.triggered:
            self.log_warning(fl_ctx, "Abort signal triggered after STEP 2 broadcast; stopping control flow")
            return

        aggregate_result = self._aggregate_or_fail(fl_ctx, step_name="STEP 2")

        # -----------------------------
        # ROUND 2 / STEP 3 – contributors only
        # -----------------------------
        fl_ctx.set_prop(key="CURRENT_ROUND", value=2)

        self._broadcast_task(
            task_name=tc.TASK_CLIENT_STEP3_COMPUTE_METRICS,
            data=aggregate_result,
            result_cb=self._accept_site_result,
            targets=self._contributors,
            min_responses=min_resp_contrib,
            fl_ctx=fl_ctx,
            abort_signal=abort_signal,
        )
        if abort_signal.triggered:
            self.log_warning(fl_ctx, "Abort signal triggered after STEP 3 broadcast; stopping control flow")
            return

        # ROUND 2 aggregate() writes results.zip and returns metadata in outgoing_shareable["result"]
        results_meta_shareable = self._aggregate_or_fail(fl_ctx, step_name="STEP 3 / FINAL AGGREGATION")
        self.log_info(fl_ctx, "Aggregator returned final results metadata shareable")

        # -----------------------------
        # ROUND 3 / FINAL – lifecycle ACK task to ALL clients
        # -----------------------------
        fl_ctx.set_prop(key="CURRENT_ROUND", value=3)

        if self._gate_results_download:
            self._gate_until_results_ready(fl_ctx)

        self._broadcast_task(
            task_name=tc.TASK_CLIENT_RECEIVE_RESULTS,
            data=results_meta_shareable,  # <-- broadcast the metadata returned by round-2 aggregate()
            result_cb=None,
            targets=self._all_clients,
            min_responses=min_resp_all,
            fl_ctx=fl_ctx,
            abort_signal=abort_signal,
        )

    def process_result_of_unknown_task(self, task: Task, fl_ctx: FLContext) -> None:
        """Handles results for tasks that are not explicitly recognized."""
        pass

    def stop_controller(self, fl_ctx: FLContext) -> None:
        """Called when the controller stops. Override for cleanup if needed."""
        pass

    # ---------------------------------------------------------------------
    # Result callback
    # ---------------------------------------------------------------------

    def _accept_site_result(self, client_task: ClientTask, fl_ctx: FLContext) -> bool:
        """
        Callback method that processes each site's result and sends it to the aggregator.
        """
        ok = False
        try:
            ok = bool(self.srr_aggregator.accept(client_task.result, fl_ctx))
        finally:
            client_name = getattr(client_task, "client_name", None) or getattr(client_task, "peer", None) or "unknown"
            self.log_info(fl_ctx, f"Result received: client={client_name} accepted={ok}")
        return ok

    # ---------------------------------------------------------------------
    # Broadcast helper
    # ---------------------------------------------------------------------

    def _broadcast_task(
        self,
        task_name: str,
        data: Shareable,
        result_cb: Optional[Callable],
        targets: List[str],
        min_responses: int,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> None:
        """
        Broadcasts a task to specified client sites and waits for responses.
        """
        if not targets:
            raise ValueError(f"Refusing to broadcast task '{task_name}' to an empty targets list")

        task = Task(
            name=task_name,
            data=data,
            props={},
            timeout=self._task_timeout,
            result_received_cb=result_cb,
        )

        self.log_info(
            fl_ctx,
            f"Broadcasting task='{task_name}' targets={targets} min_responses={int(min_responses)} timeout={self._task_timeout}",
        )

        self.broadcast_and_wait(
            task=task,
            targets=targets,
            min_responses=int(min_responses),
            wait_time_after_min_received=self._wait_time_after_min_received,
            fl_ctx=fl_ctx,
            abort_signal=abort_signal,
        )

    # ---------------------------------------------------------------------
    # Aggregation helper
    # ---------------------------------------------------------------------

    def _aggregate_or_fail(self, fl_ctx: FLContext, step_name: str) -> Shareable:
        """
        Calls aggregator.aggregate(fl_ctx) with a clearer error message if results were missing.
        """
        try:
            return self.srr_aggregator.aggregate(fl_ctx)
        except KeyError as e:
            raise RuntimeError(
                f"{step_name}: Aggregation failed because expected site results are missing "
                f"(KeyError: {e}). This usually means:\n"
                f"  - a client errored and did not return a valid Shareable\n"
                f"  - accept() returned False for one or more clients\n"
                f"  - min_responses did not match targets and broadcast_and_wait moved on early\n"
                f"Check server logs for 'Result received: ... accepted=' lines and client stderr for exceptions."
            ) from e

    # ---------------------------------------------------------------------
    # Parameters + roles
    # ---------------------------------------------------------------------

    def _load_and_set_computation_parameters(self, fl_ctx: FLContext) -> None:
        """
        Loads computation parameters from a file and sets them in the shared context for all sites to access.
        """
        with open(get_parameters_file_path(fl_ctx), "r", encoding="utf-8") as f:
            params = json.load(f)

        fl_ctx.set_prop(
            key="COMPUTATION_PARAMETERS",
            value=params,
            private=False,
            sticky=True,
        )

    def _load_participant_roles(self, fl_ctx: FLContext) -> Dict[str, str]:
        """
        Loads participant roles using the ridge system format:
        { "user_roles": { "<site>": "contributor|observer|server", ... } }

        Fallback behavior (if participant_roles.json is not present):
        - assigns 'contributor' to all non-server clients known to the engine
        - ensures roles['server'] == 'server'
        """
        candidates: List[str] = []

        app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
        if app_root:
            candidates.append(os.path.join(str(app_root), "config", "participant_roles.json"))

        candidates.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "config", "participant_roles.json")))
        candidates.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "config", "participant_roles.json")))

        roles_path = next((p for p in candidates if os.path.isfile(p)), None)

        # -------------------------
        # Fallback: no roles file
        # -------------------------
        if not roles_path:
            roles: Dict[str, str] = {"server": "server"}

            engine = fl_ctx.get_engine()
            if engine is None:
                # No engine => best we can do is return server only
                return roles

            # NVFlare engine typically exposes get_clients(); guard for version differences.
            get_clients = getattr(engine, "get_clients", None)
            if callable(get_clients):
                try:
                    clients = get_clients()
                except Exception:
                    clients = []
            else:
                clients = []

            # Clients are objects; name is usually .name, but be defensive.
            for c in clients or []:
                name = getattr(c, "name", None)
                if not name:
                    # Try common alternatives
                    name = getattr(c, "client_name", None) or getattr(c, "identity_name", None)

                if not name:
                    continue

                site = str(name).strip()
                if not site:
                    continue

                # Treat "server" as server; everything else => contributor
                if site.lower() == "server":
                    roles["server"] = "server"
                else:
                    roles.setdefault(site, "contributor")

            return roles

        # -------------------------
        # Normal path: load file
        # -------------------------
        with open(roles_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if "user_roles" not in raw or not isinstance(raw["user_roles"], dict):
            raise ValueError("participant_roles.json must contain a 'user_roles' mapping (ridge format)")

        roles = {k: str(v).strip().lower() for k, v in raw["user_roles"].items()}
        roles.setdefault("server", "server")

        for site, role in roles.items():
            if role not in ("server", "contributor", "observer"):
                raise ValueError(f"Unknown role '{role}' for site '{site}' in participant_roles.json")

        return roles

    def _split_clients_by_role(self, roles: Dict[str, str]) -> Tuple[List[str], List[str], List[str]]:
        """
        Splits sites into contributor targets, observer targets, and all-client targets (excluding server).
        """
        contributors: List[str] = []
        observers: List[str] = []
        all_clients: List[str] = []

        for site, role in roles.items():
            if role == "server":
                continue
            all_clients.append(site)
            if role == "contributor":
                contributors.append(site)
            elif role == "observer":
                observers.append(site)

        return sorted(contributors), sorted(observers), sorted(all_clients)

    # ---------------------------------------------------------------------
    # Optional gating: wait until results are downloadable before notifying observers
    # ---------------------------------------------------------------------

    def _gate_until_results_ready(self, fl_ctx: FLContext) -> None:
        """
        Best-effort gating to reduce observer 404s:
        - Tries to build a download URL from COMPUTATION_PARAMETERS['neuroflame'] if present.
        - If cannot build URL, it logs and returns (clients should still retry).
        - If URL exists, polls until HTTP 200 or timeout.
        """
        try:
            params = fl_ctx.get_prop("COMPUTATION_PARAMETERS", default={}) or {}
            nf = params.get("neuroflame") or params.get("NeuroFLAME") or {}
            if not isinstance(nf, dict):
                nf = {}

            file_server_url = nf.get("file_server_url") or nf.get("fileServerUrl")
            consortium_id = nf.get("consortium_id") or nf.get("consortiumId")
            run_id = nf.get("run_id") or nf.get("runId")
            token = nf.get("token") or nf.get("downloadToken") or nf.get("uploadToken")

            if not (file_server_url and consortium_id and run_id and token):
                self.log_info(fl_ctx, "Gating skipped: missing neuroflame fileServer params (client-side retry will handle)")
                return

            # This route name is a guess; if yours differs, client-side retry remains the source of truth.
            # If you have a known route, hardcode it here.
            # Example common style:
            #   GET {file_server_url}/downloadResults?consortiumId=...&runId=...
            download_url = (
                f"{str(file_server_url).rstrip('/')}/downloadResults"
                f"?consortiumId={consortium_id}&runId={run_id}"
            )

            start = time.time()
            while True:
                if time.time() - start > self._gate_poll_timeout_sec:
                    self.log_warning(fl_ctx, f"Gating timed out after {self._gate_poll_timeout_sec}s; proceeding anyway")
                    return

                try:
                    req = Request(download_url, headers={"x-access-token": str(token)})
                    with urlopen(req, timeout=10) as resp:
                        code = getattr(resp, "status", 200)
                        if int(code) == 200:
                            self.log_info(fl_ctx, "Gating OK: results appear downloadable (HTTP 200)")
                            return
                except HTTPError as e:
                    # 404/403/etc: keep polling until timeout
                    self.log_info(fl_ctx, f"Gating probe HTTPError={e.code}; retrying in {self._gate_poll_interval_sec}s")
                except URLError:
                    self.log_info(fl_ctx, f"Gating probe URLError; retrying in {self._gate_poll_interval_sec}s")
                except Exception:
                    self.log_info(fl_ctx, f"Gating probe error; retrying in {self._gate_poll_interval_sec}s")

                time.sleep(self._gate_poll_interval_sec)

        except Exception:
            # Never block the run due to gating issues
            self.log_warning(fl_ctx, "Gating failed unexpectedly; proceeding without gating")
            return
