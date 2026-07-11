"""
Dirty global object for debug request logging.
"""

import json
import os
from loguru import logger
from collections import OrderedDict


class RequestLogger:
    def __init__(self):
        # First Proof's run.sh only retrieves /data/output/, so the
        # default relative ``logs/requests`` lands at /app/logs/requests
        # inside the container and is never copied off. The entrypoint
        # sets ``MATHAGENTS_REQUEST_LOG_DIR`` to a path under
        # /data/output so the harness picks the logs up with the rest
        # of the submission. Local dev (CLI scripts, tests) keeps the
        # repo-relative default.
        self.log_dir = os.environ.get("MATHAGENTS_REQUEST_LOG_DIR") or "logs/requests"
        self.comp_name = None
        self.solver_name = None
        self.batch_idx_to_problem_idx = None

    def set_metadata(self, comp_name, solver_name, batch_idx_to_problem_idx):
        self.comp_name = comp_name
        self.solver_name = solver_name
        self.batch_idx_to_problem_idx = batch_idx_to_problem_idx

    def log_request(self, ts, batch_idx, request, **info):
        if self.comp_name is None:
            problem_idx = -1
            logfile = f"{self.log_dir}/uninitialized/{ts}_idx{batch_idx}.json"
        else:
            try:
                problem_idx = self.batch_idx_to_problem_idx[batch_idx]
            except:
                problem_idx = 0
            logfile = f"{self.log_dir}/{self.comp_name}/{self.solver_name}/{ts}_p{problem_idx}_idx{batch_idx}.json"
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        if os.path.exists(logfile):
            logger.warning(f"Can't log request, log file already exists: {logfile}")
            return

        data = OrderedDict(
            {
                "comp_name": self.comp_name,
                "solver_name": self.solver_name,
                "timestamp": ts,
                "problem_idx": problem_idx,
                "batch_idx": batch_idx,
                "request_info": info,
                "request": request,
            }
        )

        with open(logfile, "w") as f:
            json.dump(data, f, indent=4, default=str)

    def log_response(self, ts, batch_idx, response=None, **info):
        if self.comp_name is None:
            logfile = f"{self.log_dir}/uninitialized/{ts}_idx{batch_idx}.json"
        else:
            try:
                problem_idx = self.batch_idx_to_problem_idx[batch_idx]
            except:
                problem_idx = 0
            logfile = f"{self.log_dir}/{self.comp_name}/{self.solver_name}/{ts}_p{problem_idx}_idx{batch_idx}.json"
        if not os.path.exists(logfile):
            logger.warning(f"Can't log response, log file does not exist: {logfile}")
            return

        try:
            with open(logfile, "r") as f:
                data = json.load(f, object_pairs_hook=OrderedDict)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                f"Recovering malformed request log {logfile}: {type(e).__name__}: {e}"
            )
            data = OrderedDict(
                {
                    "comp_name": self.comp_name,
                    "solver_name": self.solver_name,
                    "timestamp": ts,
                    "problem_idx": problem_idx if self.comp_name is not None else -1,
                    "batch_idx": batch_idx,
                    "request_log_recovery": {
                        "type": type(e).__name__,
                        "msg": str(e),
                    },
                }
            )

        # Update the data with the response information
        data["response_info"] = info
        if response is not None:
            data["response"] = response
        with open(logfile, "w") as f:
            json.dump(data, f, indent=4, default=str)


request_logger = RequestLogger()
