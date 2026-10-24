#!/usr/bin/env python3
"""Run the Gleipnir API and web dev servers together with labelled, live logs.

Usage: ``make dev`` (or ``python scripts/dev.py`` inside the project env).

The API reloads on changes under ``backend/`` and ``src/``, and also under the
mjollnir checkout when mjollnir is installed in editable mode. The web app
hot-reloads through the React dev server, which also proxies ``/api`` to the
API (frontend/src/setupProxy.js), so a browser on another machine needs only
the web port. Ctrl-C stops both.

Environment overrides: ``GLEIPNIR_API_PORT`` (default 8000) and
``GLEIPNIR_WEB_PORT`` (default 3000).
"""

from __future__ import annotations

import getpass
import importlib.util
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import IO

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
POLL_INTERVAL_SECONDS = 0.2
READY_TIMEOUT_SECONDS = 180.0
SHUTDOWN_TIMEOUT_SECONDS = 5.0

USE_COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ
LABEL_COLORS = {"api": "\033[36m", "web": "\033[35m", "dev": "\033[32m"}
RESET = "\033[0m"

print_lock = threading.Lock()


def _say(label: str, line: str) -> None:
    prefix = f"[{label}]".ljust(6)
    if USE_COLOR:
        prefix = f"{LABEL_COLORS[label]}{prefix}{RESET}"
    with print_lock:
        print(f"{prefix} {line}", flush=True)


def _editable_mjollnir_dir() -> Path | None:
    """Return the mjollnir checkout to watch, or None when it is a pinned install."""
    spec = importlib.util.find_spec("mjollnir")
    if spec is None or spec.origin is None:
        return None
    package_dir = Path(spec.origin).resolve().parent
    if "site-packages" in package_dir.parts:
        return None
    return package_dir.parent


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _check_prerequisites(api_port: int, web_port: int) -> str:
    missing = [name for name in ("uvicorn", "fastapi", "mjollnir") if importlib.util.find_spec(name) is None]
    if missing:
        return f"missing Python packages: {', '.join(missing)}. Run `make install` first."
    if shutil.which("npm") is None:
        return "npm was not found on PATH. Install Node.js, then run `make install`."
    if not (FRONTEND / "node_modules").is_dir():
        return "frontend dependencies are missing. Run `make install` first."
    for name, port in (("API", api_port), ("web", web_port)):
        if _port_in_use(port):
            return (
                f"port {port} for the {name} server is already in use. Stop whatever is "
                f"listening there or set GLEIPNIR_{name.upper()}_PORT."
            )
    return ""


def _spawn(command: list[str], cwd: Path, env: dict[str, str]) -> subprocess.Popen[str]:
    return subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,  # own process group, so shutdown reaches reloader children
    )


def _pump(label: str, stream: IO[str]) -> None:
    for line in stream:
        _say(label, line.rstrip("\n"))


def _signal_group(process: subprocess.Popen[str], sig: signal.Signals) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def _stop_all(processes: dict[str, subprocess.Popen[str]]) -> None:
    for process in processes.values():
        _signal_group(process, signal.SIGTERM)
    deadline = time.monotonic() + SHUTDOWN_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if all(process.poll() is not None for process in processes.values()):
            return
        time.sleep(POLL_INTERVAL_SECONDS)
    for process in processes.values():
        _signal_group(process, signal.SIGKILL)
        process.wait()


def _responds(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=1.0) as response:
            return response.status < 500
    except (urllib.error.URLError, ConnectionError, TimeoutError):
        return False


def _announce_when_ready(api_port: int, web_port: int, stop: threading.Event) -> None:
    api_url = f"http://localhost:{api_port}"
    web_url = f"http://localhost:{web_port}"
    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    pending = {"api": f"{api_url}/health", "web": web_url}
    while pending and not stop.is_set() and time.monotonic() < deadline:
        for name, url in list(pending.items()):
            if _responds(url):
                del pending[name]
        time.sleep(0.5)
    if stop.is_set():
        return
    if pending:
        _say("dev", f"still waiting on {', '.join(pending)} after {READY_TIMEOUT_SECONDS:.0f}s; check the logs above.")
        return
    _say("dev", f"ready  web {web_url}  |  api docs {api_url}/api/v1/docs  |  Ctrl-C stops both")
    ssh = os.environ.get("SSH_CONNECTION", "").split()
    if len(ssh) == 4:
        host = ssh[2]
        _say("dev", f"over SSH: open http://{host}:{web_port} on the same network, or tunnel with")
        _say("dev", f"  ssh -N -L {web_port}:localhost:{web_port} {getpass.getuser()}@{host}  ->  {web_url}")
        _say("dev", f"  (if {web_port} is busy on your machine, any free local port works: -L 3300:localhost:{web_port})")


def main() -> int:
    api_port = int(os.environ.get("GLEIPNIR_API_PORT", "8000"))
    web_port = int(os.environ.get("GLEIPNIR_WEB_PORT", "3000"))
    problem = _check_prerequisites(api_port, web_port)
    if problem:
        _say("dev", f"error: {problem}")
        return 1

    reload_dirs = [ROOT / "backend", ROOT / "src"]
    mjollnir_dir = _editable_mjollnir_dir()
    if mjollnir_dir is not None:
        reload_dirs.append(mjollnir_dir)
    api_command = [sys.executable, "-m", "uvicorn", "backend.main:app", "--reload", "--port", str(api_port)]
    for directory in reload_dirs:
        api_command += ["--reload-dir", str(directory)]

    api_env = {"LAB_WARM_UP": "background", **os.environ}  # reloads serve at once
    web_env = {
        **os.environ,
        "PORT": str(web_port),
        "BROWSER": os.environ.get("BROWSER", "none"),
        "REACT_APP_API_URL": "/api/v1",  # same origin; setupProxy.js forwards to the API
        "GLEIPNIR_API_PORT": str(api_port),
    }
    npm = shutil.which("npm")
    assert npm is not None  # checked in _check_prerequisites

    _say("dev", f"mjollnir: {'editable ' + str(mjollnir_dir) if mjollnir_dir else 'pinned install'}")

    requested_signal: int | None = None

    def request_shutdown(signum: int, _frame: object) -> None:
        nonlocal requested_signal
        requested_signal = signum

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)

    processes: dict[str, subprocess.Popen[str]] = {}
    stop = threading.Event()
    try:
        processes["api"] = _spawn(api_command, ROOT, api_env)
        processes["web"] = _spawn([npm, "start"], FRONTEND, web_env)
        for label, process in processes.items():
            assert process.stdout is not None
            threading.Thread(target=_pump, args=(label, process.stdout), daemon=True).start()
        threading.Thread(target=_announce_when_ready, args=(api_port, web_port, stop), daemon=True).start()

        while requested_signal is None:
            for label, process in processes.items():
                code = process.poll()
                if code is not None:
                    _say("dev", f"{label} server exited with status {code}; stopping the other one.")
                    return code or 1
            time.sleep(POLL_INTERVAL_SECONDS)
        return 128 + requested_signal
    finally:
        stop.set()
        _stop_all(processes)


if __name__ == "__main__":
    raise SystemExit(main())
