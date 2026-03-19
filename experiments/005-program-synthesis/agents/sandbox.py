"""Safe sandbox for executing model-generated Python programs.

Variables persist across calls within a session — the model can
define functions, build data structures, and reference them later.
The `grid` variable is updated before each call.
"""

import builtins
import collections
import functools
import io
import json
import math
import signal
import traceback
from contextlib import redirect_stdout, redirect_stderr


# Pre-import numpy and force internal submodules to load
try:
    import numpy as np
    # Force lazy-loaded internals so sandbox exec doesn't trigger blocked imports
    _ = np.array([0]).mean()
    _ = np.zeros(1)
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False


def safe_import(name, *args, **kwargs):
    allowed = {
        "collections": collections,
        "math": math,
        "functools": functools,
        "json": json,
    }
    if HAS_NUMPY:
        allowed["numpy"] = np
        allowed["np"] = np
    if name in allowed:
        return allowed[name]
    raise ImportError(f"Module '{name}' not available. Available: {list(allowed.keys())}")


class _Timeout:
    """Context manager for signal-based timeout (Unix only)."""

    def __init__(self, seconds: int) -> None:
        self.seconds = seconds

    def __enter__(self) -> "_Timeout":
        self._old_handler = signal.signal(signal.SIGALRM, self._handler)
        signal.alarm(self.seconds)
        return self

    def __exit__(self, *args) -> None:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self._old_handler)

    def _handler(self, signum, frame):
        raise TimeoutError(f"Code execution exceeded {self.seconds}s limit")


class Sandbox:
    """Persistent sandbox — variables survive across run() calls."""

    def __init__(self) -> None:
        # Start with real builtins, then overlay safe_import
        safe_builtins = dict(vars(builtins))
        safe_builtins["__import__"] = safe_import
        # Remove dangerous builtins
        for name in ("open", "exec", "eval", "compile", "__loader__",
                     "exit", "quit", "breakpoint", "input"):
            safe_builtins.pop(name, None)

        self.globals: dict = {
            "__builtins__": safe_builtins,
            "grid": [],
            "ROWS": 64,
            "COLS": 64,
            "collections": collections,
            "defaultdict": collections.defaultdict,
            "deque": collections.deque,
            "Counter": collections.Counter,
            "math": math,
            "functools": functools,
            "json": json,
        }
        if HAS_NUMPY:
            self.globals["np"] = np
            self.globals["numpy"] = np

    def run(self, code: str, grid: list[list[int]], timeout: int = 5) -> str:
        """Execute code with the current grid. Variables persist across calls.

        Returns either the value of `result` (if set), captured stdout,
        or an error traceback. Output is truncated to 2000 chars.
        """
        self.globals["grid"] = grid
        self.globals["ROWS"] = len(grid)
        self.globals["COLS"] = len(grid[0]) if grid else 0

        # Clear any previous result
        self.globals.pop("result", None)

        stdout_capture = io.StringIO()

        try:
            with _Timeout(timeout), \
                 redirect_stdout(stdout_capture), \
                 redirect_stderr(io.StringIO()):
                exec(code, self.globals)
        except Exception as e:
            tb = traceback.format_exc()
            if len(tb) > 500:
                tb = "..." + tb[-500:]
            return f"ERROR: {type(e).__name__}: {e}\n{tb}"

        # Prefer `result` variable if set, otherwise use stdout
        if "result" in self.globals and self.globals["result"] is not None:
            output = str(self.globals["result"])
        else:
            output = stdout_capture.getvalue()

        if not output:
            output = "(no output — set `result` or use print() to return data)"

        # Truncate to prevent prompt bloat
        if len(output) > 2000:
            output = output[:2000] + "\n...(truncated to 2000 chars)"

        return output
