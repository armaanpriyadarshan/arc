"""Safe sandbox for executing model-generated Python programs.

Variables persist across calls within a session — the model can
define functions, build data structures, and reference them later.
The `grid` variable is updated before each call.

Uses threading-based timeout (works on Windows and Unix).
"""

import builtins
import collections
import functools
import io
import json
import math
import threading
import traceback
from contextlib import redirect_stdout, redirect_stderr


try:
    import numpy as np
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


class Sandbox:
    """Persistent sandbox — variables survive across run() calls."""

    def __init__(self) -> None:
        safe_builtins = dict(vars(builtins))
        safe_builtins["__import__"] = safe_import
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
        """Execute code with the current grid. Variables persist across calls."""
        self.globals["grid"] = grid
        self.globals["ROWS"] = len(grid)
        self.globals["COLS"] = len(grid[0]) if grid else 0
        self.globals.pop("result", None)

        result_holder = {"output": "", "error": None}
        stdout_capture = io.StringIO()

        def _exec():
            try:
                with redirect_stdout(stdout_capture), redirect_stderr(io.StringIO()):
                    exec(code, self.globals)
            except Exception as e:
                result_holder["error"] = e

        thread = threading.Thread(target=_exec, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            return f"ERROR: TimeoutError: Code execution exceeded {timeout}s limit"

        if result_holder["error"] is not None:
            e = result_holder["error"]
            tb = traceback.format_exception(type(e), e, e.__traceback__)
            tb_str = "".join(tb)
            if len(tb_str) > 500:
                tb_str = "..." + tb_str[-500:]
            return f"ERROR: {type(e).__name__}: {e}\n{tb_str}"

        if "result" in self.globals and self.globals["result"] is not None:
            output = str(self.globals["result"])
        else:
            output = stdout_capture.getvalue()

        if not output:
            output = "(no output — set `result` or use print() to return data)"

        if len(output) > 2000:
            output = output[:2000] + "\n...(truncated to 2000 chars)"

        return output
