"""Safe sandbox for executing model-generated Python programs.

Variables persist across calls within a session — the model can
define functions, build data structures, and reference them later.
The `grid` variable is updated before each call.
"""

import collections
import functools
import io
import math
import traceback
from contextlib import redirect_stdout, redirect_stderr

import numpy as np


def safe_import(name, *args, **kwargs):
    allowed = {
        "collections": collections,
        "math": math,
        "functools": functools,
        "numpy": np,
        "np": np,
    }
    if name in allowed:
        return allowed[name]
    raise ImportError(f"Module '{name}' not available. Available: {list(allowed.keys())}")


class Sandbox:
    """Persistent sandbox — variables survive across run() calls."""

    def __init__(self) -> None:
        self.globals: dict = {
            "__builtins__": {
                "range": range, "len": len, "print": print,
                "min": min, "max": max, "abs": abs, "sum": sum,
                "sorted": sorted, "enumerate": enumerate, "zip": zip,
                "list": list, "dict": dict, "set": set, "tuple": tuple,
                "int": int, "float": float, "str": str, "bool": bool,
                "True": True, "False": False, "None": None,
                "isinstance": isinstance, "type": type,
                "map": map, "filter": filter, "any": any, "all": all,
                "reversed": reversed, "hash": hash,
                "__import__": safe_import,
            },
            "grid": [],
            "ROWS": 64,
            "COLS": 64,
            "collections": collections,
            "defaultdict": collections.defaultdict,
            "deque": collections.deque,
            "Counter": collections.Counter,
            "math": math,
            "functools": functools,
            "np": np,
            "numpy": np,
        }

    def run(self, code: str, grid: list[list[int]]) -> str:
        """Execute code with the current grid. Variables persist across calls."""
        self.globals["grid"] = grid
        self.globals["ROWS"] = len(grid)
        self.globals["COLS"] = len(grid[0]) if grid else 0
        # Also provide a flat version and numpy array for convenience
        self.globals["grid_flat"] = [cell for row in grid for cell in row]
        self.globals["grid_np"] = np.array(grid, dtype=np.int32)

        stdout_capture = io.StringIO()

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(io.StringIO()):
                exec(code, self.globals)
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()[-300:]}"

        output = stdout_capture.getvalue()
        if len(output) > 2000:
            output = output[:2000] + "\n...(truncated)"
        if not output:
            output = "(no output — use print() to return results)"
        return output
