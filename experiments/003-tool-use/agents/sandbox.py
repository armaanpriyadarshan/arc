"""Safe sandbox for executing model-generated Python programs.

The model writes code that operates on the grid. We execute it
in a restricted environment and return the output.
"""

import io
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr


def run_program(code: str, grid: list[list[int]], timeout_sec: int = 5) -> str:
    """Execute a Python program with `grid` available as a variable.

    The program can print() results. Returns stdout or error message.
    Max output: 2000 chars.
    """
    # Restricted globals — no file access, no imports except math/collections
    safe_globals = {
        "__builtins__": {
            "range": range, "len": len, "print": print,
            "min": min, "max": max, "abs": abs, "sum": sum,
            "sorted": sorted, "enumerate": enumerate, "zip": zip,
            "list": list, "dict": dict, "set": set, "tuple": tuple,
            "int": int, "float": float, "str": str, "bool": bool,
            "True": True, "False": False, "None": None,
            "isinstance": isinstance, "type": type,
            "map": map, "filter": filter, "any": any, "all": all,
            "reversed": reversed,
        },
        "grid": grid,
        "ROWS": len(grid),
        "COLS": len(grid[0]) if grid else 0,
    }

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, safe_globals)
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}"

    output = stdout_capture.getvalue()
    if len(output) > 2000:
        output = output[:2000] + "\n... (truncated)"
    if not output:
        output = "(no output — use print() to return results)"
    return output
