# Testing & Development

## Running Tests

```bash
cd ARC-AGI-3-Agents

# All tests
uv run pytest

# By marker
uv run pytest -m unit           # fast, no external dependencies
uv run pytest -m integration    # may use real APIs
uv run pytest -m slow           # slow-running tests

# Single file
uv run pytest tests/unit/test_core.py
uv run pytest tests/unit/test_recorder.py
uv run pytest tests/unit/test_swarm.py

# Single test
uv run pytest -k test_agent_init
uv run pytest -k "TestRandomAgent and test_agent_action_logic"
```

### Test Structure

```
tests/
├── conftest.py           # Shared fixtures
├── recordings/           # Auto-created, auto-cleaned per session
└── unit/
    ├── test_core.py      # GameAction, ActionInput, Card, Scorecard, FrameData, Random agent, LangGraphRandom agent
    ├── test_recorder.py  # Recorder init, file ops, timestamps, class methods, error handling, concurrency
    └── test_swarm.py     # Swarm init, scorecard open/close, agent threading, cleanup, tags
```

### Key Fixtures (conftest.py)

- `clean_test_recordings` (session-scoped, autouse): Sets `RECORDINGS_DIR` to `tests/recordings/`, cleans it before each session.
- `temp_recordings_dir`: Per-test isolated recordings directory.
- `sample_frame`: A pre-built `FrameData` with `state=NOT_FINISHED, score=5`.
- `use_env_vars`: Sets up test environment variables (`ARC_API_KEY`, `OPENAI_API_KEY`, connection params).

Note: Tests import `agents.structs` for `FrameData`, `GameAction`, etc. — this module re-exports types from `arcengine`.

## Linting & Formatting

```bash
cd ARC-AGI-3-Agents

# Check
uv run ruff check .
uv run ruff format --check .

# Fix
uv run ruff check --fix .
uv run ruff format .

# Type checking (strict mode, excludes tests/)
uv run mypy .
```

Ruff config: `extend-select = ["I"]` (import sorting).

## Creating a New Agent

1. Create a new file in `agents/templates/` (or add a class to an existing file).
2. Subclass `Agent` (for full control) or `LLM` (to get the prompt/observe/act loop for free).
3. Implement `choose_action()` and `is_done()`.
4. Import your class in `agents/__init__.py` and it auto-registers.

Minimal example:

```python
# agents/templates/my_agent.py
from typing import Any
from arcengine import FrameData, GameAction, GameState
from ..agent import Agent

class MyAgent(Agent):
    MAX_ACTIONS = 100

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            return GameAction.RESET
        # Your logic here
        return GameAction.ACTION1
```

Then in `agents/__init__.py`, add:
```python
from .templates.my_agent import MyAgent
```

Run it: `uv run main.py --agent=myagent --game=ls20`

## Experiment Workflow

1. Run an agent → it produces a `.recording.jsonl` file.
2. Replay anytime: `uv run main.py --agent=<recording-filename> --game=ls20`
3. Compare scorecards online at the ARC website if using online mode.
4. Use `--tags` to categorize experiments: `--tags=v2,new-prompt,high-reasoning`
5. Logs go to `logs.log` (overwritten each run) and stdout.
