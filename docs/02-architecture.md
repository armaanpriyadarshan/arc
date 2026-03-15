# Architecture Deep Dive

## The Execution Flow

```
main.py (CLI entry point)
  │
  ├─ Parses --agent, --game, --tags
  ├─ Fetches game list from API (or derives from recording filename)
  ├─ Creates a Swarm
  │
  └─ Swarm.main()
       ├─ Opens a scorecard via Arcade SDK
       ├─ Creates one Agent instance per game
       ├─ Runs each agent in its own thread
       ├─ Waits for all threads to complete
       ├─ Closes the scorecard
       └─ Calls cleanup on all agents
```

### main.py — The CLI

`main.py` loads `.env.example` then `.env` (with override), sets up logging (to both stdout and `logs.log`), and parses three CLI args:
- `--agent` / `-a`: Required. Must be a key in `AVAILABLE_AGENTS` (auto-populated from `Agent` subclasses + recording files).
- `--game` / `-g`: Optional. Comma-separated prefix filter on the full game list fetched from the API.
- `--tags` / `-t`: Optional. Comma-separated tags attached to the scorecard.

The agent runs in a daemon thread. A SIGINT handler calls `cleanup()` to close the scorecard gracefully on Ctrl+C.

### Swarm — Orchestration Layer

**File:** `agents/swarm.py`

`Swarm` is the orchestrator. It receives the agent name, resolves it to a class via `AVAILABLE_AGENTS`, then:

1. **Opens a scorecard** via `Arcade.open_scorecard(tags)` — this is a session ID that groups all game results together.
2. **Creates agents** — one instance of the agent class per game, each with its own `EnvironmentWrapper` from `Arcade.make(game_id, scorecard_id)`.
3. **Threads** — each agent gets its own `Thread(target=a.main, daemon=True)`. All threads start, then all are joined.
4. **Closes scorecard** — after all agents finish, the scorecard is closed and results are logged.
5. **Cleanup** — calls `cleanup()` on each agent (saves recording metadata, logs stats).

Key: `Swarm` creates an `Arcade()` instance which manages local game environments. The `Arcade` comes from the `arc-agi` package.

### Agent — The Base Class

**File:** `agents/agent.py`

`Agent` is the abstract base class. Every agent must implement two methods:
- `choose_action(frames, latest_frame) -> GameAction`: Decide what to do next.
- `is_done(frames, latest_frame) -> bool`: Decide if the game is over (typically check for `GameState.WIN`).

The `main()` loop (decorated with `@trace_agent_session` for AgentOps tracing):

```python
while not self.is_done(...) and self.action_counter <= self.MAX_ACTIONS:
    action = self.choose_action(frames, latest_frame_raw)
    frame = self.take_action(action)       # sends action to environment
    self.append_frame(frame)               # stores frame + records to JSONL
    self.action_counter += 1
self.cleanup()
```

**Key properties:**
- `MAX_ACTIONS` (default 80): Safety limit to prevent infinite loops.
- `frames`: List of all `FrameData` received so far.
- `action_counter`: How many actions have been taken.
- `fps`: Actions per second.
- `state`: Current `GameState` from the last frame.
- `recorder`: A `Recorder` instance that writes JSONL.

**Action execution:** `take_action(action)` calls `do_action_request(action)` which calls `self.arc_env.step(action, data, reasoning)`. The environment wrapper comes from `arc-agi` and can be either local or online.

### FrameData and GameAction — Core Types

These come from the `arcengine` package (installed as part of `arc-agi`).

**GameAction** — an enum:
- `RESET` (id 0): Start/restart the game.
- `ACTION1` through `ACTION5` (ids 1–5): Simple actions (no parameters). In LockSmith: up, down, left, right, enter/spacebar.
- `ACTION6` (id 6): Complex action requiring `x, y` coordinates (0–63).
- `ACTION7` (id 7): Another action.

Actions have methods:
- `is_simple()` / `is_complex()`: Check type.
- `set_data(dict)`: Set parameters (game_id, x, y).
- `from_id(int)` / `from_name(str)`: Construct from int or string.
- `.reasoning`: Attach arbitrary reasoning metadata (string or dict).

**FrameData** — a Pydantic model containing:
- `game_id`: Which game this frame is from.
- `frame`: `list[list[list[int]]]` — a 3D array. Each element is a 2D grid (64×64 of ints 0–15). Usually one grid per frame, but can have multiple (animation frames).
- `state`: `GameState` enum — `NOT_PLAYED`, `NOT_FINISHED`, `GAME_OVER`, `WIN`.
- `levels_completed` (aliased as `score` in some places): 0–254, how many levels beaten.
- `win_levels`: How many levels needed to win.
- `available_actions`: List of `GameAction` available in this state.
- `guid`: Unique identifier for the game session.
- `full_reset`: Whether the game fully reset (new level).
- `action_input`: `ActionInput` — what action was taken to produce this frame.

### Recorder — JSONL Gameplay Logging

**File:** `agents/recorder.py`

Every agent automatically records its gameplay as JSONL (one JSON object per line). The `Recorder`:
- Creates files named `{game}.{agent}.{max_actions}.{uuid}.recording.jsonl` in the `RECORDINGS_DIR`.
- Each line has `{"timestamp": "...", "data": {...}}`.
- `data` contains either a frame dump, token counts, or scorecard results.

Recording files are automatically discovered and can be replayed:
```bash
uv run main.py --agent=ls20-cb3b57cc.random.80.uuid-here.recording.jsonl --game=ls20
```

### Playback — Replaying Recordings

**File:** `agents/agent.py` (class `Playback`)

`Playback` extends `Agent`. It loads a JSONL recording and replays the actions at `PLAYBACK_FPS` (5 fps). It extracts actions from the `action_input` field in each recorded event.

### Tracing — AgentOps Integration

**File:** `agents/tracing.py`

Optional observability. If `agentops` is installed and `AGENTOPS_API_KEY` is set:
- Each agent's `main()` is wrapped with `@trace_agent_session`.
- Creates a trace per agent execution with tags.
- Sets status to "Success", "Indeterminate" (hit MAX_ACTIONS), or "Error".

If `agentops` is not installed, a `NoOpAgentOps` class provides no-op stubs so the code runs without it.

### agents/__init__.py — Agent Registry

**File:** `agents/__init__.py`

This file builds `AVAILABLE_AGENTS`, a dict mapping lowercase class names to agent classes:
1. Gets all direct subclasses of `Agent`.
2. Excludes `Playback`.
3. Adds all `.recording.jsonl` files from `recordings/` as `Playback` entries.
4. Manually adds `ReasoningAgent` (since it's a grandchild of `Agent`, not a direct subclass).

This dict is what `--agent` validates against.
