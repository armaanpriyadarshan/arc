# Agent Templates — Detailed Breakdown

All templates live in `agents/templates/`. Each is a subclass of `Agent` (or of `LLM`, which itself extends `Agent`).

## 1. Random Agent (`random_agent.py`)

**Class:** `Random`
**MAX_ACTIONS:** 80

The simplest agent. Seeds a random number generator and picks random `GameAction` values each turn. If the game state is `NOT_PLAYED` or `GAME_OVER`, it sends `RESET`. Otherwise it picks a random non-RESET action.

For complex actions (like `ACTION6`), it generates random x,y coordinates in the 0–63 range.

**`is_done`:** Stops only on `GameState.WIN`. There's a commented-out option to also stop on `GAME_OVER` (which would make it single-attempt).

**Good for:** Baselines, testing the pipeline, verifying environment connectivity.

---

## 2. LLM Agent Family (`llm_agents.py`)

These are the core OpenAI-based agents. The base class `LLM` provides a sophisticated prompt → observe → act loop.

### LLM (base)

**Model:** `gpt-4o-mini` | **MAX_ACTIONS:** 80 | **DO_OBSERVATION:** True

The `choose_action` flow:

1. **First turn:** Sends a user prompt explaining the game context, then manually injects a RESET action as the first function call. Returns `GameAction.RESET`.

2. **Subsequent turns:**
   a. Constructs a function response message containing the frame state (grid pretty-printed as text), score, and game state.
   b. If `DO_OBSERVATION` is True, makes an LLM call **without tools** so the model can observe and reason about the frame in plain text.
   c. Sends the user prompt asking for an action.
   d. Makes an LLM call **with tools** (function calling). The model picks one of the GameAction functions.
   e. Parses the function call response into a `GameAction`.

**Two calling conventions:**
- `MODEL_REQUIRES_TOOLS = False` (default): Uses the deprecated OpenAI `functions`/`function_call` API.
- `MODEL_REQUIRES_TOOLS = True`: Uses the modern `tools`/`tool_choice` API.

**Message management:** `push_message()` keeps a sliding window of the last `MESSAGE_LIMIT` (10) messages. When using tools mode, it ensures the window doesn't start with a `tool` role message (which would cause API errors).

**Tool definitions:** `build_functions()` creates JSON descriptions of each `GameAction`:
- RESET, ACTION1–5 are "simple" (no parameters).
- ACTION6 is "complex" (requires x, y coordinates as strings 0–63).

**Token tracking:** Each LLM response is logged via `track_tokens()`, which also writes to the JSONL recording.

### FastLLM

Extends `LLM` with `DO_OBSERVATION = False` — skips the observation step, going directly from frame data to action selection. Faster but less context for the model.

### ReasoningLLM

**Model:** `o4-mini` | **MODEL_REQUIRES_TOOLS:** True

Uses OpenAI's reasoning model. Overrides `choose_action` to attach reasoning metadata (model name, reasoning tokens, game context) to `action.reasoning`. Captures reasoning token counts from `response.usage.completion_tokens_details.reasoning_tokens`.

### GuidedLLM

**Model:** `o3` | **REASONING_EFFORT:** `"high"` | **MODEL_REQUIRES_TOOLS:** True

The most hand-tuned agent. The user prompt contains **explicit LockSmith game rules**:
- ACTION1=up, ACTION2=down, ACTION3=left, ACTION4=right
- Player shape, wall colors, floor colors, key mechanics, rotators, energy pills, door structure
- Strategy hints and grid row references for score/energy indicators

This demonstrates how game-specific knowledge dramatically improves agent performance.

### MyCustomLLM (template)

A commented-out template showing where to add custom instructions. Inherit from `LLM` and override `build_user_prompt()`.

---

## 3. Reasoning Agent (`reasoning_agent.py`)

**Class:** `ReasoningAgent` (extends `ReasoningLLM`)
**Model:** `o4-mini` | **MAX_ACTIONS:** 400 | **REASONING_EFFORT:** `"high"`

A hypothesis-driven exploration agent. The key innovation: it **renders the game grid as a PNG image** and sends it to the LLM alongside the raw grid data.

**Data model:** Uses `ReasoningActionResponse` (Pydantic BaseModel) for structured output:
- `name`: Which action to take
- `reason`: Detailed reasoning (10–2000 chars)
- `short_description`: Brief action description
- `hypothesis`: Current hypothesis about game mechanics
- `aggregated_findings`: Running summary of discoveries

**Flow:**
1. First action is always RESET (hardcoded initial response).
2. On `full_reset` (new level), clears all history.
3. Otherwise calls `define_next_action()`:
   - Renders current grid to a colored PNG with zone overlays (16×16 zones, gold borders, coordinate labels).
   - Sends the LLM: system prompt, previous screen image, current screen image, previous action history, raw grid text.
   - Gets back a structured `ReasoningActionResponse` via tool calling.
   - Stores up to 10 screen images for comparison.

**Key config:** `ZONE_SIZE = 16` — the grid is divided into 4×4 zones for spatial reference. Cell size is 40px per grid cell in the rendered image.

---

## 4. MultiModal LLM (`multimodal.py`)

**Class:** `MultiModalLLM` (extends `Agent` directly, not `LLM`)
**Model:** `gpt-4o-mini` | **MAX_ACTIONS:** 40

The most sophisticated agent architecture. It has a **self-programming memory** system:

**Three-phase loop per turn:**

1. **Analysis phase** (if there was a previous action):
   - Sends the LLM: previous frame images, previous prompt, previous response, new frame images, and a pixel-diff image (red highlights on black showing what changed).
   - The LLM analyzes what happened and returns text split by `---`:
     - Before `---`: Analysis of what happened.
     - After `---`: Updated memory prompt (game rules, action log, goals — the LLM rewrites its own instructions).

2. **Action decision phase:**
   - Sends the LLM: current frame images, the (possibly updated) memory prompt, and an instruction to return JSON with `human_action`, `reasoning`, and `expected_result`.
   - The LLM decides a high-level "human action" like "Click on the red square near the bottom left corner".

3. **Action mapping phase:**
   - A second LLM call maps the human-readable action to a specific `GameAction` + coordinates.
   - Returns JSON with `action`, `x`, `y`.

**Image handling:**
- Uses `grid_to_image()` to convert each grid to a PIL Image with a 16-color palette.
- Uses `image_diff()` to create a red-on-black diff image between frames.
- All images are sent as base64 inline PNGs.

**Memory prompt:** Starts with a template containing `{{human.inputs}}` (replaced with available actions on first turn), a "Current Goal", "Game Rules", and "Action Log". The LLM updates this after every turn, effectively programming its own context.

---

## 5. LangGraph Functional Agent (`langgraph_functional_agent.py`)

**Classes:** `LangGraphFunc`, `LangGraphTextOnly`
**Model:** `o4-mini`

Uses LangGraph's functional API (`@entrypoint`) with an `InMemorySaver` checkpointer. This means **conversation state persists across turns** without manual message management.

**`build_agent()` function** creates the graph:
- `prompt()`: Builds the user message. First turn: `user` role. Subsequent turns: `tool` role with the tool_call_id from the previous response.
- `llm()`: Calls OpenAI with tools and `tool_choice="required"`.
- `agent()`: The entrypoint. Takes state + previous messages from checkpointer, builds prompt, calls LLM, truncates to single tool call, returns final value + saves conversation.

Each game gets a unique `thread_id` (UUID5 from game_id) so checkpointer keeps separate conversations per game.

`LangGraphTextOnly` is identical but with `USE_IMAGE = False` — sends text grid data instead of images.

Frame formatting:
- Image mode: Renders grids to PNG via `g2im()` (a compact grid-to-image function with a 16-color palette).
- Text mode: Pretty-prints grids as numbered text.

Both use LangSmith `@traceable` decorators for observability.

---

## 6. LangGraph Random Agent (`langgraph_random_agent.py`)

**Class:** `LangGraphRandom`
**MAX_ACTIONS:** 80

Demonstrates LangGraph's StateGraph API. Builds a simple two-node graph:
- `START → choose_action → END`
- The `choose_action` node does the same random logic as the `Random` agent.

Useful as a template for building more complex LangGraph workflows.

---

## 7. LangGraph Thinking Agent (`agents/templates/langgraph_thinking/`)

**Class:** `LangGraphThinking`
**Model:** GPT-4.1 (via LangChain's `ChatOpenAI`) | **MAX_ACTIONS:** 20

The most complex agent — a full multi-node LangGraph workflow with tool use and persistent memory. This is a **modular, multi-file package**.

### Files:

- **`schema.py`**: Defines `AgentState` (TypedDict), `KeyCheck`, `Observation`, `LLM` enum.
- **`llm.py`**: Factory function `get_llm()` mapping enum to `ChatOpenAI` instance.
- **`vision.py`**: `render_frame()` creates annotated PNG images with grid lines, coordinate labels, and auto-detected highlights for Player, Door, Rotator, Key, Health, Lives.
- **`prompts.py`**: Prompt templates for system, frame delta analysis, key checking, and image/text message parts.
- **`tools.py`**: Four LangChain tools: `act`, `think`, `observe`, `delete_observation`.
- **`nodes.py`**: Graph node functions: `init`, `check_key`, `analyze_frame_delta`, `act`.
- **`agent.py`**: The `LangGraphThinking` class wiring everything together.

### Graph Flow:

```
START → init → (if RESET: END, else: check_key → analyze_frame_delta → act → END)
```

### Nodes in Detail:

1. **`init`**: If game state is `NOT_PLAYED` or `GAME_OVER`, returns RESET action (→ END). Otherwise passes through.

2. **`check_key`**: Uses structured output (`KeyCheck` schema) to ask the LLM to compare the key shape (bottom-left of grid) with the door shape. Returns `key_matches_door: bool`.

3. **`analyze_frame_delta`**: Compares pixel-by-pixel between current and previous frame. Categorizes changes (health changes, energy changes, movement). Sends both frame images + a text delta summary to the LLM for interpretation. Appends the analysis to `context`.

4. **`act`**: The main decision node. Builds a prompt with:
   - Current frame image
   - Previous action
   - Key match status
   - Accumulated context from analysis
   - System prompt with observations from memory store + thoughts

   Then enters a tool-calling loop (up to 5 iterations):
   - LLM can call `act(action)` → executes a GameAction (terminates loop)
   - LLM can call `think(thought)` → stores a thought for the current session
   - LLM can call `observe(observation)` → stores in SqliteStore (persists across games!)
   - LLM can call `delete_observation(id)` → removes from store

### Persistent Memory:

Uses `langgraph.store.sqlite.SqliteStore` backed by `memory.db`. Observations stored here survive across game sessions, so the agent accumulates knowledge about game mechanics over time. This is unique to this agent — all others start fresh.

### LockSmith-Specific Knowledge:

The prompts are heavily tailored to LockSmith: they describe the player sprite, walls, door, rotator, key, energy, lives, and the strategy of rotating the key to match the door then touching it.
