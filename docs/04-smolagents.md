# Smolagents Integration (`smolagents.py`)

Two HuggingFace `smolagents` wrappers that take a fundamentally different approach: instead of the agent base class's `choose_action` loop, they hand control entirely to the smolagents framework.

## SmolCodingAgent

**Model:** `o4-mini` | **MAX_ACTIONS:** 100

Uses `smolagents.CodeAgent` — the LLM writes **Python code** to call tools, enabling complex reasoning (BFS, DFS, algorithms).

**Key difference:** Overrides `main()` entirely:
1. Resets the game manually.
2. Builds a prompt with initial game state.
3. Calls `agent.run(prompt, max_steps=MAX_ACTIONS)` — smolagents takes over completely.
4. `is_done()` and `choose_action()` are defined but only used internally.

**Tool creation:** `create_smolagents_tool()` dynamically converts each `GameAction` into a smolagents `@tool`:
- Simple actions: No parameters, returns text description of game state.
- Complex actions: Takes `x, y` as integers, validates 0–63 range.

Each tool call executes `_execute_action()` which calls `self.take_action()` and returns the function response prompt (state, score, action count, frame grid).

Uses `planning_interval=10` — the model re-plans every 10 steps.

## SmolVisionAgent

**Model:** `o4-mini` | **MAX_ACTIONS:** 100

Uses `smolagents.ToolCallingAgent` instead of `CodeAgent`. The key difference: tools return `AgentImage` (PIL Images) instead of text, so the LLM sees the game visually.

Has its own `grid_to_image()` method that renders the grid with a 16-color palette, stacking multiple grid layers horizontally with separators.

Both agents use the `LLM` class's `build_functions()` for tool metadata but create new smolagents-compatible tools.
