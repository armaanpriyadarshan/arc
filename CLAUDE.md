# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ARC-AGI-3 competition workspace. Building agents that play unknown turn-based games on 64×64 grids.

- **Root project** (`/`) — Workspace with `arc-agi` SDK (Python 3.14, uv-managed)
- **ARC-AGI-3-Agents** (`/ARC-AGI-3-Agents/`) — Official upstream framework (read-only reference, don't modify)
- **Experiments** (`/experiments/NNN-description/`) — Each experiment is self-contained: code, config, logs, analysis

### Key types (from `arc-agi` / `arcengine`)

- `GameAction` — enum: RESET, ACTION1–ACTION7. Simple (no params) or complex (x, y coords 0–63).
- `FrameData` — frame grid (`list[list[list[int]]]`, 64×64 of ints 0–15), state, levels_completed, available_actions.
- `GameState` — NOT_PLAYED, NOT_FINISHED, GAME_OVER, WIN.

### API

- Actions map to keyboard/mouse inputs: up, down, left, right, spacebar, click(x,y), undo
- `available_actions` from FrameData tells which actions are valid for the current game
- Reasoning is passed to `env.step(action, reasoning=...)` and shows in the ARC replay

### Models

- GPT-5.4: Use Responses API (`client.responses.create`). Image format: `{"type": "input_image", "image_url": "data:image/png;base64,..."}`. Text: `{"type": "input_text", "text": "..."}`. Output: `response.output_text`.
- GPT-5.4-mini: Same API, faster/cheaper, no vision.
- o4-mini: Chat Completions API. Uses `max_completion_tokens` not `max_tokens`. Supports `response_format: json_object`.
- gpt-4o: Chat Completions API. Uses `max_tokens`. Good vision.

## Experiment Philosophy

**Build fast, run, analyze after.** An experiment is a complete agent run against a game. The goal is to get data as quickly as possible, then think carefully about what it means.

Each experiment produces a **write-up** (`analysis.md`) that becomes a permanent reference. Future experiments cite past write-ups by number to build on prior findings. The write-ups are the real artifact — code changes between experiments, but the analysis accumulates.

**What makes a new experiment:** a change in agent logic, prompts, stage budgets, perception approach, or game target. Not a bug fix or refactor.

### Structure

Each experiment is **one self-contained folder** with everything in it — code, config, logs, analysis. No subdirectories, no shared code outside the folder.

```
experiments/
├── 001-staged-explorer-v1/
│   ├── pyproject.toml   # dependencies
│   ├── run.py           # entry point
│   ├── config.py        # params, scorecard ID, result summary
│   ├── analysis.md      # post-run write-up
│   ├── experiment.log   # raw output
│   └── agents/          # agent code for this experiment
├── 002-redesigned-vlm/
│   └── analysis.md      # references experiment 001
```

### Workflow

1. **Build** the agent code inside `experiments/NNN-description/`.
2. **Run** it from inside that directory: `uv run python run.py --agent=explorer --game=ls20`
3. **Analyze** the logs and outcomes.
4. **Write up** what happened in `analysis.md`.
5. **Reference** prior experiment numbers in the next experiment's analysis.

## Current Best Architecture (Experiment 004)

The most effective approach so far: **observe-then-act with auto-probe, symbolic state, and hypothesis-driven action sequences.**

### Components

- **Auto-probe** (4 actions, 0 LLM calls): Take one of each action on startup. Code records structured symbolic diffs. Results are fed as ESTABLISHED FACTS to every LLM call — the model never has to guess what actions do.
- **Symbolic state** (`symbolic.py`): Connected component analysis on the grid. Outputs objects with color, shape, size, position, bounding box, center. Plus spatial relations (above, below, adjacent).
- **Symbolic diff** (`diff_symbolic`): Compares two symbolic states. Reports which objects moved, appeared, disappeared, or changed size — no interpretation, just facts.
- **Side-by-side image** (`vision.py`): Previous frame vs current with red outlines on changed cells. Sent to the model every turn.
- **GPT-5.4** (Responses API): One call per turn. Sees: probe facts + symbolic state + symbolic diff + image + its own notes. Outputs: observation, hypotheses (with test_actions), notes.
- **Hypothesis-driven actions**: The model outputs structured hypotheses, each with a `test_actions` sequence. The first `testing` hypothesis drives the actions. Execution stops on BLOCKED or unexpected events.
- **Detailed unusual-event reporting**: When changes > 80 (beyond normal movement), the full symbolic diff is included in action results so the model sees exactly what happened.

### What works

- Probe facts eliminate the "what do actions do" problem that killed experiments 001-003
- Symbolic state + diff gives the model precise spatial data without pixel-squinting
- Hypothesis-driven action sequences reduce LLM calls (batch actions per hypothesis test)
- GPT-5.4 produces consistently good spatial reasoning from symbolic data + images

### What doesn't work yet

- The model navigates well but doesn't investigate game objects deeply enough
- Hypotheses tend to be about paths ("go up to reach X") rather than mechanics ("interacting with X does Y")
- Energy/timer depletion is noticed but not acted on with urgency
- Inconsistent across runs — initial interpretation can vary

## Game-Agnostic Principle

**The competition evaluates on unknown games. The agent must discover everything through observation. NEVER introduce game-specific knowledge into prompts, code, comments, variable names, or experiment configs.**

Any assumption that happens to be true for one game will break on another. The agent's entire value comes from its ability to generalize. If you find yourself writing something that only makes sense for a specific game, stop — you're leaking knowledge the agent should discover on its own.

### Violations to avoid

- **Color semantics.** Don't assume what colors represent. No "color 1 = wall", "the green cells are safe", "red means danger". Colors are just integers 0–15 with no inherent meaning across games.
- **Avatar / player assumptions.** Don't assume there is a player-controlled avatar, what it looks like, how big it is, or which object it is. Some games may have no avatar at all — the player might click cells or control the whole grid.
- **Entity labels.** Don't label objects as "enemies", "items", "doors", "keys", "coins", "health pickups" in prompts or code. Use neutral terms like "object at (x,y)" or "color-3 region".
- **Mechanic assumptions.** Don't assume the game has health, energy, score, lives, timers, gravity, collision, or any specific mechanic. Don't tell the model "avoid enemies" or "collect items".
- **Goal assumptions.** Don't assume the win condition. Don't say "reach the exit" or "clear all enemies". The agent must figure out what winning means.
- **Spatial assumptions.** Don't assume top-down vs side-scrolling, grid-based vs free movement, or that actions map to specific directions consistently across games.

### What to do instead

- Let the LLM **form and test its own hypotheses** about game semantics.
- Use **neutral, observation-based language** in prompts: "observe changes", "describe what you see", "what happened after that action".

## Design Principles

- **Game-agnostic — see dedicated section above.** No game-specific knowledge anywhere.
- **Don't hardcode position tracking or entity detection.** The model decides what info it needs. Only raw diffs and symbolic state are computed by code.
- **Action budget is sacred.** Every action must either gather information or make progress.
- **LLM calls are expensive.** Use hypothesis-driven action sequences, not LLM-per-action.
- **Upstream is read-only.** `ARC-AGI-3-Agents/` is a reference only.

## Environment Setup

API keys in `.env`:
- `ARC_API_KEY` — required, from https://three.arcprize.org/
- `OPENAI_API_KEY` — required for LLM-based stages

Local game environments in `environment_files/`.

## Reference Documentation

See `docs/` for upstream framework walkthroughs:
- `docs/01-overview.md` through `docs/06-testing-and-development.md`
