# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an ARC-AGI-3 competition workspace for building agents that play ARC-AGI-3 games. It contains two Python projects:

- **Root project** (`/`) — Personal experimentation space using the `arc-agi` SDK directly (Python 3.14, uv-managed)
- **ARC-AGI-3-Agents** (`/ARC-AGI-3-Agents/`) — The official ARC-AGI-3 agent framework (Python 3.12+, uv-managed), cloned from `arcprize/ARC-AGI-3-Agents`

## Commands

### Root project
```bash
uv run main.py          # run main entry point
uv run my-play.py       # run playground script
```

### ARC-AGI-3-Agents
```bash
cd ARC-AGI-3-Agents

# Run an agent against a game
uv run main.py --agent=random --game=ls20
uv run main.py --agent=llm --game=ls20 --tags=experiment,v1

# Tests
uv run pytest                              # all tests
uv run pytest tests/unit/test_core.py      # single file
uv run pytest -k test_name                 # single test by name
uv run pytest -m unit                      # only unit tests
uv run pytest -m integration               # only integration tests

# Linting & formatting (ruff)
uv run ruff check .
uv run ruff format .

# Type checking
uv run mypy .
```

## Architecture (ARC-AGI-3-Agents)

### Agent system

`Agent` (abstract base in `agents/agent.py`) defines the game loop: repeatedly call `choose_action()` then `take_action()` until `is_done()` returns true or `MAX_ACTIONS` (80) is reached. Subclasses must implement `choose_action()` and `is_done()`.

`Swarm` (`agents/swarm.py`) orchestrates multiple agents across games — creates an agent per game, runs each in its own thread, manages scorecards via `Arcade` from the `arc-agi` SDK.

`Playback` (in `agent.py`) replays recorded JSONL sessions.

### Agent templates (`agents/templates/`)

- `random_agent.py` — random actions
- `llm_agents.py` — LLM-based (LLM, FastLLM, GuidedLLM, ReasoningLLM) using OpenAI
- `reasoning_agent.py` — chain-of-thought reasoning agent
- `multimodal.py` — multimodal (vision) LLM agent
- `langgraph_*.py` — LangGraph-based agents
- `smolagents.py` — HuggingFace smolagents integration

### Key types (from `arc-agi` / `arcengine` packages)

- `GameAction` — enum of actions (ACTION1–ACTION7, RESET)
- `FrameData` — per-frame state (frame grid, levels_completed, win_levels, state, available_actions)
- `GameState` — game lifecycle state
- `Arcade` / `EnvironmentWrapper` — game environment management

### Recordings

Gameplay is recorded as JSONL files in `recordings/`. Agents auto-register recording files as playback agents.

## Environment Setup

Both projects require API keys in `.env`:
- `ARC_API_KEY` — required, from https://three.arcprize.org/
- `AGENTOPS_API_KEY` — optional, for observability
- `ONLINE_ONLY=True` — to force online API instead of local execution

Local game environments are stored in `environment_files/`.

## Detailed Documentation

See `docs/` for comprehensive walkthroughs:
- `docs/01-overview.md` — Project overview, how to run agents, all CLI options
- `docs/02-architecture.md` — Execution flow, Swarm, Agent base class, Recorder, Playback, Tracing
- `docs/03-agent-templates.md` — Every agent template in detail (Random, LLM family, ReasoningAgent, MultiModalLLM, LangGraph agents)
- `docs/04-smolagents.md` — HuggingFace smolagents integration (SmolCodingAgent, SmolVisionAgent)
- `docs/05-game-concepts.md` — Frame structure, GameState lifecycle, color palette, LockSmith rules
- `docs/06-testing-and-development.md` — Test structure, creating new agents, experiment workflow
