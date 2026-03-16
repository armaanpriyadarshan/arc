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
├── 002-fixed-perception/
│   ├── ...              # copies and modifies what it needs from 001
│   └── analysis.md      # references experiment 001
```

### Workflow

1. **Build** the agent code inside `experiments/NNN-description/`.
2. **Run** it from inside that directory: `uv run python run.py --agent=explorer --game=ls20`
3. **Analyze** the logs and outcomes.
4. **Write up** what happened in `analysis.md`.
5. **Reference** prior experiment numbers in the next experiment's analysis.

## Design Principles

- **Perception is code, reasoning is LLM.** Frame diffing and object detection are deterministic Python. LLM calls only for interpretation and synthesis.
- **Action budget is sacred.** Every action must either gather information or make progress.
- **World model is test-driven.** The test suite grows monotonically. Fixing one mechanic must not break another.
- **LLM calls are expensive.** Use plan-then-execute (LLM outputs action sequences), not LLM-per-action.
- **Upstream is read-only.** `ARC-AGI-3-Agents/` is a reference only.

## Environment Setup

API keys in `.env`:
- `ARC_API_KEY` — required, from https://three.arcprize.org/
- `OPENAI_API_KEY` — required for LLM-based stages

Local game environments in `environment_files/`.

## Reference Documentation

See `docs/` for upstream framework walkthroughs:
- `docs/01-overview.md` through `docs/06-testing-and-development.md`
