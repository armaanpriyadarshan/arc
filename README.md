# ARC-AGI-3 Agent Development

Building autonomous agents that discover and play unknown turn-based games on 64x64 grids with 16 colors. Part of the [ARC-AGI-3](https://three.arcprize.org/) competition.

## The Challenge

The agent receives a 64x64 grid of integers (0-15) and can take actions (directional movement, clicks, interact). It must:
- **Discover** what each action does through experimentation
- **Understand** game mechanics (switches, gates, keys, patterns) from observation alone
- **Plan** multi-step strategies to complete levels and increase score
- **Generalize** across completely different games without hardcoded knowledge

No game rules are provided. The agent must figure everything out from scratch.

## Repository Structure

```
arc/
├── experiments/          # Self-contained experiment folders
│   ├── 001-011           # Numbered experiments with code, logs, analysis
│   └── ...
├── ARC-AGI-3-Agents/     # Official upstream framework (read-only reference)
├── environment_files/    # Local game environments (ls20, ft09, vc33)
├── synthetic_games/      # Custom games for testing
├── docs/                 # Framework documentation
└── CLAUDE.md             # Project guidelines
```

Each experiment is a complete, self-contained folder with its own `run.py`, `pyproject.toml`, agent code, logs, and analysis writeup.

## Key Techniques

### Perception Pipeline
- **Auto-probe**: On startup, take one of each action and record what changes. Establishes action-to-effect mappings without game knowledge.
- **Symbolic state extraction**: Connected component analysis on the grid produces objects with color, shape, size, position, orientation, symmetry, and spatial relations.
- **Symbolic diffing**: Frame-to-frame comparison reports which objects moved, appeared, disappeared, or changed properties.
- **Side-by-side diff images**: Previous vs current frame with red outlines on changed cells.

### Reasoning Architectures
- **Reactive loop** (004-v2, 008): One LLM call per action, break on BLOCKED. Simple but effective for navigation.
- **DAG reasoning** (007): Structured chain of nodes (reflect → observe → strategize → verify → act) forces systematic thinking.
- **Batch planning with queue** (010, 011): Analyzer outputs 3-5 actions at once, queue drains them with zero LLM calls. Score changes flush the queue.
- **Tool calling** (004-v5, 011): Model can read/grep its own game log to review past turns, enabling long-term memory beyond the context window.

### Hypothesis Management
- **Competing hypotheses**: Model maintains 2-3 alternative explanations with predictions for what would confirm/reject each.
- **Bayesian probabilities** (004-v3+): Each hypothesis carries a probability updated by evidence.
- **Interaction journal**: Every significant interaction recorded as fact, informing future plans.
- **Verified rules**: Universal game mechanics confirmed across levels, persisted with deduplication.

### Spatial Reasoning
- **ASCII grid rendering**: Text-based grid representation the model can read spatially (unlike images which VLMs struggle with).
- **State-action memory**: Remembers which actions from which grid states caused changes vs no-ops, preventing re-testing.
- **Grid diff tracking**: Shows exact cell-level changes between frames.
- **Mandatory code analysis**: Model writes Python code to analyze the grid computationally every turn.

### Learning & Memory
- **Cross-level rule persistence**: Verified game rules carry across level transitions.
- **Memory bank**: Persistent storage for reflections and insights across levels.
- **Conversational sliding window**: Model maintains conversation history for context continuity.
- **Run log with Read/Grep tools**: Model can query its own history via tool calls.

## Models Used

- **GPT-5.4** (OpenAI Responses API): Primary model for most experiments. Supports vision, reasoning effort levels, and structured outputs.
- **Claude Opus 4.6** (via Claude Code CLI): Used in experiment 011. Built-in Read/Grep/Bash tools for analyzing game logs.
- **o3** (OpenAI): Used by the upstream GuidedLLM agent with hardcoded game rules.

## How to Run

### Prerequisites
- Python 3.12+ with [uv](https://docs.astral.sh/uv/)
- API keys in `.env`: `ARC_API_KEY` (from [arcprize.org](https://three.arcprize.org/)), `OPENAI_API_KEY`
- For experiment 011: [Claude Code CLI](https://claude.ai/claude-code) installed

### Running an experiment

```bash
cd experiments/008-lightweight    # or any experiment folder
uv run python run.py --game=ls20
```

### Running the upstream framework agents

```bash
cd ARC-AGI-3-Agents
uv run main.py --agent=guidedllm --game=ls20
```

### Running the RGB-Agent (requires Docker)

```bash
cd RGB-Agent
cd docker/opencode-sandbox && bash build.sh && cd ../..
uv run rgb-swarm --game ls20 --model openai/gpt-5.2 --max-actions 200
```

## Synthetic Games

12 custom games in `synthetic_games/` for controlled testing. Same 64x64 grid, 16 colors, and ACTION1-7 interface as real ARC-AGI-3 games.

| # | Game | Mechanics |
|---|------|-----------|
| 01 | Courier | Navigation, colored package delivery, locked doors, fuel management |
| 02 | Wiring | Click-to-place wires, signal propagation, color mixing |
| 03 | Alchemist | Ingredient collection, hidden recipe discovery, crafting |
| 04 | Mirror Maze | Rotate mirrors to reflect colored beams to matching receptors |
| 05 | Terraformer | Place terrain with chain reactions (lava+water=stone, grass spreads) |
| 06 | Conductor | Toggle track switches to route trains to colored stations |
| 07 | Architect | Place tetrominoes with gravity/rotation to fill a target silhouette |
| 08 | Cipher | Deduce hidden transformation rules from input-output examples |
| 09 | Ecosystem | Place species, simulate interactions to hit population targets |
| 10 | Shapeshifter | Shift between forms with different abilities to solve gated puzzles |
| 11 | Maze | Pure navigation (baseline) |
| 12 | Maze2 | Navigation with dead-ends (baseline) |

```bash
# Watch an agent play visually
uv run python synthetic_games/visual_agent_play.py --game courier --agent random --seed 0

# Headless run across all games
uv run python synthetic_games/play.py --game all --agent random --seed 42
```

## Key Findings

1. **Text-based spatial representations outperform visual ones.** ASCII grid rendering and symbolic state give the model more reliable spatial understanding than raw images.

2. **Auto-probe eliminates the bootstrapping problem.** Systematically testing each action on startup and recording structured diffs gives the model immediate ground truth about the action space.

3. **Adaptive reasoning effort is critical.** Higher reasoning effort produces better strategic decisions; lower effort keeps latency manageable. Using high effort selectively (e.g. first few calls, after major state changes) balances quality and speed.

4. **Tool calling enables long-term memory.** Letting the model read/grep its own game log is more effective than sliding window context or persistent notes for maintaining coherent strategy across many turns.

5. **State-action memory prevents redundant exploration.** Recording which actions caused changes from which grid states eliminates repetitive behavior and focuses the agent on novel interactions.

6. **Batch planning with queue draining reduces API cost.** Having the analyzer output multi-step plans that execute without additional LLM calls significantly reduces the number of API calls per game.

## References

- [ARC-AGI-3 Competition](https://three.arcprize.org/)
- [RGB-Agent](https://github.com/alexisfox7/RGB-Agent) — Read-Grep-Bash agent architecture
- [ARC-AGI-3 Claude Code SDK](https://github.com/ThariqS/ARC-AGI-3-ClaudeCode-SDK)
