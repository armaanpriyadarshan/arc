# ARC-AGI-3 Agents — Complete Walkthrough

## What is ARC-AGI-3?

ARC-AGI-3 is a competition where you build **agents** that play unknown, dynamic, turn-based games on a 64×64 grid. Each cell holds an integer 0–15 (representing colors). Your agent takes actions (move up/down/left/right, click coordinates, etc.) and receives back a new frame. The goal is to **WIN** while minimizing the number of actions.

The known game so far is **LockSmith** (`ls20`), a puzzle where you navigate a maze, rotate a key to match an exit door, and walk through that door. There are 6 levels with an energy system.

## Two Projects in This Repo

| Directory | Purpose | Python |
|-----------|---------|--------|
| `/` (root) | Your personal experimentation space using `arc-agi` SDK directly | 3.14 |
| `/ARC-AGI-3-Agents/` | Official agent framework (cloned from `arcprize/ARC-AGI-3-Agents`) | 3.12+ |

The root project has minimal code (`main.py`, `my-play.py`). All the real agent infrastructure lives in `ARC-AGI-3-Agents/`.

## How to Run an Agent

```bash
cd ARC-AGI-3-Agents

# Make sure .env has your ARC_API_KEY (and optionally OPENAI_API_KEY for LLM agents)
cp .env.example .env
# Edit .env with your keys

# Run any agent against any game
uv run main.py --agent=random --game=ls20
uv run main.py --agent=llm --game=ls20
uv run main.py --agent=reasoningagent --game=ls20
uv run main.py --agent=multimodalllm --game=ls20
uv run main.py --agent=langgraphfunc --game=ls20

# Tags for scorecard organization
uv run main.py --agent=random --game=ls20 --tags=experiment,baseline

# Run against multiple games (comma-separated prefix filter)
uv run main.py --agent=random --game=ls20,locksmith
```

### Available Agent Names

Agent names are auto-registered from class names (lowercased). Current built-ins:

| CLI name | Class | Description |
|----------|-------|-------------|
| `random` | `Random` | Random actions — good baseline |
| `llm` | `LLM` | GPT-4o-mini with text observations |
| `fastllm` | `FastLLM` | GPT-4o-mini, skips observation step |
| `guidedllm` | `GuidedLLM` | o3 with explicit LockSmith rules |
| `reasoningllm` | `ReasoningLLM` | o4-mini with reasoning metadata |
| `reasoningagent` | `ReasoningAgent` | o4-mini with vision + hypothesis tracking |
| `multimodalllm` | `MultiModalLLM` | Vision-based with self-programming memory |
| `langgraphrandom` | `LangGraphRandom` | Random agent as LangGraph workflow |
| `langgraphfunc` | `LangGraphFunc` | o4-mini via LangGraph functional API |
| `langgraphtextonly` | `LangGraphTextOnly` | Same as above, text only (no images) |
| `langgraphthinking` | `LangGraphThinking` | LangGraph with tool use, memory, vision |
| `smolcodingagent` | `SmolCodingAgent` | HuggingFace smolagents CodeAgent |
| `smolvisionagent` | `SmolVisionAgent` | HuggingFace smolagents with vision |

Additionally, any `.recording.jsonl` file in `recordings/` is auto-registered as a playback agent you can replay.

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `ARC_API_KEY` | Yes | API key from https://three.arcprize.org/ |
| `OPENAI_API_KEY` | For LLM agents | OpenAI API key |
| `OPENAI_SECRET_KEY` | For MultiModalLLM | Alternative OpenAI key name used by that agent |
| `AGENTOPS_API_KEY` | No | AgentOps observability |
| `ONLINE_ONLY` | No | Set `True` to force online API (vs. local execution) |
| `RECORDINGS_DIR` | No | Override where JSONL recordings are saved |
| `DEBUG` | No | Set `True` for debug-level logging |
| `SCHEME`, `HOST`, `PORT` | No | Override API connection (defaults: `http`, `localhost`, `8001`) |
