"""Benchmark: v4 vs v5 agent on game_11_maze (30 steps, real API calls).

Measures wall-clock time to quantify the impact of v5's 5 optimizations.
Both agents use the same synthetic game and real GPT-5.4 API calls.

Usage:
    cd experiments/004-world-model-induction-v5
    uv run python benchmark.py
"""

import importlib.util
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Load env for API keys
_here = Path(__file__).resolve().parent
load_dotenv(_here / ".env")
load_dotenv(_here / "../../.env")
load_dotenv(_here / "../../ARC-AGI-3-Agents/.env")

from arcengine import ActionInput, GameAction, GameState

# ---------------------------------------------------------------------------
# Load MazeGame from synthetic_games/
# ---------------------------------------------------------------------------

SYNTHETIC_DIR = _here / "../../synthetic_games"


def _load_maze_game():
    """Import and instantiate MazeGame from synthetic_games/game_11_maze.py."""
    spec = importlib.util.spec_from_file_location(
        "game_11_maze", SYNTHETIC_DIR / "game_11_maze.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MazeGame(seed=0)


# ---------------------------------------------------------------------------
# LocalGameEnv: wraps a synthetic game for the ToolUseAgent
# ---------------------------------------------------------------------------

class LocalGameEnv:
    """Adapts game.perform_action(ActionInput) to the env.step(action) interface
    that ToolUseAgent._step expects."""

    def __init__(self, game):
        self.game = game

    def step(self, action: GameAction, reasoning: str = "", **kwargs):
        """Match the interface: env.step(action, reasoning=...) -> raw frame."""
        if action.is_complex() and hasattr(action, 'action_data') and action.action_data:
            data = {
                "x": getattr(action.action_data, 'x', 0),
                "y": getattr(action.action_data, 'y', 0),
            }
            ai = ActionInput(id=action, data=data)
        else:
            ai = ActionInput(id=action)
        return self.game.perform_action(ai)


# ---------------------------------------------------------------------------
# Load v4 agent via importlib
# ---------------------------------------------------------------------------

def _load_v4_agent_class():
    """Import ToolUseAgent from the v4 experiment directory."""
    v4_dir = _here / "../004-world-model-induction-v4"

    # Load symbolic module first (dependency)
    sym_spec = importlib.util.spec_from_file_location(
        "v4_agents.symbolic", v4_dir / "agents/symbolic.py")
    sym_mod = importlib.util.module_from_spec(sym_spec)
    sys.modules["v4_agents.symbolic"] = sym_mod
    sym_spec.loader.exec_module(sym_mod)

    # Load vision module
    vis_spec = importlib.util.spec_from_file_location(
        "v4_agents.vision", v4_dir / "agents/vision.py")
    vis_mod = importlib.util.module_from_spec(vis_spec)
    sys.modules["v4_agents.vision"] = vis_mod
    vis_spec.loader.exec_module(vis_mod)

    # Load agent module with patched imports
    agent_spec = importlib.util.spec_from_file_location(
        "v4_agents.agent", v4_dir / "agents/agent.py",
        submodule_search_locations=[])
    agent_mod = importlib.util.module_from_spec(agent_spec)

    # Patch the relative imports the agent module expects
    import types
    v4_pkg = types.ModuleType("v4_agents")
    v4_pkg.__path__ = [str(v4_dir / "agents")]
    sys.modules["v4_agents"] = v4_pkg
    v4_pkg.symbolic = sym_mod
    v4_pkg.vision = vis_mod

    # Temporarily redirect .symbolic and .vision to our loaded modules
    agent_mod.__package__ = "v4_agents"
    sys.modules["v4_agents.agent"] = agent_mod
    agent_spec.loader.exec_module(agent_mod)

    return agent_mod.ToolUseAgent


# ---------------------------------------------------------------------------
# Load v5 agent directly
# ---------------------------------------------------------------------------

def _load_v5_agent_class():
    """Import v5 ToolUseAgent from this experiment's agents/."""
    from agents.agent import ToolUseAgent
    return ToolUseAgent


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_agent(agent_cls, label: str, max_actions: int = 10):
    """Run an agent on MazeGame, return timing data."""
    game = _load_maze_game()
    env = LocalGameEnv(game)

    agent = agent_cls(game_id="game_11_maze", env=env)
    agent.MAX_ACTIONS = max_actions

    # Instrument _think_and_act for per-call timing
    original_think = agent._think_and_act
    call_times = []

    def timed_think(frame):
        t0 = time.perf_counter()
        result = original_think(frame)
        call_times.append(time.perf_counter() - t0)
        return result

    agent._think_and_act = timed_think

    print(f"\n{'='*60}")
    print(f"Running {label} ({max_actions} max actions)")
    print(f"{'='*60}")

    t_start = time.perf_counter()
    agent.run()
    t_total = time.perf_counter() - t_start

    return {
        "label": label,
        "total_time": t_total,
        "actions": agent.action_counter,
        "llm_calls": agent.llm_calls,
        "call_times": call_times,
        "avg_per_action": t_total / max(agent.action_counter, 1),
        "avg_per_llm_call": (sum(call_times) / len(call_times)) if call_times else 0,
        "model_calls": getattr(agent, 'model_calls', {}),
    }


def main():
    MAX_ACTIONS = 30

    # Silence third-party noise
    for name in ("openai", "httpx", "httpcore", "arc_agi", "urllib3", "arcengine"):
        logging.getLogger(name).setLevel(logging.WARNING)

    # Suppress agent logs during benchmark (avoid I/O overhead skewing results)
    agent_logger = logging.getLogger("agents")
    agent_logger.setLevel(logging.WARNING)

    print("=" * 60)
    print("BENCHMARK: v4 vs v5 on game_11_maze")
    print(f"Max actions: {MAX_ACTIONS}")
    print(f"Real GPT-5.4 API calls")
    print("=" * 60)

    # Run v4
    v4_cls = _load_v4_agent_class()
    v4_results = run_agent(v4_cls, "v4 (baseline)", MAX_ACTIONS)

    # Run v5
    v5_cls = _load_v5_agent_class()
    v5_results = run_agent(v5_cls, "v5 (optimized)", MAX_ACTIONS)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'Metric':<30} {'v4':>12} {'v5':>12} {'Change':>12}")
    print("-" * 66)

    for key, label, fmt in [
        ("total_time", "Total time (s)", ".2f"),
        ("actions", "Actions taken", "d"),
        ("llm_calls", "LLM calls", "d"),
        ("avg_per_action", "Avg time/action (s)", ".2f"),
        ("avg_per_llm_call", "Avg time/LLM call (s)", ".2f"),
    ]:
        v4_val = v4_results[key]
        v5_val = v5_results[key]
        if isinstance(v4_val, float) and v4_val > 0:
            change = f"{((v5_val - v4_val) / v4_val) * 100:+.1f}%"
        elif isinstance(v4_val, int) and v4_val > 0:
            change = f"{((v5_val - v4_val) / v4_val) * 100:+.1f}%"
        else:
            change = "N/A"
        print(f"{label:<30} {v4_val:>{12}{fmt}} {v5_val:>{12}{fmt}} {change:>12}")

    # Per-LLM-call breakdown
    if v4_results["call_times"] and v5_results["call_times"]:
        print(f"\n{'LLM Call Timing Breakdown':}")
        print(f"  v4 calls: {[f'{t:.2f}s' for t in v4_results['call_times']]}")
        print(f"  v5 calls: {[f'{t:.2f}s' for t in v5_results['call_times']]}")

    # Model routing breakdown
    v5_models = v5_results.get("model_calls", {})
    if v5_models:
        print(f"\nv5 Model Routing:")
        for model, count in sorted(v5_models.items()):
            print(f"  {model}: {count} calls")

    # Time saved
    saved = v4_results["total_time"] - v5_results["total_time"]
    pct = (saved / v4_results["total_time"]) * 100 if v4_results["total_time"] > 0 else 0
    print(f"\nTime saved: {saved:.2f}s ({pct:.1f}%)")
    print(f"LLM calls saved: {v4_results['llm_calls'] - v5_results['llm_calls']}")


if __name__ == "__main__":
    main()
