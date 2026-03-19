"""Play a synthetic game with an agent.

Usage:
    # List available games and agents
    python play.py --list

    # Run a specific game with a built-in agent
    python play.py --game courier --agent random --seed 42
    python play.py --game courier --agent random --seed 42 --max-actions 500
    python play.py --game alchemist --agent interactive

    # Run ALL games and get a summary
    python play.py --game all --agent random --seed 42

    # Run with a custom agent class from a file
    python play.py --game courier --agent path/to/my_agent.py:MyAgentClass

    # Verbose mode shows progress every 50 actions
    python play.py --game courier --agent random --seed 42 -v

Agent interface:
    Custom agents must implement two methods:

        class MyAgent:
            def choose_action(self, frame_data):
                '''Return action int (1-7) or tuple (6, x, y) for clicks.
                   frame_data is arcengine.FrameData.'''
                ...

            def is_done(self, frame_data) -> bool:
                '''Return True to stop playing. Optional — defaults to False.'''
                ...

    The constructor receives (game, available_actions, config):
        def __init__(self, game, available_actions, config):
            ...

    Where `game` is the ARCBaseGame instance, `available_actions` is a list of
    int action IDs, and `config` is a dict with seed, max_actions, etc.
"""

import argparse
import importlib.util
import logging
import os
import random
import sys
import time
import types
from typing import Optional, Type

import numpy as np
from arcengine import ActionInput, GameAction, FrameData

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Game registry — auto-discover game_*.py files
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))

# Map short names -> (module_name, class_name)
GAME_REGISTRY: dict[str, tuple[str, str]] = {}


def _discover_games():
    """Scan this directory for game_*.py files and register them."""
    for fname in sorted(os.listdir(_HERE)):
        if fname.startswith("game_") and fname.endswith(".py"):
            # e.g. game_01_courier.py -> short name "courier"
            parts = fname[:-3].split("_", 2)  # ['game', '01', 'courier']
            if len(parts) >= 3:
                short_name = parts[2]
            else:
                short_name = fname[:-3]

            # Find the game class in the module
            module_path = os.path.join(_HERE, fname)
            spec = importlib.util.spec_from_file_location(fname[:-3], module_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            # Look for a class that inherits from ARCBaseGame
            from arcengine import ARCBaseGame
            for attr_name in dir(mod):
                attr = getattr(mod, attr_name)
                if (isinstance(attr, type) and issubclass(attr, ARCBaseGame)
                        and attr is not ARCBaseGame):
                    GAME_REGISTRY[short_name] = (module_path, attr_name)
                    break


def _load_game(name: str, seed: int):
    """Instantiate a game by short name."""
    if name not in GAME_REGISTRY:
        raise ValueError(
            f"Unknown game '{name}'. Available: {', '.join(sorted(GAME_REGISTRY))}"
        )
    module_path, class_name = GAME_REGISTRY[name]
    spec = importlib.util.spec_from_file_location("game_mod", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cls = getattr(mod, class_name)
    return cls(seed=seed)


# ---------------------------------------------------------------------------
# Built-in agents
# ---------------------------------------------------------------------------

class RandomAgent:
    """Picks a random valid action each step. Returns (action_id, x, y) for clicks."""

    def __init__(self, game, available_actions, config):
        self._actions = available_actions
        self._rng = random.Random(config.get("seed", 0))

    def choose_action(self, frame_data: FrameData):
        action_id = self._rng.choice(self._actions)
        if action_id == 6:
            x = self._rng.randint(0, 63)
            y = self._rng.randint(0, 63)
            return (action_id, x, y)
        return action_id

    def is_done(self, frame_data: FrameData) -> bool:
        return False


class InteractiveAgent:
    """Human-in-the-loop: prints the grid and prompts for action input."""

    # Color palette for terminal display (ANSI 256-color approximations)
    COLOR_CHARS = {
        0: " ",   # empty/bg
        1: "#",   # wall
        2: "@",   # player
        3: "&",   # special object
        4: "D",   # door/border
        5: "=",   # HUD bg
        6: "b",   # blue
        7: "y",   # yellow
        8: "r",   # red
        9: "p",   # purple
        10: "g",  # green
        11: "o",  # orange
        12: ".",  # dimmed/gray
        13: "~",  # teal/beam
        14: "k",  # pink
        15: "+",  # HUD border
    }

    def __init__(self, game, available_actions, config):
        self._actions = available_actions
        self._game = game

    def _render_grid(self, frame_data: FrameData):
        """Print a compact ASCII representation of the 64x64 grid."""
        frame = np.array(frame_data.frame)
        if frame.ndim == 3:
            frame = frame[0]

        # Downsample 2x for readability (64->32 cols)
        print("\n" + "=" * 34)
        for r in range(0, 64, 2):
            row_chars = []
            for c in range(0, 64, 2):
                # Pick the most "interesting" (non-zero) pixel in the 2x2 block
                block = [frame[r][c], frame[r][c+1], frame[r+1][c], frame[r+1][c+1]]
                val = max(block, key=lambda v: (v != 0, v))
                row_chars.append(self.COLOR_CHARS.get(val, "?"))
            print("|" + "".join(row_chars) + "|")
        print("=" * 34)

    def _print_action_map(self):
        action_names = {
            1: "W/Up", 2: "S/Down", 3: "A/Left", 4: "D/Right",
            5: "Space/Interact", 6: "Click(x,y)", 7: "Undo",
        }
        available = [f"  {a} = {action_names.get(a, f'ACTION{a}')}"
                     for a in sorted(self._actions)]
        print("Actions:\n" + "\n".join(available))

    def choose_action(self, frame_data: FrameData):
        self._render_grid(frame_data)
        state = frame_data.state.name if hasattr(frame_data.state, 'name') else str(frame_data.state)
        print(f"State: {state} | Levels completed: {frame_data.levels_completed}")
        self._print_action_map()

        while True:
            try:
                raw = input("Action (number, or '6 x y' for click, or 'q'): ").strip().lower()
                if raw in ("q", "quit", "exit"):
                    return -1  # signal to quit
                parts = raw.split()
                action_id = int(parts[0])
                if action_id not in self._actions:
                    print(f"Invalid. Choose from: {sorted(self._actions)}")
                    continue
                if action_id == 6:
                    if len(parts) == 3:
                        x, y = int(parts[1]), int(parts[2])
                        return (6, x, y)
                    print("Click requires coordinates: '6 x y' (0-63)")
                    continue
                return action_id
            except (ValueError, EOFError):
                print(f"Enter a number from: {sorted(self._actions)}")

    def is_done(self, frame_data: FrameData) -> bool:
        return False


BUILTIN_AGENTS: dict[str, Type] = {
    "random": RandomAgent,
    "interactive": InteractiveAgent,
}


# ---------------------------------------------------------------------------
# Custom agent loader
# ---------------------------------------------------------------------------

def _load_custom_agent(spec_str: str) -> Type:
    """Load agent class from 'path/to/file.py:ClassName' format.

    Supports agents that use relative imports (e.g. ``from .symbolic import …``)
    by registering the containing directory as a package before loading.
    """
    if ":" not in spec_str:
        raise ValueError(
            f"Custom agent spec must be 'path/to/file.py:ClassName', got '{spec_str}'"
        )
    path, class_name = spec_str.rsplit(":", 1)
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Agent file not found: {path}")

    pkg_dir = os.path.dirname(path)
    pkg_name = os.path.basename(pkg_dir)
    mod_name = os.path.splitext(os.path.basename(path))[0]
    fq_name = f"{pkg_name}.{mod_name}"           # e.g. "agents.agent"

    # Ensure the package's parent is on sys.path so sibling imports resolve.
    parent_dir = os.path.dirname(pkg_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    # Register the containing directory as a package if not already imported.
    if pkg_name not in sys.modules:
        init_path = os.path.join(pkg_dir, "__init__.py")
        if os.path.isfile(init_path):
            pkg_spec = importlib.util.spec_from_file_location(
                pkg_name, init_path,
                submodule_search_locations=[pkg_dir],
            )
            pkg_mod = importlib.util.module_from_spec(pkg_spec)
            sys.modules[pkg_name] = pkg_mod
            pkg_spec.loader.exec_module(pkg_mod)
        else:
            # No __init__.py — create a namespace package.
            pkg_mod = types.ModuleType(pkg_name)
            pkg_mod.__path__ = [pkg_dir]
            sys.modules[pkg_name] = pkg_mod

    mod_spec = importlib.util.spec_from_file_location(fq_name, path,
                                                       submodule_search_locations=[])
    mod = importlib.util.module_from_spec(mod_spec)
    mod.__package__ = pkg_name
    sys.modules[fq_name] = mod
    mod_spec.loader.exec_module(mod)

    if not hasattr(mod, class_name):
        raise AttributeError(f"Class '{class_name}' not found in {path}")
    return getattr(mod, class_name)


# ---------------------------------------------------------------------------
# Game action helpers
# ---------------------------------------------------------------------------

_GAME_ACTIONS = {
    0: GameAction.RESET,
    1: GameAction.ACTION1,
    2: GameAction.ACTION2,
    3: GameAction.ACTION3,
    4: GameAction.ACTION4,
    5: GameAction.ACTION5,
    6: GameAction.ACTION6,
    7: GameAction.ACTION7,
}


def _make_action_input(action_id: int, click_x: int = 0, click_y: int = 0) -> ActionInput:
    """Convert an integer action ID to an ActionInput."""
    ga = _GAME_ACTIONS.get(action_id)
    if ga is None:
        raise ValueError(f"Unknown action ID: {action_id}")
    if action_id == 6:
        return ActionInput(id=ga, data={"x": click_x, "y": click_y})
    return ActionInput(id=ga)


# ---------------------------------------------------------------------------
# Main play loop
# ---------------------------------------------------------------------------

def play(game_name: str, agent_spec: str, seed: int = 0,
         max_actions: int = 500, verbose: bool = False):
    """Run a synthetic game with an agent."""

    game = _load_game(game_name, seed)
    config = {"seed": seed, "max_actions": max_actions}

    # Resolve agent
    if agent_spec in BUILTIN_AGENTS:
        agent_cls = BUILTIN_AGENTS[agent_spec]
    else:
        agent_cls = _load_custom_agent(agent_spec)

    # Get available actions from the game
    available_actions = game._available_actions if hasattr(game, '_available_actions') else [1, 2, 3, 4, 5]
    agent = agent_cls(game, available_actions, config)

    # Initial reset
    frame = game.perform_action(ActionInput(id=GameAction.RESET))
    action_count = 0
    start_time = time.time()
    levels_at_start = frame.levels_completed

    print(f"Playing '{game_name}' with agent '{agent_spec}' (seed={seed}, max_actions={max_actions})")
    print(f"Available actions: {available_actions}")
    print(f"Grid: 64x64, Colors: 0-15")
    print("-" * 50)

    while action_count < max_actions:
        # Check game state
        state_name = frame.state.name if hasattr(frame.state, 'name') else str(frame.state)

        if state_name == "WIN":
            elapsed = time.time() - start_time
            print(f"\nWIN after {action_count} actions ({elapsed:.1f}s)")
            print(f"Levels completed: {frame.levels_completed}")
            break

        if state_name == "GAME_OVER":
            print(f"\nGAME OVER at action {action_count} (levels: {frame.levels_completed})")
            # Reset and continue
            frame = game.perform_action(ActionInput(id=GameAction.RESET))
            action_count += 1
            continue

        # Check if agent wants to stop
        if hasattr(agent, 'is_done') and agent.is_done(frame):
            print(f"\nAgent stopped after {action_count} actions")
            break

        # Agent chooses action — can return int or (int, x, y) for clicks
        choice = agent.choose_action(frame)
        if isinstance(choice, tuple):
            action_id, click_x, click_y = choice
        else:
            action_id = choice
            click_x, click_y = 0, 0

        if action_id == -1:  # quit signal from interactive agent
            print("\nQuitting.")
            break

        # Execute action
        action_input = _make_action_input(action_id, click_x, click_y)
        frame = game.perform_action(action_input)
        action_count += 1

        # Progress reporting
        new_levels = frame.levels_completed
        if new_levels > levels_at_start:
            print(f"  Level completed! ({new_levels} total) at action {action_count}")
            levels_at_start = new_levels

        if verbose and action_count % 50 == 0:
            print(f"  Action {action_count}: state={state_name}, levels={frame.levels_completed}")

    else:
        elapsed = time.time() - start_time
        print(f"\nMax actions ({max_actions}) reached ({elapsed:.1f}s)")
        print(f"Final state: {frame.state.name}, levels: {frame.levels_completed}")

    return frame


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    _discover_games()

    parser = argparse.ArgumentParser(
        description="Play a synthetic ARC game with an agent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python play.py --list
  python play.py --game courier --agent random
  python play.py --game courier --agent interactive
  python play.py --game all --agent random --seed 42
  python play.py --game alchemist --agent random --seed 7 --max-actions 1000
  python play.py --game courier --agent ../experiments/my_agent.py:MyAgent
""",
    )
    parser.add_argument("--list", action="store_true",
                        help="List available games and built-in agents")
    parser.add_argument("-g", "--game", type=str, default=None,
                        help="Game short name (e.g. courier, wiring, alchemist)")
    parser.add_argument("-a", "--agent", type=str, default="random",
                        help="Built-in agent name or path/to/file.py:ClassName")
    parser.add_argument("-s", "--seed", type=int, default=0,
                        help="Random seed for game generation (default: 0)")
    parser.add_argument("-n", "--max-actions", type=int, default=500,
                        help="Maximum actions before stopping (default: 500)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print progress every 50 actions")

    args = parser.parse_args()

    if args.list:
        print("Available games:")
        for name in sorted(GAME_REGISTRY):
            print(f"  {name}")
        print("\nBuilt-in agents:")
        for name, cls in sorted(BUILTIN_AGENTS.items()):
            print(f"  {name:15s} — {cls.__doc__.strip().split(chr(10))[0]}")
        print("\nCustom agents:")
        print("  path/to/file.py:ClassName")
        return

    if args.game is None:
        parser.error("--game is required (use --list to see available games, or --game all)")

    if args.game == "all":
        # Run agent on every game and print a summary
        results = {}
        for name in sorted(GAME_REGISTRY):
            print(f"\n{'='*50}")
            frame = play(name, args.agent, seed=args.seed,
                         max_actions=args.max_actions, verbose=args.verbose)
            results[name] = (frame.state.name, frame.levels_completed)

        print(f"\n{'='*50}")
        print("SUMMARY")
        print(f"{'='*50}")
        for name, (state, levels) in results.items():
            print(f"  {name:15s}  state={state:15s}  levels={levels}")
    else:
        play(args.game, args.agent, seed=args.seed,
             max_actions=args.max_actions, verbose=args.verbose)


if __name__ == "__main__":
    main()
