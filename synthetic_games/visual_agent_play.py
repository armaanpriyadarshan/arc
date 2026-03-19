"""Visually watch an agent play a synthetic game via pygame.

Combines visual_play.py's pygame rendering with play.py's agent loop.
The agent decides in a background thread; the main thread owns the game
and pygame (executes actions, renders frames).

Usage:
    # Random agent (fast, good for testing the visualizer)
    uv run python synthetic_games/visual_agent_play.py --game courier --agent random --seed 0

    # Reasoning agent (slower, makes LLM calls)
    uv run python synthetic_games/visual_agent_play.py --game wiring --agent reasoning_agent.py:ReasoningAgent

    # Any custom agent
    uv run python synthetic_games/visual_agent_play.py --game conductor --agent path/to/my_agent.py:MyAgent

    # Standalone experiment agent (owns its own game loop via arc_agi.Arcade)
    uv run python synthetic_games/visual_agent_play.py --game wiring --standalone --agent experiments/004-world-model-induction-v2/agents/agent.py:ToolUseAgent

Controls:
    ESC / Q = Quit

Requires: pip install pygame
"""

import argparse
import importlib.util
import os
import queue
import sys
import threading
import time
import traceback

import numpy as np

# ---------------------------------------------------------------------------
# Reuse from play.py: action helpers, agent loading, built-in agents
# ---------------------------------------------------------------------------

from play import (
    BUILTIN_AGENTS,
    _GAME_ACTIONS,
    _load_custom_agent,
    _make_action_input,
)

# ---------------------------------------------------------------------------
# ARC color palette (same as visual_play.py)
# ---------------------------------------------------------------------------

PALETTE = {
    0:  (0,   0,   0),
    1:  (70,  70,  70),
    2:  (0,   120, 255),
    3:  (50,  180, 50),
    4:  (200, 200, 0),
    5:  (30,  30,  30),
    6:  (0,   80,  200),
    7:  (230, 220, 50),
    8:  (220, 50,  50),
    9:  (150, 50,  200),
    10: (50,  200, 80),
    11: (240, 150, 30),
    12: (160, 160, 160),
    13: (0,   180, 180),
    14: (230, 100, 180),
    15: (240, 240, 240),
}

# ---------------------------------------------------------------------------
# Game discovery + loading (same as visual_play.py / play.py)
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
GAME_REGISTRY = {}


def _discover_games():
    for fname in sorted(os.listdir(_HERE)):
        if fname.startswith("game_") and fname.endswith(".py"):
            parts = fname[:-3].split("_", 2)
            short_name = parts[2] if len(parts) >= 3 else fname[:-3]

            module_path = os.path.join(_HERE, fname)
            spec = importlib.util.spec_from_file_location(fname[:-3], module_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            from arcengine import ARCBaseGame
            for attr_name in dir(mod):
                attr = getattr(mod, attr_name)
                if (isinstance(attr, type) and issubclass(attr, ARCBaseGame)
                        and attr is not ARCBaseGame):
                    GAME_REGISTRY[short_name] = (module_path, attr_name)
                    break


def _load_game(name, seed):
    if name not in GAME_REGISTRY:
        raise ValueError(f"Unknown game '{name}'. Available: {', '.join(sorted(GAME_REGISTRY))}")
    module_path, class_name = GAME_REGISTRY[name]
    spec = importlib.util.spec_from_file_location("game_mod", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)(seed=seed)


# ---------------------------------------------------------------------------
# Agent decision thread
#
# Only calls agent.choose_action() — never touches the game or pygame.
# Communicates with the main thread via two queues:
#   frame_in_q:  main -> agent  (FrameData for the agent to observe)
#   action_out_q: agent -> main  (chosen action, or sentinel on error/done)
# ---------------------------------------------------------------------------

_SENTINEL_DONE = object()
_SENTINEL_ERROR = object()


def _agent_decide_loop(agent, frame_in_q, action_out_q, stop_event):
    """Background thread: repeatedly read a frame, call agent, send action."""
    try:
        while not stop_event.is_set():
            # Wait for the next frame from the main thread
            try:
                frame = frame_in_q.get(timeout=0.1)
            except queue.Empty:
                continue

            if frame is _SENTINEL_DONE:
                return

            # Check if agent wants to stop
            if hasattr(agent, "is_done") and agent.is_done(frame):
                action_out_q.put(_SENTINEL_DONE)
                return

            # Ask agent for an action (this may block for seconds on LLM calls)
            choice = agent.choose_action(frame)
            if stop_event.is_set():
                return
            action_out_q.put(choice)

    except Exception:
        traceback.print_exc()
        action_out_q.put(_SENTINEL_ERROR)


# ---------------------------------------------------------------------------
# Standalone mode — experiment agents that own their game loop
# ---------------------------------------------------------------------------

class _SyntheticEnvAdapter:
    """Wraps a synthetic ARCBaseGame to match the ARC API's env.step() interface.

    The experiment agents call ``self.env.step(action, reasoning=...)``.
    Synthetic games use ``game.perform_action(ActionInput(id=action))``.
    This adapter bridges the two.
    """

    def __init__(self, game):
        self._game = game

    def step(self, action, reasoning=""):
        from arcengine import ActionInput
        action_input = ActionInput(id=action)
        return self._game.perform_action(action_input)


def _run_standalone(args, pygame):
    """Run an experiment agent that owns its own game loop.

    The agent manages its own arc_agi.Arcade / env internally.  We monkey-
    patch its ``_step`` method to publish every FrameData onto a queue so the
    main (pygame) thread can render it.

    If the game name matches a synthetic game, we create the synthetic game
    and inject it as an env adapter so the agent doesn't need the ARC API.
    """
    # Load .env so API keys (OPENAI_API_KEY, etc.) are available
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(_HERE, "..", ".env"))
    except ImportError:
        pass

    # Enable logging so agent reasoning traces are printed to the console
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    agent_cls = _load_custom_agent(args.agent)

    # If the game is a synthetic game, create it locally and inject via env=
    if args.game in GAME_REGISTRY:
        game = _load_game(args.game, args.seed)
        env = _SyntheticEnvAdapter(game)
        agent = agent_cls(args.game, env=env)
    else:
        agent = agent_cls(args.game)

    # Queue of (action_name: str, frame: FrameData) produced by the agent
    frame_q = queue.Queue()
    agent_done = threading.Event()
    agent_error: list[str] = []       # holds traceback if agent crashes
    delay = args.delay

    # --- Wrap _step to capture every frame ---
    original_step = agent._step

    def _step_with_render(action, reasoning=""):
        frame = original_step(action, reasoning=reasoning)
        action_name = action.name if hasattr(action, "name") else str(action)
        frame_q.put((action_name, frame))
        if delay > 0:
            time.sleep(delay)
        return frame

    agent._step = _step_with_render

    # --- Agent thread ---
    def _agent_thread():
        try:
            agent.run()
        except Exception:
            agent_error.append(traceback.format_exc())
            traceback.print_exc()
        finally:
            agent_done.set()

    t = threading.Thread(target=_agent_thread, daemon=True)
    t.start()

    # --- Init pygame ---
    scale = args.scale
    win_size = 64 * scale
    pygame.init()
    screen = pygame.display.set_mode((win_size, win_size))
    pygame.display.set_caption(f"[standalone] {args.agent} on {args.game}")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)

    grid_surface = pygame.Surface((64, 64))
    palette_array = np.zeros((16, 3), dtype=np.uint8)
    for idx, rgb in PALETTE.items():
        palette_array[idx] = rgb

    turn = 0
    last_action = ""
    last_state = ""
    last_levels = 0

    def render(frame_data, action_name):
        nonlocal last_state, last_levels
        frame = np.array(frame_data.frame)
        if frame.ndim == 3:
            frame = frame[-1]  # last grid in the sequence

        clipped = np.clip(frame, 0, 15).astype(np.uint8)
        rgb = palette_array[clipped]
        pygame.surfarray.blit_array(grid_surface, rgb.transpose(1, 0, 2))
        pygame.transform.scale(grid_surface, (win_size, win_size), screen)

        state = frame_data.state.name if hasattr(frame_data.state, "name") else str(frame_data.state)
        last_state = state
        last_levels = frame_data.levels_completed
        status = f"Turn {turn} | {action_name} | {state} | Levels: {last_levels}"
        text_surf = font.render(status, True, (255, 255, 255), (0, 0, 0))
        screen.blit(text_surf, (4, win_size - 18))
        pygame.display.flip()
        pygame.event.pump()

    # --- Main render loop ---
    running = True
    while running:
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False

        if not running:
            break

        # Drain all available frames, render the latest
        latest = None
        while True:
            try:
                action_name, frame_data = frame_q.get_nowait()
                turn += 1
                latest = (action_name, frame_data)
                last_action = action_name
                print(f"[standalone] turn {turn}: {action_name}")
            except queue.Empty:
                break

        if latest:
            render(latest[1], latest[0])

        # Check if agent is done
        if agent_done.is_set() and frame_q.empty():
            if agent_error:
                pygame.display.set_caption(
                    f"[standalone] ERROR after {turn} actions — {args.game} (ESC to quit)")
                print(f"[standalone] agent error after {turn} actions")
            else:
                pygame.display.set_caption(
                    f"[standalone] {last_state} after {turn} actions, levels={last_levels} — {args.game} (ESC to quit)")
                print(f"[standalone] agent finished — {last_state} after {turn} actions, levels={last_levels}")

            # Keep window open until user closes it
            while running:
                clock.tick(10)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key in (pygame.K_ESCAPE, pygame.K_q):
                            running = False

    pygame.quit()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _discover_games()

    parser = argparse.ArgumentParser(description="Visually watch an agent play a synthetic ARC game.")
    parser.add_argument("--list", action="store_true", help="List available games and agents")
    parser.add_argument("-g", "--game", type=str, default=None)
    parser.add_argument("-a", "--agent", type=str, default="random",
                        help="Built-in agent name or path/to/file.py:ClassName")
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-n", "--max-actions", type=int, default=500)
    parser.add_argument("--scale", type=int, default=8,
                        help="Pixel scale factor (default: 8 -> 512x512 window)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print grid diff around click point after each action")
    parser.add_argument("--standalone", action="store_true",
                        help="Standalone mode: the agent owns its game loop (for experiment agents)")
    parser.add_argument("--delay", type=float, default=0.15,
                        help="Delay in seconds between rendered frames in standalone mode (default: 0.15)")
    args = parser.parse_args()

    if args.list:
        print("Available games:")
        for name in sorted(GAME_REGISTRY):
            print(f"  {name}")
        print("\nBuilt-in agents:")
        for name, cls in sorted(BUILTIN_AGENTS.items()):
            doc = cls.__doc__ or ""
            print(f"  {name:15s} — {doc.strip().split(chr(10))[0]}")
        print("\nCustom agents:")
        print("  path/to/file.py:ClassName")
        return

    if args.game is None:
        parser.error("--game is required (use --list to see options)")

    try:
        import pygame
    except ImportError:
        print("pygame is required. Install it:")
        print("  pip install pygame")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Standalone mode: the experiment agent owns its own game loop.
    # We just wrap _step to capture frames and render them in the main thread.
    # -----------------------------------------------------------------------
    if args.standalone:
        _run_standalone(args, pygame)
        return

    from arcengine import ActionInput, GameAction

    # --- Init game (main thread owns the game) ---
    game = _load_game(args.game, args.seed)
    available_actions = game._available_actions if hasattr(game, "_available_actions") else [1, 2, 3, 4, 5]
    config = {"seed": args.seed, "max_actions": args.max_actions}

    # --- Init agent ---
    if args.agent in BUILTIN_AGENTS:
        agent_cls = BUILTIN_AGENTS[args.agent]
    else:
        agent_cls = _load_custom_agent(args.agent)
    agent = agent_cls(game, available_actions, config)

    # --- Init pygame ---
    scale = args.scale
    win_size = 64 * scale
    pygame.init()
    screen = pygame.display.set_mode((win_size, win_size))
    pygame.display.set_caption(f"Agent: {args.agent} | {args.game} (seed={args.seed})")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)

    # Pre-build a 64x64 surface; we draw pixels here then scale to the window.
    grid_surface = pygame.Surface((64, 64))

    # Build a numpy lookup: palette index -> (r, g, b)
    palette_array = np.zeros((16, 3), dtype=np.uint8)
    for idx, rgb in PALETTE.items():
        palette_array[idx] = rgb

    def render_frame(frame_data, turn_number):
        """Draw the 64x64 grid scaled up, plus an overlay status bar."""
        frame = np.array(frame_data.frame)
        if frame.ndim == 3:
            frame = frame[0]

        # Map palette indices -> RGB via numpy (shape 64x64x3, uint8)
        clipped = np.clip(frame, 0, 15).astype(np.uint8)
        rgb = palette_array[clipped]  # (64, 64, 3)

        # Blit the pixel data onto the 64x64 surface.
        # surfarray expects (width, height, 3) — so transpose axes 0 and 1.
        pygame.surfarray.blit_array(grid_surface, rgb.transpose(1, 0, 2))

        # Scale the 64x64 surface up to the window
        pygame.transform.scale(grid_surface, (win_size, win_size), screen)

        # Overlay status bar
        state = frame_data.state.name if hasattr(frame_data.state, "name") else str(frame_data.state)
        status = f"Turn {turn_number} | {state} | Levels: {frame_data.levels_completed}"
        text_surf = font.render(status, True, (255, 255, 255), (0, 0, 0))
        screen.blit(text_surf, (4, win_size - 18))

        pygame.display.flip()
        # Pump events so macOS processes the display update
        pygame.event.pump()

    def dump_grid_region(frame_data, cx, cy, radius=5, label=""):
        """Print a small text region of the grid around (cx, cy)."""
        frame = np.array(frame_data.frame)
        if frame.ndim == 3:
            frame = frame[0]
        y0 = max(0, cy - radius)
        y1 = min(64, cy + radius + 1)
        x0 = max(0, cx - radius)
        x1 = min(64, cx + radius + 1)
        if label:
            print(f"  [{label}] grid[{y0}:{y1}, {x0}:{x1}]  (click @ x={cx}, y={cy})")
        for y in range(y0, y1):
            row = []
            for x in range(x0, x1):
                val = int(frame[y][x])
                marker = f"[{val:X}]" if (x == cx and y == cy) else f" {val:X} "
                row.append(marker)
            print(f"    y={y:2d}: {''.join(row)}")

    def count_nonzero(frame_data):
        """Count non-zero cells in the grid."""
        frame = np.array(frame_data.frame)
        if frame.ndim == 3:
            frame = frame[0]
        return int(np.count_nonzero(frame))

    # --- Initial reset (main thread) ---
    frame = game.perform_action(ActionInput(id=GameAction.RESET))
    turn = 0
    render_frame(frame, turn)
    nz = count_nonzero(frame)
    print(f"[visual] initial frame rendered ({nz} non-zero cells)")

    # --- Start agent decision thread ---
    frame_in_q = queue.Queue()    # main -> agent: frames to observe
    action_out_q = queue.Queue()  # agent -> main: chosen actions
    stop_event = threading.Event()

    agent_thread = threading.Thread(
        target=_agent_decide_loop,
        args=(agent, frame_in_q, action_out_q, stop_event),
        daemon=True,
    )
    agent_thread.start()

    # Send the initial frame to the agent
    frame_in_q.put(frame)

    # --- Main loop: render + execute actions from agent ---
    running = True
    done = False

    while running:
        clock.tick(30)

        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False

        if not running:
            break

        # Check if agent has produced an action
        if not done:
            try:
                choice = action_out_q.get_nowait()
            except queue.Empty:
                continue

            # Handle sentinels
            if choice is _SENTINEL_DONE:
                done = True
                state = frame.state.name if hasattr(frame.state, "name") else str(frame.state)
                pygame.display.set_caption(
                    f"Agent done ({state}) after {turn} actions — {args.game} (ESC to quit)")
                print(f"[visual] agent done — {state} after {turn} actions")
                continue
            if choice is _SENTINEL_ERROR:
                done = True
                pygame.display.set_caption(
                    f"Agent ERROR after {turn} actions — {args.game} (ESC to quit)")
                print(f"[visual] agent thread error after {turn} actions")
                continue

            # Parse the agent's choice
            if isinstance(choice, tuple):
                action_id, click_x, click_y = choice
            else:
                action_id = choice
                click_x, click_y = 0, 0

            if action_id == -1:
                done = True
                pygame.display.set_caption(
                    f"Agent quit after {turn} actions — {args.game} (ESC to quit)")
                continue

            # Execute the action on the main thread
            try:
                prev_frame = frame
                action_input = _make_action_input(action_id, click_x, click_y)
                frame = game.perform_action(action_input)
                turn += 1
            except Exception as e:
                print(f"[visual] error executing action {choice}: {e}")
                traceback.print_exc()
                done = True
                continue

            # Render the new frame
            render_frame(frame, turn)
            nz = count_nonzero(frame)
            print(f"[visual] turn {turn} rendered (action={choice}, {nz} non-zero cells)")
            if args.verbose and action_id == 6:
                dump_grid_region(prev_frame, click_x, click_y, label="BEFORE")
                dump_grid_region(frame, click_x, click_y, label="AFTER")

            # Check game state
            state = frame.state.name if hasattr(frame.state, "name") else str(frame.state)

            if state == "WIN":
                done = True
                pygame.display.set_caption(
                    f"WIN after {turn} actions — {args.game} (ESC to quit)")
                print(f"[visual] WIN after {turn} actions")
            elif state == "GAME_OVER":
                # Auto-reset
                frame = game.perform_action(ActionInput(id=GameAction.RESET))
                turn += 1
                render_frame(frame, turn)
                print(f"[visual] GAME_OVER -> RESET at turn {turn}")
                pygame.display.set_caption(
                    f"Turn {turn} | RESET after GAME_OVER | {args.agent} on {args.game}")
                # Send the reset frame to the agent
                frame_in_q.put(frame)
            elif turn >= args.max_actions:
                done = True
                pygame.display.set_caption(
                    f"Max actions ({args.max_actions}) reached — {args.game} (ESC to quit)")
                print(f"[visual] max actions reached")
            else:
                pygame.display.set_caption(
                    f"Turn {turn} | {state} | {args.agent} on {args.game}")
                # Send the new frame to the agent for its next decision
                frame_in_q.put(frame)

    # Cleanup
    stop_event.set()
    frame_in_q.put(_SENTINEL_DONE)  # unblock agent if waiting on queue
    pygame.quit()


if __name__ == "__main__":
    main()
