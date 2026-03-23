"""Run experiment 010: RGB-Agent against synthetic games (Codex CLI variant).

Uses the RGB-Agent framework with OpenAI Codex CLI as the analyzer.
Codex runs natively (no Docker required) with OS-level sandboxing.

Usage:
    # Headless
    uv run python run_codex.py --game wiring
    uv run python run_codex.py --game wiring --model gpt-5.4

    # With pygame visualization
    uv run python run_codex.py --game wiring --visual
    uv run python run_codex.py --game conductor --visual --delay 0.2

    # List available games
    uv run python run_codex.py --list

Prerequisites:
    - Codex CLI installed: npm install -g @openai/codex
    - API key in .env: OPENAI_API_KEY
"""

import argparse
import logging
import os
import queue
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths and env
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent

# Add RGB-Agent to sys.path so rgb_agent package is importable
sys.path.insert(0, str(_HERE / "RGB-Agent"))

import numpy as np
from dotenv import load_dotenv

load_dotenv(_HERE / ".env")
load_dotenv(_PROJECT_ROOT / ".env")
load_dotenv(_PROJECT_ROOT / "ARC-AGI-3-Agents" / ".env")

# ---------------------------------------------------------------------------
# Game ID mapping (short name -> full game_id in environment_files/)
# ---------------------------------------------------------------------------

GAME_MAP = {
    "wiring":    "wiri-00000001",
    "alchemist": "alch-00000001",
    "conductor": "cond-00000001",
}

# ---------------------------------------------------------------------------
# Visual env wrapper — intercepts step/reset to capture frames for pygame
# ---------------------------------------------------------------------------

class VisualEnvWrapper:
    """Wraps ArcAgi3Env to capture frames for visualization."""

    def __init__(self, env, frame_queue):
        self._env = env
        self._queue = frame_queue

    def reset(self, **kwargs):
        result = self._env.reset(**kwargs)
        obs = result[0] if isinstance(result, tuple) else result
        self._queue.put(("RESET", obs))
        return result

    def step(self, action_payload):
        result = self._env.step(action_payload)
        obs = result[0] if isinstance(result, tuple) else result
        action_name = "?"
        if isinstance(action_payload, dict):
            action = action_payload.get("action")
            if hasattr(action, "name"):
                action_name = action.name
            else:
                action_name = str(action)
        self._queue.put((action_name, obs))
        return result

    def __getattr__(self, name):
        return getattr(self._env, name)


# ---------------------------------------------------------------------------
# Pygame visualization
# ---------------------------------------------------------------------------

PALETTE = {
    0:  (255, 255, 255),  1:  (204, 204, 204),
    2:  (153, 153, 153),  3:  (102, 102, 102),
    4:  (51,  51,  51),   5:  (0,   0,   0),
    6:  (229, 58,  163),  7:  (255, 123, 204),
    8:  (249, 60,  49),   9:  (30,  147, 255),
    10: (136, 216, 241),  11: (255, 220, 0),
    12: (255, 133, 27),   13: (146, 18,  49),
    14: (79,  204, 48),   15: (163, 86,  214),
}


def _run_visual(runner_func, frame_queue, args):
    """Main-thread pygame render loop. runner_func runs in a background thread."""
    import pygame

    agent_done = threading.Event()
    agent_error = []

    def _runner_thread():
        try:
            runner_func()
        except Exception:
            agent_error.append(traceback.format_exc())
            traceback.print_exc()
        finally:
            agent_done.set()

    t = threading.Thread(target=_runner_thread, daemon=True)
    t.start()

    scale = args.scale
    win_size = 64 * scale
    pygame.init()
    screen = pygame.display.set_mode((win_size, win_size))
    pygame.display.set_caption(f"[010-codex] {args.model} on {args.game}")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)

    grid_surface = pygame.Surface((64, 64))
    palette_array = np.zeros((16, 3), dtype=np.uint8)
    for idx, rgb in PALETTE.items():
        palette_array[idx] = rgb

    turn = 0
    last_state = ""
    last_score = 0

    def render(obs_dict, action_name):
        nonlocal last_state, last_score
        frame = obs_dict.get("frame", [])
        if not frame:
            return
        grid = np.array(frame)
        if grid.ndim == 3:
            grid = grid[-1]

        clipped = np.clip(grid, 0, 15).astype(np.uint8)
        rgb = palette_array[clipped]
        pygame.surfarray.blit_array(grid_surface, rgb.transpose(1, 0, 2))
        pygame.transform.scale(grid_surface, (win_size, win_size), screen)

        last_state = obs_dict.get("state", "?")
        last_score = obs_dict.get("score", 0)
        status = f"Turn {turn} | {action_name} | {last_state} | Score: {last_score}"
        text_surf = font.render(status, True, (255, 255, 255), (0, 0, 0))
        screen.blit(text_surf, (4, win_size - 18))
        pygame.display.flip()
        pygame.event.pump()

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

        # Drain frames, render latest
        latest = None
        while True:
            try:
                action_name, obs = frame_queue.get_nowait()
                turn += 1
                latest = (action_name, obs)
                if args.delay > 0:
                    time.sleep(args.delay)
            except queue.Empty:
                break

        if latest:
            render(latest[1], latest[0])

        if agent_done.is_set() and frame_queue.empty():
            if agent_error:
                pygame.display.set_caption(f"[010-codex] ERROR after {turn} actions (ESC to quit)")
            else:
                pygame.display.set_caption(
                    f"[010-codex] {last_state} score={last_score} after {turn} actions (ESC to quit)")
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
    import arc_agi
    from arc_agi import OperationMode
    from rgb_agent.environment import ArcAgi3Env
    from rgb_agent.environment.runner import GameRunner
    from rgb_agent.agent import CodexAgent

    parser = argparse.ArgumentParser(description="Experiment 010: Codex CLI agent on synthetic games")
    parser.add_argument("-g", "--game", default=None,
                        help=f"Game to play: {', '.join(sorted(GAME_MAP))}. Or a full game_id.")
    parser.add_argument("-m", "--model", default="gpt-5.4",
                        help="Analyzer model (default: gpt-5.4)")
    parser.add_argument("-n", "--max-actions", type=int, default=500)
    parser.add_argument("--interval", type=int, default=10,
                        help="Actions per analyzer batch plan (default: 10)")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--visual", action="store_true", help="Enable pygame visualization")
    parser.add_argument("--delay", type=float, default=0.05,
                        help="Delay between frames in visual mode (default: 0.05s)")
    parser.add_argument("--scale", type=int, default=8, help="Pixel scale (default: 8 -> 512x512)")
    parser.add_argument("--list", action="store_true", help="List available games and exit")
    args = parser.parse_args()

    if args.list:
        print("Available synthetic games:")
        for name, gid in sorted(GAME_MAP.items()):
            print(f"  {name:15s} -> {gid}")
        return

    if args.game is None:
        parser.error("--game is required (use --list to see options)")

    # Resolve game name -> full game_id
    game_id = GAME_MAP.get(args.game, args.game)

    # Logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logging.getLogger("arc_agi").propagate = False

    logs_dir = _HERE / "logs"
    logs_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = logs_dir / f"{args.game}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    game_log_dir = run_dir / game_id.split("-")[0]
    game_log_dir.mkdir(parents=True, exist_ok=True)
    prompts_log_path = game_log_dir / "logs.txt"
    prompts_log_path.write_text("")

    fh = logging.FileHandler(run_dir / "run.log", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logging.getLogger().addHandler(fh)

    logging.info("Game: %s (%s)", args.game, game_id)
    logging.info("Model: %s", args.model)
    logging.info("Max actions: %d", args.max_actions)
    logging.info("Logs: %s", run_dir)

    # Silence noisy loggers
    for name in ("openai", "httpx", "httpcore", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)

    # NORMAL mode: local environment_files/ for synthetic games + ARC API for official games
    arcade = arc_agi.Arcade(
        operation_mode=OperationMode.NORMAL,
        environments_dir=str(_HERE / "environment_files"),
    )

    card_id = arcade.open_scorecard(tags=[f"010-{args.game}"])
    logging.info("Scorecard: %s", card_id)

    # Create env
    env = ArcAgi3Env.from_arcade(
        arcade=arcade,
        game_id=game_id,
        scorecard_id=card_id,
        max_actions=args.max_actions,
    )

    # Wrap for visualization if requested
    frame_queue = None
    if args.visual:
        frame_queue = queue.Queue()
        env = VisualEnvWrapper(env, frame_queue)

    # Create analyzer (Codex CLI — no Docker required)
    agent = CodexAgent(
        model=args.model,
        plan_size=args.interval,
    )
    logging.info("Analyzer: CodexAgent (model=%s, plan_size=%d)", args.model, args.interval)

    # Create runner (the real RGB-Agent game loop)
    inner_agent_kwargs = {"name": "rgb_agent", "plan_size": args.interval}

    runner = GameRunner(
        env=env,
        game_id=game_id,
        agent_name="rgb_agent",
        max_actions_per_game=args.max_actions,
        tags=[f"010-{args.game}"],
        prompts_log_path=prompts_log_path,
        analyzer=agent.analyze,
        log_post_board=True,
        analyzer_retries=args.retries,
        agent_kwargs=inner_agent_kwargs,
    )

    def run_game():
        try:
            metrics = runner.run()
            logging.info("Final: status=%s score=%d actions=%d",
                         metrics.status.name, metrics.final_score, metrics.run_total_actions)
        finally:
            try:
                env.close()
            except Exception:
                pass
            try:
                scorecard = arcade.close_scorecard(card_id)
                if scorecard:
                    logging.info("Scorecard closed: score=%.1f", scorecard.score)
            except Exception:
                pass

    if args.visual:
        try:
            _run_visual(run_game, frame_queue, args)
        except ImportError:
            print("pygame required for --visual: pip install pygame")
            sys.exit(1)
    else:
        run_game()


if __name__ == "__main__":
    main()
