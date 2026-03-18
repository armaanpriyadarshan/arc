"""Visually play a synthetic game in a window using pygame.

Usage:
    uv run python synthetic_games/visual_play.py --game courier
    uv run python synthetic_games/visual_play.py --game wiring --seed 7
    uv run python synthetic_games/visual_play.py --list

Controls:
    W / Up Arrow    = ACTION1 (move up)
    S / Down Arrow  = ACTION2 (move down)
    A / Left Arrow  = ACTION3 (move left)
    D / Right Arrow = ACTION4 (move right)
    Space           = ACTION5 (interact / toggle / pause)
    Left Click      = ACTION6 (click at grid position)
    Z               = ACTION7 (undo)
    R               = Restart current level
    Shift+R         = Full reset (back to level 0)
    ESC / Q         = Quit

Requires: pip install pygame
"""

import argparse
import importlib.util
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# ARC color palette (indices 0-15) -> RGB
# Based on the standard ARC-AGI palette
# ---------------------------------------------------------------------------
PALETTE = {
    0:  (0,   0,   0),      # black / background
    1:  (70,  70,  70),     # dark gray / wall
    2:  (0,   120, 255),    # bright blue / player
    3:  (50,  180, 50),     # green
    4:  (200, 200, 0),      # yellow
    5:  (30,  30,  30),     # near-black / HUD bg
    6:  (0,   80,  200),    # blue
    7:  (230, 220, 50),     # yellow
    8:  (220, 50,  50),     # red
    9:  (150, 50,  200),    # purple
    10: (50,  200, 80),     # green
    11: (240, 150, 30),     # orange
    12: (160, 160, 160),    # gray
    13: (0,   180, 180),    # teal
    14: (230, 100, 180),    # pink
    15: (240, 240, 240),    # white / HUD border
}

# ---------------------------------------------------------------------------
# Game discovery (same as play.py)
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
# Main
# ---------------------------------------------------------------------------

def main():
    _discover_games()

    parser = argparse.ArgumentParser(description="Visually play a synthetic ARC game.")
    parser.add_argument("--list", action="store_true", help="List available games")
    parser.add_argument("-g", "--game", type=str, default=None)
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("--scale", type=int, default=8,
                        help="Pixel scale factor (default: 8 -> 512x512 window)")
    args = parser.parse_args()

    if args.list:
        print("Available games:")
        for name in sorted(GAME_REGISTRY):
            print(f"  {name}")
        return

    if args.game is None:
        parser.error("--game is required (use --list to see options)")

    try:
        import pygame
    except ImportError:
        print("pygame is required for visual play. Install it:")
        print("  pip install pygame")
        print("  # or: uv add pygame")
        sys.exit(1)

    from arcengine import ActionInput, GameAction

    # Init game
    game = _load_game(args.game, args.seed)
    available = game._available_actions if hasattr(game, '_available_actions') else [1,2,3,4,5]

    # Init pygame
    scale = args.scale
    win_size = 64 * scale
    pygame.init()
    screen = pygame.display.set_mode((win_size, win_size))
    pygame.display.set_caption(f"ARC Synthetic: {args.game} (seed={args.seed})")
    clock = pygame.time.Clock()

    # Key mappings (R and Shift+R handled separately for level/full reset)
    key_to_action = {
        pygame.K_w: GameAction.ACTION1,
        pygame.K_UP: GameAction.ACTION1,
        pygame.K_s: GameAction.ACTION2,
        pygame.K_DOWN: GameAction.ACTION2,
        pygame.K_a: GameAction.ACTION3,
        pygame.K_LEFT: GameAction.ACTION3,
        pygame.K_d: GameAction.ACTION4,
        pygame.K_RIGHT: GameAction.ACTION4,
        pygame.K_SPACE: GameAction.ACTION5,
        pygame.K_z: GameAction.ACTION7,
    }

    def render_frame(frame_data):
        frame = np.array(frame_data.frame)
        if frame.ndim == 3:
            frame = frame[0]

        for y in range(64):
            for x in range(64):
                color = PALETTE.get(int(frame[y][x]), (255, 0, 255))
                pygame.draw.rect(screen, color,
                                 (x * scale, y * scale, scale, scale))

        # Status text
        state = frame_data.state.name if hasattr(frame_data.state, 'name') else str(frame_data.state)
        font = pygame.font.SysFont("monospace", 14)
        status = f"{state} | Levels: {frame_data.levels_completed} | Actions: {', '.join(str(a) for a in available)}"
        text_surf = font.render(status, True, (255, 255, 255), (0, 0, 0))
        screen.blit(text_surf, (4, win_size - 18))

        pygame.display.flip()

    # Initial frame
    frame = game.perform_action(ActionInput(id=GameAction.RESET))
    render_frame(frame)

    running = True
    while running:
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False

                elif event.key == pygame.K_r:
                    mods = pygame.key.get_mods()
                    if mods & pygame.KMOD_SHIFT:
                        # Shift+R: full reset (back to level 0)
                        frame = game.perform_action(ActionInput(id=GameAction.RESET))
                    else:
                        # R: restart current level
                        game.level_reset()
                        frame = game.perform_action(ActionInput(id=GameAction.RESET))
                    pygame.display.set_caption(f"ARC Synthetic: {args.game} (seed={args.seed})")
                    render_frame(frame)

                elif event.key in key_to_action:
                    # Block gameplay actions if game is over or won
                    cur_state = frame.state.name if hasattr(frame.state, 'name') else str(frame.state)
                    if cur_state == "WIN":
                        pygame.display.set_caption(
                            f"YOU WIN! — {args.game} — All levels complete! (R to play again)")
                        continue
                    if cur_state == "GAME_OVER":
                        pygame.display.set_caption(
                            f"GAME OVER — {args.game} (R to restart level, Shift+R for level 0)")
                        continue

                    ga = key_to_action[event.key]
                    action_id = ga.value if hasattr(ga, 'value') else ga
                    if action_id in available:
                        frame = game.perform_action(ActionInput(id=ga))
                        render_frame(frame)

                        new_state = frame.state.name if hasattr(frame.state, 'name') else str(frame.state)
                        if new_state == "WIN":
                            pygame.display.set_caption(
                                f"YOU WIN! — {args.game} — All levels complete! (R to play again)")
                        elif new_state == "GAME_OVER":
                            pygame.display.set_caption(
                                f"GAME OVER — {args.game} (R to restart level, Shift+R for level 0)")

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Block clicks if game is over or won
                cur_state = frame.state.name if hasattr(frame.state, 'name') else str(frame.state)
                if cur_state in ("WIN", "GAME_OVER"):
                    continue
                # Left click -> ACTION6
                if 6 in available:
                    mx, my = event.pos
                    gx = mx // scale
                    gy = my // scale
                    frame = game.perform_action(
                        ActionInput(id=GameAction.ACTION6, data={"x": gx, "y": gy})
                    )
                    render_frame(frame)

    pygame.quit()


if __name__ == "__main__":
    main()
