"""
Game 05: Terraformer

Mechanics:
  1. Grid movement (WASD) on a 62x60 tile world rendered onto a 64x64 grid
  2. Terraform tool cycles through 4 terrain types (sand, water, grass, stone);
     ACTION5 places current tool terrain on the player's tile then advances
  3. Environmental chain reactions (run every 2 steps):
     - Lava + adjacent water -> both become stone
     - Lava + adjacent grass -> grass becomes fire
     - Fire spreads to adjacent grass, then burns out to floor after 3 ticks
     - Sand adjacent to water slowly grows into grass (after ~8 simulation ticks)
     Water does NOT spread — it stays where placed. Stone blocks nothing
     special but is impassable.
  4. Target zones: marked cells that must contain a specific terrain type
  5. Goal flag: step on it once all targets are met to advance
  6. Limited terraform charges on later levels (levels 0-1 are unlimited)

How to win:
  Place terrain on every target cell to match the required type, then walk
  onto the goal flag. Chain reactions can help or hinder — e.g. place water
  next to sand to grow grass, or place lava next to water to create stone.

Actions:
  ACTION1 = Move Up    (W)
  ACTION2 = Move Down  (S)
  ACTION3 = Move Left  (A)
  ACTION4 = Move Right (D)
  ACTION5 = Place current tool terrain on player's tile + cycle tool

Color key (palette indices used for rendering):
  0  = white        (empty/void)
  1  = off-white     (floor / bare earth)
  3  = neutral gray  (stone)
  5  = black         (wall, UI background)
  7  = pink/yellow   (fire)
  8  = red/orange    (lava)
  9  = blue          (water)
  10 = light blue    (player)
  11 = yellow        (sand)
  12 = orange        (goal flag)
  14 = green         (grass)
  15 = purple        (unmet target border indicator)

Display:
  Row 63: current tool color swatch + all tool options with selection marker
  Row 62: charges remaining bar (green=available, red=used) — hidden if unlimited
  Row 61: level progress dots
"""

import random
from typing import List, Tuple

import numpy as np

from arcengine import (
    ARCBaseGame,
    Camera,
    Level,
    Sprite,
)

# ── Terrain type constants ────────────────────────────────────────
EMPTY = 0
SAND = 1
WATER = 2
GRASS = 3
STONE = 4
WALL = 5
LAVA = 6
FIRE = 7
FLOOR = 8

# ── Palette color mapping ─────────────────────────────────────────
C_EMPTY = 0
C_SAND = 11
C_WATER = 9
C_GRASS = 14
C_STONE = 3
C_WALL = 5
C_LAVA = 8
C_FIRE = 7
C_FLOOR = 1
C_PLAYER = 10
C_GOAL = 12
C_TARGET_MARKER = 15
C_UI_BG = 5

TERRAIN_TO_COLOR = {
    EMPTY: C_EMPTY, SAND: C_SAND, WATER: C_WATER, GRASS: C_GRASS,
    STONE: C_STONE, WALL: C_WALL, LAVA: C_LAVA, FIRE: C_FIRE, FLOOR: C_FLOOR,
}

# Tool ordering: player cycles through these with ACTION5
# Lava is NOT player-placeable — it only appears as a level hazard.
TOOL_TERRAINS = [SAND, WATER, GRASS, STONE]
TOOL_COLORS = [C_SAND, C_WATER, C_GRASS, C_STONE]

GRID_W = 62
GRID_H = 60
DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]


# ── Level generation ──────────────────────────────────────────────
def _generate_level(seed: int, level_idx: int) -> dict:
    rng = random.Random(seed + level_idx * 1000)
    w, h = GRID_W, GRID_H
    terrain = [[FLOOR for _ in range(w)] for _ in range(h)]

    # Outer walls
    for x in range(w):
        terrain[0][x] = WALL
        terrain[h - 1][x] = WALL
    for y in range(h):
        terrain[y][0] = WALL
        terrain[y][w - 1] = WALL

    # Interior wall segments
    for _ in range(3 + level_idx * 2):
        wx, wy = rng.randint(3, w - 4), rng.randint(3, h - 4)
        length = rng.randint(2, 5 + level_idx)
        horiz = rng.choice([True, False])
        for i in range(length):
            nx = wx + (i if horiz else 0)
            ny = wy + (0 if horiz else i)
            if 1 <= nx < w - 1 and 1 <= ny < h - 1:
                terrain[ny][nx] = WALL

    # Sand patches
    for _ in range(4 + level_idx * 2):
        sx, sy = rng.randint(2, w - 3), rng.randint(2, h - 3)
        for ddx in range(-1, 2):
            for ddy in range(-1, 2):
                nx, ny = sx + ddx, sy + ddy
                if 1 <= nx < w - 1 and 1 <= ny < h - 1 and terrain[ny][nx] == FLOOR:
                    if rng.random() < 0.7:
                        terrain[ny][nx] = SAND

    # Pre-placed water (from level 1+)
    if level_idx >= 1:
        for _ in range(1 + level_idx):
            wx, wy = rng.randint(3, w - 4), rng.randint(3, h - 4)
            if terrain[wy][wx] in (FLOOR, SAND):
                terrain[wy][wx] = WATER

    # Pre-placed lava (from level 3+)
    if level_idx >= 3:
        for _ in range(level_idx - 1):
            lx, ly = rng.randint(3, w - 4), rng.randint(3, h - 4)
            if terrain[ly][lx] == FLOOR:
                terrain[ly][lx] = LAVA

    # Target zones
    num_targets = 2 + level_idx
    if level_idx <= 1:
        target_choices = [WATER, GRASS, SAND]
    else:
        target_choices = [WATER, GRASS, SAND, STONE]
    targets: List[Tuple[int, int, int]] = []
    for _ in range(300):
        if len(targets) >= num_targets:
            break
        tx, ty = rng.randint(3, w - 4), rng.randint(3, h - 4)
        if terrain[ty][tx] == WALL:
            continue
        if any(t[0] == tx and t[1] == ty for t in targets):
            continue
        req = rng.choice(target_choices)
        if terrain[ty][tx] == req:
            terrain[ty][tx] = FLOOR
        targets.append((tx, ty, req))

    # Player start
    px, py = 2, 2
    for _ in range(200):
        if terrain[py][px] == FLOOR:
            break
        px, py = rng.randint(2, w - 3), rng.randint(2, h - 3)

    # Goal flag
    gx, gy = w - 3, h - 3
    for _ in range(200):
        if terrain[gy][gx] == FLOOR and not (gx == px and gy == py):
            break
        gx, gy = rng.randint(2, w - 3), rng.randint(2, h - 3)

    # Charges
    if level_idx <= 1:
        charges = -1  # unlimited
    elif level_idx == 2:
        charges = num_targets * 4
    elif level_idx == 3:
        charges = num_targets * 3
    else:
        charges = num_targets * 3 + 2

    return {
        "terrain": terrain,
        "targets": targets,
        "player_start": (px, py),
        "goal_pos": (gx, gy),
        "charges": charges,
    }


# ══════════════════════════════════════════════════════════════════
class Terraformer(ARCBaseGame):
    """Terraformer game — transform terrain to match target patterns."""

    def __init__(self, seed: int = 42) -> None:
        self._game_seed = seed
        self._num_levels = 5
        self._level_data = [_generate_level(seed, i) for i in range(self._num_levels)]

        levels = []
        for i in range(self._num_levels):
            bg = Sprite(
                pixels=[[-1] * 64 for _ in range(64)],
                name=f"bg_{i}",
                visible=True,
                collidable=False,
                layer=-10,
                tags=["background"],
            )
            levels.append(Level(
                sprites=[bg], grid_size=(64, 64),
                data={"level_idx": i}, name=f"level_{i}",
            ))

        super().__init__(
            game_id="terraformer",
            levels=levels,
            camera=Camera(0, 0, 64, 64, C_UI_BG, C_UI_BG),
            debug=False,
            win_score=1,
            available_actions=[1, 2, 3, 4, 5],
            seed=seed,
        )

    # ── Level setup ────────────────────────────────────────────────
    def on_set_level(self, level: Level) -> None:
        idx = level.get_data("level_idx") or 0
        data = self._level_data[idx]

        self._terrain = [row[:] for row in data["terrain"]]
        self._targets = list(data["targets"])
        self._px, self._py = data["player_start"]
        self._gx, self._gy = data["goal_pos"]
        self._charges = data["charges"]
        self._max_charges = data["charges"]
        self._tool_idx = 0
        self._step_counter = 0
        self._fire_timers: dict[Tuple[int, int], int] = {}
        self._grass_timers: dict[Tuple[int, int], int] = {}

        self._render_to_bg()

    # ── Chain-reaction simulation (NO water spreading) ─────────────
    def _simulate_terrain(self) -> None:
        w, h = GRID_W, GRID_H
        changes: dict[Tuple[int, int], Tuple[int, int]] = {}

        def add_change(cx: int, cy: int, new_t: int, prio: int) -> None:
            key = (cx, cy)
            if key not in changes or changes[key][1] < prio:
                changes[key] = (new_t, prio)

        for y in range(1, h - 1):
            for x in range(1, w - 1):
                t = self._terrain[y][x]

                if t == WATER:
                    # Water does NOT spread. Only reacts with adjacent lava.
                    for ddx, ddy in DIRS:
                        nx, ny = x + ddx, y + ddy
                        if 0 <= nx < w and 0 <= ny < h:
                            if self._terrain[ny][nx] == LAVA:
                                add_change(nx, ny, STONE, 3)
                                add_change(x, y, STONE, 3)

                elif t == LAVA:
                    for ddx, ddy in DIRS:
                        nx, ny = x + ddx, y + ddy
                        if 0 <= nx < w and 0 <= ny < h:
                            adj = self._terrain[ny][nx]
                            if adj == GRASS:
                                add_change(nx, ny, FIRE, 4)
                            elif adj == WATER:
                                add_change(nx, ny, STONE, 3)
                                add_change(x, y, STONE, 3)

                elif t == FIRE:
                    key = (x, y)
                    if key not in self._fire_timers:
                        self._fire_timers[key] = 3
                    self._fire_timers[key] -= 1
                    if self._fire_timers[key] <= 0:
                        add_change(x, y, FLOOR, 5)
                        del self._fire_timers[key]
                    else:
                        for ddx, ddy in DIRS:
                            nx, ny = x + ddx, y + ddy
                            if 0 <= nx < w and 0 <= ny < h:
                                if self._terrain[ny][nx] == GRASS:
                                    add_change(nx, ny, FIRE, 4)

                elif t == SAND:
                    # Sand adjacent to water slowly becomes grass
                    has_water = any(
                        0 <= x + ddx < w and 0 <= y + ddy < h
                        and self._terrain[y + ddy][x + ddx] == WATER
                        for ddx, ddy in DIRS
                    )
                    if has_water:
                        key = (x, y)
                        if key not in self._grass_timers:
                            self._grass_timers[key] = 8
                        self._grass_timers[key] -= 1
                        if self._grass_timers[key] <= 0:
                            add_change(x, y, GRASS, 2)
                            self._grass_timers.pop(key, None)

        # Apply changes, but protect target cells that already have correct terrain
        protected = set()
        for tx, ty, req in self._targets:
            if self._terrain[ty][tx] == req:
                protected.add((tx, ty))

        for (cx, cy), (new_t, _) in changes.items():
            if self._terrain[cy][cx] != WALL and (cx, cy) not in protected:
                self._terrain[cy][cx] = new_t

    # ── Win condition ──────────────────────────────────────────────
    def _all_targets_met(self) -> bool:
        return all(self._terrain[ty][tx] == req for tx, ty, req in self._targets)

    # ── Render full frame ──────────────────────────────────────────
    def _render_to_bg(self) -> None:
        bg_list = self.current_level.get_sprites_by_tag("background")
        if not bg_list:
            return
        bg = bg_list[0]

        pixels = np.full((64, 64), C_UI_BG, dtype=np.int8)

        # Terrain
        for y in range(GRID_H):
            for x in range(GRID_W):
                pixels[y][x] = TERRAIN_TO_COLOR.get(self._terrain[y][x], C_EMPTY)

        # Target zone indicators (blink between target color and marker)
        for tx, ty, req in self._targets:
            if 0 <= tx < 64 and 0 <= ty < 64:
                if self._terrain[ty][tx] != req:
                    req_color = TERRAIN_TO_COLOR.get(req, C_EMPTY)
                    pixels[ty][tx] = req_color if self._step_counter % 4 < 2 else C_TARGET_MARKER

        # Goal flag
        if 0 <= self._gx < 64 and 0 <= self._gy < 64:
            show_goal = self._all_targets_met() or self._step_counter % 6 < 3
            if show_goal:
                pixels[self._gy][self._gx] = C_GOAL

        # Player
        if 0 <= self._px < 64 and 0 <= self._py < 64:
            pixels[self._py][self._px] = C_PLAYER

        # UI row 63: tool indicator
        tool_color = TOOL_COLORS[self._tool_idx]
        for x in range(5):
            pixels[63][x] = tool_color
        for i, tc in enumerate(TOOL_COLORS):
            bx = 7 + i * 3
            if bx < 64:
                pixels[63][bx] = tc
                if i == self._tool_idx:
                    if bx + 1 < 64:
                        pixels[63][bx + 1] = C_PLAYER

        # UI row 62: charge bar
        if self._max_charges > 0:
            bar_len = min(self._max_charges, 50)
            filled = int(bar_len * max(0, self._charges) / self._max_charges)
            for x in range(bar_len):
                pixels[62][x] = 14 if x < filled else 8

        # UI row 61: level dots
        level_idx = self.current_level.get_data("level_idx") or 0
        for i in range(level_idx + 1):
            if i < 64:
                pixels[61][i] = C_GOAL

        bg.pixels = pixels

    # ── Step ───────────────────────────────────────────────────────
    def step(self) -> None:
        action_id = self.action.id.value
        dx, dy = 0, 0

        if action_id == 1:
            dy = -1
        elif action_id == 2:
            dy = 1
        elif action_id == 3:
            dx = -1
        elif action_id == 4:
            dx = 1
        elif action_id == 5:
            # Place current tool terrain on player tile, then cycle
            if self._charges != 0:
                self._terrain[self._py][self._px] = TOOL_TERRAINS[self._tool_idx]
                if self._charges > 0:
                    self._charges -= 1
            self._tool_idx = (self._tool_idx + 1) % len(TOOL_TERRAINS)

        # Movement
        if dx or dy:
            nx, ny = self._px + dx, self._py + dy
            if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                dest = self._terrain[ny][nx]
                if dest not in (WALL, STONE, LAVA):
                    self._px, self._py = nx, ny

        # Simulation runs only on movement actions (not tool cycling)
        # This prevents chain reactions during rapid tool cycling
        self._step_counter += 1
        if (dx or dy) and self._step_counter % 2 == 0:
            self._simulate_terrain()

        # Win check
        if self._px == self._gx and self._py == self._gy and self._all_targets_met():
            self._render_to_bg()
            self.next_level()
            self.complete_action()
            return

        self._render_to_bg()
        self.complete_action()


# ── Standalone test ────────────────────────────────────────────────
if __name__ == "__main__":
    from arcengine import ActionInput, GameAction

    game = Terraformer(seed=42)
    frame = game.perform_action(ActionInput(id=GameAction.RESET))
    print(f"Game started. State: {frame.state}, Levels: {frame.levels_completed}/{frame.win_levels}")

    rng = random.Random(12345)
    actions = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
               GameAction.ACTION4, GameAction.ACTION5]
    for i in range(500):
        action = rng.choice(actions)
        frame = game.perform_action(ActionInput(id=action))
        if frame.state.value == "WIN":
            print(f"  WIN after {i+1} actions!")
            break
        if frame.state.value == "GAME_OVER":
            print(f"  GAME_OVER after {i+1} actions!")
            break
        if i % 100 == 0:
            print(f"  Step {i}: state={frame.state}, levels={frame.levels_completed}/{frame.win_levels}")
    else:
        print(f"  Finished 500 actions. State: {frame.state}, "
              f"Levels: {frame.levels_completed}/{frame.win_levels}")
