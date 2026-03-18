"""
Game 03: Alchemist

Mechanics: Navigation + item collection + crafting/combining + recipe discovery + goal matching

The player navigates a grid collecting colored ingredients. An anvil sprite on
the map lets the player combine two held ingredients (ACTION5 when adjacent) into
a new item based on hidden recipes. A display area shows what item is needed to
unlock the exit. The player must figure out which combinations produce which
results. Later levels require multi-step crafting (combine A+B=C, then C+D=E).
The player can hold at most 2 items at a time. Dropping an item (ACTION5 on
empty ground away from anvil/ingredient) is allowed.

Actions:
    ACTION1 = Move Up (W)
    ACTION2 = Move Down (S)
    ACTION3 = Move Left (A)
    ACTION4 = Move Right (D)
    ACTION5 = Context action: pick up ingredient / combine at anvil / drop first held item

Display:
    Row 60-63: HUD — held item slots (left), target item indicator (right)

Win condition: Craft the target item and walk onto the exit door.

Color key:
    0  = empty / background floor
    1  = wall
    2  = player
    3  = anvil
    4  = exit door border
    5  = black / HUD bg
    6  = ingredient: blue
    7  = ingredient: yellow
    8  = ingredient: red
    9  = ingredient: purple  (red + blue)
    10 = ingredient: green   (blue + yellow)
    11 = ingredient: orange  (red + yellow)
    12 = ingredient: white   (purple + green, or orange + green)
    13 = ingredient: teal    (blue + green)
    14 = ingredient: pink    (red + purple, or purple + orange)
    15 = HUD label / slot border
"""

import random
from typing import List, Optional, Tuple

import numpy as np
from arcengine import (
    ARCBaseGame,
    Camera,
    Level,
    RenderableUserDisplay,
    Sprite,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GRID = 64
CELL = 3          # player & objects occupy 3x3 pixels
MOVE = 3          # movement step size
BG_COLOR = 0
WALL_COLOR = 1
PLAYER_COLOR = 2
ANVIL_COLOR = 3
EXIT_COLOR = 4
HUD_BG = 5
HUD_BORDER = 15

# Ingredient colors
BLUE = 6
YELLOW = 7
RED = 8
PURPLE = 9
GREEN = 10
ORANGE = 11
WHITE = 12
TEAL = 13
PINK = 14

BASE_INGREDIENTS = [BLUE, YELLOW, RED]

# Hidden recipes: frozenset({colorA, colorB}) -> resultColor
RECIPES = {
    frozenset([RED, BLUE]): PURPLE,
    frozenset([BLUE, YELLOW]): GREEN,
    frozenset([RED, YELLOW]): ORANGE,
    frozenset([PURPLE, GREEN]): WHITE,
    frozenset([BLUE, GREEN]): TEAL,
    frozenset([RED, PURPLE]): PINK,
    frozenset([ORANGE, GREEN]): WHITE,
    frozenset([PURPLE, ORANGE]): PINK,
    frozenset([TEAL, YELLOW]): GREEN,
    frozenset([TEAL, RED]): PURPLE,
}


# ---------------------------------------------------------------------------
# Sprite pixel helpers
# ---------------------------------------------------------------------------

def _player_pixels() -> List[List[int]]:
    c = PLAYER_COLOR
    return [
        [-1, c, -1],
        [c,  c,  c],
        [-1, c, -1],
    ]


def _anvil_pixels() -> List[List[int]]:
    c = ANVIL_COLOR
    return [
        [c,  c,  c],
        [-1, c, -1],
        [c,  c,  c],
    ]


def _exit_pixels(target_color: int) -> List[List[int]]:
    e = EXIT_COLOR
    t = target_color
    return [
        [e,  e,  e,  e,  e],
        [e,  t,  t,  t,  e],
        [e,  t,  t,  t,  e],
        [e,  t,  t,  t,  e],
        [e,  e,  e,  e,  e],
    ]


def _ingredient_pixels(color: int) -> List[List[int]]:
    return [
        [-1,   color, -1],
        [color, color, color],
        [-1,   color, -1],
    ]


# ---------------------------------------------------------------------------
# Maze generation
# ---------------------------------------------------------------------------

def _generate_maze(width: int, height: int, rng: random.Random) -> List[List[int]]:
    """Generate a maze on a grid. Returns 2D array where 1=wall, 0=floor."""
    grid = [[WALL_COLOR] * width for _ in range(height)]

    cols = width // CELL
    rows = height // CELL
    visited = [[False] * cols for _ in range(rows)]

    def _carve(cr: int, cc: int):
        for dy in range(CELL):
            for dx in range(CELL):
                py, px = cr * CELL + dy, cc * CELL + dx
                if 0 <= py < height and 0 <= px < width:
                    grid[py][px] = BG_COLOR

    # Start from cell (1,1) and carve using randomised DFS
    start_r, start_c = 1, 1
    visited[start_r][start_c] = True
    _carve(start_r, start_c)
    stack = [(start_r, start_c)]

    while stack:
        cr, cc = stack[-1]
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = cr + dr, cc + dc
            if 1 <= nr < rows - 1 and 1 <= nc < cols - 1 and not visited[nr][nc]:
                neighbors.append((nr, nc))
        if not neighbors:
            stack.pop()
            continue
        nr, nc = rng.choice(neighbors)
        visited[nr][nc] = True
        # Carve passage: clear all pixels between current cell and neighbor
        min_r, min_c = min(cr, nr), min(cc, nc)
        max_r, max_c = max(cr, nr), max(cc, nc)
        for r in range(min_r * CELL, (max_r + 1) * CELL):
            for c in range(min_c * CELL, (max_c + 1) * CELL):
                if 0 <= r < height and 0 <= c < width:
                    grid[r][c] = BG_COLOR
        _carve(nr, nc)
        stack.append((nr, nc))

    # Enforce wall borders
    for y in range(height):
        for bx in range(CELL):
            grid[y][bx] = WALL_COLOR
            if width - 1 - bx >= 0:
                grid[y][width - 1 - bx] = WALL_COLOR
    for x in range(width):
        for by in range(CELL):
            grid[by][x] = WALL_COLOR
            if height - 1 - by >= 0:
                grid[height - 1 - by][x] = WALL_COLOR

    return grid


def _find_open_positions(
    maze: List[List[int]], count: int, rng: random.Random,
    obj_size: int = CELL,
) -> List[Tuple[int, int]]:
    """Find `count` non-overlapping open positions in the maze."""
    h = len(maze)
    w = len(maze[0]) if h else 0
    min_gap = obj_size + 2  # minimum distance between placed objects

    candidates = []
    for y in range(CELL, h - CELL - obj_size + 1):
        for x in range(CELL, w - CELL - obj_size + 1):
            ok = True
            for dy in range(obj_size):
                for dx in range(obj_size):
                    if maze[y + dy][x + dx] != BG_COLOR:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                candidates.append((x, y))

    rng.shuffle(candidates)

    chosen: List[Tuple[int, int]] = []
    for cx, cy in candidates:
        if len(chosen) >= count:
            break
        overlap = False
        for ox, oy in chosen:
            if abs(cx - ox) < min_gap and abs(cy - oy) < min_gap:
                overlap = True
                break
        if not overlap:
            chosen.append((cx, cy))

    return chosen


# ---------------------------------------------------------------------------
# Level builder
# ---------------------------------------------------------------------------

def _build_level(
    level_num: int,
    seed: int,
    ingredients_needed: List[int],
    target_color: int,
) -> Level:
    rng = random.Random(seed + level_num * 1000)

    play_h = 57  # rows 0-56 playable, rows 57+ behind HUD
    play_w = 60

    maze = _generate_maze(play_w, play_h, rng)

    num_objects = 3 + len(ingredients_needed)  # player + anvil + exit + ingredients
    positions = _find_open_positions(maze, num_objects, rng, obj_size=CELL)

    # Fallback if not enough positions found
    while len(positions) < num_objects:
        positions.append((CELL * 2 + len(positions) * CELL, CELL * 2))

    player_pos = positions[0]
    anvil_pos = positions[1]
    exit_pos = positions[2]
    ingredient_positions = positions[3:]

    sprites_list: List[Sprite] = []

    # Build wall sprites (horizontal runs for efficiency)
    for y in range(play_h):
        x = 0
        while x < play_w:
            if maze[y][x] == WALL_COLOR:
                run_start = x
                while x < play_w and maze[y][x] == WALL_COLOR:
                    x += 1
                wall_sprite = Sprite(
                    pixels=[[WALL_COLOR] * (x - run_start)],
                    name=f"wall_{y}_{run_start}",
                    x=run_start, y=y,
                    layer=-1, visible=True, collidable=True,
                    tags=["wall"],
                )
                sprites_list.append(wall_sprite)
            else:
                x += 1

    # Player
    sprites_list.append(Sprite(
        pixels=_player_pixels(), name="player",
        x=player_pos[0], y=player_pos[1],
        layer=10, visible=True, collidable=True,
        tags=["player"],
    ))

    # Anvil (collidable — player bumps into it, interacts adjacently)
    sprites_list.append(Sprite(
        pixels=_anvil_pixels(), name="anvil",
        x=anvil_pos[0], y=anvil_pos[1],
        layer=5, visible=True, collidable=True,
        tags=["anvil"],
    ))

    # Exit door (NOT collidable — player walks onto it, win checked on overlap)
    sprites_list.append(Sprite(
        pixels=_exit_pixels(target_color), name="exit",
        x=exit_pos[0], y=exit_pos[1],
        layer=-2, visible=True, collidable=False,
        tags=["exit"],
    ))

    # Ingredients (not collidable — player walks over them)
    for i, ipos in enumerate(ingredient_positions):
        color = ingredients_needed[i] if i < len(ingredients_needed) else rng.choice(BASE_INGREDIENTS)
        sprites_list.append(Sprite(
            pixels=_ingredient_pixels(color), name=f"ingredient_{i}",
            x=ipos[0], y=ipos[1],
            layer=3, visible=True, collidable=False,
            tags=["ingredient"],
        ))

    return Level(
        sprites=sprites_list,
        grid_size=(GRID, GRID),
        data={"target_color": target_color, "player_start": player_pos},
        name=f"level_{level_num}",
    )


# ---------------------------------------------------------------------------
# HUD overlay
# ---------------------------------------------------------------------------

class AlchemistHUD(RenderableUserDisplay):
    """Renders held items and target on rows 60-63 of the 64x64 grid."""

    def __init__(self, game: "AlchemistGame"):
        self.game = game

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        frame[60:64, :] = HUD_BG

        # Two held-item slots on the left
        for slot_idx in range(2):
            sx = 1 + slot_idx * 5
            frame[60, sx:sx + 4] = HUD_BORDER
            frame[63, sx:sx + 4] = HUD_BORDER
            frame[60:64, sx] = HUD_BORDER
            frame[60:64, sx + 3] = HUD_BORDER
            if slot_idx < len(self.game.held_items):
                frame[61:63, sx + 1:sx + 3] = self.game.held_items[slot_idx]

        # Target slot on the right
        tx = 55
        frame[60, tx:tx + 5] = HUD_BORDER
        frame[63, tx:tx + 5] = HUD_BORDER
        frame[60:64, tx] = HUD_BORDER
        frame[60:64, tx + 4] = HUD_BORDER
        if self.game.target_color is not None:
            frame[61:63, tx + 1:tx + 4] = self.game.target_color

        # Arrow from slots toward target
        frame[61:63, 13] = HUD_BORDER
        frame[61, 50:53] = HUD_BORDER
        frame[62, 51] = HUD_BORDER

        return frame


# ---------------------------------------------------------------------------
# Game class
# ---------------------------------------------------------------------------

class AlchemistGame(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self.held_items: List[int] = []
        self.target_color: Optional[int] = None
        self.hud = AlchemistHUD(self)
        self._seed = seed
        self._drop_counter = 0

        level_configs = self._get_level_configs()
        built_levels = []
        for i, cfg in enumerate(level_configs):
            built_levels.append(_build_level(
                level_num=i, seed=seed,
                ingredients_needed=cfg["ingredients"],
                target_color=cfg["target"],
            ))

        camera = Camera(0, 0, 64, 64, BG_COLOR, HUD_BG, [self.hud])

        super().__init__(
            game_id="alchemist",
            levels=built_levels,
            camera=camera,
            debug=False,
            win_score=len(built_levels),
            available_actions=[1, 2, 3, 4, 5],
            seed=seed,
        )

    def _get_level_configs(self) -> List[dict]:
        return [
            # Level 1: Simple — red + blue = purple
            {"ingredients": [RED, BLUE], "target": PURPLE},
            # Level 2: Simple — blue + yellow = green
            {"ingredients": [BLUE, YELLOW], "target": GREEN},
            # Level 3: Choose right pair (3 ingredients, target = orange = red+yellow)
            {"ingredients": [RED, YELLOW, BLUE], "target": ORANGE},
            # Level 4: Two-step — need purple (red+blue) then white (purple+green where green=blue+yellow)
            {"ingredients": [RED, BLUE, BLUE, YELLOW], "target": WHITE},
            # Level 5: Two-step with distractors — pink = red+purple, purple = red+blue
            {"ingredients": [RED, BLUE, YELLOW, RED, YELLOW], "target": PINK},
        ]

    # ------------------------------------------------------------------
    # Level lifecycle
    # ------------------------------------------------------------------

    def on_set_level(self, level: Level) -> None:
        self.held_items = []
        self.target_color = level.get_data("target_color")
        self.player = self.current_level.get_sprites_by_tag("player")[0]
        self._drop_counter = 0

    # ------------------------------------------------------------------
    # Proximity helpers
    # ------------------------------------------------------------------

    def _sprites_within(self, tag: str, max_dist: int) -> List[Sprite]:
        """Return all visible sprites with `tag` whose position is within max_dist of the player."""
        px, py = self.player.x, self.player.y
        result = []
        for s in self.current_level.get_sprites_by_tag(tag):
            if not s.is_visible:
                continue
            if abs(s.x - px) <= max_dist and abs(s.y - py) <= max_dist:
                result.append(s)
        return result

    def _nearest_sprite(self, tag: str, max_dist: int) -> Optional[Sprite]:
        """Return the closest visible sprite with `tag` within max_dist, or None."""
        matches = self._sprites_within(tag, max_dist)
        if not matches:
            return None
        px, py = self.player.x, self.player.y
        return min(matches, key=lambda s: abs(s.x - px) + abs(s.y - py))

    # ------------------------------------------------------------------
    # Crafting
    # ------------------------------------------------------------------

    def _try_craft(self) -> bool:
        if len(self.held_items) != 2:
            return False
        key = frozenset([int(c) for c in self.held_items])
        result = RECIPES.get(key)
        if result is not None:
            self.held_items = [result]
            return True
        return False

    # ------------------------------------------------------------------
    # Win check
    # ------------------------------------------------------------------

    def _check_exit_win(self) -> bool:
        """Check if player is on the exit with the correct item."""
        exits = self._sprites_within("exit", CELL)
        if not exits:
            return False
        if len(self.held_items) == 1 and int(self.held_items[0]) == self.target_color:
            return True
        return False

    # ------------------------------------------------------------------
    # Main game step
    # ------------------------------------------------------------------

    def step(self) -> None:
        action_id = self.action.id.value

        # ---- Movement (ACTION1-4) ----
        if action_id in (1, 2, 3, 4):
            dx, dy = 0, 0
            if action_id == 1:
                dy = -MOVE
            elif action_id == 2:
                dy = MOVE
            elif action_id == 3:
                dx = -MOVE
            elif action_id == 4:
                dx = MOVE

            self.try_move_sprite(self.player, dx, dy)

            # Check win after every move
            if self._check_exit_win():
                self.next_level()

            self.complete_action()
            return

        # ---- ACTION5: context-sensitive interact ----
        if action_id == 5:
            # Priority 1: If adjacent to anvil and holding 2 items → craft
            anvil = self._nearest_sprite("anvil", CELL + 1)
            if anvil is not None and len(self.held_items) == 2:
                self._try_craft()
                self.complete_action()
                return

            # Priority 2: If overlapping/adjacent ingredient and not full → pick up
            ingredient = self._nearest_sprite("ingredient", CELL - 1)
            if ingredient is not None and len(self.held_items) < 2:
                color = int(ingredient.pixels[1][1])
                if color < 0:
                    color = int(ingredient.pixels[0][1])
                self.held_items.append(color)
                self.current_level.remove_sprite(ingredient)
                self.complete_action()
                return

            # Priority 3: Drop first held item (only if not on an ingredient)
            nearby_ing = self._nearest_sprite("ingredient", CELL - 1)
            if nearby_ing is None and len(self.held_items) > 0:
                dropped_color = self.held_items.pop(0)
                # Place dropped item slightly offset so it doesn't overlap player exactly
                drop_x = self.player.x + CELL
                drop_y = self.player.y
                dropped = Sprite(
                    pixels=_ingredient_pixels(dropped_color),
                    name=f"dropped_{self._drop_counter}",
                    x=drop_x, y=drop_y,
                    layer=3, visible=True, collidable=False,
                    tags=["ingredient"],
                )
                self._drop_counter += 1
                self.current_level.add_sprite(dropped)
                self.complete_action()
                return

            self.complete_action()
            return

        self.complete_action()
