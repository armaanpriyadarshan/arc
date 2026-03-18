"""
Game 04: Mirror Maze

Mechanics: Navigation + beam reflection + mirror rotation + beam-activated doors + limited moves

A beam of light enters the grid from one edge. Mirror sprites (diagonal 3x3
blocks) are placed on the grid. The player can rotate mirrors by moving adjacent
to them and pressing ACTION5. The beam reflects off mirrors and must be directed
to hit a target receptor. When all beams hit their matching receptors, the exit
door opens and the player can walk onto it to advance. The player has a limited
move counter. Later levels have multiple beams of different colors and mirrors
that only reflect certain colors.

Actions:
    ACTION1 = Move Up (W)
    ACTION2 = Move Down (S)
    ACTION3 = Move Left (A)
    ACTION4 = Move Right (D)
    ACTION5 = Rotate nearest adjacent mirror 90 degrees

Display:
    Row 62: move counter (filled bar = remaining moves)
    Row 63: receptor status dots (lit = beam hitting)

Win condition: All beams hitting their matching receptors, then player walks
onto the exit door.

Color key:
    0  = empty / floor
    1  = wall
    2  = player
    3  = mirror frame (backslash orientation)
    4  = mirror frame (forward-slash orientation)
    5  = HUD background
    6  = blue (beam / emitter / receptor)
    7  = yellow (beam / emitter / receptor)
    8  = red (beam / emitter / receptor)
    9  = purple (beam / emitter / receptor)
    10 = green (exit door open)
    11 = exit door closed
    12 = dimmed / inactive indicator
    13 = beam path pixel
    14 = receptor border (correct)
    15 = HUD border
"""

import random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from arcengine import (
    ARCBaseGame,
    Camera,
    GameAction,
    Level,
    Sprite,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GRID = 64
CELL = 3
MOVE = 3
BG = 0
WALL = 1
PLAYER_COLOR = 2
MIRROR_BS = 3      # backslash mirror \
MIRROR_FS = 4      # forward-slash mirror /
HUD_BG = 5
BLUE = 6
YELLOW = 7
RED = 8
PURPLE = 9
EXIT_OPEN = 10
EXIT_CLOSED = 11
DIMMED = 12
BEAM_PATH = 13
RECEPTOR_LIT = 14
HUD_BORDER = 15

# Beam directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

BEAM_COLORS = [RED, BLUE, YELLOW, PURPLE]


# ---------------------------------------------------------------------------
# Mirror orientation: 0 = backslash (\), 1 = forward-slash (/)
# Reflection rules (dx, dy) -> (dx, dy):
#   Backslash \:  RIGHT->(0,1)=DOWN, LEFT->(0,-1)=UP, DOWN->(1,0)=RIGHT, UP->(-1,0)=LEFT
#   Forward  /:  RIGHT->(0,-1)=UP, LEFT->(0,1)=DOWN, DOWN->(-1,0)=LEFT, UP->(1,0)=RIGHT
# ---------------------------------------------------------------------------

def _reflect(dx: int, dy: int, orientation: int) -> Tuple[int, int]:
    """Reflect a beam direction off a mirror. Returns new (dx, dy)."""
    if orientation == 0:  # backslash
        return (dy, dx)
    else:  # forward-slash
        return (-dy, -dx)


# ---------------------------------------------------------------------------
# Sprite builders
# ---------------------------------------------------------------------------

def _make_player() -> Sprite:
    px = [
        [PLAYER_COLOR, PLAYER_COLOR, PLAYER_COLOR],
        [PLAYER_COLOR, PLAYER_COLOR, PLAYER_COLOR],
        [PLAYER_COLOR, PLAYER_COLOR, PLAYER_COLOR],
    ]
    return Sprite(pixels=px, name="player", visible=True, collidable=True)


def _make_wall() -> Sprite:
    px = [[WALL] * CELL for _ in range(CELL)]
    return Sprite(pixels=px, name="wall", visible=True, collidable=True)


def _make_mirror(orientation: int, idx: int, color_filter: Optional[int] = None) -> Sprite:
    """3x3 mirror sprite. Orientation 0=backslash, 1=forward-slash.
    color_filter: if set, mirror only reflects this beam color."""
    if orientation == 0:
        px = [
            [MIRROR_BS, BG, BG],
            [BG, MIRROR_BS, BG],
            [BG, BG, MIRROR_BS],
        ]
    else:
        px = [
            [BG, BG, MIRROR_FS],
            [BG, MIRROR_FS, BG],
            [MIRROR_FS, BG, BG],
        ]
    # If color-filtered, tint the center pixel
    if color_filter is not None:
        px[1][1] = color_filter
    tags = ["mirror"]
    return Sprite(pixels=px, name=f"mirror_{idx}", visible=True, collidable=True, tags=tags)


def _make_emitter(color: int, idx: int) -> Sprite:
    """3x3 beam emitter."""
    px = [
        [color, color, color],
        [color, BG, color],
        [color, color, color],
    ]
    return Sprite(pixels=px, name=f"emitter_{idx}", visible=True, collidable=True,
                  tags=["emitter"])


def _make_receptor(color: int, idx: int) -> Sprite:
    """3x3 receptor that needs to be hit by matching beam."""
    px = [
        [DIMMED, DIMMED, DIMMED],
        [DIMMED, color, DIMMED],
        [DIMMED, DIMMED, DIMMED],
    ]
    return Sprite(pixels=px, name=f"receptor_{idx}", visible=True, collidable=True,
                  tags=["receptor"])


def _make_exit(is_open: bool) -> Sprite:
    color = EXIT_OPEN if is_open else EXIT_CLOSED
    px = [[color] * CELL for _ in range(CELL)]
    return Sprite(pixels=px, name="exit", visible=True, collidable=not is_open,
                  tags=["exit"])


# ---------------------------------------------------------------------------
# Maze generation (reused from game_01)
# ---------------------------------------------------------------------------

def _carve_maze(rng: random.Random, cols: int, rows: int) -> np.ndarray:
    maze = np.ones((rows, cols), dtype=int)

    def _neighbors(r, c):
        dirs = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        rng.shuffle(dirs)
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr, nc] == 1:
                yield nr, nc, r + dr // 2, c + dc // 2

    stack = [(1, 1)]
    maze[1, 1] = 0
    while stack:
        r, c = stack[-1]
        found = False
        for nr, nc, wr, wc in _neighbors(r, c):
            maze[nr, nc] = 0
            maze[wr, wc] = 0
            stack.append((nr, nc))
            found = True
            break
        if not found:
            stack.pop()
    return maze


# ---------------------------------------------------------------------------
# Level configs
# ---------------------------------------------------------------------------

def _level_configs():
    return [
        # Level 0: 1 beam, 2 mirrors, generous moves, small grid
        {
            "cell_cols": 11, "cell_rows": 15,
            "num_beams": 1, "num_mirrors": 2, "max_moves": 60,
            "color_filters": False,
        },
        # Level 1: 1 beam, 3 mirrors, moderate moves
        {
            "cell_cols": 13, "cell_rows": 17,
            "num_beams": 1, "num_mirrors": 3, "max_moves": 50,
            "color_filters": False,
        },
        # Level 2: 2 beams, 4 mirrors
        {
            "cell_cols": 13, "cell_rows": 17,
            "num_beams": 2, "num_mirrors": 4, "max_moves": 60,
            "color_filters": False,
        },
        # Level 3: 2 beams, 5 mirrors, some color-filtered
        {
            "cell_cols": 15, "cell_rows": 19,
            "num_beams": 2, "num_mirrors": 5, "max_moves": 55,
            "color_filters": True,
        },
        # Level 4: 3 beams, 6 mirrors, tight move budget, color filters
        {
            "cell_cols": 17, "cell_rows": 19,
            "num_beams": 3, "num_mirrors": 6, "max_moves": 45,
            "color_filters": True,
        },
    ]


# ---------------------------------------------------------------------------
# Game class
# ---------------------------------------------------------------------------

class MirrorMazeGame(ARCBaseGame):
    """Game 04: Mirror Maze — navigate, rotate mirrors, direct beams to receptors."""

    def __init__(self, seed: int = 0):
        self._seed = seed
        self._rng = random.Random(seed)
        self._cfgs = _level_configs()

        # Per-level state
        self._moves_left = 0
        self._max_moves = 0
        self._mirror_orientations: Dict[str, int] = {}  # mirror_name -> 0 or 1
        self._mirror_color_filters: Dict[str, Optional[int]] = {}  # mirror_name -> color or None
        self._emitter_dirs: Dict[str, Tuple[int, int]] = {}  # emitter_name -> (dx, dy)
        self._emitter_colors: Dict[str, int] = {}
        self._receptor_colors: Dict[str, int] = {}
        self._receptor_hit: Dict[str, bool] = {}
        self._exit_open = False
        self._num_receptors = 0

        levels = [Level(sprites=[], grid_size=(GRID, GRID)) for _ in self._cfgs]
        camera = Camera(0, 0, GRID, GRID, BG, HUD_BG)
        super().__init__(
            game_id="game_04_mirror_maze",
            levels=levels,
            camera=camera,
            available_actions=[1, 2, 3, 4, 5],
            win_score=len(self._cfgs),
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Level setup
    # ------------------------------------------------------------------

    def on_set_level(self, level):
        level_idx = self._levels.index(level)
        rng = random.Random(self._seed * 100 + level_idx)
        cfg = self._cfgs[level_idx]

        cell_cols = cfg["cell_cols"]
        cell_rows = cfg["cell_rows"]
        num_beams = cfg["num_beams"]
        num_mirrors = cfg["num_mirrors"]
        self._max_moves = cfg["max_moves"]
        self._moves_left = cfg["max_moves"]
        use_color_filters = cfg["color_filters"]

        # Generate open maze
        maze = _carve_maze(rng, cell_cols, cell_rows)

        # Gather floor cells
        floor_cells = []
        for r in range(cell_rows):
            for c in range(cell_cols):
                if maze[r, c] == 0:
                    floor_cells.append((r, c))
        rng.shuffle(floor_cells)
        used = set()

        def _take():
            while floor_cells:
                cell = floor_cells.pop()
                if cell not in used:
                    used.add(cell)
                    return cell
            return None

        # Place walls
        for r in range(cell_rows):
            for c in range(cell_cols):
                if maze[r, c] == 1:
                    w = _make_wall()
                    w.set_position(c * CELL, r * CELL)
                    level.add_sprite(w)

        # Place player
        player_cell = _take()
        player = _make_player()
        player.set_position(player_cell[1] * CELL, player_cell[0] * CELL)
        level.add_sprite(player)

        # Choose beam colors
        beam_colors = list(BEAM_COLORS[:num_beams])

        # Place emitters on left edge (column 0 floor cells, or nearest floor)
        self._emitter_dirs = {}
        self._emitter_colors = {}
        edge_floors = [(r, c) for r, c in floor_cells if c <= 1 and (r, c) not in used]
        if not edge_floors:
            edge_floors = [(r, c) for r, c in floor_cells if (r, c) not in used]

        for i, color in enumerate(beam_colors):
            if edge_floors:
                cell = edge_floors.pop(0)
                used.add(cell)
            else:
                cell = _take()
            if cell is None:
                continue
            emitter = _make_emitter(color, i)
            emitter.set_position(cell[1] * CELL, cell[0] * CELL)
            level.add_sprite(emitter)
            self._emitter_dirs[f"emitter_{i}"] = RIGHT  # beams go right
            self._emitter_colors[f"emitter_{i}"] = color

        # Place receptors on right side
        self._receptor_colors = {}
        self._receptor_hit = {}
        right_floors = [(r, c) for r, c in floor_cells
                        if c >= cell_cols - 3 and (r, c) not in used]
        if not right_floors:
            right_floors = [(r, c) for r, c in floor_cells if (r, c) not in used]

        for i, color in enumerate(beam_colors):
            if right_floors:
                cell = right_floors.pop(0)
                used.add(cell)
            else:
                cell = _take()
            if cell is None:
                continue
            receptor = _make_receptor(color, i)
            receptor.set_position(cell[1] * CELL, cell[0] * CELL)
            level.add_sprite(receptor)
            self._receptor_colors[f"receptor_{i}"] = color
            self._receptor_hit[f"receptor_{i}"] = False

        self._num_receptors = len(self._receptor_colors)

        # Place mirrors
        self._mirror_orientations = {}
        self._mirror_color_filters = {}
        for i in range(num_mirrors):
            cell = _take()
            if cell is None:
                break
            orientation = rng.randint(0, 1)
            color_filter = None
            if use_color_filters and i < len(beam_colors) and rng.random() < 0.4:
                color_filter = beam_colors[i % len(beam_colors)]
            mirror = _make_mirror(orientation, i, color_filter)
            mirror.set_position(cell[1] * CELL, cell[0] * CELL)
            level.add_sprite(mirror)
            self._mirror_orientations[f"mirror_{i}"] = orientation
            self._mirror_color_filters[f"mirror_{i}"] = color_filter

        # Place exit door
        exit_cell = _take()
        self._exit_open = False
        if exit_cell:
            exit_sprite = _make_exit(False)
            exit_sprite.set_position(exit_cell[1] * CELL, exit_cell[0] * CELL)
            level.add_sprite(exit_sprite)

        # Initial beam trace and HUD
        self._trace_all_beams()
        self._render_hud()

    # ------------------------------------------------------------------
    # Beam tracing
    # ------------------------------------------------------------------

    def _trace_all_beams(self):
        """Trace all beams from emitters through mirrors. Update receptor hit status."""
        level = self.current_level

        # Remove old beam path sprites
        for s in list(level.get_sprites_by_tag("beam_path")):
            level.remove_sprite(s)

        # Reset receptor hits
        for key in self._receptor_hit:
            self._receptor_hit[key] = False

        # Build a spatial map: cell (cx, cy) -> sprite name for mirrors/receptors/walls
        sprite_map: Dict[Tuple[int, int], Sprite] = {}
        for s in level._sprites:
            if s.is_visible and s.is_collidable:
                cx = s.x // CELL
                cy = s.y // CELL
                if "mirror" in s.tags or "receptor" in s.tags or "emitter" in s.tags:
                    sprite_map[(cx, cy)] = s
                elif s.name == "wall":
                    sprite_map[(cx, cy)] = s

        beam_idx = 0
        for ename, direction in self._emitter_dirs.items():
            emitters = level.get_sprites_by_name(ename)
            if not emitters:
                continue
            emitter = emitters[0]
            color = self._emitter_colors[ename]
            cx = emitter.x // CELL
            cy = emitter.y // CELL
            dx, dy = direction

            # Trace beam
            visited = set()
            max_steps = 200
            for _ in range(max_steps):
                cx += dx
                cy += dy

                # Out of bounds
                if cx < 0 or cx >= GRID // CELL or cy < 0 or cy >= 20:
                    break

                state_key = (cx, cy, dx, dy)
                if state_key in visited:
                    break  # loop detection
                visited.add(state_key)

                hit = sprite_map.get((cx, cy))
                if hit is not None:
                    if hit.name == "wall" or hit.name == "player":
                        break
                    if "emitter" in hit.tags:
                        break
                    if "receptor" in hit.tags:
                        # Check if beam color matches
                        rname = hit.name
                        if rname in self._receptor_colors:
                            if self._receptor_colors[rname] == color:
                                self._receptor_hit[rname] = True
                                # Light up the receptor
                                hit.pixels[0] = [RECEPTOR_LIT, RECEPTOR_LIT, RECEPTOR_LIT]
                                hit.pixels[2] = [RECEPTOR_LIT, RECEPTOR_LIT, RECEPTOR_LIT]
                        break
                    if "mirror" in hit.tags:
                        mname = hit.name
                        m_filter = self._mirror_color_filters.get(mname)
                        if m_filter is not None and m_filter != color:
                            # Mirror doesn't reflect this color — beam passes through
                            # Draw beam pixel and continue same direction
                            pass
                        else:
                            orientation = self._mirror_orientations[mname]
                            dx, dy = _reflect(dx, dy, orientation)
                        continue

                # Draw beam path pixel at this cell
                bpx = [[color]]
                bp = Sprite(pixels=bpx, name=f"beam_{beam_idx}",
                            visible=True, collidable=False, tags=["beam_path"])
                bp.set_position(cx * CELL + CELL // 2, cy * CELL + CELL // 2)
                level.add_sprite(bp)
                beam_idx += 1

        # Check if all receptors are hit -> open exit
        all_hit = all(self._receptor_hit.values()) and self._num_receptors > 0
        if all_hit != self._exit_open:
            self._exit_open = all_hit
            exits = level.get_sprites_by_tag("exit")
            if exits:
                old_exit = exits[0]
                ex, ey = old_exit.x, old_exit.y
                level.remove_sprite(old_exit)
                new_exit = _make_exit(all_hit)
                new_exit.set_position(ex, ey)
                level.add_sprite(new_exit)

    # ------------------------------------------------------------------
    # HUD
    # ------------------------------------------------------------------

    def _render_hud(self):
        level = self.current_level

        for s in list(level.get_sprites_by_tag("hud")):
            level.remove_sprite(s)

        # HUD background
        hud_bg = [[HUD_BG] * GRID for _ in range(4)]
        bg_sprite = Sprite(pixels=hud_bg, name="hud_bg", visible=True,
                           collidable=False, tags=["hud"])
        bg_sprite.set_position(0, 60)
        level.add_sprite(bg_sprite)

        # Move counter bar on row 62
        if self._max_moves > 0:
            fill_width = max(0, int(50 * self._moves_left / self._max_moves))
        else:
            fill_width = 0

        border_px = [[HUD_BORDER] * 52]
        border = Sprite(pixels=border_px, name="move_border", visible=True,
                        collidable=False, tags=["hud"])
        border.set_position(6, 62)
        level.add_sprite(border)

        if fill_width > 0:
            # Color shifts from green to red as moves deplete
            bar_color = EXIT_OPEN if self._moves_left > self._max_moves // 4 else RED
            fill_px = [[bar_color] * fill_width]
            fill = Sprite(pixels=fill_px, name="move_fill", visible=True,
                          collidable=False, tags=["hud"])
            fill.set_position(7, 62)
            level.add_sprite(fill)

        # Receptor status dots on row 63
        for i, (rname, is_hit) in enumerate(self._receptor_hit.items()):
            color = self._receptor_colors[rname] if is_hit else DIMMED
            dot_px = [[color, color]]
            dot = Sprite(pixels=dot_px, name=f"rdot_{i}", visible=True,
                         collidable=False, tags=["hud"])
            dot.set_position(6 + i * 5, 63)
            level.add_sprite(dot)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_player(self) -> Optional[Sprite]:
        sprites = self.current_level.get_sprites_by_name("player")
        return sprites[0] if sprites else None

    def _find_adjacent_mirror(self, player: Sprite) -> Optional[str]:
        """Find a mirror sprite adjacent to the player (within 1 cell)."""
        px, py = player.x // CELL, player.y // CELL
        for s in self.current_level.get_sprites_by_tag("mirror"):
            mx, my = s.x // CELL, s.y // CELL
            if abs(mx - px) + abs(my - py) == 1:
                return s.name
        return None

    # ------------------------------------------------------------------
    # Game loop
    # ------------------------------------------------------------------

    def step(self):
        player = self._get_player()
        if player is None:
            self.complete_action()
            return

        action = self.action.id

        if action == GameAction.ACTION5:
            # Rotate adjacent mirror
            mirror_name = self._find_adjacent_mirror(player)
            if mirror_name is not None:
                old_orient = self._mirror_orientations[mirror_name]
                new_orient = 1 - old_orient
                self._mirror_orientations[mirror_name] = new_orient

                # Update mirror sprite visually
                mirrors = self.current_level.get_sprites_by_name(mirror_name)
                if mirrors:
                    old_mirror = mirrors[0]
                    mx, my = old_mirror.x, old_mirror.y
                    self.current_level.remove_sprite(old_mirror)

                    idx = int(mirror_name.split("_")[1])
                    cf = self._mirror_color_filters.get(mirror_name)
                    new_mirror = _make_mirror(new_orient, idx, cf)
                    new_mirror.set_position(mx, my)
                    self.current_level.add_sprite(new_mirror)

                # Re-trace beams
                self._trace_all_beams()
                self._render_hud()

                # Uses a move
                self._moves_left -= 1
                if self._moves_left <= 0:
                    self.lose()

            self.complete_action()
            return

        # Movement
        dx, dy = 0, 0
        if action == GameAction.ACTION1:
            dy = -MOVE
        elif action == GameAction.ACTION2:
            dy = MOVE
        elif action == GameAction.ACTION3:
            dx = -MOVE
        elif action == GameAction.ACTION4:
            dx = MOVE
        else:
            self.complete_action()
            return

        collided = self.try_move("player", dx, dy)

        if not collided:
            # Successful move
            self._moves_left -= 1

            # Check if player stepped on open exit
            if self._exit_open:
                exits = self.current_level.get_sprites_by_tag("exit")
                if exits:
                    ex_sprite = exits[0]
                    if player.x == ex_sprite.x and player.y == ex_sprite.y:
                        self.next_level()
                        self.complete_action()
                        return

            if self._moves_left <= 0:
                self.lose()
                self.complete_action()
                return

            # Re-trace beams (player position blocks beams)
            self._trace_all_beams()

        self._render_hud()
        self.complete_action()
