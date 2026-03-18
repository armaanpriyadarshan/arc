"""
Game 04: Mirror Maze

Mechanics: Navigation + beam reflection + mirror rotation + beam-activated doors + limited moves

An open grid with beam emitters on one edge and receptors on another. Mirrors
are scattered on the grid. The player walks around and presses ACTION5 when
adjacent to a mirror to rotate it (flips between backslash and forward-slash).
Beams travel from emitters, reflect off mirrors, and must hit matching-color
receptors. When all receptors are lit, the exit door opens. Walk onto it to
advance. Moves are limited — running out resets the level.

Later levels have multiple beams of different colors and mirrors that only
reflect certain colors (indicated by a colored center pixel).

Actions:
    ACTION1 = Move Up (W)
    ACTION2 = Move Down (S)
    ACTION3 = Move Left (A)
    ACTION4 = Move Right (D)
    ACTION5 = Rotate nearest adjacent mirror 90 degrees

Display:
    Row 62: move counter (filled bar = remaining moves)
    Row 63: receptor status dots (colored = hit, gray = not hit)

Win condition: All beams hitting their matching receptors, then walk onto exit.

Color key:
    0  = empty / floor
    1  = wall (border only)
    2  = player
    3  = mirror (backslash orientation)
    4  = mirror (forward-slash orientation)
    5  = HUD background
    6  = blue (beam / emitter / receptor)
    7  = yellow (beam / emitter / receptor)
    8  = red (beam / emitter / receptor)
    9  = purple (beam / emitter / receptor)
    10 = green (exit door open)
    11 = exit door closed
    12 = dimmed / inactive indicator
    14 = receptor border (lit)
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
    RenderableUserDisplay,
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
MIRROR_BS = 3
MIRROR_FS = 4
HUD_BG = 5
BLUE = 6
YELLOW = 7
RED = 8
PURPLE = 9
EXIT_OPEN = 10
EXIT_CLOSED = 11
DIMMED = 12
RECEPTOR_LIT = 14
HUD_BORDER = 15

BEAM_COLORS = [RED, BLUE, YELLOW, PURPLE]

# Playfield: 20 columns x 20 rows of cells (60x60 pixels, rows 0-59)
PLAY_COLS = 20
PLAY_ROWS = 20


# ---------------------------------------------------------------------------
# Reflection
# ---------------------------------------------------------------------------

def _reflect(dx: int, dy: int, orientation: int) -> Tuple[int, int]:
    """Reflect beam direction off mirror. 0=backslash, 1=forward-slash."""
    if orientation == 0:  # backslash: swap dx,dy
        return (dy, dx)
    else:  # forward-slash: swap and negate
        return (-dy, -dx)


# ---------------------------------------------------------------------------
# Sprite builders
# ---------------------------------------------------------------------------

def _make_player() -> Sprite:
    px = [[PLAYER_COLOR] * CELL for _ in range(CELL)]
    return Sprite(pixels=px, name="player", visible=True, collidable=True)


def _make_wall() -> Sprite:
    px = [[WALL] * CELL for _ in range(CELL)]
    return Sprite(pixels=px, name="wall", visible=True, collidable=True)


def _make_mirror(orientation: int, idx: int, color_filter: Optional[int] = None) -> Sprite:
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
    if color_filter is not None:
        px[1][1] = color_filter
    return Sprite(pixels=px, name=f"mirror_{idx}", visible=True, collidable=True,
                  tags=["mirror"])


def _make_emitter(color: int, idx: int) -> Sprite:
    """Emitter: colored border with black center, indicates beam source."""
    px = [
        [color, color, color],
        [color, BG, color],
        [color, color, color],
    ]
    return Sprite(pixels=px, name=f"emitter_{idx}", visible=True, collidable=True,
                  tags=["emitter"])


def _make_receptor(color: int, idx: int) -> Sprite:
    """Receptor: dimmed border with colored center showing required color."""
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
# Hand-crafted level layouts (open grids, no maze)
# ---------------------------------------------------------------------------
# Each level: emitters, receptors, mirrors, player_pos, exit_pos, max_moves
# Positions in cell coords (col, row). Emitters shoot RIGHT by default.
# Beams are meant to be solvable by rotating the right mirrors.

def _level_configs():
    """Hand-crafted levels with verified beam paths.

    Reflection rules:
      \\ (orient 0): RIGHT->DOWN, DOWN->RIGHT, LEFT->UP, UP->LEFT
      /  (orient 1): RIGHT->UP,   UP->RIGHT,   LEFT->DOWN, DOWN->LEFT

    Each mirror's start orientation is set so the puzzle requires rotating
    some mirrors to solve. The "needs" comment shows the correct orientation.
    """
    return [
        # Level 0: 1 beam, 2 mirrors — tutorial
        # Path: (1,4) RIGHT -> (10,4) \\ -> DOWN -> (10,12) \\ -> RIGHT -> receptor (17,12)
        # Solution: rotate mirror 0 from / to \\
        {
            "emitters": [(1, 4, RED, (1, 0))],
            "receptors": [(17, 12, RED)],
            "mirrors": [
                (10, 4, 1, None),    # starts /, needs \\ — player rotates this
                (10, 12, 0, None),   # starts \\ (correct)
            ],
            "player": (5, 8),
            "exit": (3, 16),
            "max_moves": 50,
        },
        # Level 1: 1 beam, 3 mirrors — 3 bounces
        # Path: (1,3) RIGHT -> (8,3) \\ -> DOWN -> (8,10) \\ -> RIGHT -> (15,10) \\ -> DOWN -> receptor (15,16)
        # Solution: rotate mirrors 0, 1, 2 from / to \\
        {
            "emitters": [(1, 3, RED, (1, 0))],
            "receptors": [(15, 16, RED)],
            "mirrors": [
                (8, 3, 1, None),     # starts /, needs \\
                (8, 10, 1, None),    # starts /, needs \\
                (15, 10, 1, None),   # starts /, needs \\
            ],
            "player": (4, 13),
            "exit": (2, 17),
            "max_moves": 50,
        },
        # Level 2: 2 beams, 4 mirrors — two separate paths
        # Red:  (1,3) RIGHT -> (8,3) \\ -> DOWN -> (8,9) \\ -> RIGHT -> receptor (16,9)
        # Blue: (1,13) RIGHT -> (8,13) \\ -> DOWN -> (8,17) \\ -> RIGHT -> receptor (16,17)
        # Solution: rotate mirrors 0,2 from / to \\
        {
            "emitters": [
                (1, 3, RED, (1, 0)),
                (1, 13, BLUE, (1, 0)),
            ],
            "receptors": [
                (16, 9, RED),
                (16, 17, BLUE),
            ],
            "mirrors": [
                (8, 3, 1, None),     # red path, starts /, needs \\
                (8, 9, 0, None),     # red path, starts \\ (correct)
                (8, 13, 1, None),    # blue path, starts /, needs \\
                (8, 17, 0, None),    # blue path, starts \\ (correct)
            ],
            "player": (4, 11),
            "exit": (2, 18),
            "max_moves": 50,
        },
        # Level 3: 2 beams, 5 mirrors (1 color-filtered)
        # Red:  (1,4) RIGHT -> (7,4) \\ -> DOWN -> (7,11) \\ -> RIGHT -> receptor (17,11)
        # Blue: (1,15) RIGHT -> (12,15) / -> UP -> (12,8) / -> RIGHT -> receptor (17,8)
        # Separate columns (7 vs 12) so beams don't cross each other's mirrors
        # Mirror 4 at (15,6) is RED-only filter (decoy)
        # Solution: rotate mirrors 0,1 to \\; rotate mirrors 2,3 to /
        {
            "emitters": [
                (1, 4, RED, (1, 0)),
                (1, 15, BLUE, (1, 0)),
            ],
            "receptors": [
                (17, 11, RED),
                (17, 8, BLUE),
            ],
            "mirrors": [
                (7, 4, 1, None),     # red, starts /, needs \\
                (7, 11, 1, None),    # red, starts /, needs \\
                (12, 15, 0, None),   # blue, starts \\, needs /
                (12, 8, 0, None),    # blue, starts \\, needs /
                (15, 6, 1, RED),     # color-filtered decoy
            ],
            "player": (5, 10),
            "exit": (2, 18),
            "max_moves": 50,
        },
        # Level 4: 3 beams, 6 mirrors
        # Red:    (1,2) RIGHT -> (7,2) \\ -> DOWN -> (7,7) \\ -> RIGHT -> receptor (17,7)
        # Blue:   (1,10) RIGHT -> (11,10) \\ -> DOWN -> (11,14) \\ -> RIGHT -> receptor (17,14)
        # Yellow: (1,17) RIGHT -> (14,17) / -> UP -> (14,4) / -> RIGHT -> receptor (17,4)
        # Solution: rotate 0,1,2,3 to \\; rotate 4,5 to /
        {
            "emitters": [
                (1, 2, RED, (1, 0)),
                (1, 10, BLUE, (1, 0)),
                (1, 17, YELLOW, (1, 0)),
            ],
            "receptors": [
                (17, 7, RED),
                (17, 14, BLUE),
                (17, 4, YELLOW),
            ],
            "mirrors": [
                (7, 2, 1, None),     # red, starts /, needs \\
                (7, 7, 1, None),     # red, starts /, needs \\
                (11, 10, 1, None),   # blue, starts /, needs \\
                (11, 14, 1, None),   # blue, starts /, needs \\
                (14, 17, 0, None),   # yellow, starts \\, needs /
                (14, 4, 0, None),    # yellow, starts \\, needs /
            ],
            "player": (5, 18),
            "exit": (2, 18),
            "max_moves": 50,
        },
    ]


# ---------------------------------------------------------------------------
# Game class
# ---------------------------------------------------------------------------

class MirrorHUD(RenderableUserDisplay):
    """Writes move bar and receptor dots directly onto the frame buffer."""

    def __init__(self, game: "MirrorMazeGame"):
        self.game = game

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        # Clear HUD area
        frame[60:64, :] = HUD_BG

        # Move bar: 60px wide, drops 6px per 10% used
        g = self.game
        pct = g._moves_left / g._max_moves if g._max_moves > 0 else 0
        tens = int(pct * 10)  # 10=full, 0=empty
        fill_w = tens * 6
        bar_color = EXIT_OPEN if tens > 2 else RED
        for px in range(60):
            frame[62, 2 + px] = bar_color if px < fill_w else WALL

        # Receptor status dots on row 63
        for i, (rname, is_hit) in enumerate(g._receptor_hit.items()):
            color = g._receptor_colors[rname] if is_hit else DIMMED
            frame[63, 6 + i * 5] = color
            frame[63, 7 + i * 5] = color

        return frame


class MirrorMazeGame(ARCBaseGame):
    """Game 04: Mirror Maze — navigate open grid, rotate mirrors, direct beams."""

    def __init__(self, seed: int = 0):
        self._seed = seed
        self._cfgs = _level_configs()

        self._moves_left = 0
        self._max_moves = 0
        self._mirror_orientations: Dict[str, int] = {}
        self._mirror_color_filters: Dict[str, Optional[int]] = {}
        self._emitter_data: List[Tuple[int, int, int, Tuple[int, int]]] = []
        self._receptor_colors: Dict[str, int] = {}
        self._receptor_hit: Dict[str, bool] = {}
        self._exit_open = False
        self._num_receptors = 0
        self._hud = MirrorHUD(self)

        levels = [Level(sprites=[], grid_size=(GRID, GRID)) for _ in self._cfgs]
        camera = Camera(0, 0, GRID, GRID, BG, HUD_BG, [self._hud])
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
        cfg = self._cfgs[level_idx]

        self._max_moves = cfg["max_moves"]
        self._moves_left = cfg["max_moves"]
        self._exit_open = False

        # Border walls (top, bottom, left, right of play area)
        for c in range(PLAY_COLS):
            for r in [0, PLAY_ROWS - 1]:
                w = _make_wall()
                w.set_position(c * CELL, r * CELL)
                level.add_sprite(w)
        for r in range(PLAY_ROWS):
            for c in [0, PLAY_COLS - 1]:
                w = _make_wall()
                w.set_position(c * CELL, r * CELL)
                level.add_sprite(w)

        # Place emitters
        self._emitter_data = []
        for i, (cx, cy, color, direction) in enumerate(cfg["emitters"]):
            emitter = _make_emitter(color, i)
            emitter.set_position(cx * CELL, cy * CELL)
            level.add_sprite(emitter)
            self._emitter_data.append((cx, cy, color, direction))

        # Place receptors
        self._receptor_colors = {}
        self._receptor_hit = {}
        for i, (cx, cy, color) in enumerate(cfg["receptors"]):
            receptor = _make_receptor(color, i)
            receptor.set_position(cx * CELL, cy * CELL)
            level.add_sprite(receptor)
            self._receptor_colors[f"receptor_{i}"] = color
            self._receptor_hit[f"receptor_{i}"] = False
        self._num_receptors = len(self._receptor_colors)

        # Place mirrors
        self._mirror_orientations = {}
        self._mirror_color_filters = {}
        for i, (cx, cy, orient, cf) in enumerate(cfg["mirrors"]):
            mirror = _make_mirror(orient, i, cf)
            mirror.set_position(cx * CELL, cy * CELL)
            level.add_sprite(mirror)
            self._mirror_orientations[f"mirror_{i}"] = orient
            self._mirror_color_filters[f"mirror_{i}"] = cf

        # Place player
        px, py = cfg["player"]
        player = _make_player()
        player.set_position(px * CELL, py * CELL)
        level.add_sprite(player)

        # Place exit
        ex, ey = cfg["exit"]
        exit_sprite = _make_exit(False)
        exit_sprite.set_position(ex * CELL, ey * CELL)
        level.add_sprite(exit_sprite)

        self._trace_all_beams()
        # HUD auto-renders via MirrorHUD RenderableUserDisplay

    # ------------------------------------------------------------------
    # Beam tracing
    # ------------------------------------------------------------------

    def _trace_all_beams(self):
        """Trace beams from emitters through mirrors. Render beam paths as
        full CELL-wide colored strips so they're clearly visible."""
        level = self.current_level

        # Remove old beam sprites
        for s in list(level.get_sprites_by_tag("beam_path")):
            level.remove_sprite(s)

        # Reset receptor visuals and hit state
        for rname in self._receptor_hit:
            self._receptor_hit[rname] = False
        for s in level.get_sprites_by_tag("receptor"):
            color = self._receptor_colors.get(s.name, DIMMED)
            s.pixels[0] = [DIMMED, DIMMED, DIMMED]
            s.pixels[2] = [DIMMED, DIMMED, DIMMED]

        # Build spatial map of collidable objects
        sprite_map: Dict[Tuple[int, int], Sprite] = {}
        for s in level._sprites:
            if s.is_visible and s.is_collidable:
                cx, cy = s.x // CELL, s.y // CELL
                if any(tag in s.tags for tag in ["mirror", "receptor", "emitter"]) or s.name == "wall":
                    sprite_map[(cx, cy)] = s

        beam_idx = 0
        for i, (ecx, ecy, color, direction) in enumerate(self._emitter_data):
            cx, cy = ecx, ecy
            dx, dy = direction

            visited: Set[Tuple[int, int, int, int]] = set()
            for _ in range(100):
                cx += dx
                cy += dy

                # Out of bounds
                if cx < 0 or cx >= PLAY_COLS or cy < 0 or cy >= PLAY_ROWS:
                    break

                state_key = (cx, cy, dx, dy)
                if state_key in visited:
                    break
                visited.add(state_key)

                hit = sprite_map.get((cx, cy))
                if hit is not None:
                    if hit.name == "wall" or hit.name == "player":
                        break
                    if "emitter" in hit.tags:
                        break
                    if "receptor" in hit.tags:
                        rname = hit.name
                        if rname in self._receptor_colors and self._receptor_colors[rname] == color:
                            self._receptor_hit[rname] = True
                            hit.pixels[0] = [RECEPTOR_LIT, RECEPTOR_LIT, RECEPTOR_LIT]
                            hit.pixels[2] = [RECEPTOR_LIT, RECEPTOR_LIT, RECEPTOR_LIT]
                        break
                    if "mirror" in hit.tags:
                        mname = hit.name
                        m_filter = self._mirror_color_filters.get(mname)
                        if m_filter is not None and m_filter != color:
                            pass  # beam passes through
                        else:
                            dx, dy = _reflect(dx, dy, self._mirror_orientations[mname])
                        continue

                # Draw beam as full 3x1 or 1x3 strip depending on direction
                if dx != 0:  # horizontal beam
                    bpx = [[color], [color], [color]]
                else:  # vertical beam
                    bpx = [[color, color, color]]
                bp = Sprite(pixels=bpx, name=f"beam_{beam_idx}",
                            visible=True, collidable=False, tags=["beam_path"])
                bp.set_position(cx * CELL + (0 if dx != 0 else 0),
                                cy * CELL + (0 if dy != 0 else 0))
                level.add_sprite(bp)
                beam_idx += 1

        # Update exit door
        all_hit = all(self._receptor_hit.values()) and self._num_receptors > 0
        if all_hit != self._exit_open:
            self._exit_open = all_hit
            exits = level.get_sprites_by_tag("exit")
            if exits:
                old = exits[0]
                ex, ey = old.x, old.y
                level.remove_sprite(old)
                new_exit = _make_exit(all_hit)
                new_exit.set_position(ex, ey)
                level.add_sprite(new_exit)

    # ------------------------------------------------------------------
    # HUD
    # ------------------------------------------------------------------

    # HUD is rendered by MirrorHUD (RenderableUserDisplay) — writes directly
    # to frame buffer after sprites are rendered, so no sprite layering issues.

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_player(self) -> Optional[Sprite]:
        sprites = self.current_level.get_sprites_by_name("player")
        return sprites[0] if sprites else None

    def _find_adjacent_mirror(self, player: Sprite) -> Optional[str]:
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

        # RESET doesn't cost a move
        if action == GameAction.RESET:
            self.complete_action()
            return

        # Every other action costs a move (even blocked ones)
        self._moves_left -= 1

        # Check depletion immediately — reset and stop
        if self._moves_left <= 0:
            self.level_reset()
            self.complete_action()
            return

        if action == GameAction.ACTION5:
            mirror_name = self._find_adjacent_mirror(player)
            if mirror_name is not None:
                new_orient = 1 - self._mirror_orientations[mirror_name]
                self._mirror_orientations[mirror_name] = new_orient

                mirrors = self.current_level.get_sprites_by_name(mirror_name)
                if mirrors:
                    old = mirrors[0]
                    mx, my = old.x, old.y
                    self.current_level.remove_sprite(old)
                    idx = int(mirror_name.split("_")[1])
                    cf = self._mirror_color_filters.get(mirror_name)
                    new_m = _make_mirror(new_orient, idx, cf)
                    new_m.set_position(mx, my)
                    self.current_level.add_sprite(new_m)

                self._trace_all_beams()

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
            if self._exit_open:
                exits = self.current_level.get_sprites_by_tag("exit")
                if exits and player.x == exits[0].x and player.y == exits[0].y:
                    self.next_level()
                    self.complete_action()
                    return
            self._trace_all_beams()

        self.complete_action()
