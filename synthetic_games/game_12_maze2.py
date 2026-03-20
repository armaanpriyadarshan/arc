"""
Game 12: Maze2

Mechanics: Pure navigation through a trident-shaped corridor.

A shorter U-shaped path (half the length of Maze1) with an additional
center prong that is a dead end. The overall shape resembles a trident:
three vertical prongs connected by a horizontal bar at the bottom.
The player starts at the top of the left prong; the goal is at the top
of the right prong. The center prong is a dead-end distractor.

Actions:
    ACTION1 = Move Up (W)
    ACTION2 = Move Down (S)
    ACTION3 = Move Left (A)
    ACTION4 = Move Right (D)

Win condition: Player reaches the goal marker.

Color key:
    0  = background (void / off-limits)
    1  = wall
    5  = floor (walkable path)
    2  = player
    3  = goal
"""

from arcengine import (
    ARCBaseGame,
    Camera,
    Level,
    Sprite,
)

GRID = 64
CELL = 3          # player & objects occupy 3x3 pixels

BG_COLOR = 0
WALL_COLOR = 1
PLAYER_COLOR = 2
GOAL_COLOR = 3
FLOOR_COLOR = 5

# Trident layout in cell coordinates (each cell = 3 pixels).
#
#      4  5  6  7  8  9 10 11 12 13 14
#  4:  W  W  W  W  W  W  W  W  W  W  W
#  5:  W  P  W  W  W  .  W  W  W  G  W
#  6:  W  |  W  W  W  |  W  W  W  |  W
#  7:  W  |  W  W  W  |  W  W  W  |  W
#  8:  W  |  W  W  W  |  W  W  W  |  W
#  9:  W  |  W  W  W  |  W  W  W  |  W
# 10:  W  |  W  W  W  |  W  W  W  |  W
# 11:  W  |  W  W  W  |  W  W  W  |  W
# 12:  W  |  W  W  W  |  W  W  W  |  W
# 13:  W  +--+--+--+--+--+--+--+--+  W
# 14:  W  W  W  W  W  W  W  W  W  W  W
#
# Left arm:      col  5, rows 5-13  (player starts here)
# Bottom bar:    cols 5-13, row 13
# Right arm:     col 13, rows 5-13  (goal here)
# Center prong:  col  9, rows 5-12  (dead end)

PATH_LEFT_COL = 5
PATH_RIGHT_COL = 13
PATH_CENTER_COL = 9
PATH_TOP_ROW = 5
PATH_BOTTOM_ROW = 13
PATH_CENTER_TOP = 5      # center prong goes up to here (dead end)
PATH_CENTER_BOTTOM = 12  # center prong bottom (connects to bar at row 13)

PLAYER_START = (PATH_LEFT_COL, PATH_TOP_ROW)
GOAL_POS = (PATH_RIGHT_COL, PATH_TOP_ROW)


def _cell_to_px(cx: int, cy: int) -> tuple[int, int]:
    """Convert cell coordinate to top-left pixel coordinate."""
    return cx * CELL, cy * CELL


def _is_path_cell(cx: int, cy: int) -> bool:
    """Return True if the cell coordinate is on the walkable trident path."""
    # Left arm
    if cx == PATH_LEFT_COL and PATH_TOP_ROW <= cy <= PATH_BOTTOM_ROW:
        return True
    # Bottom connector
    if cy == PATH_BOTTOM_ROW and PATH_LEFT_COL <= cx <= PATH_RIGHT_COL:
        return True
    # Right arm
    if cx == PATH_RIGHT_COL and PATH_TOP_ROW <= cy <= PATH_BOTTOM_ROW:
        return True
    # Center prong (dead end)
    if cx == PATH_CENTER_COL and PATH_CENTER_TOP <= cy <= PATH_CENTER_BOTTOM:
        return True
    return False


def _make_floor_sprite(w_cells: int, h_cells: int, name: str) -> Sprite:
    pw, ph = w_cells * CELL, h_cells * CELL
    px = [[FLOOR_COLOR] * pw for _ in range(ph)]
    return Sprite(pixels=px, name=name, visible=True, collidable=False, layer=0)


def _make_wall_sprite(w_cells: int, h_cells: int, name: str) -> Sprite:
    pw, ph = w_cells * CELL, h_cells * CELL
    px = [[WALL_COLOR] * pw for _ in range(ph)]
    return Sprite(pixels=px, name=name, visible=True, collidable=False, layer=0)


def _make_entity(color: int, name: str, tags: list[str] | None = None) -> Sprite:
    px = [[color] * CELL for _ in range(CELL)]
    return Sprite(pixels=px, name=name, visible=True, collidable=False,
                  tags=tags or [], layer=2)


class Maze2Game(ARCBaseGame):
    """Game 12: Maze2 — navigate a trident-shaped corridor to reach the goal."""

    def __init__(self, seed: int = 0):
        self._seed = seed

        levels = [Level(sprites=[], grid_size=(GRID, GRID))]
        camera = Camera(0, 0, GRID, GRID, BG_COLOR, BG_COLOR)
        super().__init__(
            game_id="game_12_maze2",
            levels=levels,
            camera=camera,
            available_actions=[1, 2, 3, 4],
            win_score=1,
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        for s in list(level._sprites):
            level.remove_sprite(s)

        # --- Outer walls ---
        # Top border: cols 4-14, row 4  (11 wide)
        s = _make_wall_sprite(11, 1, "wall_top")
        s.set_position(*_cell_to_px(4, 4))
        level.add_sprite(s)

        # Bottom border: cols 4-14, row 14
        s = _make_wall_sprite(11, 1, "wall_bottom")
        s.set_position(*_cell_to_px(4, 14))
        level.add_sprite(s)

        # Left border: col 4, rows 5-13  (1 wide, 9 tall)
        s = _make_wall_sprite(1, 9, "wall_left")
        s.set_position(*_cell_to_px(4, 5))
        level.add_sprite(s)

        # Right border: col 14, rows 5-13
        s = _make_wall_sprite(1, 9, "wall_right")
        s.set_position(*_cell_to_px(14, 5))
        level.add_sprite(s)

        # --- Interior walls (between prongs) ---
        # Between left arm and center prong: cols 6-8, rows 5-12
        s = _make_wall_sprite(3, 8, "wall_inner_left")
        s.set_position(*_cell_to_px(6, 5))
        level.add_sprite(s)

        # Between center prong and right arm: cols 10-12, rows 5-12
        s = _make_wall_sprite(3, 8, "wall_inner_right")
        s.set_position(*_cell_to_px(10, 5))
        level.add_sprite(s)

        # --- Floor (trident path) ---
        # Left arm: col 5, rows 5-13
        s = _make_floor_sprite(1, 9, "floor_left")
        s.set_position(*_cell_to_px(PATH_LEFT_COL, PATH_TOP_ROW))
        level.add_sprite(s)

        # Right arm: col 13, rows 5-13
        s = _make_floor_sprite(1, 9, "floor_right")
        s.set_position(*_cell_to_px(PATH_RIGHT_COL, PATH_TOP_ROW))
        level.add_sprite(s)

        # Center prong: col 9, rows 5-12 (dead end)
        s = _make_floor_sprite(1, 8, "floor_center")
        s.set_position(*_cell_to_px(PATH_CENTER_COL, PATH_CENTER_TOP))
        level.add_sprite(s)

        # Bottom connector: cols 5-13, row 13
        s = _make_floor_sprite(9, 1, "floor_bottom")
        s.set_position(*_cell_to_px(PATH_LEFT_COL, PATH_BOTTOM_ROW))
        level.add_sprite(s)

        # --- Goal ---
        goal = _make_entity(GOAL_COLOR, "goal", tags=["goal"])
        goal.set_position(*_cell_to_px(*GOAL_POS))
        level.add_sprite(goal)

        # --- Player ---
        player = _make_entity(PLAYER_COLOR, "player", tags=["player"])
        player.set_position(*_cell_to_px(*PLAYER_START))
        level.add_sprite(player)

        self._player = player
        self._goal = goal
        self._player_cx, self._player_cy = PLAYER_START

    def step(self) -> None:
        dx, dy = 0, 0
        if self.action.id.value == 1:
            dy = -1
        elif self.action.id.value == 2:
            dy = 1
        elif self.action.id.value == 3:
            dx = -1
        elif self.action.id.value == 4:
            dx = 1
        else:
            self.complete_action()
            return

        ncx = self._player_cx + dx
        ncy = self._player_cy + dy

        if _is_path_cell(ncx, ncy):
            self._player_cx = ncx
            self._player_cy = ncy
            self._player.set_position(*_cell_to_px(ncx, ncy))

            if (ncx, ncy) == GOAL_POS:
                self.next_level()

        self.complete_action()
