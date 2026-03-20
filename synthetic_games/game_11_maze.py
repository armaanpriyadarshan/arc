"""
Game 11: Maze

Mechanics: Pure navigation through a U-shaped corridor.

The player starts at the top-left end of a U-shaped path and must travel
down, across, and back up to reach the goal at the top-right end. There
are no other mechanics — no items, enemies, fuel, or doors. This is a
minimal test of spatial navigation and path-following.

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
MOVE = 3          # movement step size

BG_COLOR = 0
WALL_COLOR = 1
PLAYER_COLOR = 2
GOAL_COLOR = 3
FLOOR_COLOR = 5

# Path layout in cell coordinates (each cell = 3 pixels).
# The U shape: left arm down, bottom across, right arm up.
#
# Cell grid (playfield rows 0-17, cols 0-17):
#   P . . . . . . . . . . . . . . G   (row 1)
#   | . . . . . . . . . . . . . . |
#   | . . . . . . . . . . . . . . |
#   | . . . . . . . . . . . . . . |
#   +------------------------------+   (row 16)
#
# Left arm:  col 1,     rows 1-16
# Bottom:    cols 2-15,  row 16
# Right arm: col 16,    rows 1-16

PATH_LEFT_COL = 1
PATH_RIGHT_COL = 16
PATH_TOP_ROW = 1
PATH_BOTTOM_ROW = 16

PLAYER_START = (PATH_LEFT_COL, PATH_TOP_ROW)      # cell coords
GOAL_POS = (PATH_RIGHT_COL, PATH_TOP_ROW)          # cell coords


def _cell_to_px(cx: int, cy: int) -> tuple[int, int]:
    """Convert cell coordinate to top-left pixel coordinate."""
    return cx * CELL, cy * CELL


def _is_path_cell(cx: int, cy: int) -> bool:
    """Return True if the cell coordinate is on the walkable U-path."""
    # Left arm
    if cx == PATH_LEFT_COL and PATH_TOP_ROW <= cy <= PATH_BOTTOM_ROW:
        return True
    # Bottom connector
    if cy == PATH_BOTTOM_ROW and PATH_LEFT_COL <= cx <= PATH_RIGHT_COL:
        return True
    # Right arm
    if cx == PATH_RIGHT_COL and PATH_TOP_ROW <= cy <= PATH_BOTTOM_ROW:
        return True
    return False


def _make_floor_sprite(w_cells: int, h_cells: int, name: str) -> Sprite:
    """Create a floor rectangle sprite sized in cells."""
    pw, ph = w_cells * CELL, h_cells * CELL
    px = [[FLOOR_COLOR] * pw for _ in range(ph)]
    return Sprite(pixels=px, name=name, visible=True, collidable=False, layer=0)


def _make_wall_sprite(w_cells: int, h_cells: int, name: str) -> Sprite:
    """Create a wall rectangle sprite sized in cells."""
    pw, ph = w_cells * CELL, h_cells * CELL
    px = [[WALL_COLOR] * pw for _ in range(ph)]
    return Sprite(pixels=px, name=name, visible=True, collidable=False, layer=0)


def _make_entity(color: int, name: str, tags: list[str] | None = None) -> Sprite:
    """Create a 3x3 entity sprite."""
    px = [[color] * CELL for _ in range(CELL)]
    return Sprite(pixels=px, name=name, visible=True, collidable=False,
                  tags=tags or [], layer=2)


class MazeGame(ARCBaseGame):
    """Game 11: Maze — navigate a U-shaped corridor to reach the goal."""

    def __init__(self, seed: int = 0):
        self._seed = seed

        levels = [Level(sprites=[], grid_size=(GRID, GRID))]
        camera = Camera(0, 0, GRID, GRID, BG_COLOR, BG_COLOR)
        super().__init__(
            game_id="game_11_maze",
            levels=levels,
            camera=camera,
            available_actions=[1, 2, 3, 4],
            win_score=1,
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        # Clear any existing sprites
        for s in list(level._sprites):
            level.remove_sprite(s)

        # --- Walls (border around the U-path) ---
        # Top border: full width
        s = _make_wall_sprite(18, 1, "wall_top")
        s.set_position(0, 0)
        level.add_sprite(s)

        # Bottom border
        s = _make_wall_sprite(18, 1, "wall_bottom")
        s.set_position(0, 17 * CELL)
        level.add_sprite(s)

        # Left border
        s = _make_wall_sprite(1, 16, "wall_left")
        s.set_position(0, 1 * CELL)
        level.add_sprite(s)

        # Right border
        s = _make_wall_sprite(1, 16, "wall_right")
        s.set_position(17 * CELL, 1 * CELL)
        level.add_sprite(s)

        # Center block (the interior of the U)
        s = _make_wall_sprite(14, 15, "wall_center")
        s.set_position(2 * CELL, 1 * CELL)
        level.add_sprite(s)

        # --- Floor (the U-path) ---
        # Left arm: col 1, rows 1-16
        s = _make_floor_sprite(1, 16, "floor_left")
        s.set_position(*_cell_to_px(PATH_LEFT_COL, PATH_TOP_ROW))
        level.add_sprite(s)

        # Bottom connector: cols 2-15, row 16
        s = _make_floor_sprite(14, 1, "floor_bottom")
        s.set_position(*_cell_to_px(2, PATH_BOTTOM_ROW))
        level.add_sprite(s)

        # Right arm: col 16, rows 1-16
        s = _make_floor_sprite(1, 16, "floor_right")
        s.set_position(*_cell_to_px(PATH_RIGHT_COL, PATH_TOP_ROW))
        level.add_sprite(s)

        # --- Goal ---
        goal = _make_entity(GOAL_COLOR, "goal", tags=["goal"])
        goal.set_position(*_cell_to_px(*GOAL_POS))
        level.add_sprite(goal)

        # --- Player ---
        player = _make_entity(PLAYER_COLOR, "player", tags=["player"])
        player.set_position(*_cell_to_px(*PLAYER_START))
        level.add_sprite(player)

        # Cache references
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
