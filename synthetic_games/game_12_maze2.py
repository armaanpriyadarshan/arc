"""
Game 12: Maze2

Mechanics: Pure navigation through a corridor grid.

A horizontal hallway with three vertical prongs jutting upward and three
downward, connected in the middle. The shape looks like a horizontal line
with three half-lines branching off on each side. The player starts at
the top of the top-left prong; the goal is at the bottom of the
bottom-right prong.

Actions:
    ACTION1 = Move Up (W)
    ACTION2 = Move Down (S)
    ACTION3 = Move Left (A)
    ACTION4 = Move Right (D)

Win condition: Player reaches the goal marker.

Color key:
    1  = background (light grey)
    4  = wall (dark grey)
    3  = floor / walkable path (mid grey)
    12 = player (orange)
    14 = goal (green)
"""

from arcengine import (
    ARCBaseGame,
    Camera,
    Level,
    Sprite,
)

GRID = 64
CELL = 3          # player & objects occupy 3x3 pixels

BG_COLOR = 1       # light grey (204, 204, 204)
WALL_COLOR = 4     # dark grey  (51, 51, 51)
PLAYER_COLOR = 12  # orange     (255, 133, 27)
GOAL_COLOR = 14    # green      (79, 204, 48)
FLOOR_COLOR = 3    # mid grey   (102, 102, 102)

# Layout in cell coordinates (each cell = 3 pixels).
#
#      4  5  6  7  8  9 10 11 12 13 14 15 16 17 18
#  1:     W  W  W           W  W  W           W  W  W
#  2:     W  P  W           W  .  W           W  .  W
#  3:     W  .  W           W  .  W           W  .  W
#  4:     W  .  W           W  .  W           W  .  W
#  5:     W  .  W           W  .  W           W  .  W
#  6:     W  .  W           W  .  W           W  .  W
#  7:     W  .  W           W  .  W           W  .  W
#  8:     W  .  W           W  .  W           W  .  W
#  9:  W  W  .  W  W  W  W  W  .  W  W  W  W  W  .  W  W
# 10:  W  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  W
# 11:  W  W  .  W  W  W  W  W  .  W  W  W  W  W  .  W  W
# 12:     W  .  W           W  .  W           W  .  W
# 13:     W  .  W           W  .  W           W  .  W
# 14:     W  .  W           W  .  W           W  .  W
# 15:     W  .  W           W  .  W           W  .  W
# 16:     W  .  W           W  .  W           W  .  W
# 17:     W  .  W           W  .  W           W  .  W
# 18:     W  .  W           W  .  W           W  G  W
# 19:     W  W  W           W  W  W           W  W  W

# Three prong columns
PRONG_COLS = [6, 11, 16]

# Horizontal bar
BAR_ROW = 10
BAR_COL_START = 5
BAR_COL_END = 17

# Vertical prong ranges (exclusive of bar row)
TOP_PRONG_ROW_START = 2
TOP_PRONG_ROW_END = 9
BOTTOM_PRONG_ROW_START = 11
BOTTOM_PRONG_ROW_END = 18

PLAYER_START = (PRONG_COLS[0], TOP_PRONG_ROW_START)    # (6, 2)
GOAL_POS = (PRONG_COLS[2], BOTTOM_PRONG_ROW_END)       # (16, 18)


def _cell_to_px(cx: int, cy: int) -> tuple[int, int]:
    """Convert cell coordinate to top-left pixel coordinate."""
    return cx * CELL, cy * CELL


def _is_path_cell(cx: int, cy: int) -> bool:
    """Return True if the cell coordinate is on the walkable path."""
    # Horizontal bar
    if cy == BAR_ROW and BAR_COL_START <= cx <= BAR_COL_END:
        return True
    # Top prongs
    if cx in PRONG_COLS and TOP_PRONG_ROW_START <= cy <= TOP_PRONG_ROW_END:
        return True
    # Bottom prongs
    if cx in PRONG_COLS and BOTTOM_PRONG_ROW_START <= cy <= BOTTOM_PRONG_ROW_END:
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
    """Game 12: Maze2 — navigate a corridor grid to reach the goal."""

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

        # --- Floor (walkable path) ---
        # Horizontal bar: cols 5-17, row 10  (13 wide, 1 tall)
        s = _make_floor_sprite(13, 1, "floor_bar")
        s.set_position(*_cell_to_px(BAR_COL_START, BAR_ROW))
        level.add_sprite(s)

        # Top & bottom prongs
        for i, c in enumerate(PRONG_COLS):
            label = ["left", "center", "right"][i]
            # Top prong: rows 2-9  (8 tall)
            s = _make_floor_sprite(1, 8, f"floor_top_{label}")
            s.set_position(*_cell_to_px(c, TOP_PRONG_ROW_START))
            level.add_sprite(s)
            # Bottom prong: rows 11-18  (8 tall)
            s = _make_floor_sprite(1, 8, f"floor_bot_{label}")
            s.set_position(*_cell_to_px(c, BOTTOM_PRONG_ROW_START))
            level.add_sprite(s)

        # --- Walls ---
        # Walls around each top prong (col c, rows 1-9)
        for i, c in enumerate(PRONG_COLS):
            label = ["left", "center", "right"][i]
            # Left wall: col c-1, rows 1-9  (9 tall)
            s = _make_wall_sprite(1, 9, f"tw_l_{label}")
            s.set_position(*_cell_to_px(c - 1, 1))
            level.add_sprite(s)
            # Right wall: col c+1, rows 1-9  (9 tall)
            s = _make_wall_sprite(1, 9, f"tw_r_{label}")
            s.set_position(*_cell_to_px(c + 1, 1))
            level.add_sprite(s)
            # Top cap: col c, row 1  (1x1)
            s = _make_wall_sprite(1, 1, f"tw_cap_{label}")
            s.set_position(*_cell_to_px(c, 1))
            level.add_sprite(s)

        # Walls around each bottom prong (col c, rows 11-19)
        for i, c in enumerate(PRONG_COLS):
            label = ["left", "center", "right"][i]
            # Left wall: col c-1, rows 11-19  (9 tall)
            s = _make_wall_sprite(1, 9, f"bw_l_{label}")
            s.set_position(*_cell_to_px(c - 1, 11))
            level.add_sprite(s)
            # Right wall: col c+1, rows 11-19  (9 tall)
            s = _make_wall_sprite(1, 9, f"bw_r_{label}")
            s.set_position(*_cell_to_px(c + 1, 11))
            level.add_sprite(s)
            # Bottom cap: col c, row 19  (1x1)
            s = _make_wall_sprite(1, 1, f"bw_cap_{label}")
            s.set_position(*_cell_to_px(c, 19))
            level.add_sprite(s)

        # Bar end walls
        # Left end: col 4, rows 9-11  (1 wide, 3 tall)
        s = _make_wall_sprite(1, 3, "bar_end_left")
        s.set_position(*_cell_to_px(4, 9))
        level.add_sprite(s)
        # Right end: col 18, rows 9-11  (1 wide, 3 tall)
        s = _make_wall_sprite(1, 3, "bar_end_right")
        s.set_position(*_cell_to_px(18, 9))
        level.add_sprite(s)

        # Bar gap walls (fill between prong walls on rows 9 and 11)
        # Row 9: cols 8-9 and cols 13-14
        s = _make_wall_sprite(2, 1, "bar_gap_top_1")
        s.set_position(*_cell_to_px(8, 9))
        level.add_sprite(s)
        s = _make_wall_sprite(2, 1, "bar_gap_top_2")
        s.set_position(*_cell_to_px(13, 9))
        level.add_sprite(s)
        # Row 11: cols 8-9 and cols 13-14
        s = _make_wall_sprite(2, 1, "bar_gap_bot_1")
        s.set_position(*_cell_to_px(8, 11))
        level.add_sprite(s)
        s = _make_wall_sprite(2, 1, "bar_gap_bot_2")
        s.set_position(*_cell_to_px(13, 11))
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
