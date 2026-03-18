"""
Game 10: Shapeshifter

Mechanics: Player transformation + ability-gating + environmental puzzles + state memory

The player navigates a grid and can shapeshift between 3 forms using ACTION5:
  - Small (1x1): fits through narrow 1-cell gaps
  - Medium (2x2): can push moveable blocks
  - Large (3x3): can break cracked walls, but can't fit in tight spaces

Each form has a different color. Some floor tiles are pressure plates that only
activate when the player stands on them at a specific size. Activating the right
combination of pressure plates opens doors. Some areas require being small to
enter, then shifting to large inside to break a cracked wall, then shifting to
small again to exit. Later levels add shift cooldown (can only transform every
N moves) and form-locked zones (areas where shifting is disabled).

Actions:
    ACTION1 = Move Up (W)
    ACTION2 = Move Down (S)
    ACTION3 = Move Left (A)
    ACTION4 = Move Right (D)
    ACTION5 = Shapeshift to next form (Space)

Display:
    Row 63: Current form indicator (color block) + shift cooldown counter
    Row 62: Shift cooldown bar

Win condition: Reach the exit door.

Color key:
    0  = empty / floor
    1  = wall
    2  = player (small form)
    3  = player (medium form)
    4  = player (large form)
    5  = exit door (closed)
    6  = exit door (open / green)
    7  = pressure plate (small-activated, cyan)
    8  = pressure plate (medium-activated, pink)
    9  = pressure plate (large-activated, purple)
    10 = cracked wall (brownish)
    11 = moveable block (orange)
    12 = form-locked zone floor (dark gray)
    13 = HUD background
    14 = cooldown bar fill (yellow)
    15 = HUD border
"""

import random
from typing import List, Optional, Tuple, Set

import numpy as np
from arcengine import (
    ARCBaseGame,
    Camera,
    Level,
    Sprite,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GRID = 64
CELL = 3          # base cell size in pixels
MOVE = 3          # movement step
BG_COLOR = 0
WALL_COLOR = 1

# Player form colors
SMALL_COLOR = 2
MEDIUM_COLOR = 3
LARGE_COLOR = 4

# Objects
EXIT_CLOSED = 5
EXIT_OPEN = 6
PLATE_SMALL = 7   # cyan — needs small form
PLATE_MEDIUM = 8  # pink — needs medium form
PLATE_LARGE = 9   # purple — needs large form
CRACKED_WALL = 10
MOVEABLE_BLOCK = 11
LOCKED_ZONE = 12
HUD_BG = 13
COOLDOWN_FILL = 14
HUD_BORDER = 15

# Form sizes in cells (each cell = CELL pixels)
FORM_SMALL = 0
FORM_MEDIUM = 1
FORM_LARGE = 2

FORM_PIXEL_SIZES = {
    FORM_SMALL: 1,   # 1x1 pixels (actually CELL x CELL but we use 1-cell)
    FORM_MEDIUM: 2,  # 2x2 cells
    FORM_LARGE: 3,   # 3x3 cells
}

FORM_COLORS = {
    FORM_SMALL: SMALL_COLOR,
    FORM_MEDIUM: MEDIUM_COLOR,
    FORM_LARGE: LARGE_COLOR,
}

PLATE_FORM_MAP = {
    PLATE_SMALL: FORM_SMALL,
    PLATE_MEDIUM: FORM_MEDIUM,
    PLATE_LARGE: FORM_LARGE,
}

HUD_ROW_COOLDOWN = 62
HUD_ROW_FORM = 63

# ---------------------------------------------------------------------------
# Sprite builders
# ---------------------------------------------------------------------------

def _make_player(form: int) -> Sprite:
    """Build a player sprite for the given form."""
    color = FORM_COLORS[form]
    size = FORM_PIXEL_SIZES[form] * CELL
    px = [[color] * size for _ in range(size)]
    return Sprite(pixels=px, name="player", visible=True, collidable=True)


def _make_wall(w_cells: int = 1, h_cells: int = 1) -> Sprite:
    """Wall sprite spanning w x h cells."""
    pw = w_cells * CELL
    ph = h_cells * CELL
    px = [[WALL_COLOR] * pw for _ in range(ph)]
    return Sprite(pixels=px, name="wall", visible=True, collidable=True, tags=["wall"])


def _make_cracked_wall() -> Sprite:
    """Cracked wall: breakable by large form. 1 cell."""
    px = [
        [CRACKED_WALL, WALL_COLOR, CRACKED_WALL],
        [WALL_COLOR, CRACKED_WALL, WALL_COLOR],
        [CRACKED_WALL, WALL_COLOR, CRACKED_WALL],
    ]
    return Sprite(pixels=px, name="cracked_wall", visible=True, collidable=True,
                  tags=["cracked_wall"])


def _make_moveable_block() -> Sprite:
    """Moveable block: pushable by medium or large form. 1 cell."""
    px = [[MOVEABLE_BLOCK] * CELL for _ in range(CELL)]
    return Sprite(pixels=px, name="moveable_block", visible=True, collidable=True,
                  tags=["moveable_block"])


def _make_pressure_plate(plate_type: int) -> Sprite:
    """Pressure plate — collidable=False so player can stand on it."""
    px = [[plate_type] * CELL for _ in range(CELL)]
    return Sprite(pixels=px, name=f"plate_{plate_type}", visible=True, collidable=False,
                  tags=["pressure_plate", f"plate_{plate_type}"])


def _make_exit(opened: bool = False) -> Sprite:
    """Exit door sprite."""
    color = EXIT_OPEN if opened else EXIT_CLOSED
    px = [
        [color, color, color],
        [color, BG_COLOR, color],
        [color, color, color],
    ]
    return Sprite(pixels=px, name="exit", visible=True,
                  collidable=(not opened), tags=["exit"])


def _make_locked_zone_tile() -> Sprite:
    """Floor tile marking a form-locked zone (shifting disabled)."""
    px = [[LOCKED_ZONE] * CELL for _ in range(CELL)]
    return Sprite(pixels=px, name="locked_zone", visible=True, collidable=False,
                  tags=["locked_zone"])


# ---------------------------------------------------------------------------
# Room-based level generation
# ---------------------------------------------------------------------------

def _generate_room_layout(rng: random.Random, grid_w: int, grid_h: int,
                          num_rooms: int) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
    """Generate a grid with rooms connected by corridors of varying widths.
    Returns (grid, rooms) where grid is 0=floor, 1=wall and rooms is list of (x,y,w,h) in cells."""
    grid = np.ones((grid_h, grid_w), dtype=int)
    rooms = []

    # Generate rooms with random sizes (3-5 cells wide/tall)
    for _ in range(num_rooms * 3):  # try more times to place rooms
        if len(rooms) >= num_rooms:
            break
        rw = rng.randint(3, 5)
        rh = rng.randint(3, 5)
        rx = rng.randint(1, grid_w - rw - 1)
        ry = rng.randint(1, grid_h - rh - 1)

        # Check overlap with existing rooms (with 1-cell margin)
        overlap = False
        for ox, oy, ow, oh in rooms:
            if (rx - 1 < ox + ow + 1 and rx + rw + 1 > ox - 1 and
                ry - 1 < oy + oh + 1 and ry + rh + 1 > oy - 1):
                overlap = True
                break
        if overlap:
            continue

        rooms.append((rx, ry, rw, rh))
        for r in range(ry, ry + rh):
            for c in range(rx, rx + rw):
                grid[r, c] = 0

    # Connect rooms with corridors of varying widths
    for i in range(len(rooms) - 1):
        x1 = rooms[i][0] + rooms[i][2] // 2
        y1 = rooms[i][1] + rooms[i][3] // 2
        x2 = rooms[i + 1][0] + rooms[i + 1][2] // 2
        y2 = rooms[i + 1][1] + rooms[i + 1][3] // 2

        # Corridor width: 1 (small only), 2 (medium ok), or 3 (large ok)
        width = rng.choice([1, 2, 3])

        # Carve L-shaped corridor
        # Horizontal segment
        sx, ex = min(x1, x2), max(x1, x2)
        for c in range(sx, ex + 1):
            for w in range(width):
                row = y1 + w
                if 0 < row < grid_h - 1 and 0 < c < grid_w - 1:
                    grid[row, c] = 0

        # Vertical segment
        sy, ey = min(y1, y2), max(y1, y2)
        for r in range(sy, ey + 1):
            for w in range(width):
                col = x2 + w
                if 0 < r < grid_h - 1 and 0 < col < grid_w - 1:
                    grid[r, col] = 0

    return grid, rooms


# ---------------------------------------------------------------------------
# Level configuration
# ---------------------------------------------------------------------------
# (num_rooms, num_plates, num_cracked, num_moveable, cooldown, has_locked_zones)
LEVEL_CONFIGS = [
    (3, 2, 1, 0, 0, False),   # Level 1: 3 rooms, 2 plates, 1 cracked wall, no cooldown
    (4, 3, 1, 1, 0, False),   # Level 2: add moveable block
    (4, 3, 2, 1, 3, False),   # Level 3: add shift cooldown
    (5, 4, 2, 2, 3, True),    # Level 4: add form-locked zones
    (5, 5, 3, 2, 4, True),    # Level 5: harder — more plates, tighter cooldown
]


# ---------------------------------------------------------------------------
# Game class
# ---------------------------------------------------------------------------

class Game10Shapeshifter(ARCBaseGame):
    def __init__(self, seed=0):
        self._seed = seed

        # Player state
        self._form = FORM_SMALL
        self._cooldown = 0          # turns until shifting allowed
        self._cooldown_max = 0      # max cooldown for current level
        self._move_count = 0
        self._activated_plates: Set[str] = set()  # track activated plate sprite names

        levels = [Level(sprites=[], grid_size=(GRID, GRID)) for _ in LEVEL_CONFIGS]
        camera = Camera(0, 0, GRID, GRID, BG_COLOR, HUD_BG)
        super().__init__(
            game_id="game_10_shapeshifter",
            levels=levels,
            camera=camera,
            available_actions=[1, 2, 3, 4, 5],
            win_score=len(LEVEL_CONFIGS),
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Level setup
    # ------------------------------------------------------------------

    def on_set_level(self, level):
        """Procedurally generate the level layout."""
        level_idx = self._levels.index(level)
        rng = random.Random(self._seed * 100 + level_idx)

        cfg = LEVEL_CONFIGS[level_idx]
        num_rooms, num_plates, num_cracked, num_moveable, cooldown, has_locked = cfg

        self._form = FORM_SMALL
        self._cooldown = 0
        self._cooldown_max = cooldown
        self._move_count = 0
        self._activated_plates = set()

        # Grid in cells: use 20x20 cell grid (= 60 pixels, leaving room for HUD at rows 62-63)
        cell_w, cell_h = 20, 20
        grid, rooms = _generate_room_layout(rng, cell_w, cell_h, num_rooms)
        mh, mw = grid.shape

        # Collect floor cells
        floor_cells = []
        for r in range(mh):
            for c in range(mw):
                if grid[r, c] == 0:
                    floor_cells.append((r, c))
        rng.shuffle(floor_cells)

        used_cells = set()

        def _take_cell():
            while floor_cells:
                cell = floor_cells.pop()
                if cell not in used_cells:
                    used_cells.add(cell)
                    return cell
            return None

        def _cell_to_px(r, c):
            """Convert maze coord to pixel position."""
            return c * CELL, r * CELL

        # Place walls
        for r in range(mh):
            for c in range(mw):
                if grid[r, c] == 1:
                    wall = _make_wall()
                    px_x, px_y = _cell_to_px(r, c)
                    wall.set_position(px_x, px_y)
                    level.add_sprite(wall)

        # Place player (small form) at start
        player_cell = _take_cell()
        if player_cell:
            player = _make_player(FORM_SMALL)
            px_x, px_y = _cell_to_px(*player_cell)
            player.set_position(px_x, px_y)
            level.add_sprite(player)

        # Place exit at a distant cell
        exit_cell = _take_cell()
        if exit_cell:
            exit_sprite = _make_exit(opened=False)
            px_x, px_y = _cell_to_px(*exit_cell)
            exit_sprite.set_position(px_x, px_y)
            level.add_sprite(exit_sprite)

        # Place pressure plates with varied types
        plate_types = [PLATE_SMALL, PLATE_MEDIUM, PLATE_LARGE]
        for i in range(num_plates):
            cell = _take_cell()
            if cell is None:
                break
            pt = plate_types[i % len(plate_types)]
            plate = _make_pressure_plate(pt)
            px_x, px_y = _cell_to_px(*cell)
            plate.set_position(px_x, px_y)
            # Give each plate a unique name for tracking activation
            plate._name = f"plate_{i}"
            plate._tags = ["pressure_plate", f"plate_{pt}"]
            level.add_sprite(plate)

        # Place cracked walls (replacing some existing walls adjacent to floor)
        cracked_placed = 0
        # Find wall cells adjacent to floor cells
        wall_candidates = []
        for r in range(1, mh - 1):
            for c in range(1, mw - 1):
                if grid[r, c] == 1:
                    # Check if adjacent to a floor cell on both sides (acts as barrier)
                    adj_floor = 0
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < mh and 0 <= nc < mw and grid[nr, nc] == 0:
                            adj_floor += 1
                    if adj_floor >= 2:
                        wall_candidates.append((r, c))
        rng.shuffle(wall_candidates)

        for r, c in wall_candidates:
            if cracked_placed >= num_cracked:
                break
            # Remove existing wall at this position
            px_x, px_y = _cell_to_px(r, c)
            existing = level.get_sprite_at(px_x + 1, px_y + 1)
            if existing and "wall" in existing.tags:
                level.remove_sprite(existing)
                cracked = _make_cracked_wall()
                cracked.set_position(px_x, px_y)
                cracked._name = f"cracked_{cracked_placed}"
                level.add_sprite(cracked)
                cracked_placed += 1

        # Place moveable blocks on floor cells
        for i in range(num_moveable):
            cell = _take_cell()
            if cell is None:
                break
            block = _make_moveable_block()
            px_x, px_y = _cell_to_px(*cell)
            block.set_position(px_x, px_y)
            block._name = f"block_{i}"
            level.add_sprite(block)

        # Place locked zone tiles (form-locked areas)
        if has_locked:
            num_locked = min(3, len(floor_cells))
            for i in range(num_locked):
                cell = _take_cell()
                if cell is None:
                    break
                lz = _make_locked_zone_tile()
                px_x, px_y = _cell_to_px(*cell)
                lz.set_position(px_x, px_y)
                lz._name = f"locked_{i}"
                level.add_sprite(lz)

        # Render HUD
        self._render_hud()

    # ------------------------------------------------------------------
    # HUD
    # ------------------------------------------------------------------

    def _render_hud(self):
        """Update HUD: current form indicator + cooldown bar."""
        level = self.current_level

        # Remove old HUD elements
        for s in list(level.get_sprites_by_tag("hud_element")):
            level.remove_sprite(s)

        # -- Form indicator on row 63 --
        form_color = FORM_COLORS[self._form]
        # Show 3 blocks: current form highlighted, others dimmed
        for i, f in enumerate([FORM_SMALL, FORM_MEDIUM, FORM_LARGE]):
            c = FORM_COLORS[f]
            if f == self._form:
                block_px = [[HUD_BORDER, HUD_BORDER, HUD_BORDER, HUD_BORDER, HUD_BORDER],
                            [HUD_BORDER, c, c, c, HUD_BORDER]]
            else:
                block_px = [[BG_COLOR, BG_COLOR, BG_COLOR, BG_COLOR, BG_COLOR],
                            [BG_COLOR, c, c, c, BG_COLOR]]
            block = Sprite(pixels=block_px, name=f"hud_form_{i}",
                           visible=True, collidable=False, tags=["hud_element"])
            block.set_position(2 + i * 7, HUD_ROW_FORM)
            level.add_sprite(block)

        # -- Cooldown bar on row 62 --
        if self._cooldown_max > 0:
            # Border
            bar_width = 40
            border_px = [[HUD_BORDER] * (bar_width + 2)]
            border = Sprite(pixels=border_px, name="cooldown_border",
                            visible=True, collidable=False, tags=["hud_element"])
            border.set_position(2, HUD_ROW_COOLDOWN)
            level.add_sprite(border)

            # Fill: shows remaining cooldown (more fill = more cooldown remaining)
            if self._cooldown > 0:
                fill_w = max(1, int(bar_width * self._cooldown / self._cooldown_max))
                fill_px = [[COOLDOWN_FILL] * fill_w]
                fill = Sprite(pixels=fill_px, name="cooldown_fill",
                              visible=True, collidable=False, tags=["hud_element"])
                fill.set_position(3, HUD_ROW_COOLDOWN)
                level.add_sprite(fill)

        # -- Activated plates indicator --
        total_plates = len(list(level.get_sprites_by_tag("pressure_plate")))
        activated = len(self._activated_plates)
        for i in range(total_plates + activated):
            dot_color = EXIT_OPEN if i < activated else EXIT_CLOSED
            dot_px = [[dot_color, dot_color]]
            dot = Sprite(pixels=dot_px, name=f"plate_dot_{i}",
                         visible=True, collidable=False, tags=["hud_element"])
            dot.set_position(50 + i * 3, HUD_ROW_FORM)
            level.add_sprite(dot)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_player(self) -> Optional[Sprite]:
        sprites = self.current_level.get_sprites_by_name("player")
        return sprites[0] if sprites else None

    def _player_on_locked_zone(self, player: Sprite) -> bool:
        """Check if player is on a form-locked zone tile."""
        for lz in self.current_level.get_sprites_by_tag("locked_zone"):
            if lz.x == player.x and lz.y == player.y:
                return True
        return False

    def _would_fit(self, x: int, y: int, form: int) -> bool:
        """Check if the player in the given form would fit at position (x, y)
        without colliding with walls or going out of bounds."""
        size = FORM_PIXEL_SIZES[form] * CELL
        # Check bounds
        if x < 0 or y < 0 or x + size > GRID or y + size > HUD_ROW_COOLDOWN:
            return False
        # Check collisions with wall sprites
        level = self.current_level
        for wall in level.get_sprites_by_tag("wall"):
            # Simple AABB check
            if (x < wall.x + CELL and x + size > wall.x and
                y < wall.y + CELL and y + size > wall.y):
                return False
        # Check cracked walls
        for cw in level.get_sprites_by_tag("cracked_wall"):
            if (x < cw.x + CELL and x + size > cw.x and
                y < cw.y + CELL and y + size > cw.y):
                return False
        # Check moveable blocks
        for block in level.get_sprites_by_tag("moveable_block"):
            if (x < block.x + CELL and x + size > block.x and
                y < block.y + CELL and y + size > block.y):
                return False
        # Check closed exit
        for ex in level.get_sprites_by_tag("exit"):
            if ex.is_collidable:  # closed exit
                if (x < ex.x + CELL and x + size > ex.x and
                    y < ex.y + CELL and y + size > ex.y):
                    return False
        return True

    def _replace_player(self, x: int, y: int, form: int):
        """Remove current player and place a new one at (x, y) with the given form."""
        player = self._get_player()
        if player:
            self.current_level.remove_sprite(player)
        new_player = _make_player(form)
        new_player.set_position(x, y)
        self.current_level.add_sprite(new_player)

    def _check_plates(self, player: Sprite):
        """Check if player is standing on a pressure plate of matching form."""
        px, py = player.x, player.y
        player_size = FORM_PIXEL_SIZES[self._form] * CELL

        for plate in list(self.current_level.get_sprites_by_tag("pressure_plate")):
            # Check overlap
            if (px <= plate.x < px + player_size and
                py <= plate.y < py + player_size):
                # Determine what form the plate requires
                required_form = None
                if f"plate_{PLATE_SMALL}" in plate.tags:
                    required_form = FORM_SMALL
                elif f"plate_{PLATE_MEDIUM}" in plate.tags:
                    required_form = FORM_MEDIUM
                elif f"plate_{PLATE_LARGE}" in plate.tags:
                    required_form = FORM_LARGE

                if required_form is not None and self._form == required_form:
                    self._activated_plates.add(plate.name)

        # Check if all plates activated → open exit
        total_plates = (len(list(self.current_level.get_sprites_by_tag("pressure_plate")))
                        + len(self._activated_plates))
        # Total plates = remaining unactivated + activated
        # Actually, plates stay on the grid. Let's count them differently.
        # All plates always exist. We just track which ones got activated.
        all_plate_names = set()
        for plate in self.current_level.get_sprites_by_tag("pressure_plate"):
            all_plate_names.add(plate.name)

        if all_plate_names and all_plate_names.issubset(self._activated_plates):
            self._open_exit()

    def _open_exit(self):
        """Open the exit door."""
        for ex in list(self.current_level.get_sprites_by_tag("exit")):
            self.current_level.remove_sprite(ex)
            opened = _make_exit(opened=True)
            opened.set_position(ex.x, ex.y)
            self.current_level.add_sprite(opened)

    def _check_exit(self, player: Sprite) -> bool:
        """Check if player reached an open exit."""
        player_size = FORM_PIXEL_SIZES[self._form] * CELL
        for ex in self.current_level.get_sprites_by_tag("exit"):
            if not ex.is_collidable:  # open exit
                if (player.x <= ex.x < player.x + player_size and
                    player.y <= ex.y < player.y + player_size):
                    return True
                if (ex.x <= player.x < ex.x + CELL and
                    ex.y <= player.y < ex.y + CELL):
                    return True
        return False

    def _try_push_block(self, block: Sprite, dx: int, dy: int) -> bool:
        """Try to push a moveable block. Returns True if push succeeded."""
        new_x = block.x + dx
        new_y = block.y + dy

        # Check bounds
        if new_x < 0 or new_y < 0 or new_x + CELL > GRID or new_y + CELL > HUD_ROW_COOLDOWN:
            return False

        # Check collision with walls
        for wall in self.current_level.get_sprites_by_tag("wall"):
            if (new_x < wall.x + CELL and new_x + CELL > wall.x and
                new_y < wall.y + CELL and new_y + CELL > wall.y):
                return False

        # Check collision with cracked walls
        for cw in self.current_level.get_sprites_by_tag("cracked_wall"):
            if (new_x < cw.x + CELL and new_x + CELL > cw.x and
                new_y < cw.y + CELL and new_y + CELL > cw.y):
                return False

        # Check collision with other moveable blocks
        for other in self.current_level.get_sprites_by_tag("moveable_block"):
            if other is block:
                continue
            if (new_x < other.x + CELL and new_x + CELL > other.x and
                new_y < other.y + CELL and new_y + CELL > other.y):
                return False

        # Move the block
        block.set_position(new_x, new_y)
        return True

    def _try_break_wall(self, cx: int, cy: int) -> bool:
        """If large form, try to break a cracked wall at (cx, cy). Returns True if broken."""
        if self._form != FORM_LARGE:
            return False
        for cw in list(self.current_level.get_sprites_by_tag("cracked_wall")):
            if cw.x == cx and cw.y == cy:
                self.current_level.remove_sprite(cw)
                return True
        return False

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def _do_move(self, dx: int, dy: int):
        """Handle player movement with form-specific abilities."""
        player = self._get_player()
        if player is None:
            return

        player_size = FORM_PIXEL_SIZES[self._form] * CELL
        new_x = player.x + dx
        new_y = player.y + dy

        # Bounds check
        if new_x < 0 or new_y < 0 or new_x + player_size > GRID or new_y + player_size > HUD_ROW_COOLDOWN:
            return

        # Check what we'd collide with
        blocked = False

        # Walls
        for wall in self.current_level.get_sprites_by_tag("wall"):
            if (new_x < wall.x + CELL and new_x + player_size > wall.x and
                new_y < wall.y + CELL and new_y + player_size > wall.y):
                blocked = True
                break

        # Cracked walls — large form breaks them
        if not blocked:
            for cw in list(self.current_level.get_sprites_by_tag("cracked_wall")):
                if (new_x < cw.x + CELL and new_x + player_size > cw.x and
                    new_y < cw.y + CELL and new_y + player_size > cw.y):
                    if self._form == FORM_LARGE:
                        self.current_level.remove_sprite(cw)
                        # Don't block — we break through
                    else:
                        blocked = True
                        break

        # Moveable blocks — medium/large form can push
        if not blocked:
            for block in list(self.current_level.get_sprites_by_tag("moveable_block")):
                if (new_x < block.x + CELL and new_x + player_size > block.x and
                    new_y < block.y + CELL and new_y + player_size > block.y):
                    if self._form >= FORM_MEDIUM:
                        pushed = self._try_push_block(block, dx, dy)
                        if not pushed:
                            blocked = True
                            break
                    else:
                        blocked = True
                        break

        # Closed exit
        if not blocked:
            for ex in self.current_level.get_sprites_by_tag("exit"):
                if ex.is_collidable:
                    if (new_x < ex.x + CELL and new_x + player_size > ex.x and
                        new_y < ex.y + CELL and new_y + player_size > ex.y):
                        blocked = True
                        break

        if blocked:
            return

        # Move player
        self._replace_player(new_x, new_y, self._form)
        self._move_count += 1

        # Decrement cooldown
        if self._cooldown > 0:
            self._cooldown -= 1

        # Check interactions
        new_player = self._get_player()
        if new_player:
            self._check_plates(new_player)
            if self._check_exit(new_player):
                self.next_level()

    # ------------------------------------------------------------------
    # Shapeshift
    # ------------------------------------------------------------------

    def _do_shapeshift(self):
        """Cycle to next form if allowed."""
        player = self._get_player()
        if player is None:
            return

        # Check cooldown
        if self._cooldown > 0:
            return

        # Check form-locked zone
        if self._player_on_locked_zone(player):
            return

        # Cycle form: small → medium → large → small
        # Try next form, then the one after if next doesn't fit
        px, py = player.x, player.y
        shifted = False
        for offset in [1, 2]:
            candidate = (self._form + offset) % 3
            if self._would_fit(px, py, candidate):
                self._form = candidate
                self._replace_player(px, py, candidate)
                shifted = True
                break

        if shifted:
            # Apply cooldown
            if self._cooldown_max > 0:
                self._cooldown = self._cooldown_max

            # Check plates and exit with new form
            new_player = self._get_player()
            if new_player:
                self._check_plates(new_player)
                if self._check_exit(new_player):
                    self.next_level()

    # ------------------------------------------------------------------
    # Game loop
    # ------------------------------------------------------------------

    def step(self):
        from arcengine import GameAction

        action = self.action.id

        if action == GameAction.ACTION1:
            self._do_move(0, -MOVE)
        elif action == GameAction.ACTION2:
            self._do_move(0, MOVE)
        elif action == GameAction.ACTION3:
            self._do_move(-MOVE, 0)
        elif action == GameAction.ACTION4:
            self._do_move(MOVE, 0)
        elif action == GameAction.ACTION5:
            self._do_shapeshift()

        self._render_hud()
        self.complete_action()
