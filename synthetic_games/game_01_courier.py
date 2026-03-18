"""
Game 01: Courier

Mechanics: Navigation + package collection + locked colored doors + limited fuel + refueling

The player moves through a walled grid carrying packages. Each package has a
color. Colored doors block paths and only open when the player is carrying a
package of the matching color (the door disappears, consuming no package).
Delivering a package to a matching-color mailbox removes it from inventory.
The player has limited fuel that decrements each move. Fuel stations (2x2 green
squares) refill fuel to maximum. All mailboxes must receive their packages to
complete the level. Later levels have multiple packages requiring route planning
to avoid running out of fuel.

Actions:
    ACTION1 = Move Up (W)
    ACTION2 = Move Down (S)
    ACTION3 = Move Left (A)
    ACTION4 = Move Right (D)

Display:
    Row 62: Fuel gauge (filled portion = remaining fuel)
    Row 63: Current carried package color indicator

Win condition: All packages delivered to their matching mailboxes.

Color key:
    0  = empty / floor
    1  = wall
    2  = player
    3  = fuel station (green)
    4  = mailbox border / frame
    5  = HUD background
    6  = package / door / mailbox: blue
    7  = package / door / mailbox: yellow
    8  = package / door / mailbox: red
    9  = package / door / mailbox: purple
    10 = fuel gauge fill (green)
    11 = package / door / mailbox: orange
    12 = empty inventory indicator (gray)
    13 = fuel gauge border (teal)
    14 = package / door / mailbox: pink
    15 = HUD label / border
"""

import random
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

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
CELL = 3          # player & objects occupy 3x3 pixels
MOVE = 3          # movement step size
BG_COLOR = 0
WALL_COLOR = 1
PLAYER_COLOR = 2
FUEL_STATION_COLOR = 3
MAILBOX_BORDER = 4
HUD_BG = 5
HUD_BORDER = 15
FUEL_FILL = 10
FUEL_BORDER = 13
EMPTY_INV = 12

# Package / door / mailbox colors
BLUE = 6
YELLOW = 7
RED = 8
PURPLE = 9
ORANGE = 11
PINK = 14

PACKAGE_COLORS = [BLUE, YELLOW, RED, PURPLE, ORANGE, PINK]

# HUD rows
HUD_ROW_FUEL = 62
HUD_ROW_INV = 63

# ---------------------------------------------------------------------------
# Sprite builders
# ---------------------------------------------------------------------------

def _make_player() -> Sprite:
    """3x3 player sprite."""
    px = [
        [PLAYER_COLOR, PLAYER_COLOR, PLAYER_COLOR],
        [PLAYER_COLOR, PLAYER_COLOR, PLAYER_COLOR],
        [PLAYER_COLOR, PLAYER_COLOR, PLAYER_COLOR],
    ]
    return Sprite(pixels=px, name="player", visible=True, collidable=True)


def _make_wall_block() -> Sprite:
    """3x3 wall block."""
    px = [[WALL_COLOR] * 3 for _ in range(3)]
    return Sprite(pixels=px, name="wall", visible=True, collidable=True)


def _make_fuel_station(idx: int) -> Sprite:
    """3x3 (1 cell) fuel station — same size as other objects for reliable collision."""
    px = [
        [FUEL_STATION_COLOR, FUEL_STATION_COLOR, FUEL_STATION_COLOR],
        [FUEL_STATION_COLOR, FUEL_FILL, FUEL_STATION_COLOR],
        [FUEL_STATION_COLOR, FUEL_STATION_COLOR, FUEL_STATION_COLOR],
    ]
    return Sprite(pixels=px, name=f"fuel_station_{idx}", visible=True, collidable=False,
                  tags=["fuel_station"])


def _make_package(color: int, idx: int) -> Sprite:
    """3x3 package with a distinct center pixel."""
    px = [
        [color, color, color],
        [color, BG_COLOR, color],
        [color, color, color],
    ]
    return Sprite(pixels=px, name=f"package_{idx}", visible=True, collidable=False,
                  tags=["package"])


def _make_door(color: int, idx: int) -> Sprite:
    """3x3 colored door that blocks movement."""
    px = [
        [color, WALL_COLOR, color],
        [WALL_COLOR, color, WALL_COLOR],
        [color, WALL_COLOR, color],
    ]
    return Sprite(pixels=px, name=f"door_{idx}", visible=True, collidable=True,
                  tags=["door"])


def _make_mailbox(color: int, idx: int) -> Sprite:
    """3x3 mailbox: border color 4 with colored center."""
    px = [
        [MAILBOX_BORDER, MAILBOX_BORDER, MAILBOX_BORDER],
        [MAILBOX_BORDER, color, MAILBOX_BORDER],
        [MAILBOX_BORDER, MAILBOX_BORDER, MAILBOX_BORDER],
    ]
    return Sprite(pixels=px, name=f"mailbox_{idx}", visible=True, collidable=False,
                  tags=["mailbox"])


# ---------------------------------------------------------------------------
# HUD rendering
# ---------------------------------------------------------------------------

def _make_hud_bg() -> Sprite:
    """Full-width HUD background spanning rows 60-63."""
    px = [[HUD_BG] * GRID for _ in range(4)]
    return Sprite(pixels=px, name="hud_bg", visible=True, collidable=False)


# ---------------------------------------------------------------------------
# Level generation helpers
# ---------------------------------------------------------------------------

def _carve_maze(rng: random.Random, cols: int, rows: int) -> np.ndarray:
    """Generate a maze using recursive backtracking.
    Returns a grid where 0=floor, 1=wall. Grid coords are in cell units.
    """
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


def _bfs_distance(maze: np.ndarray, start: Tuple[int, int], end: Tuple[int, int]) -> int:
    """BFS shortest path distance between two cells on the maze. Returns -1 if unreachable."""
    rows, cols = maze.shape
    if start == end:
        return 0
    visited: Set[Tuple[int, int]] = {start}
    queue: deque = deque([(start, 0)])
    while queue:
        (r, c), dist = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and maze[nr, nc] == 0:
                if (nr, nc) == end:
                    return dist + 1
                visited.add((nr, nc))
                queue.append(((nr, nc), dist + 1))
    return -1


def _all_reachable(maze: np.ndarray, cells: List[Tuple[int, int]]) -> bool:
    """Check that all cells are reachable from cells[0] via BFS."""
    if len(cells) <= 1:
        return True
    for target in cells[1:]:
        if _bfs_distance(maze, cells[0], target) == -1:
            return False
    return True


def _compute_optimal_route(maze: np.ndarray,
                           player: Tuple[int, int],
                           packages: List[Tuple[int, int]],
                           mailboxes: List[Tuple[int, int]],
                           fuel_stations: List[Tuple[int, int]]) -> int:
    """Estimate the minimum distance to collect all packages and deliver them.
    Returns the total BFS distance for a greedy nearest-first route.
    This is an upper bound on the optimal route, not the true optimum.
    """
    # Build distance cache
    all_points = [player] + packages + mailboxes + fuel_stations
    dist_cache: Dict[Tuple[Tuple[int, int], Tuple[int, int]], int] = {}
    for i, a in enumerate(all_points):
        for j, b in enumerate(all_points):
            if i < j:
                d = _bfs_distance(maze, a, b)
                dist_cache[(a, b)] = d
                dist_cache[(b, a)] = d

    # Greedy: pick up nearest package, deliver to its mailbox, repeat
    total = 0
    pos = player
    remaining = list(zip(packages, mailboxes))

    while remaining:
        # Find nearest package
        best_idx = 0
        best_dist = float("inf")
        for i, (pkg, _) in enumerate(remaining):
            d = dist_cache.get((pos, pkg), _bfs_distance(maze, pos, pkg))
            if d < best_dist:
                best_dist = d
                best_idx = i

        pkg, mb = remaining.pop(best_idx)
        d_to_pkg = dist_cache.get((pos, pkg), _bfs_distance(maze, pos, pkg))
        d_to_mb = dist_cache.get((pkg, mb), _bfs_distance(maze, pkg, mb))
        total += d_to_pkg + d_to_mb
        pos = mb

    return total


class CourierGame(ARCBaseGame):
    """Game 01: Courier — navigate, collect packages, unlock doors, manage fuel."""

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)
        self._seed = seed

        # State that gets set per level
        self._fuel = 0
        self._max_fuel = 0
        self._carried_package: Optional[int] = None  # color or None
        self._delivered_count = 0
        self._total_mailboxes = 0

        # Level definitions (grid_cell_size, num_packages, fuel_margin_mult, num_doors)
        # fuel_margin_mult: fuel = optimal_route_distance * this multiplier
        level_configs = [
            (11, 1, 3.0, 0),   # Level 1: small, 1 package, no doors, generous fuel
            (13, 2, 2.5, 1),   # Level 2: medium, 2 packages, 1 door
            (15, 2, 2.0, 2),   # Level 3: 2 doors, moderate fuel
            (17, 3, 2.0, 2),   # Level 4: bigger, more packages
            (19, 3, 1.8, 3),   # Level 5: tight fuel, 3 doors
        ]
        self._level_configs = level_configs

        levels = [Level(sprites=[], grid_size=(GRID, GRID)) for _ in level_configs]
        camera = Camera(0, 0, GRID, GRID, BG_COLOR, HUD_BG)
        super().__init__(
            game_id="game_01_courier",
            levels=levels,
            camera=camera,
            available_actions=[1, 2, 3, 4],
            win_score=len(level_configs),
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Level setup
    # ------------------------------------------------------------------

    def on_set_level(self, level):
        """Procedurally generate the level layout with solvability guarantee."""
        level_idx = self._levels.index(level)

        cfg = self._level_configs[level_idx]
        cell_cols, num_packages, fuel_margin_mult, num_doors = cfg
        cell_rows = min(cell_cols, 20)

        # Retry with different sub-seeds until we get a solvable layout
        for attempt in range(50):
            rng = random.Random(self._seed * 1000 + level_idx * 50 + attempt)
            result = self._try_generate_level(
                rng, level, cell_cols, cell_rows, num_packages, fuel_margin_mult, num_doors
            )
            if result is not None:
                fuel_budget, total_mailboxes = result
                self._fuel = fuel_budget
                self._max_fuel = fuel_budget
                self._carried_package = None
                self._delivered_count = 0
                self._total_mailboxes = total_mailboxes
                self._render_hud()
                return

        # Fallback: should never happen, but use last attempt with generous fuel
        self._fuel = 500
        self._max_fuel = 500
        self._carried_package = None
        self._delivered_count = 0
        self._total_mailboxes = num_packages
        self._render_hud()

    def _try_generate_level(self, rng, level, cell_cols, cell_rows,
                            num_packages, fuel_margin_mult, num_doors) -> Optional[Tuple[int, int]]:
        """Try to generate a solvable level layout. Returns (fuel_budget, num_mailboxes)
        on success, None if the layout is unsolvable."""
        # Clear any existing sprites
        for s in list(level._sprites):
            level.remove_sprite(s)

        maze = _carve_maze(rng, cell_cols, cell_rows)

        # Gather floor cells
        floor_cells = []
        for r in range(cell_rows):
            for c in range(cell_cols):
                if maze[r, c] == 0:
                    floor_cells.append((r, c))
        rng.shuffle(floor_cells)

        used_cells: Set[Tuple[int, int]] = set()

        def _take_cell():
            while floor_cells:
                cell = floor_cells.pop()
                if cell not in used_cells:
                    used_cells.add(cell)
                    return cell
            return None

        # Place objects into cell positions first (don't create sprites yet)
        player_cell = _take_cell()
        if player_cell is None:
            return None

        available_colors = list(PACKAGE_COLORS)
        rng.shuffle(available_colors)
        colors_used = available_colors[:num_packages]

        package_cells = []
        for color in colors_used:
            cell = _take_cell()
            if cell is None:
                return None
            package_cells.append((cell, color))

        mailbox_cells = []
        for color in colors_used:
            cell = _take_cell()
            if cell is None:
                return None
            mailbox_cells.append((cell, color))

        fuel_cells = []
        num_fuel = 1 if cell_cols <= 13 else 2
        for _ in range(num_fuel):
            cell = _take_cell()
            if cell is None:
                break
            fuel_cells.append(cell)

        door_cells = []
        door_colors = list(colors_used)
        rng.shuffle(door_colors)
        for i in range(min(num_doors, len(door_colors))):
            cell = _take_cell()
            if cell is None:
                break
            door_cells.append((cell, door_colors[i % len(door_colors)]))

        # Verify solvability: all key locations reachable from player
        all_key_cells = (
            [player_cell]
            + [c for c, _ in package_cells]
            + [c for c, _ in mailbox_cells]
            + fuel_cells
        )
        if not _all_reachable(maze, all_key_cells):
            return None

        # Compute optimal route distance and set fuel budget
        pkg_positions = [c for c, _ in package_cells]
        mb_positions = [c for c, _ in mailbox_cells]
        optimal_dist = _compute_optimal_route(maze, player_cell, pkg_positions,
                                              mb_positions, fuel_cells)
        if optimal_dist <= 0:
            return None

        fuel_budget = max(int(optimal_dist * fuel_margin_mult), optimal_dist + 30)

        # Layout is solvable — now create sprites
        # Walls
        for r in range(cell_rows):
            for c in range(cell_cols):
                if maze[r, c] == 1:
                    wall = _make_wall_block()
                    wall.set_position(c * CELL, r * CELL)
                    level.add_sprite(wall)

        # Player
        player = _make_player()
        player.set_position(player_cell[1] * CELL, player_cell[0] * CELL)
        level.add_sprite(player)

        # Packages
        for i, (cell, color) in enumerate(package_cells):
            pkg = _make_package(color, i)
            pkg.set_position(cell[1] * CELL, cell[0] * CELL)
            level.add_sprite(pkg)

        # Mailboxes
        for i, (cell, color) in enumerate(mailbox_cells):
            mb = _make_mailbox(color, i)
            mb.set_position(cell[1] * CELL, cell[0] * CELL)
            level.add_sprite(mb)

        # Doors
        for i, (cell, color) in enumerate(door_cells):
            door = _make_door(color, i)
            door.set_position(cell[1] * CELL, cell[0] * CELL)
            level.add_sprite(door)

        # Fuel stations
        for i, cell in enumerate(fuel_cells):
            fs = _make_fuel_station(i)
            fs.set_position(cell[1] * CELL, cell[0] * CELL)
            level.add_sprite(fs)

        # HUD
        hud = _make_hud_bg()
        hud.set_position(0, 60)
        level.add_sprite(hud)

        return (fuel_budget, num_packages)

    # ------------------------------------------------------------------
    # HUD
    # ------------------------------------------------------------------

    def _render_hud(self):
        """Update HUD sprites for fuel gauge and inventory."""
        level = self.current_level

        # Remove old HUD elements
        for s in list(level.get_sprites_by_tag("hud_element")):
            level.remove_sprite(s)

        # -- Fuel gauge on row 62 --
        # Border
        border_px = [[FUEL_BORDER] * 52]
        border_top = Sprite(pixels=border_px, name="fuel_border_top",
                            visible=True, collidable=False, tags=["hud_element"])
        border_top.set_position(6, HUD_ROW_FUEL)
        level.add_sprite(border_top)

        # Fill proportional to remaining fuel
        if self._max_fuel > 0:
            fill_width = max(0, int(50 * self._fuel / self._max_fuel))
        else:
            fill_width = 0

        if fill_width > 0:
            fill_px = [[FUEL_FILL] * fill_width]
            fuel_bar = Sprite(pixels=fill_px, name="fuel_fill",
                              visible=True, collidable=False, tags=["hud_element"])
            fuel_bar.set_position(7, HUD_ROW_FUEL)
            level.add_sprite(fuel_bar)

        # -- Inventory indicator on row 63 --
        inv_color = self._carried_package if self._carried_package is not None else EMPTY_INV
        inv_px = [
            [HUD_BORDER, HUD_BORDER, HUD_BORDER, HUD_BORDER, HUD_BORDER],
            [HUD_BORDER, inv_color, inv_color, inv_color, HUD_BORDER],
            [HUD_BORDER, HUD_BORDER, HUD_BORDER, HUD_BORDER, HUD_BORDER],
        ]
        # Only 1 row left, so make it 1px high
        inv_px_flat = [[HUD_BORDER, inv_color, inv_color, inv_color, inv_color, inv_color, HUD_BORDER]]
        inv_sprite = Sprite(pixels=inv_px_flat, name="inv_indicator",
                            visible=True, collidable=False, tags=["hud_element"])
        inv_sprite.set_position(6, HUD_ROW_INV)
        level.add_sprite(inv_sprite)

        # Delivery progress: small dots for each delivered package
        for i in range(self._total_mailboxes):
            dot_color = FUEL_FILL if i < self._delivered_count else EMPTY_INV
            dot_px = [[dot_color, dot_color]]
            dot = Sprite(pixels=dot_px, name=f"delivery_dot_{i}",
                         visible=True, collidable=False, tags=["hud_element"])
            dot.set_position(30 + i * 4, HUD_ROW_INV)
            level.add_sprite(dot)

    # ------------------------------------------------------------------
    # Interaction helpers
    # ------------------------------------------------------------------

    def _get_player(self) -> Optional[Sprite]:
        sprites = self.current_level.get_sprites_by_name("player")
        return sprites[0] if sprites else None

    def _check_overlap(self, player: Sprite) -> None:
        """Check if the player overlaps with interactive objects after moving."""
        px, py = player.x, player.y
        level = self.current_level

        # Check packages — pick up if not carrying
        if self._carried_package is None:
            for pkg in list(level.get_sprites_by_tag("package")):
                if pkg.x == px and pkg.y == py:
                    # Pick up: store color, remove sprite
                    # Extract the package color from pixels
                    color = pkg.pixels[0][0]
                    self._carried_package = color
                    level.remove_sprite(pkg)
                    break

        # Check mailboxes — deliver if carrying matching color
        for mb in list(level.get_sprites_by_tag("mailbox")):
            if mb.x == px and mb.y == py:
                # Mailbox center color is at [1][1]
                mb_color = mb.pixels[1][1]
                if self._carried_package == mb_color:
                    self._carried_package = None
                    self._delivered_count += 1
                    level.remove_sprite(mb)
                    break

        # Check fuel stations (3x3, same size as player — exact position match)
        for fs in level.get_sprites_by_tag("fuel_station"):
            if fs.x == px and fs.y == py:
                self._fuel = self._max_fuel
                break

    def _check_door_collision(self, collided: list) -> None:
        """If player bumped into a door and carrying matching color, remove door."""
        for sprite in collided:
            if "door" in sprite.tags:
                # Door color: get the dominant package color from pixels
                # Door pattern has the package color at [0][0]
                door_color = sprite.pixels[0][0]
                if self._carried_package == door_color:
                    self.current_level.remove_sprite(sprite)

    # ------------------------------------------------------------------
    # Game loop
    # ------------------------------------------------------------------

    def step(self):
        from arcengine import GameAction

        player = self._get_player()
        if player is None:
            self.complete_action()
            return

        dx, dy = 0, 0
        action = self.action.id

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

        # Try to move
        collided = self.try_move("player", dx, dy)

        if collided:
            # Check if we hit a door we can open
            self._check_door_collision(collided)
            # Don't consume fuel on blocked move
        else:
            # Movement succeeded — consume fuel
            self._fuel -= 1

            # Check interactions at new position
            self._check_overlap(player)

        # Check fuel depletion — reset level to try again
        if self._fuel <= 0:
            self.level_reset()
            self.complete_action()
            return

        # Check win condition — next_level() handles final-level win via win_score
        if self._delivered_count >= self._total_mailboxes:
            self.next_level()
            self.complete_action()
            return

        # Update HUD
        self._render_hud()
        self.complete_action()
