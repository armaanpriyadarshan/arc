"""
Game 07: Architect

Mechanics:
  1. Drag-and-place shapes: the player has tetromino-like pieces in an
     inventory panel (right side). Click a piece to select it; click on
     the play grid to place it.
  2. Rotation: ACTION5 rotates the currently selected piece 90 degrees
     clockwise.
  3. Gravity: after placement, all pieces fall downward until they rest
     on the floor or on another piece.
  4. Structural stability: pieces that overhang with nothing directly
     below any of their cells will continue to fall (no floating).
  5. Target silhouette: the level shows a target outline on the grid.
     The player must stack pieces to fill the silhouette exactly (every
     silhouette cell covered, no piece cell outside the silhouette).
  6. Piece removal: clicking on an already-placed piece removes it back
     to inventory so the player can retry placements.

Actions:
  ACTION5 = Rotate selected piece 90° clockwise
  ACTION6 = Click (select piece from inventory / place piece on grid /
            remove placed piece)

How to win:
  Fill the target silhouette exactly with the available pieces. Every
  target cell must be covered and no piece cell may extend outside
  the silhouette.

Color key (palette indices):
  0  = white / empty background
  1  = off-white / grid floor
  3  = gray / target silhouette outline
  5  = black / inventory panel background
  6  = magenta / piece color A
  7  = yellow / piece color B
  8  = red / piece color C
  9  = blue / piece color D
  10 = light blue / selection highlight
  11 = orange / piece color E
  12 = dark / inventory slot border
  14 = green / piece color F
  15 = purple / HUD text / level indicator

Display:
  Columns 0-47: play area (grid floor + target silhouette + placed pieces)
  Columns 48-63: inventory panel (available pieces)
  Row 63: level progress indicator
"""

import random
from typing import List, Optional, Tuple, Set

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
PLAY_W = 48          # play area width in pixels
PLAY_H = 62          # play area height (rows 0-61)
INV_X = 49           # inventory panel start x
INV_W = 15           # inventory panel width
FLOOR_Y = 61         # floor row (bottom of play area)

# Colors
C_EMPTY = 0
C_FLOOR = 1
C_TARGET = 3
C_INV_BG = 5
C_SELECT = 10
C_INV_BORDER = 12
C_LEVEL_IND = 15

PIECE_COLORS = [6, 7, 8, 9, 11, 14]  # magenta, yellow, red, blue, orange, green

# ---------------------------------------------------------------------------
# Tetromino / shape definitions (relative cells, origin at top-left)
# Each shape is a list of (row, col) offsets
# ---------------------------------------------------------------------------
SHAPES = [
    # I-piece (4 long)
    [(0, 0), (0, 1), (0, 2), (0, 3)],
    # L-piece
    [(0, 0), (1, 0), (2, 0), (2, 1)],
    # J-piece
    [(0, 1), (1, 1), (2, 1), (2, 0)],
    # T-piece
    [(0, 0), (0, 1), (0, 2), (1, 1)],
    # S-piece
    [(0, 1), (0, 2), (1, 0), (1, 1)],
    # Z-piece
    [(0, 0), (0, 1), (1, 1), (1, 2)],
    # O-piece (2x2)
    [(0, 0), (0, 1), (1, 0), (1, 1)],
    # Small L (3 cells)
    [(0, 0), (1, 0), (1, 1)],
    # Line-3
    [(0, 0), (0, 1), (0, 2)],
    # Corner (3 cells)
    [(0, 0), (0, 1), (1, 0)],
    # Big L (5 cells)
    [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1)],
    # Plus (5 cells)
    [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],
]

# Cell size: each logical cell = CELL_SZ x CELL_SZ pixels
CELL_SZ = 3


def _rotate_shape(cells: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Rotate shape 90 degrees clockwise: (r,c) -> (c, -r), then normalize."""
    rotated = [(c, -r) for r, c in cells]
    min_r = min(r for r, c in rotated)
    min_c = min(c for r, c in rotated)
    return sorted([(r - min_r, c - min_c) for r, c in rotated])


def _shape_dims(cells: List[Tuple[int, int]]) -> Tuple[int, int]:
    """Return (height, width) of the shape in cells."""
    max_r = max(r for r, c in cells)
    max_c = max(c for r, c in cells)
    return max_r + 1, max_c + 1


# ---------------------------------------------------------------------------
# Level definitions
# ---------------------------------------------------------------------------

def _generate_target_and_pieces(
    rng: random.Random, level_idx: int
) -> Tuple[Set[Tuple[int, int]], List[List[Tuple[int, int]]], int, int]:
    """
    Generate a target silhouette and the pieces needed to fill it.

    Returns:
        target_cells: set of (row, col) in cell coords within the play area
        pieces: list of shape definitions (cell offsets)
        grid_cols: number of cell columns in play area
        grid_rows: number of cell rows in play area
    """
    grid_cols = PLAY_W // CELL_SZ  # 16 cell columns
    grid_rows = PLAY_H // CELL_SZ  # ~20 cell rows

    # We'll build the silhouette by placing pieces bottom-up
    # Number of pieces scales with level
    num_pieces = 3 + level_idx  # 3, 4, 5, 6, 7

    # Pick shapes for this level
    available_shapes = list(range(len(SHAPES)))
    pieces = []
    for i in range(num_pieces):
        shape_idx = rng.choice(available_shapes)
        shape = list(SHAPES[shape_idx])
        # Random rotation
        rotations = rng.randint(0, 3)
        for _ in range(rotations):
            shape = _rotate_shape(shape)
        pieces.append(shape)

    # Build target by "placing" each piece bottom-up with gravity
    # We simulate placement on a grid to create a valid target
    occupied: Set[Tuple[int, int]] = set()
    floor_row = grid_rows - 1  # bottom row

    # Place pieces left-to-right at various x positions, let them fall
    # Use a narrower range to keep silhouette compact
    silhouette_width = min(8 + level_idx * 2, grid_cols - 2)
    start_col = (grid_cols - silhouette_width) // 2

    for piece_cells in pieces:
        h, w = _shape_dims(piece_cells)
        # Pick a column that fits
        max_col = start_col + silhouette_width - w
        min_col = start_col
        if max_col < min_col:
            max_col = min_col
        place_col = rng.randint(min_col, max_col)

        # Drop piece: find the lowest row it can rest at
        for drop_row in range(floor_row, -1, -1):
            placed = [(drop_row + r, place_col + c) for r, c in piece_cells]
            # Check if any cell is below floor or overlaps
            if any(r > floor_row for r, _ in placed):
                continue
            if any((r, c) in occupied for r, c in placed):
                continue
            # Check if resting: on floor or on top of occupied cell
            resting = any(
                r == floor_row or (r + 1, c) in occupied
                for r, c in placed
            )
            if resting:
                for r, c in placed:
                    occupied.add((r, c))
                break
        else:
            # Couldn't place — just put at top
            placed = [(r, place_col + c) for r, c in piece_cells]
            for r, c in placed:
                occupied.add((r, c))

    return occupied, pieces, grid_cols, grid_rows


class ArchitectGame(ARCBaseGame):
    """Game 07: Architect — place and stack shapes to match a target silhouette."""

    def __init__(self, seed: int = 0):
        self._seed = seed
        self._rng = random.Random(seed)

        # Per-level state
        self._target_cells: Set[Tuple[int, int]] = set()
        self._pieces: List[dict] = []  # available pieces in inventory
        self._placed_pieces: List[dict] = []  # pieces on the grid
        self._selected_idx: Optional[int] = None  # index into self._pieces
        self._grid_cols = 0
        self._grid_rows = 0
        self._occupied: Set[Tuple[int, int]] = set()  # cells occupied by placed pieces

        num_levels = 5
        levels = [Level(sprites=[], grid_size=(GRID, GRID)) for _ in range(num_levels)]
        camera = Camera(0, 0, GRID, GRID, C_EMPTY, C_INV_BG)
        super().__init__(
            game_id="game_07_architect",
            levels=levels,
            camera=camera,
            available_actions=[5, 6],
            win_score=num_levels,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Level setup
    # ------------------------------------------------------------------

    def on_set_level(self, level):
        level_idx = self._levels.index(level)
        rng = random.Random(self._seed * 100 + level_idx)

        target_cells, piece_shapes, grid_cols, grid_rows = (
            _generate_target_and_pieces(rng, level_idx)
        )

        self._target_cells = target_cells
        self._grid_cols = grid_cols
        self._grid_rows = grid_rows
        self._placed_pieces = []
        self._occupied = set()
        self._selected_idx = None

        # Assign colors to pieces and store them
        colors = list(PIECE_COLORS)
        rng.shuffle(colors)
        self._pieces = []
        for i, shape in enumerate(piece_shapes):
            self._pieces.append({
                "cells": list(shape),
                "color": colors[i % len(colors)],
                "placed": False,
                "rotation": 0,  # current rotation state
            })

        self._render_all()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_all(self):
        """Clear and re-render the entire grid."""
        level = self.current_level
        # Remove all sprites
        for s in list(level._sprites):
            level.remove_sprite(s)

        # 1. Play area background (floor)
        floor_px = [[C_FLOOR] * PLAY_W for _ in range(PLAY_H)]
        floor_sprite = Sprite(
            pixels=floor_px, name="floor", visible=True, collidable=False
        )
        floor_sprite.set_position(0, 0)
        level.add_sprite(floor_sprite)

        # 2. Target silhouette
        for r, c in self._target_cells:
            px_x = c * CELL_SZ
            px_y = r * CELL_SZ
            if px_x + CELL_SZ <= PLAY_W and px_y + CELL_SZ <= PLAY_H:
                target_px = [[C_TARGET] * CELL_SZ for _ in range(CELL_SZ)]
                s = Sprite(
                    pixels=target_px,
                    name=f"target_{r}_{c}",
                    visible=True,
                    collidable=False,
                    tags=["target"],
                )
                s.set_position(px_x, px_y)
                level.add_sprite(s)

        # 3. Placed pieces (rendered on top of target)
        for pi, piece in enumerate(self._placed_pieces):
            for r, c in piece["cells"]:
                px_x = c * CELL_SZ
                px_y = r * CELL_SZ
                if px_x + CELL_SZ <= PLAY_W and px_y + CELL_SZ <= PLAY_H:
                    piece_px = [[piece["color"]] * CELL_SZ for _ in range(CELL_SZ)]
                    s = Sprite(
                        pixels=piece_px,
                        name=f"placed_{pi}_{r}_{c}",
                        visible=True,
                        collidable=False,
                        tags=["placed"],
                    )
                    s.set_position(px_x, px_y)
                    level.add_sprite(s)

        # 4. Inventory panel background
        inv_bg_px = [[C_INV_BG] * INV_W for _ in range(GRID)]
        inv_bg = Sprite(
            pixels=inv_bg_px, name="inv_bg", visible=True, collidable=False
        )
        inv_bg.set_position(INV_X, 0)
        level.add_sprite(inv_bg)

        # 5. Inventory pieces
        inv_y = 2
        for i, piece in enumerate(self._pieces):
            if piece["placed"]:
                continue
            # Draw piece in a small preview box
            h, w = _shape_dims(piece["cells"])
            preview_h = h * 2
            preview_w = w * 2

            # Border for the slot
            slot_h = max(preview_h + 2, 6)
            slot_w = INV_W - 2
            border_color = C_SELECT if self._selected_idx == i else C_INV_BORDER
            slot_px = [[border_color] * slot_w for _ in range(slot_h)]
            # Fill interior
            for sy in range(1, slot_h - 1):
                for sx in range(1, slot_w - 1):
                    slot_px[sy][sx] = C_INV_BG

            # Draw piece cells inside the slot (scaled down to 2x2 per cell)
            offset_x = (slot_w - preview_w) // 2
            offset_y = (slot_h - preview_h) // 2
            for r, c in piece["cells"]:
                for dy in range(2):
                    for dx in range(2):
                        py = offset_y + r * 2 + dy
                        px = offset_x + c * 2 + dx
                        if 0 <= py < slot_h and 0 <= px < slot_w:
                            slot_px[py][px] = piece["color"]

            slot_sprite = Sprite(
                pixels=slot_px,
                name=f"inv_slot_{i}",
                visible=True,
                collidable=False,
                tags=["inv_slot"],
            )
            slot_sprite.set_position(INV_X + 1, inv_y)
            level.add_sprite(slot_sprite)

            inv_y += slot_h + 1
            if inv_y > 55:
                break  # don't overflow inventory panel

        # 6. Level indicator on row 63
        level_idx = self._levels.index(level)
        for li in range(len(self._levels)):
            dot_color = C_LEVEL_IND if li <= level_idx else C_INV_BORDER
            dot_px = [[dot_color, dot_color]]
            dot = Sprite(
                pixels=dot_px,
                name=f"level_dot_{li}",
                visible=True,
                collidable=False,
            )
            dot.set_position(2 + li * 4, 63)
            level.add_sprite(dot)

    # ------------------------------------------------------------------
    # Gravity simulation
    # ------------------------------------------------------------------

    def _apply_gravity(self):
        """Apply gravity to all placed pieces. Pieces fall until resting."""
        floor_row = self._grid_rows - 1
        changed = True
        while changed:
            changed = False
            # Rebuild occupied set
            self._occupied = set()
            for piece in self._placed_pieces:
                for r, c in piece["cells"]:
                    self._occupied.add((r, c))

            for piece in self._placed_pieces:
                # Check if this piece can fall (all cells below are free or floor)
                piece_cells = set((r, c) for r, c in piece["cells"])
                can_fall = True
                for r, c in piece["cells"]:
                    below = (r + 1, c)
                    if r + 1 > floor_row:
                        can_fall = False
                        break
                    if below in self._occupied and below not in piece_cells:
                        can_fall = False
                        break
                if can_fall:
                    piece["cells"] = [(r + 1, c) for r, c in piece["cells"]]
                    changed = True

        # Final rebuild of occupied set
        self._occupied = set()
        for piece in self._placed_pieces:
            for r, c in piece["cells"]:
                self._occupied.add((r, c))

    # ------------------------------------------------------------------
    # Win check
    # ------------------------------------------------------------------

    def _check_win(self) -> bool:
        """Win if placed pieces exactly cover the target silhouette."""
        if not self._target_cells:
            return False
        return self._occupied == self._target_cells

    # ------------------------------------------------------------------
    # Click handling
    # ------------------------------------------------------------------

    def _handle_click(self, display_x: int, display_y: int):
        """Handle a click at display coordinates."""
        coords = self.camera.display_to_grid(display_x, display_y)
        if not coords:
            return
        grid_x, grid_y = coords

        # Check if click is in inventory area
        if grid_x >= INV_X:
            self._handle_inventory_click(grid_x, grid_y)
            return

        # Click is in play area
        if grid_x < PLAY_W and grid_y < PLAY_H:
            self._handle_play_click(grid_x, grid_y)

    def _handle_inventory_click(self, grid_x: int, grid_y: int):
        """Select a piece from inventory."""
        # Find which inventory slot was clicked
        inv_y = 2
        for i, piece in enumerate(self._pieces):
            if piece["placed"]:
                continue
            h, w = _shape_dims(piece["cells"])
            slot_h = max(h * 2 + 2, 6)
            slot_top = inv_y
            slot_bottom = inv_y + slot_h

            if slot_top <= grid_y < slot_bottom:
                if self._selected_idx == i:
                    self._selected_idx = None  # deselect
                else:
                    self._selected_idx = i
                self._render_all()
                return

            inv_y += slot_h + 1
            if inv_y > 55:
                break

    def _handle_play_click(self, grid_x: int, grid_y: int):
        """Place selected piece or remove a placed piece."""
        cell_c = grid_x // CELL_SZ
        cell_r = grid_y // CELL_SZ

        # First check if clicking on a placed piece to remove it
        if self._selected_idx is None:
            for pi, piece in enumerate(self._placed_pieces):
                if (cell_r, cell_c) in [(r, c) for r, c in piece["cells"]]:
                    # Remove this piece and return it to inventory
                    removed = self._placed_pieces.pop(pi)
                    # Find the original inventory piece by color
                    for inv_piece in self._pieces:
                        if inv_piece["color"] == removed["color"] and inv_piece["placed"]:
                            inv_piece["placed"] = False
                            break
                    # Rebuild occupied and apply gravity to remaining
                    self._apply_gravity()
                    self._render_all()
                    return
            return

        # Place selected piece
        piece = self._pieces[self._selected_idx]
        if piece["placed"]:
            self._selected_idx = None
            self._render_all()
            return

        # Calculate placement cells
        cells = piece["cells"]
        placed_cells = [(cell_r + r, cell_c + c) for r, c in cells]

        # Validate: all cells must be within play area
        floor_row = self._grid_rows - 1
        for r, c in placed_cells:
            if c < 0 or c >= self._grid_cols or r < 0:
                return  # out of bounds

        # Validate: no overlap with existing pieces
        for r, c in placed_cells:
            if (r, c) in self._occupied:
                return

        # Place the piece
        self._placed_pieces.append({
            "cells": placed_cells,
            "color": piece["color"],
        })
        piece["placed"] = True
        self._selected_idx = None

        # Apply gravity
        self._apply_gravity()

        # Check win
        if self._check_win():
            self._render_all()
            self.next_level()
            return

        self._render_all()

    # ------------------------------------------------------------------
    # Rotation
    # ------------------------------------------------------------------

    def _rotate_selected(self):
        """Rotate the selected piece 90 degrees clockwise."""
        if self._selected_idx is None:
            return
        piece = self._pieces[self._selected_idx]
        if piece["placed"]:
            return
        piece["cells"] = _rotate_shape(piece["cells"])
        self._render_all()

    # ------------------------------------------------------------------
    # Game loop
    # ------------------------------------------------------------------

    def step(self):
        action = self.action.id

        if action == GameAction.ACTION5:
            self._rotate_selected()
        elif action == GameAction.ACTION6:
            x = self.action.data.get("x", 0)
            y = self.action.data.get("y", 0)
            self._handle_click(x, y)

        self.complete_action()
