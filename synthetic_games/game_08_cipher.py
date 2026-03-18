"""
Game 08: Cipher

Mechanics: Symbol observation + encoding rules + input sequence + hidden transformation + verification

The grid displays an "encoder machine." The top half shows several input→output
example pairs that demonstrate a hidden transformation rule. The bottom half has
an editable input row and a target output row. The player must place the correct
input symbols (using ACTION6 to cycle cell values at clicked positions) such that
after the hidden transformation, the output matches the target. ACTION5 submits
the current input for verification.

The transformation rule changes each level:
  Level 1: Shift each color value by +2 (mod 9, within symbol range 6-14)
  Level 2: Reverse the sequence order
  Level 3: Swap adjacent pairs (pos 0↔1, 2↔3, …); odd-length keeps last
  Level 4: Rotate sequence left by 1 (first element goes to end)
  Level 5: Each symbol is replaced by (symbol + position_index) mod 9, mapped to 6-14
  Level 6: Reverse, then shift each by +1

Actions:
    ACTION5 = Submit current input for verification
    ACTION6 = Click a cell in the input row to cycle its symbol value

Win condition: Submitted input, when transformed, matches the target output.

Color key:
    0  = empty / background
    1  = border / frame lines
    2  = arrow indicator (shows input→output direction)
    3  = machine zone / transformation area background
    4  = selection cursor / highlight
    5  = HUD background
    6  = symbol: blue
    7  = symbol: yellow
    8  = symbol: red
    9  = symbol: purple
    10 = symbol: green
    11 = symbol: orange
    12 = symbol: white
    13 = symbol: teal
    14 = symbol: pink
    15 = label / status indicator
"""

import random
from typing import List, Optional, Tuple

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
BG_COLOR = 0
BORDER_COLOR = 1
ARROW_COLOR = 2
MACHINE_COLOR = 3
CURSOR_COLOR = 4
HUD_BG = 5
LABEL_COLOR = 15

# Symbol palette (the values that get transformed)
SYMBOL_COLORS = [6, 7, 8, 9, 10, 11, 12, 13, 14]
NUM_SYMBOLS = len(SYMBOL_COLORS)  # 9

# Layout constants
CELL_SIZE = 4        # each symbol cell is 4x4 pixels
CELL_GAP = 1         # gap between cells
EXAMPLE_START_Y = 2  # y-pixel where first example row starts
ROW_SPACING = 6      # vertical spacing between example rows (cell + gap + arrow row)
INPUT_ROW_Y = 44     # y-pixel for the player's editable input row
TARGET_ROW_Y = 54    # y-pixel for the target output row
ARROW_COL_X = 30     # x-pixel for the arrow between input and output columns
INPUT_COL_X = 4      # x-pixel where input sequences start
OUTPUT_COL_X = 36    # x-pixel where output sequences start


# ---------------------------------------------------------------------------
# Transformation functions
# ---------------------------------------------------------------------------

def _shift_plus_2(seq: List[int]) -> List[int]:
    """Shift each symbol by +2 within the symbol range."""
    return [SYMBOL_COLORS[(SYMBOL_COLORS.index(s) + 2) % NUM_SYMBOLS] for s in seq]


def _reverse(seq: List[int]) -> List[int]:
    """Reverse the sequence."""
    return list(reversed(seq))


def _swap_pairs(seq: List[int]) -> List[int]:
    """Swap adjacent pairs: 0↔1, 2↔3, etc."""
    result = list(seq)
    for i in range(0, len(result) - 1, 2):
        result[i], result[i + 1] = result[i + 1], result[i]
    return result


def _rotate_left(seq: List[int]) -> List[int]:
    """Rotate sequence left by 1 position."""
    if len(seq) <= 1:
        return list(seq)
    return seq[1:] + [seq[0]]


def _shift_by_position(seq: List[int]) -> List[int]:
    """Each symbol shifted by its position index."""
    return [SYMBOL_COLORS[(SYMBOL_COLORS.index(s) + i) % NUM_SYMBOLS]
            for i, s in enumerate(seq)]


def _reverse_then_shift(seq: List[int]) -> List[int]:
    """Reverse, then shift each by +1."""
    rev = list(reversed(seq))
    return [SYMBOL_COLORS[(SYMBOL_COLORS.index(s) + 1) % NUM_SYMBOLS] for s in rev]


TRANSFORMS = [
    _shift_plus_2,
    _reverse,
    _swap_pairs,
    _rotate_left,
    _shift_by_position,
    _reverse_then_shift,
]


# ---------------------------------------------------------------------------
# Sprite builders
# ---------------------------------------------------------------------------

def _make_symbol_cell(color: int, name: str, tags: Optional[List[str]] = None) -> Sprite:
    """Create a CELL_SIZE x CELL_SIZE solid-color sprite for a symbol."""
    px = [[color] * CELL_SIZE for _ in range(CELL_SIZE)]
    return Sprite(pixels=px, name=name, visible=True, collidable=False,
                  tags=tags or [])


def _make_arrow(name: str) -> Sprite:
    """Small 3x3 right-pointing arrow sprite."""
    px = [
        [BG_COLOR, ARROW_COLOR, BG_COLOR],
        [ARROW_COLOR, ARROW_COLOR, ARROW_COLOR],
        [BG_COLOR, ARROW_COLOR, BG_COLOR],
    ]
    return Sprite(pixels=px, name=name, visible=True, collidable=False,
                  tags=["arrow"])


def _make_cursor() -> Sprite:
    """Cursor sprite: 4x4 border highlight."""
    px = [
        [CURSOR_COLOR, CURSOR_COLOR, CURSOR_COLOR, CURSOR_COLOR],
        [CURSOR_COLOR, BG_COLOR, BG_COLOR, CURSOR_COLOR],
        [CURSOR_COLOR, BG_COLOR, BG_COLOR, CURSOR_COLOR],
        [CURSOR_COLOR, CURSOR_COLOR, CURSOR_COLOR, CURSOR_COLOR],
    ]
    return Sprite(pixels=px, name="cursor", visible=False, collidable=False,
                  tags=["cursor"])


def _make_border_line(width: int, height: int, name: str) -> Sprite:
    """Horizontal or vertical border line."""
    px = [[BORDER_COLOR] * width for _ in range(height)]
    return Sprite(pixels=px, name=name, visible=True, collidable=False,
                  tags=["border"])


def _make_label_block(width: int, name: str) -> Sprite:
    """Small label indicator block."""
    px = [[LABEL_COLOR] * width for _ in range(2)]
    return Sprite(pixels=px, name=name, visible=True, collidable=False,
                  tags=["label"])


def _make_status_indicator(color: int, name: str) -> Sprite:
    """Small status dot for HUD."""
    px = [[color] * 3 for _ in range(3)]
    return Sprite(pixels=px, name=name, visible=True, collidable=False,
                  tags=["hud_element"])


# ---------------------------------------------------------------------------
# Game class
# ---------------------------------------------------------------------------

class CipherGame(ARCBaseGame):
    """Game 08: Cipher — deduce transformation rules from examples."""

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)
        self._seed = seed

        # Level configs: (sequence_length, num_examples)
        self._level_configs = [
            (4, 3),   # Level 1: short sequences, 3 examples
            (4, 3),   # Level 2: short, 3 examples
            (5, 3),   # Level 3: medium, 3 examples
            (5, 4),   # Level 4: medium, 4 examples
            (6, 3),   # Level 5: longer, 3 examples
            (6, 4),   # Level 6: longer, 4 examples
        ]

        # Per-level state
        self._seq_len = 0
        self._num_examples = 0
        self._transform_fn = None
        self._examples: List[Tuple[List[int], List[int]]] = []
        self._target_output: List[int] = []
        self._correct_input: List[int] = []
        self._player_input: List[int] = []
        self._submitted = False
        self._cursor_pos: Optional[int] = None

        levels = [Level(sprites=[], grid_size=(GRID, GRID))
                  for _ in self._level_configs]
        camera = Camera(0, 0, GRID, GRID, BG_COLOR, HUD_BG)
        super().__init__(
            game_id="game_08_cipher",
            levels=levels,
            camera=camera,
            available_actions=[5, 6],
            win_score=len(self._level_configs),
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Level setup
    # ------------------------------------------------------------------

    def on_set_level(self, level):
        """Generate the level: examples, target, and transformation rule."""
        level_idx = self._levels.index(level)
        rng = random.Random(self._seed * 100 + level_idx)

        cfg = self._level_configs[level_idx]
        self._seq_len, self._num_examples = cfg

        # Select transformation for this level
        self._transform_fn = TRANSFORMS[level_idx]

        # Generate example pairs
        self._examples = []
        for _ in range(self._num_examples):
            inp = [rng.choice(SYMBOL_COLORS) for _ in range(self._seq_len)]
            out = self._transform_fn(inp)
            self._examples.append((inp, out))

        # Generate the target: pick a random input, transform it for the target
        self._correct_input = [rng.choice(SYMBOL_COLORS) for _ in range(self._seq_len)]
        self._target_output = self._transform_fn(self._correct_input)

        # Make sure correct_input is not identical to any example input
        # (to prevent trivial copying)
        attempts = 0
        while attempts < 20:
            duplicate = False
            for ex_in, _ in self._examples:
                if ex_in == self._correct_input:
                    duplicate = True
                    break
            if not duplicate:
                break
            self._correct_input = [rng.choice(SYMBOL_COLORS) for _ in range(self._seq_len)]
            self._target_output = self._transform_fn(self._correct_input)
            attempts += 1

        # Initialize player input to first symbol color
        self._player_input = [SYMBOL_COLORS[0]] * self._seq_len
        self._submitted = False
        self._cursor_pos = None

        # Render everything
        self._render_level()

    def _render_level(self):
        """Render all visual elements onto the level."""
        level = self.current_level

        # Clear all dynamic sprites
        for tag in ["example", "arrow", "input_cell", "target_cell",
                     "border", "label", "hud_element", "cursor",
                     "section_bg", "result_cell"]:
            for s in list(level.get_sprites_by_tag(tag)):
                level.remove_sprite(s)

        # -- Separator line between examples and input area --
        sep = _make_border_line(GRID, 1, "separator")
        sep.set_position(0, 40)
        level.add_sprite(sep)

        # -- Render example pairs --
        for i, (inp, out) in enumerate(self._examples):
            y = EXAMPLE_START_Y + i * (CELL_SIZE + ROW_SPACING)

            # Input symbols
            for j, color in enumerate(inp):
                x = INPUT_COL_X + j * (CELL_SIZE + CELL_GAP)
                cell = _make_symbol_cell(color, f"ex_in_{i}_{j}", tags=["example"])
                cell.set_position(x, y)
                level.add_sprite(cell)

            # Arrow
            arrow = _make_arrow(f"arrow_{i}")
            arrow.set_position(ARROW_COL_X, y + 1)
            level.add_sprite(arrow)

            # Output symbols
            for j, color in enumerate(out):
                x = OUTPUT_COL_X + j * (CELL_SIZE + CELL_GAP)
                cell = _make_symbol_cell(color, f"ex_out_{i}_{j}", tags=["example"])
                cell.set_position(x, y)
                level.add_sprite(cell)

        # -- Render player input row --
        self._render_input_row()

        # -- Render target output row --
        for j, color in enumerate(self._target_output):
            x = OUTPUT_COL_X + j * (CELL_SIZE + CELL_GAP)
            cell = _make_symbol_cell(color, f"target_{j}", tags=["target_cell"])
            cell.set_position(x, TARGET_ROW_Y)
            level.add_sprite(cell)

        # Arrow between input and target
        arrow = _make_arrow("arrow_target")
        arrow.set_position(ARROW_COL_X, INPUT_ROW_Y + 1)
        level.add_sprite(arrow)

        # "?" label next to input row to indicate editable
        q_label = _make_label_block(2, "q_label")
        q_label.set_position(1, INPUT_ROW_Y + 1)
        level.add_sprite(q_label)

        # Target label (small block next to target row)
        t_label = _make_label_block(2, "t_label")
        t_label.set_position(OUTPUT_COL_X - 4, TARGET_ROW_Y + 1)
        level.add_sprite(t_label)

        # -- HUD: level indicator --
        level_idx = self._levels.index(level)
        for i in range(len(self._level_configs)):
            color = LABEL_COLOR if i <= level_idx else BG_COLOR
            dot = _make_status_indicator(color, f"level_dot_{i}")
            dot.set_position(2 + i * 5, 61)
            level.add_sprite(dot)

    def _render_input_row(self):
        """Render/refresh just the player's editable input cells."""
        level = self.current_level

        # Remove old input cells
        for s in list(level.get_sprites_by_tag("input_cell")):
            level.remove_sprite(s)
        for s in list(level.get_sprites_by_tag("cursor")):
            level.remove_sprite(s)
        for s in list(level.get_sprites_by_tag("result_cell")):
            level.remove_sprite(s)

        for j, color in enumerate(self._player_input):
            x = INPUT_COL_X + j * (CELL_SIZE + CELL_GAP)
            cell = _make_symbol_cell(color, f"input_{j}", tags=["input_cell"])
            cell.set_position(x, INPUT_ROW_Y)
            level.add_sprite(cell)

        # Render cursor if active
        if self._cursor_pos is not None and 0 <= self._cursor_pos < self._seq_len:
            x = INPUT_COL_X + self._cursor_pos * (CELL_SIZE + CELL_GAP)
            cursor = _make_cursor()
            cursor.set_position(x, INPUT_ROW_Y)
            cursor.set_visible(True)
            level.add_sprite(cursor)

    def _render_result(self, success: bool):
        """Show the transformed result next to the target for feedback."""
        level = self.current_level

        # Remove old result cells
        for s in list(level.get_sprites_by_tag("result_cell")):
            level.remove_sprite(s)

        # Show transformed output below target
        transformed = self._transform_fn(self._player_input)
        result_y = TARGET_ROW_Y - (CELL_SIZE + CELL_GAP + 1)

        for j, color in enumerate(transformed):
            x = OUTPUT_COL_X + j * (CELL_SIZE + CELL_GAP)
            cell = _make_symbol_cell(color, f"result_{j}", tags=["result_cell"])
            cell.set_position(x, result_y)
            level.add_sprite(cell)

        # Status indicator
        status_color = 10 if success else 8  # green for success, red for fail
        status = _make_status_indicator(status_color, "status")
        status.set_position(58, INPUT_ROW_Y + 1)
        level.add_sprite(status)

    # ------------------------------------------------------------------
    # Click detection
    # ------------------------------------------------------------------

    def _get_input_cell_at(self, grid_x: int, grid_y: int) -> Optional[int]:
        """Return the index of the input cell at grid coords, or None."""
        if not (INPUT_ROW_Y <= grid_y < INPUT_ROW_Y + CELL_SIZE):
            return None

        for j in range(self._seq_len):
            x_start = INPUT_COL_X + j * (CELL_SIZE + CELL_GAP)
            if x_start <= grid_x < x_start + CELL_SIZE:
                return j
        return None

    # ------------------------------------------------------------------
    # Game loop
    # ------------------------------------------------------------------

    def step(self):
        action = self.action.id

        if action == GameAction.ACTION6:
            # Click to cycle input cell value
            x = self.action.data.get("x", 0)
            y = self.action.data.get("y", 0)
            coords = self.camera.display_to_grid(x, y)
            if coords:
                grid_x, grid_y = coords
                cell_idx = self._get_input_cell_at(grid_x, grid_y)
                if cell_idx is not None:
                    # Cycle to next symbol color
                    current = self._player_input[cell_idx]
                    current_idx = SYMBOL_COLORS.index(current)
                    next_idx = (current_idx + 1) % NUM_SYMBOLS
                    self._player_input[cell_idx] = SYMBOL_COLORS[next_idx]
                    self._cursor_pos = cell_idx
                    self._submitted = False
                    self._render_input_row()

        elif action == GameAction.ACTION5:
            # Submit: check if transformed input matches target
            transformed = self._transform_fn(self._player_input)
            self._submitted = True

            if transformed == self._target_output:
                self._render_result(True)
                self.next_level()
            else:
                self._render_result(False)

        self.complete_action()
