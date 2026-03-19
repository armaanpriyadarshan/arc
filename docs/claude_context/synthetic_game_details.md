# Task: Create 10 Synthetic ARC-AGI-3 Games

## Goal

I'm running an experiment to build an "intuition module" for an LLM agent playing ARC-AGI-3. The idea: an LLM plays synthetic games, writes its learnings into a structured JSON file, and that JSON gets fed as context when the LLM plays real ARC-AGI-3 games — giving it priors about game mechanics that humans naturally have but LLMs don't.

Your job is to create 10 synthetic games using the ARCEngine library. Each game should layer **multiple interacting mechanics** that the player must discover through exploration. These are NOT simple single-mechanic toys — they should be comparable in complexity to real ARC-AGI-3 games.

## Complexity Reference

Here's an example of real ARC-AGI-3 game complexity (LockSmith):
- Player navigates a grid with walls (movement mechanic)
- Limited energy that depletes per move (resource management)
- Energy pills scattered on the map (resource collection)
- A key with shape AND color properties (multi-dimensional state)
- A "shape rotator" object that cycles key shape on contact (interaction object)
- A "color rotator" object that cycles key color on contact (interaction object)
- An exit door containing a target key pattern (goal matching)
- Player must cycle both shape and color to match the door's key, THEN touch the door (multi-step goal chaining)
- 6 levels of increasing maze complexity
- Score and energy displayed on the grid itself (embedded UI)

Each of your games should have **at least 3-4 interacting mechanics** and require the player to **discover how objects interact** through experimentation. No game should be solvable by a single repeated action.

## Setup

```bash
pip install arcengine
```

## ARCEngine API Reference

All games subclass `ARCBaseGame` and implement `step()`. Here's the structure:

```python
from arcengine import (
    ARCBaseGame,
    ActionInput,
    Camera,
    GameAction,
    Level,
    Sprite,
    BlockingMode,
    InteractionMode,
)
```

### Key Concepts

- **Sprite**: 2D pixel array (palette indices 0-15, negative = transparent). Has position (x,y), scale, rotation (0/90/180/270), layer, mirroring, collision modes (NOT_BLOCKED, BOUNDING_BOX, PIXEL_PERFECT), interaction modes (TANGIBLE, INTANGIBLE, INVISIBLE, REMOVED), and tags.
- **Level**: Collection of sprites + optional grid_size (width, height) + metadata dict via `data={}`. Methods: `add_sprite()`, `remove_sprite()`, `get_sprite_at(x, y)`, `get_sprites_by_tag(tag)`, `get_sprites_by_name(name)`, `collides_with(sprite)`.
- **Camera**: Viewport that always outputs 64x64. Smaller viewports get upscaled nearest-neighbor + letterboxed. `display_to_grid(x, y)` converts display coords to grid coords. Constructor: `Camera(x, y, width, height, background, letter_box)`.
- **ARCBaseGame**: Base class. Constructor: `__init__(game_id, levels, camera, available_actions=[1,2,3,4,5,6], seed=0)`. Key methods: `complete_action()`, `next_level()`, `win()`, `lose()`, `try_move(sprite_name, dx, dy)` (returns list of collided sprites, moves sprite if no collisions), `set_level(index)`, `full_reset()`, `level_reset()`. Override `step()` for game logic, `on_set_level(level)` for level setup.
- **GameAction**: `RESET` (0), `ACTION1` (up/W), `ACTION2` (down/S), `ACTION3` (left/A), `ACTION4` (right/D), `ACTION5` (space/interact), `ACTION6` (click at x,y — access via `self.action.data["x"]` and `self.action.data["y"]`), `ACTION7` (undo/Z).
- **perform_action(ActionInput(id=GameAction.ACTION1))** runs the game loop and returns frame data.
- Sprite methods: `clone()`, `set_position(x,y)`, `move(dx,dy)`, `set_scale(s)`, `set_rotation(r)`, `rotate(delta)`, `color_remap(old, new)`, `set_visible(bool)`, `set_collidable(bool)`, `merge(other)`, `set_mirror_lr(bool)`, `set_mirror_ud(bool)`.

### Example Game (click-to-remove sprites)

```python
import random
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

sprites = {
    "sprite-1": Sprite(pixels=[[9]], name="sprite-1", visible=True, collidable=True),
}

levels = [
    Level(sprites=[], grid_size=(8, 8)),
    Level(sprites=[], grid_size=(16, 16)),
    Level(sprites=[], grid_size=(32, 32)),
]

class Ab12(ARCBaseGame):
    def __init__(self, seed=0):
        self._rng = random.Random(seed)
        camera = Camera(background=0, letter_box=4, width=8, height=8)
        super().__init__(game_id="ab12", levels=levels, camera=camera)

    def generate_sprites(self):
        cell_count = self.current_level.grid_size[0] * self.current_level.grid_size[1]
        sprite_count = cell_count // 64
        for idx in range(sprite_count):
            scale = self._rng.randint(1, 4)
            color = self._rng.choice([6, 7, 8, 9, 10, 11])
            x = self._rng.randint(0, self.current_level.grid_size[0] - 1)
            y = self._rng.randint(0, self.current_level.grid_size[1] - 1)
            sprite = sprites["sprite-1"].clone().color_remap(None, color).set_scale(scale).set_position(x, y)
            self.current_level.add_sprite(sprite)

    def on_set_level(self, level):
        self.generate_sprites()

    def _check_win(self):
        return len(self.current_level._sprites) == 0

    def step(self):
        if self.action.id == GameAction.ACTION6:
            x = self.action.data.get("x", 0)
            y = self.action.data.get("y", 0)
            coords = self.camera.display_to_grid(x, y)
            if coords:
                grid_x, grid_y = coords
                clicked_sprite = self.current_level.get_sprite_at(grid_x, grid_y)
                if clicked_sprite:
                    self.current_level.remove_sprite(clicked_sprite)
                    if self._check_win():
                        self.next_level()
        self.complete_action()
```

## The 10 Games To Build

Each game below describes the **layered mechanics** the player must discover. Every game should have 4-6 levels with increasing complexity (more objects, larger grids, trickier layouts). Use `seed` for reproducibility. Keep each game in a single file.

---

### Game 01: `game_01_courier.py` — Courier

**Mechanics:** Navigation + key collection + locked doors + limited fuel + refueling

The player moves through a walled grid carrying packages. Each package has a color. Colored doors block paths and only open when the player is carrying a package of the matching color. Delivering a package to a matching mailbox removes it from inventory. The player has limited fuel (decrements each move). Fuel stations (2x2 green squares) refill fuel. All mailboxes must receive their packages to complete the level. Later levels have multiple packages requiring route planning to avoid running out of fuel.

- Actions: WASD movement only (ACTION1-4)
- Display: fuel gauge on row 62, current carried package color on row 63
- Win: all packages delivered

---

### Game 02: `game_02_wiring.py` — Wiring

**Mechanics:** Click-to-place + signal propagation + color mixing + toggle switches

The grid shows power sources (colored squares on the left edge) and light bulbs (on the right edge) that need to be lit. The player clicks empty cells (ACTION6) to place wire segments. Wires propagate the color of whatever power source they connect to. If two different-colored signals meet at a junction, they mix (e.g., red + blue = purple). Each bulb requires a specific color. Toggle switches (ACTION5 to cycle) on the grid flip which power sources are active. Player must wire the grid AND set switches correctly so all bulbs receive the right color.

- Actions: ACTION5 (toggle switch, must be adjacent), ACTION6 (place wire)
- Win: all bulbs lit with correct colors

---

### Game 03: `game_03_alchemist.py` — Alchemist

**Mechanics:** Navigation + item collection + crafting/combining + recipe discovery + goal matching

The player navigates a grid collecting ingredients (colored sprites). An "anvil" sprite on the map lets the player combine their two held ingredients (ACTION5 when adjacent) into a new item based on hidden recipes (e.g., red+blue=purple, green+yellow=orange). A display area shows what item is needed to unlock the exit. The player must figure out which combinations produce which results. Later levels require multi-step crafting (combine A+B=C, then C+D=E). The player can only hold 2 items at a time. Dropping items (ACTION5 on empty ground) is allowed.

- Actions: WASD movement, ACTION5 (combine at anvil / pick up / drop)
- Display: held items shown on rows 62-63, target item shown near exit
- Win: craft the target item and bring it to the exit

---

### Game 04: `game_04_mirror_maze.py` — Mirror Maze

**Mechanics:** Navigation + beam reflection + mirror rotation + beam-activated doors + limited moves

A beam of light enters the grid from one edge. Mirror sprites (diagonal 2x2 blocks) are placed on the grid. The player can rotate mirrors by moving onto them and pressing ACTION5. The beam reflects off mirrors and must be directed to hit a target receptor. When the beam hits the receptor, a door opens revealing the exit. The player has a limited move counter. Later levels have multiple beams of different colors that must each hit their matching receptor, and mirrors that only reflect certain colors.

- Actions: WASD movement, ACTION5 (rotate mirror when adjacent)
- Display: move counter on row 62, beam paths rendered as colored lines on the grid
- Win: all beams hitting their matching receptors

---

### Game 05: `game_05_terraformer.py` — Terraformer

**Mechanics:** Click-to-paint terrain + adjacency rules + growth simulation + target matching

The grid starts with a terrain pattern. The player clicks cells (ACTION6) to change terrain type, cycling through: water(blue) → grass(green) → sand(yellow) → rock(gray) → water. But terrain follows adjacency rules: grass dies (becomes sand) if not adjacent to water. Water spreads to adjacent sand after each action. Rock blocks water spread. The game simulates one step of these rules after each player action. A target pattern is shown (on a mini-display in the corner). The player must paint and let the simulation evolve until the grid matches the target.

- Actions: ACTION6 (cycle terrain at clicked cell), ACTION5 (advance simulation one step without painting)
- Win: grid matches target pattern

---

### Game 06: `game_06_conductor.py` — Conductor

**Mechanics:** Timing + sequencing + multiple agents + command queuing

The grid has multiple "train" sprites on rail tracks. Each train moves forward automatically each turn. The player uses ACTION6 to click on track switches to toggle their direction (left/right fork). Trains must each reach their matching colored station. If two trains collide, the level resets. If a train goes off the edge, the level resets. Later levels have trains entering at different times, requiring the player to toggle switches at precise moments. ACTION5 pauses/unpauses the simulation so the player can plan.

- Actions: ACTION5 (pause/unpause), ACTION6 (click to toggle track switch)
- Win: all trains reach their matching stations without collision

---

### Game 07: `game_07_architect.py` — Architect

**Mechanics:** Drag-and-place shapes + rotation + gravity + structural stability + target silhouette

The player has a set of tetromino-like shapes in an inventory panel. Using ACTION6, they click to select a piece, then click to place it on the grid. ACTION5 rotates the selected piece. Placed pieces fall due to gravity until they rest on the ground or another piece. Unsupported pieces (overhanging with nothing below) will topple/fall. The level shows a target silhouette outline. The player must stack pieces to fill the silhouette exactly. Later levels have irregular shapes and require specific placement order since pieces interact physically.

- Actions: ACTION5 (rotate selected piece), ACTION6 (select piece / place piece)
- Display: available pieces in top-right panel, target silhouette outline on the grid
- Win: placed pieces match the target silhouette

---

### Game 08: `game_08_cipher.py` — Cipher

**Mechanics:** Symbol observation + encoding rules + input sequence + hidden transformation + verification

The grid displays an "encoder machine" — sprites enter from the left, pass through transformation zones, and exit on the right changed. The player observes several input→output examples shown on the top half of the grid. The bottom half has an empty input row and a target output row. The player must place the correct input symbols (using ACTION6 to cycle cell values, ACTION5 to submit) such that after the transformation the output matches the target. The transformation rule changes each level (e.g., shift color by +2, swap even/odd positions, mirror the sequence, apply a substitution cipher). The player must deduce the rule from examples.

- Actions: ACTION5 (submit/confirm input), ACTION6 (cycle cell value at position)
- Win: submitted input produces the target output when transformed

---

### Game 09: `game_09_ecosystem.py` — Ecosystem

**Mechanics:** Entity placement + predator/prey simulation + population targets + environmental modifiers

The grid represents a habitat. The player places animals and plants (using ACTION6 to select type, then click to place). After placement, pressing ACTION5 runs a simulation step: herbivores move toward plants, predators move toward herbivores, animals reproduce if well-fed, animals die if starving. Environmental tiles (water, desert, forest) modify movement speed and reproduction rates. A target panel shows required population counts for each species after N simulation steps. The player must figure out initial placements that evolve to match the target. Later levels have more species and environmental complexity.

- Actions: ACTION1-4 (cycle selected entity type), ACTION5 (run simulation step), ACTION6 (place entity)
- Display: population counts on row 62-63, target counts in corner panel
- Win: population counts match targets after required simulation steps

---

### Game 10: `game_10_shapeshifter.py` — Shapeshifter

**Mechanics:** Player transformation + ability-gating + environmental puzzles + state memory

The player navigates a grid but can shapeshift between 3 forms using ACTION5: small (1x1, can fit through narrow gaps), medium (2x2, can push moveable blocks), large (4x4, can break cracked walls but can't fit in tight spaces). Each form has a different color. Some floor tiles are pressure plates that only activate under specific weights (sizes). Activating the right combination of pressure plates opens doors. Some areas require being small to enter, then shifting to large inside to break a wall, then shifting to small again to exit. Later levels add "shift cooldown" (can only transform every N moves) and "form-locked zones" (areas where shifting is disabled).

- Actions: WASD movement (ACTION1-4), ACTION5 (shapeshift to next form)
- Display: current form indicator on row 63, shift cooldown counter on row 62
- Win: reach the exit door

---

## Requirements

- Each game must have 4-6 levels with increasing difficulty.
- Each game must use `seed` for reproducible procedural generation of layouts.
- Each game should render meaningful state to the 64x64 grid — walls, objects, indicators, status bars. The grid IS the display.
- Keep each game self-contained in a single file.
- Use colors 0-15 deliberately and consistently within each game (document the color key in the docstring).
- Each game must have a clear win condition via `next_level()` or `win()`.
- Include a docstring at the top of each file explaining: the mechanic, how to win, what each action does, and the color key.
- Do NOT include any instructions or rules in the game code that get shown to the player — the whole point is that the player must discover the rules through experimentation.
- Create a `run_all.py` script that instantiates each game and runs 500 random actions, printing level progress and whether the game was completed. This is for sanity-checking.

## File Structure

```
synthetic_games/
├── game_01_courier.py
├── game_02_wiring.py
├── game_03_alchemist.py
├── game_04_mirror_maze.py
├── game_05_terraformer.py
├── game_06_conductor.py
├── game_07_architect.py
├── game_08_cipher.py
├── game_09_ecosystem.py
├── game_10_shapeshifter.py
└── run_all.py
```
