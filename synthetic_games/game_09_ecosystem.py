"""
Game 09: Ecosystem

Mechanics:
  1. Entity placement — player selects an entity type, then clicks to place it
     on the grid. ACTION1-4 cycle through entity types. ACTION6 places the
     currently selected entity at the clicked position.
  2. Predator/prey simulation — ACTION5 advances one simulation step:
     - Plants grow: each plant has a chance to spread to an adjacent empty land cell
     - Herbivores move toward the nearest plant, eat it on contact (gain food)
     - Predators move toward the nearest herbivore, eat it on contact (gain food)
     - Well-fed animals (food >= reproduction threshold) reproduce into an adjacent
       empty cell and reset their food counter
     - Starving animals (food <= 0) die and are removed
     - Each step, every animal loses 1 food (hunger)
  3. Environmental modifiers — terrain tiles affect entity behavior:
     - Water (blue): impassable to all land entities
     - Desert (yellow): animals lose 2 food per step instead of 1
     - Forest (green): plants spread faster; herbivores gain +1 bonus food when eating
     - Marsh (teal): predators move slower (skip every other step)
  4. Population targets — a target panel shows required counts for each species
     after a required number of simulation steps. Player must figure out initial
     placements that evolve to match the targets.
  5. Win condition — after running the required number of sim steps, if population
     counts match targets exactly (within a tolerance window), the level advances.

How to win:
  Place entities strategically on the grid, then run simulation steps (ACTION5).
  After the required number of steps, the population of each species must match
  the target counts. Environmental tiles affect survival and reproduction, so
  placement location matters.

Actions:
  ACTION1 = Cycle entity type backward
  ACTION2 = Cycle entity type forward
  ACTION3 = Decrease placement count (for batch info)
  ACTION4 = Increase placement count (for batch info)
  ACTION5 = Run one simulation step
  ACTION6 = Place selected entity at clicked (x, y)

Color key (palette indices):
  0  = black          (empty/void)
  1  = dark blue      (water terrain)
  2  = red            (predator entity)
  3  = green          (plant entity)
  4  = yellow         (desert terrain)
  5  = gray           (UI background)
  6  = magenta        (herbivore entity - alt)
  7  = orange         (herbivore entity)
  8  = teal/cyan      (marsh terrain)
  9  = blue           (water deep)
  10 = light green    (forest terrain)
  11 = yellow-green   (plant on forest)
  12 = light orange   (target indicator)
  13 = purple         (predator-alt / fish)
  14 = bright green   (selected entity highlight)
  15 = white          (UI text / markers)

Display:
  Rows 0-1: target population panel (species color + count bars)
  Row 62: simulation step counter + steps remaining
  Row 63: selected entity type indicator + entity palette
"""

import random
from typing import List, Tuple, Optional, Dict

import numpy as np

from arcengine import (
    ARCBaseGame,
    Camera,
    Level,
    Sprite,
)

# ── Entity type constants ─────────────────────────────────────────
ENT_NONE = 0
ENT_PLANT = 1
ENT_HERBIVORE = 2
ENT_PREDATOR = 3
ENT_FISH = 4  # aquatic herbivore, only in later levels

NUM_ENTITY_TYPES_BY_LEVEL = [3, 3, 4, 4, 4]  # how many types available per level

# ── Terrain constants ─────────────────────────────────────────────
TERRAIN_EMPTY = 0
TERRAIN_WATER = 1
TERRAIN_DESERT = 2
TERRAIN_FOREST = 3
TERRAIN_MARSH = 4

# ── Color palette ─────────────────────────────────────────────────
C_VOID = 0       # black - empty land
C_WATER = 1      # dark blue
C_PREDATOR = 2   # red
C_PLANT = 3      # green
C_DESERT = 4     # yellow
C_UI_BG = 5      # gray
C_HERB_ALT = 6   # magenta
C_HERBIVORE = 7  # orange
C_MARSH = 8      # teal/cyan
C_WATER_DEEP = 9 # blue
C_FOREST = 10    # light green
C_PLANT_FOREST = 11  # yellow-green (plant on forest)
C_TARGET = 12    # light orange
C_FISH = 13      # purple
C_SELECTED = 14  # bright green highlight
C_WHITE = 15     # white markers

TERRAIN_COLORS = {
    TERRAIN_EMPTY: C_VOID,
    TERRAIN_WATER: C_WATER,
    TERRAIN_DESERT: C_DESERT,
    TERRAIN_FOREST: C_FOREST,
    TERRAIN_MARSH: C_MARSH,
}

ENTITY_COLORS = {
    ENT_PLANT: C_PLANT,
    ENT_HERBIVORE: C_HERBIVORE,
    ENT_PREDATOR: C_PREDATOR,
    ENT_FISH: C_FISH,
}

ENTITY_NAMES = {
    ENT_PLANT: "Plant",
    ENT_HERBIVORE: "Herbivore",
    ENT_PREDATOR: "Predator",
    ENT_FISH: "Fish",
}

DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
DIRS_8 = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]

# ── Grid dimensions ───────────────────────────────────────────────
GRID_W = 60
GRID_H = 58  # rows 0-1 for target panel, rows 62-63 for UI => play area rows 2-59

PLAY_Y_START = 3   # first row of play area
PLAY_Y_END = 61    # last row + 1 of play area
PLAY_X_START = 1
PLAY_X_END = 61


# ── Animal entity ─────────────────────────────────────────────────
class Animal:
    __slots__ = ("etype", "x", "y", "food", "alive", "skip_next")

    def __init__(self, etype: int, x: int, y: int, food: int = 3):
        self.etype = etype
        self.x = x
        self.y = y
        self.food = food
        self.alive = True
        self.skip_next = False


# ── Level generation ──────────────────────────────────────────────
def _generate_level(seed: int, level_idx: int) -> dict:
    rng = random.Random(seed + level_idx * 7919)
    pw = PLAY_X_END - PLAY_X_START  # play area width
    ph = PLAY_Y_END - PLAY_Y_START  # play area height

    # Generate terrain grid (play area only)
    terrain = [[TERRAIN_EMPTY for _ in range(pw)] for _ in range(ph)]

    # Water bodies
    num_water = 2 + level_idx
    for _ in range(num_water):
        cx = rng.randint(3, pw - 4)
        cy = rng.randint(3, ph - 4)
        size = rng.randint(2, 3 + level_idx // 2)
        for dx in range(-size, size + 1):
            for dy in range(-size, size + 1):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < pw and 0 <= ny < ph:
                    dist = abs(dx) + abs(dy)
                    if dist <= size and rng.random() < 0.7:
                        terrain[ny][nx] = TERRAIN_WATER

    # Desert patches (from level 1+)
    if level_idx >= 1:
        num_desert = 1 + level_idx
        for _ in range(num_desert):
            cx = rng.randint(2, pw - 3)
            cy = rng.randint(2, ph - 3)
            size = rng.randint(1, 2 + level_idx // 2)
            for dx in range(-size, size + 1):
                for dy in range(-size, size + 1):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < pw and 0 <= ny < ph and terrain[ny][nx] == TERRAIN_EMPTY:
                        if abs(dx) + abs(dy) <= size and rng.random() < 0.6:
                            terrain[ny][nx] = TERRAIN_DESERT

    # Forest patches
    num_forest = 2 + level_idx
    for _ in range(num_forest):
        cx = rng.randint(2, pw - 3)
        cy = rng.randint(2, ph - 3)
        size = rng.randint(2, 3 + level_idx // 2)
        for dx in range(-size, size + 1):
            for dy in range(-size, size + 1):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < pw and 0 <= ny < ph and terrain[ny][nx] == TERRAIN_EMPTY:
                    if abs(dx) + abs(dy) <= size and rng.random() < 0.5:
                        terrain[ny][nx] = TERRAIN_FOREST

    # Marsh patches (from level 2+)
    if level_idx >= 2:
        num_marsh = level_idx - 1
        for _ in range(num_marsh):
            cx = rng.randint(2, pw - 3)
            cy = rng.randint(2, ph - 3)
            size = rng.randint(1, 2)
            for dx in range(-size, size + 1):
                for dy in range(-size, size + 1):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < pw and 0 <= ny < ph and terrain[ny][nx] == TERRAIN_EMPTY:
                        if abs(dx) + abs(dy) <= size and rng.random() < 0.5:
                            terrain[ny][nx] = TERRAIN_MARSH

    # Determine target populations
    # We'll set reasonable targets that a good placement could achieve
    sim_steps_required = 5 + level_idx * 3  # more steps needed in later levels

    # Entity type count for this level
    num_types = NUM_ENTITY_TYPES_BY_LEVEL[min(level_idx, len(NUM_ENTITY_TYPES_BY_LEVEL) - 1)]

    # Target populations: plants, herbivores, predators, (fish if level >= 2)
    targets: Dict[int, int] = {}
    if level_idx == 0:
        targets[ENT_PLANT] = rng.randint(6, 10)
        targets[ENT_HERBIVORE] = rng.randint(3, 5)
        targets[ENT_PREDATOR] = rng.randint(1, 3)
    elif level_idx == 1:
        targets[ENT_PLANT] = rng.randint(8, 14)
        targets[ENT_HERBIVORE] = rng.randint(4, 7)
        targets[ENT_PREDATOR] = rng.randint(2, 4)
    elif level_idx == 2:
        targets[ENT_PLANT] = rng.randint(10, 16)
        targets[ENT_HERBIVORE] = rng.randint(5, 9)
        targets[ENT_PREDATOR] = rng.randint(2, 5)
        targets[ENT_FISH] = rng.randint(2, 4)
    elif level_idx == 3:
        targets[ENT_PLANT] = rng.randint(12, 20)
        targets[ENT_HERBIVORE] = rng.randint(6, 10)
        targets[ENT_PREDATOR] = rng.randint(3, 6)
        targets[ENT_FISH] = rng.randint(3, 5)
    else:
        targets[ENT_PLANT] = rng.randint(14, 22)
        targets[ENT_HERBIVORE] = rng.randint(7, 12)
        targets[ENT_PREDATOR] = rng.randint(4, 7)
        targets[ENT_FISH] = rng.randint(4, 6)

    # Placement budget: how many entities the player can place
    total_target = sum(targets.values())
    placement_budget = total_target + 2 + level_idx  # some slack

    # Tolerance: how close populations must be to targets
    tolerance = max(1, 2 - level_idx // 3)

    return {
        "terrain": terrain,
        "targets": targets,
        "sim_steps_required": sim_steps_required,
        "placement_budget": placement_budget,
        "tolerance": tolerance,
        "num_types": num_types,
    }


# ══════════════════════════════════════════════════════════════════
class Ecosystem(ARCBaseGame):
    """Ecosystem game — place entities and simulate predator/prey dynamics."""

    def __init__(self, seed: int = 42) -> None:
        self._game_seed = seed
        self._num_levels = 5
        self._level_data = [_generate_level(seed, i) for i in range(self._num_levels)]

        levels = []
        for i in range(self._num_levels):
            bg = Sprite(
                pixels=[[-1] * 64 for _ in range(64)],
                name=f"bg_{i}",
                visible=True,
                collidable=False,
                layer=-10,
                tags=["background"],
            )
            levels.append(Level(
                sprites=[bg], grid_size=(64, 64),
                data={"level_idx": i}, name=f"level_{i}",
            ))

        super().__init__(
            game_id="ecosystem",
            levels=levels,
            camera=Camera(0, 0, 64, 64, C_UI_BG, C_UI_BG),
            debug=False,
            win_score=1,
            available_actions=[1, 2, 3, 4, 5, 6],
            seed=seed,
        )

    # ── Level setup ───────────────────────────────────────────────
    def on_set_level(self, level: Level) -> None:
        idx = level.get_data("level_idx") or 0
        data = self._level_data[idx]

        self._terrain = [row[:] for row in data["terrain"]]
        self._targets: Dict[int, int] = dict(data["targets"])
        self._sim_steps_required: int = data["sim_steps_required"]
        self._placement_budget: int = data["placement_budget"]
        self._tolerance: int = data["tolerance"]
        self._num_types: int = data["num_types"]

        # Entity lists
        self._animals: List[Animal] = []
        self._plants: List[Tuple[int, int]] = []  # (x, y) positions

        # State
        self._sim_steps_done: int = 0
        self._placements_used: int = 0
        self._selected_entity: int = ENT_PLANT  # start with plant selected
        self._sim_rng = random.Random(self._game_seed + idx * 3571)

        self._render_to_bg()

    # ── Helpers ────────────────────────────────────────────────────
    def _play_to_grid(self, px: int, py: int) -> Tuple[int, int]:
        """Convert play-area coords to grid coords."""
        return px + PLAY_X_START, py + PLAY_Y_START

    def _grid_to_play(self, gx: int, gy: int) -> Optional[Tuple[int, int]]:
        """Convert grid coords to play-area coords, or None if out of bounds."""
        px = gx - PLAY_X_START
        py = gy - PLAY_Y_START
        pw = PLAY_X_END - PLAY_X_START
        ph = PLAY_Y_END - PLAY_Y_START
        if 0 <= px < pw and 0 <= py < ph:
            return px, py
        return None

    def _is_land(self, px: int, py: int) -> bool:
        """Check if play-area cell is passable land (not water)."""
        pw = PLAY_X_END - PLAY_X_START
        ph = PLAY_Y_END - PLAY_Y_START
        if 0 <= px < pw and 0 <= py < ph:
            return self._terrain[py][px] != TERRAIN_WATER
        return False

    def _is_water(self, px: int, py: int) -> bool:
        """Check if play-area cell is water."""
        pw = PLAY_X_END - PLAY_X_START
        ph = PLAY_Y_END - PLAY_Y_START
        if 0 <= px < pw and 0 <= py < ph:
            return self._terrain[py][px] == TERRAIN_WATER
        return False

    def _is_in_bounds(self, px: int, py: int) -> bool:
        pw = PLAY_X_END - PLAY_X_START
        ph = PLAY_Y_END - PLAY_Y_START
        return 0 <= px < pw and 0 <= py < ph

    def _entity_at(self, px: int, py: int) -> bool:
        """Check if any living entity occupies this play-area cell."""
        for a in self._animals:
            if a.alive and a.x == px and a.y == py:
                return True
        if (px, py) in self._plants:
            return True
        return False

    def _get_terrain(self, px: int, py: int) -> int:
        if self._is_in_bounds(px, py):
            return self._terrain[py][px]
        return TERRAIN_WATER  # out of bounds = water

    def _available_entity_types(self) -> List[int]:
        types = [ENT_PLANT, ENT_HERBIVORE, ENT_PREDATOR]
        if self._num_types >= 4:
            types.append(ENT_FISH)
        return types

    # ── Population counting ───────────────────────────────────────
    def _count_populations(self) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        counts[ENT_PLANT] = len(self._plants)
        for etype in [ENT_HERBIVORE, ENT_PREDATOR, ENT_FISH]:
            counts[etype] = sum(1 for a in self._animals if a.alive and a.etype == etype)
        return counts

    # ── Simulation step ───────────────────────────────────────────
    def _run_simulation_step(self) -> None:
        rng = self._sim_rng
        pw = PLAY_X_END - PLAY_X_START
        ph = PLAY_Y_END - PLAY_Y_START

        # Build occupancy set for quick lookup
        occupied: set = set()
        for a in self._animals:
            if a.alive:
                occupied.add((a.x, a.y))
        plant_set = set(self._plants)

        # --- Plant growth ---
        new_plants: List[Tuple[int, int]] = []
        for px, py in list(self._plants):
            terrain = self._get_terrain(px, py)
            spread_chance = 0.15
            if terrain == TERRAIN_FOREST:
                spread_chance = 0.30  # forest helps plants grow

            if rng.random() < spread_chance:
                # Try to spread to a random adjacent cell
                shuffled = list(DIRS)
                rng.shuffle(shuffled)
                for ddx, ddy in shuffled:
                    nx, ny = px + ddx, py + ddy
                    if (self._is_in_bounds(nx, ny)
                            and self._is_land(nx, ny)
                            and (nx, ny) not in plant_set
                            and (nx, ny) not in occupied):
                        new_plants.append((nx, ny))
                        plant_set.add((nx, ny))
                        break

        self._plants.extend(new_plants)

        # --- Fish movement and feeding (aquatic herbivores) ---
        # Fish eat plants that are adjacent to water
        for animal in self._animals:
            if not animal.alive or animal.etype != ENT_FISH:
                continue

            # Fish can only be in water
            if not self._is_water(animal.x, animal.y):
                animal.alive = False
                continue

            # Fish skip in marsh? No, marsh affects land predators.
            # Fish try to eat plants adjacent to water
            ate = False
            for ddx, ddy in DIRS:
                nx, ny = animal.x + ddx, animal.y + ddy
                if (nx, ny) in plant_set and self._is_land(nx, ny):
                    # Eat the plant from the water
                    plant_set.discard((nx, ny))
                    if (nx, ny) in self._plants:
                        self._plants.remove((nx, ny))
                    animal.food += 2
                    ate = True
                    break

            if not ate:
                # Move toward nearest plant adjacent to water
                best_dist = 999
                best_pos = None
                for ppx, ppy in list(plant_set):
                    # Check if plant is adjacent to water
                    adj_water = any(
                        self._is_water(ppx + ddx, ppy + ddy) for ddx, ddy in DIRS
                    )
                    if adj_water:
                        dist = abs(ppx - animal.x) + abs(ppy - animal.y)
                        if dist < best_dist:
                            best_dist = dist
                            best_pos = (ppx, ppy)

                if best_pos is not None:
                    # Move one step toward it (staying in water)
                    tx, ty = best_pos
                    dx = 1 if tx > animal.x else (-1 if tx < animal.x else 0)
                    dy = 1 if ty > animal.y else (-1 if ty < animal.y else 0)

                    # Try dx first, then dy
                    moved = False
                    for attempt_dx, attempt_dy in [(dx, 0), (0, dy), (dx, dy)]:
                        nx, ny = animal.x + attempt_dx, animal.y + attempt_dy
                        if self._is_in_bounds(nx, ny) and self._is_water(nx, ny) and (nx, ny) not in occupied:
                            occupied.discard((animal.x, animal.y))
                            animal.x, animal.y = nx, ny
                            occupied.add((nx, ny))
                            moved = True
                            break

            # Hunger
            animal.food -= 1
            if animal.food <= 0:
                animal.alive = False
                occupied.discard((animal.x, animal.y))

        # --- Herbivore movement and feeding ---
        for animal in self._animals:
            if not animal.alive or animal.etype != ENT_HERBIVORE:
                continue

            terrain = self._get_terrain(animal.x, animal.y)

            # Check if standing on a plant
            if (animal.x, animal.y) in plant_set:
                plant_set.discard((animal.x, animal.y))
                if (animal.x, animal.y) in self._plants:
                    self._plants.remove((animal.x, animal.y))
                food_gain = 2
                if terrain == TERRAIN_FOREST:
                    food_gain = 3  # forest bonus
                animal.food += food_gain
            else:
                # Move toward nearest plant
                best_dist = 999
                best_pos = None
                for ppx, ppy in list(plant_set):
                    if self._is_land(ppx, ppy):
                        dist = abs(ppx - animal.x) + abs(ppy - animal.y)
                        if dist < best_dist:
                            best_dist = dist
                            best_pos = (ppx, ppy)

                if best_pos is not None:
                    tx, ty = best_pos
                    dx = 1 if tx > animal.x else (-1 if tx < animal.x else 0)
                    dy = 1 if ty > animal.y else (-1 if ty < animal.y else 0)

                    for attempt_dx, attempt_dy in [(dx, 0), (0, dy)]:
                        nx, ny = animal.x + attempt_dx, animal.y + attempt_dy
                        if (self._is_in_bounds(nx, ny)
                                and self._is_land(nx, ny)
                                and (nx, ny) not in occupied):
                            occupied.discard((animal.x, animal.y))
                            animal.x, animal.y = nx, ny
                            occupied.add((nx, ny))
                            break

                    # Check if we landed on a plant
                    if (animal.x, animal.y) in plant_set:
                        plant_set.discard((animal.x, animal.y))
                        if (animal.x, animal.y) in self._plants:
                            self._plants.remove((animal.x, animal.y))
                        food_gain = 2
                        if self._get_terrain(animal.x, animal.y) == TERRAIN_FOREST:
                            food_gain = 3
                        animal.food += food_gain

            # Hunger (desert = double hunger)
            hunger = 2 if terrain == TERRAIN_DESERT else 1
            animal.food -= hunger

            if animal.food <= 0:
                animal.alive = False
                occupied.discard((animal.x, animal.y))

        # --- Predator movement and feeding ---
        for animal in self._animals:
            if not animal.alive or animal.etype != ENT_PREDATOR:
                continue

            terrain = self._get_terrain(animal.x, animal.y)

            # Marsh: predators skip every other turn
            if terrain == TERRAIN_MARSH:
                if animal.skip_next:
                    animal.skip_next = False
                    animal.food -= 1
                    if animal.food <= 0:
                        animal.alive = False
                        occupied.discard((animal.x, animal.y))
                    continue
                else:
                    animal.skip_next = True

            # Find nearest herbivore
            best_dist = 999
            best_prey = None
            for prey in self._animals:
                if prey.alive and prey.etype in (ENT_HERBIVORE, ENT_FISH):
                    dist = abs(prey.x - animal.x) + abs(prey.y - animal.y)
                    if dist < best_dist:
                        best_dist = dist
                        best_prey = prey

            if best_prey is not None:
                if best_dist <= 1:
                    # Eat the prey
                    best_prey.alive = False
                    occupied.discard((best_prey.x, best_prey.y))
                    animal.food += 3
                else:
                    # Move toward prey
                    tx, ty = best_prey.x, best_prey.y
                    dx = 1 if tx > animal.x else (-1 if tx < animal.x else 0)
                    dy = 1 if ty > animal.y else (-1 if ty < animal.y else 0)

                    for attempt_dx, attempt_dy in [(dx, 0), (0, dy)]:
                        nx, ny = animal.x + attempt_dx, animal.y + attempt_dy
                        if (self._is_in_bounds(nx, ny)
                                and self._is_land(nx, ny)
                                and (nx, ny) not in occupied):
                            occupied.discard((animal.x, animal.y))
                            animal.x, animal.y = nx, ny
                            occupied.add((nx, ny))
                            break

            # Hunger
            hunger = 2 if terrain == TERRAIN_DESERT else 1
            animal.food -= hunger

            if animal.food <= 0:
                animal.alive = False
                occupied.discard((animal.x, animal.y))

        # --- Reproduction ---
        new_animals: List[Animal] = []
        repro_threshold = 5

        for animal in self._animals:
            if not animal.alive or animal.food < repro_threshold:
                continue

            # Find empty adjacent cell
            shuffled = list(DIRS)
            rng.shuffle(shuffled)
            for ddx, ddy in shuffled:
                nx, ny = animal.x + ddx, animal.y + ddy

                if animal.etype == ENT_FISH:
                    # Fish reproduce in water
                    if self._is_in_bounds(nx, ny) and self._is_water(nx, ny) and (nx, ny) not in occupied:
                        baby = Animal(ENT_FISH, nx, ny, food=3)
                        new_animals.append(baby)
                        occupied.add((nx, ny))
                        animal.food -= 3  # cost of reproduction
                        break
                else:
                    # Land animals reproduce on land
                    if (self._is_in_bounds(nx, ny)
                            and self._is_land(nx, ny)
                            and (nx, ny) not in occupied
                            and (nx, ny) not in plant_set):
                        baby = Animal(animal.etype, nx, ny, food=3)
                        new_animals.append(baby)
                        occupied.add((nx, ny))
                        animal.food -= 3
                        break

        self._animals.extend(new_animals)

        # Clean up dead animals
        self._animals = [a for a in self._animals if a.alive]

    # ── Win check ─────────────────────────────────────────────────
    def _check_win(self) -> bool:
        if self._sim_steps_done < self._sim_steps_required:
            return False

        counts = self._count_populations()
        for etype, target in self._targets.items():
            actual = counts.get(etype, 0)
            if abs(actual - target) > self._tolerance:
                return False
        return True

    # ── Render ────────────────────────────────────────────────────
    def _render_to_bg(self) -> None:
        bg_list = self.current_level.get_sprites_by_tag("background")
        if not bg_list:
            return
        bg = bg_list[0]

        pixels = np.full((64, 64), C_UI_BG, dtype=np.int8)

        pw = PLAY_X_END - PLAY_X_START
        ph = PLAY_Y_END - PLAY_Y_START

        # Render terrain
        for py in range(ph):
            for px in range(pw):
                gx, gy = self._play_to_grid(px, py)
                if 0 <= gx < 64 and 0 <= gy < 64:
                    t = self._terrain[py][px]
                    pixels[gy][gx] = TERRAIN_COLORS.get(t, C_VOID)

        # Render plants
        for px, py in self._plants:
            gx, gy = self._play_to_grid(px, py)
            if 0 <= gx < 64 and 0 <= gy < 64:
                terrain = self._get_terrain(px, py)
                if terrain == TERRAIN_FOREST:
                    pixels[gy][gx] = C_PLANT_FOREST
                else:
                    pixels[gy][gx] = C_PLANT

        # Render animals
        for animal in self._animals:
            if animal.alive:
                gx, gy = self._play_to_grid(animal.x, animal.y)
                if 0 <= gx < 64 and 0 <= gy < 64:
                    pixels[gy][gx] = ENTITY_COLORS.get(animal.etype, C_WHITE)

        # ── Target panel (rows 0-2) ──────────────────────────────
        # Format: for each target species, show color swatch + count bar
        col = 1
        for etype in [ENT_PLANT, ENT_HERBIVORE, ENT_PREDATOR, ENT_FISH]:
            if etype not in self._targets:
                continue
            target_count = self._targets[etype]
            current_count = self._count_populations().get(etype, 0)
            color = ENTITY_COLORS.get(etype, C_WHITE)

            # Row 0: entity color swatch (2 pixels wide)
            if col < 62:
                pixels[0][col] = color
                pixels[0][col + 1] = color

            # Row 1: target count as bar (white pixels)
            for i in range(min(target_count, 20)):
                if col + i < 64:
                    pixels[1][col + i] = C_WHITE

            # Row 2: current count as bar (entity color)
            for i in range(min(current_count, 20)):
                if col + i < 64:
                    pixels[2][col + i] = color

            col += max(target_count, current_count, 3) + 2

        # ── Steps remaining indicator (row 62) ───────────────────
        steps_remaining = max(0, self._sim_steps_required - self._sim_steps_done)
        # Show steps done vs required as a bar
        bar_max = min(self._sim_steps_required, 50)
        if bar_max > 0:
            filled = int(bar_max * self._sim_steps_done / self._sim_steps_required)
            for x in range(bar_max):
                pixels[62][x + 1] = C_PLANT if x < filled else C_PREDATOR

        # Show steps remaining as number-like dots
        for i in range(min(steps_remaining, 10)):
            if 55 + i < 64:
                pixels[62][55 + i] = C_WHITE

        # ── Entity selection + budget (row 63) ────────────────────
        # Show selected entity type
        sel_color = ENTITY_COLORS.get(self._selected_entity, C_WHITE)
        for x in range(4):
            pixels[63][x + 1] = sel_color
        pixels[63][5] = C_SELECTED  # selection marker

        # Show all available types
        available = self._available_entity_types()
        for i, etype in enumerate(available):
            bx = 8 + i * 4
            if bx < 60:
                ec = ENTITY_COLORS.get(etype, C_WHITE)
                pixels[63][bx] = ec
                pixels[63][bx + 1] = ec
                if etype == self._selected_entity:
                    pixels[63][bx + 2] = C_SELECTED

        # Placement budget remaining
        budget_remaining = self._placement_budget - self._placements_used
        for i in range(min(budget_remaining, 15)):
            if 45 + i < 64:
                pixels[63][45 + i] = C_WHITE

        bg.pixels = pixels

    # ── Step ──────────────────────────────────────────────────────
    def step(self) -> None:
        action_id = self.action.id.value

        if action_id == 1:
            # Cycle entity type backward
            available = self._available_entity_types()
            idx = available.index(self._selected_entity) if self._selected_entity in available else 0
            idx = (idx - 1) % len(available)
            self._selected_entity = available[idx]

        elif action_id == 2:
            # Cycle entity type forward
            available = self._available_entity_types()
            idx = available.index(self._selected_entity) if self._selected_entity in available else 0
            idx = (idx + 1) % len(available)
            self._selected_entity = available[idx]

        elif action_id == 5:
            # Run simulation step
            self._run_simulation_step()
            self._sim_steps_done += 1

            # Check win after simulation
            if self._check_win():
                self._render_to_bg()
                self.next_level()
                self.complete_action()
                return

        elif action_id == 6:
            # Place entity at clicked position
            x = self.action.data.get("x", 0)
            y = self.action.data.get("y", 0)
            coords = self.camera.display_to_grid(x, y)
            if coords:
                gx, gy = coords
                play_coords = self._grid_to_play(gx, gy)
                if play_coords and self._placements_used < self._placement_budget:
                    px, py = play_coords

                    if self._selected_entity == ENT_PLANT:
                        if self._is_land(px, py) and not self._entity_at(px, py):
                            self._plants.append((px, py))
                            self._placements_used += 1

                    elif self._selected_entity == ENT_FISH:
                        if self._is_water(px, py) and not self._entity_at(px, py):
                            self._animals.append(Animal(ENT_FISH, px, py, food=3))
                            self._placements_used += 1

                    elif self._selected_entity in (ENT_HERBIVORE, ENT_PREDATOR):
                        if self._is_land(px, py) and not self._entity_at(px, py):
                            self._animals.append(Animal(self._selected_entity, px, py, food=3))
                            self._placements_used += 1

        # Actions 3, 4 are unused but reserved

        self._render_to_bg()
        self.complete_action()


# ── Standalone test ───────────────────────────────────────────────
if __name__ == "__main__":
    from arcengine import ActionInput, GameAction

    game = Ecosystem(seed=42)
    frame = game.perform_action(ActionInput(id=GameAction.RESET))
    print(f"Game started. State: {frame.state}, Levels: {frame.levels_completed}/{frame.win_levels}")

    rng = random.Random(12345)
    actions = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION5]
    # Also include some placement actions
    for i in range(500):
        if rng.random() < 0.4:
            # Place entity at random position
            x = rng.randint(0, 63)
            y = rng.randint(0, 63)
            frame = game.perform_action(ActionInput(id=GameAction.ACTION6, data={"x": x, "y": y}))
        else:
            action = rng.choice(actions)
            frame = game.perform_action(ActionInput(id=action))
        if frame.state.value == "WIN":
            print(f"  WIN after {i+1} actions!")
            break
        if frame.state.value == "GAME_OVER":
            print(f"  GAME_OVER after {i+1} actions!")
            break
        if i % 100 == 0:
            print(f"  Step {i}: state={frame.state}, levels={frame.levels_completed}/{frame.win_levels}")
    else:
        print(f"  Finished 500 actions. State: {frame.state}, "
              f"Levels: {frame.levels_completed}/{frame.win_levels}")
