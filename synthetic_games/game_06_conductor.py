"""
Game 06: Conductor

Mechanics: Timing + sequencing + multiple trains + track switches + pause

The grid has multiple train sprites on rail tracks. Each train moves forward
automatically each turn along its track. The player clicks on track switches
(ACTION6) to toggle their direction (fork left vs fork right). Trains must each
reach their matching colored station. If two trains collide or a train goes off
the edge of the track network, the level resets. Later levels have trains that
enter at different times, requiring the player to toggle switches at precise
moments. ACTION5 pauses/unpauses the simulation so the player can plan.

Actions:
    ACTION5 = Pause / unpause simulation
    ACTION6 = Click to toggle a track switch

Display:
    Row 62: turn counter
    Row 63: train status indicators (colored dots, filled = arrived)

Win condition: All trains reach their matching colored stations.

Color key:
    0  = empty / background
    1  = track (rail)
    2  = switch (fork) — state A
    3  = switch (fork) — state B
    4  = station border
    5  = HUD background
    6  = blue (train / station)
    7  = yellow (train / station)
    8  = red (train / station)
    9  = purple (train / station)
    10 = green (arrived indicator)
    11 = orange (train / station)
    12 = track highlight / active rail
    13 = pause indicator
    14 = collision indicator
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
    Sprite,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GRID = 64
CELL = 3
BG = 0
TRACK = 1
SWITCH_A = 2
SWITCH_B = 3
STATION_BORDER = 4
HUD_BG = 5
BLUE = 6
YELLOW = 7
RED = 8
PURPLE = 9
GREEN = 10
ORANGE = 11
TRACK_HI = 12
PAUSE_IND = 13
COLLISION_IND = 14
HUD_BORDER = 15

TRAIN_COLORS = [RED, BLUE, YELLOW, PURPLE, ORANGE]

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)


# ---------------------------------------------------------------------------
# Track network: a graph of cells with connections
# ---------------------------------------------------------------------------

class TrackNetwork:
    """Represents a network of track cells on a grid.

    Each cell (col, row) can connect to neighbors. Switches are cells where
    two outgoing directions exist; the switch state picks which one is active.
    """

    def __init__(self, cols: int, rows: int):
        self.cols = cols
        self.rows = rows
        # connections[cell] = list of neighbor cells reachable from this cell
        self.connections: Dict[Tuple[int,int], List[Tuple[int,int]]] = {}
        # switches: cell -> (option_A_next, option_B_next)
        self.switches: Dict[Tuple[int,int], Tuple[Tuple[int,int], Tuple[int,int]]] = {}
        # switch states: cell -> 0 (A) or 1 (B)
        self.switch_states: Dict[Tuple[int,int], int] = {}
        # stations: cell -> color
        self.stations: Dict[Tuple[int,int], int] = {}
        # All track cells (for rendering)
        self.track_cells: Set[Tuple[int,int]] = set()

    def add_track(self, path: List[Tuple[int,int]]):
        """Add a sequence of connected cells as track."""
        for i, cell in enumerate(path):
            self.track_cells.add(cell)
            if cell not in self.connections:
                self.connections[cell] = []
            if i + 1 < len(path):
                nxt = path[i + 1]
                if nxt not in self.connections[cell]:
                    self.connections[cell].append(nxt)

    def add_switch(self, cell: Tuple[int,int], option_a: Tuple[int,int], option_b: Tuple[int,int]):
        """Mark a cell as a switch with two possible next cells."""
        self.switches[cell] = (option_a, option_b)
        self.switch_states[cell] = 0
        self.track_cells.add(cell)
        # Make sure both options are in connections
        if cell not in self.connections:
            self.connections[cell] = []
        for opt in (option_a, option_b):
            if opt not in self.connections[cell]:
                self.connections[cell].append(opt)

    def add_station(self, cell: Tuple[int,int], color: int):
        """Place a station (destination) at a cell."""
        self.stations[cell] = color
        self.track_cells.add(cell)

    def get_next(self, cell: Tuple[int,int], prev: Optional[Tuple[int,int]] = None) -> Optional[Tuple[int,int]]:
        """Get the next cell a train should move to from `cell`, given it came from `prev`."""
        if cell in self.switches:
            state = self.switch_states[cell]
            opt_a, opt_b = self.switches[cell]
            return opt_a if state == 0 else opt_b

        neighbors = self.connections.get(cell, [])
        if not neighbors:
            return None
        # If we came from prev, don't go back
        forward = [n for n in neighbors if n != prev]
        if forward:
            return forward[0]
        # Dead end or start — just pick first
        return neighbors[0] if neighbors else None

    def toggle_switch(self, cell: Tuple[int,int]):
        """Toggle a switch between state A and B."""
        if cell in self.switch_states:
            self.switch_states[cell] = 1 - self.switch_states[cell]


# ---------------------------------------------------------------------------
# Train state
# ---------------------------------------------------------------------------

class Train:
    def __init__(self, color: int, start: Tuple[int,int], spawn_turn: int = 0):
        self.color = color
        self.pos = start
        self.prev: Optional[Tuple[int,int]] = None
        self.start = start
        self.spawn_turn = spawn_turn
        self.arrived = False
        self.active = False  # becomes active at spawn_turn


# ---------------------------------------------------------------------------
# Level builder
# ---------------------------------------------------------------------------

def _build_level(rng: random.Random, cfg: dict) -> Tuple[TrackNetwork, List[Train]]:
    """Build a track network and trains from a level config.

    Config keys:
        grid_cols, grid_rows: cell dimensions of the playfield
        tracks: list of paths (list of (col, row) tuples)
        switches: list of (cell, option_a, option_b)
        stations: list of (cell, color)
        trains: list of (color, start_cell, spawn_turn)
    """
    net = TrackNetwork(cfg["grid_cols"], cfg["grid_rows"])

    for path in cfg["tracks"]:
        net.add_track(path)

    for cell, opt_a, opt_b in cfg["switches"]:
        net.add_switch(cell, opt_a, opt_b)

    for cell, color in cfg["stations"]:
        net.add_station(cell, color)

    trains = []
    for color, start, spawn in cfg["trains"]:
        trains.append(Train(color, start, spawn))

    return net, trains


def _level_configs():
    """Hand-crafted level configs with increasing difficulty."""
    return [
        # Level 0: 1 train, 1 switch, 1 station — just learn to toggle
        {
            "grid_cols": 16, "grid_rows": 16,
            "tracks": [
                # Main horizontal track
                [(1,4), (2,4), (3,4), (4,4), (5,4), (6,4), (7,4), (8,4), (9,4), (10,4)],
                # Branch up from switch at (5,4)
                [(5,4), (5,3), (5,2), (6,2), (7,2), (8,2)],
                # Branch continues straight
                [(5,4), (6,4), (7,4), (8,4), (9,4), (10,4)],
            ],
            "switches": [
                ((5,4), (6,4), (5,3)),  # switch: straight=(6,4), branch=(5,3)
            ],
            "stations": [
                ((8,2), RED),   # station at end of branch
            ],
            "trains": [
                (RED, (1,4), 0),
            ],
        },
        # Level 1: 2 trains, 1 switch — route each to correct station
        {
            "grid_cols": 18, "grid_rows": 16,
            "tracks": [
                # Shared entry track
                [(1,6), (2,6), (3,6), (4,6), (5,6)],
                # Upper branch
                [(5,6), (5,5), (5,4), (6,4), (7,4), (8,4), (9,4), (10,4)],
                # Lower branch (straight)
                [(5,6), (6,6), (7,6), (8,6), (9,6), (10,6)],
            ],
            "switches": [
                ((5,6), (6,6), (5,5)),  # straight=lower, branch=upper
            ],
            "stations": [
                ((10,4), RED),
                ((10,6), BLUE),
            ],
            "trains": [
                (RED, (1,6), 0),
                (BLUE, (1,6), 8),  # second train enters later
            ],
        },
        # Level 2: 2 trains, 2 switches — both need routing
        {
            "grid_cols": 18, "grid_rows": 18,
            "tracks": [
                # Entry from left
                [(1,5), (2,5), (3,5), (4,5), (5,5), (6,5)],
                # Switch 1 at (6,5): up or straight
                [(6,5), (6,4), (6,3), (7,3), (8,3), (9,3), (10,3)],
                [(6,5), (7,5), (8,5), (9,5)],
                # Switch 2 at (9,5): down or straight
                [(9,5), (10,5), (11,5), (12,5)],
                [(9,5), (9,6), (9,7), (10,7), (11,7), (12,7)],
                # Entry from top
                [(5,1), (5,2), (5,3), (5,4), (5,5)],
            ],
            "switches": [
                ((6,5), (7,5), (6,4)),
                ((9,5), (10,5), (9,6)),
            ],
            "stations": [
                ((10,3), YELLOW),
                ((12,5), RED),
                ((12,7), BLUE),
            ],
            "trains": [
                (RED, (1,5), 0),
                (BLUE, (1,5), 6),
                (YELLOW, (5,1), 3),
            ],
        },
        # Level 3: 3 trains, 3 switches, tighter timing
        {
            "grid_cols": 20, "grid_rows": 18,
            "tracks": [
                # Horizontal main line
                [(1,8), (2,8), (3,8), (4,8), (5,8), (6,8), (7,8), (8,8),
                 (9,8), (10,8), (11,8), (12,8), (13,8), (14,8)],
                # Switch 1 at (4,8): up branch
                [(4,8), (4,7), (4,6), (5,6), (6,6), (7,6)],
                # Switch 2 at (8,8): down branch
                [(8,8), (8,9), (8,10), (9,10), (10,10), (11,10)],
                # Switch 3 at (12,8): up branch
                [(12,8), (12,7), (12,6), (13,6), (14,6)],
            ],
            "switches": [
                ((4,8), (5,8), (4,7)),
                ((8,8), (9,8), (8,9)),
                ((12,8), (13,8), (12,7)),
            ],
            "stations": [
                ((7,6), RED),
                ((11,10), BLUE),
                ((14,6), YELLOW),
            ],
            "trains": [
                (RED, (1,8), 0),
                (BLUE, (1,8), 5),
                (YELLOW, (1,8), 10),
            ],
        },
        # Level 4: 4 trains, 4 switches, converging paths — collision risk
        {
            "grid_cols": 20, "grid_rows": 20,
            "tracks": [
                # Left entry
                [(1,6), (2,6), (3,6), (4,6), (5,6), (6,6), (7,6), (8,6)],
                # Switch 1 at (5,6): up
                [(5,6), (5,5), (5,4), (6,4), (7,4), (8,4), (9,4)],
                # Switch 2 at (8,6): down
                [(8,6), (8,7), (8,8), (9,8), (10,8), (11,8)],
                # Continue straight from switch 2
                [(8,6), (9,6), (10,6), (11,6), (12,6)],
                # Top entry
                [(8,1), (8,2), (8,3), (8,4)],
                # Switch 3 at (8,4): left or right
                [(8,4), (9,4)],
                [(8,4), (7,4), (6,4)],
                # Bottom entry
                [(4,12), (5,12), (6,12), (7,12), (8,12)],
                # Switch 4 at (8,12): up or right
                [(8,12), (8,11), (8,10), (8,9), (8,8)],
                [(8,12), (9,12), (10,12), (11,12)],
            ],
            "switches": [
                ((5,6), (6,6), (5,5)),
                ((8,6), (9,6), (8,7)),
                ((8,4), (9,4), (7,4)),
                ((8,12), (8,11), (9,12)),
            ],
            "stations": [
                ((9,4), RED),
                ((12,6), BLUE),
                ((11,8), YELLOW),
                ((11,12), PURPLE),
            ],
            "trains": [
                (RED, (1,6), 0),
                (BLUE, (1,6), 7),
                (YELLOW, (8,1), 2),
                (PURPLE, (4,12), 4),
            ],
        },
    ]


# ---------------------------------------------------------------------------
# Game class
# ---------------------------------------------------------------------------

class ConductorGame(ARCBaseGame):
    """Game 06: Conductor — route trains through switches to matching stations."""

    def __init__(self, seed: int = 0):
        self._seed = seed
        self._cfgs = _level_configs()

        # Per-level state
        self._network: Optional[TrackNetwork] = None
        self._trains: List[Train] = []
        self._paused = False
        self._turn = 0
        self._level_failed = False

        levels = [Level(sprites=[], grid_size=(GRID, GRID)) for _ in self._cfgs]
        camera = Camera(0, 0, GRID, GRID, BG, HUD_BG)
        super().__init__(
            game_id="game_06_conductor",
            levels=levels,
            camera=camera,
            available_actions=[5, 6],
            win_score=len(self._cfgs),
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Level setup
    # ------------------------------------------------------------------

    def on_set_level(self, level):
        level_idx = self._levels.index(level)
        cfg = self._cfgs[level_idx]
        self._network, self._trains = _build_level(
            random.Random(self._seed * 100 + level_idx), cfg
        )
        self._paused = False
        self._turn = 0
        self._level_failed = False

        # Activate trains that spawn at turn 0
        for t in self._trains:
            if t.spawn_turn == 0:
                t.active = True

        self._render_full()

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _advance_trains(self):
        """Move all active trains one step along their tracks."""
        net = self._network
        self._turn += 1

        # Activate trains whose spawn turn has arrived
        for t in self._trains:
            if not t.active and not t.arrived and t.spawn_turn <= self._turn:
                t.active = True
                t.pos = t.start
                t.prev = None

        # Move each active train
        new_positions: Dict[Tuple[int,int], List[Train]] = {}
        for t in self._trains:
            if not t.active or t.arrived:
                continue

            nxt = net.get_next(t.pos, t.prev)
            if nxt is None or nxt not in net.track_cells:
                # Off the edge — level reset
                self._level_failed = True
                return

            t.prev = t.pos
            t.pos = nxt

            # Check if arrived at matching station
            if nxt in net.stations and net.stations[nxt] == t.color:
                t.arrived = True
                t.active = False
                continue

            # Track position for collision detection
            if nxt not in new_positions:
                new_positions[nxt] = []
            new_positions[nxt].append(t)

        # Also add non-moving trains to collision check
        for t in self._trains:
            if t.active and not t.arrived:
                if t.pos in new_positions:
                    if t not in new_positions[t.pos]:
                        new_positions[t.pos].append(t)

        # Check collisions
        for pos, trains_at in new_positions.items():
            if len(trains_at) > 1:
                self._level_failed = True
                return

    def _check_win(self) -> bool:
        """Check if all trains have arrived."""
        return all(t.arrived for t in self._trains)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_full(self):
        """Clear and re-render the entire board."""
        level = self.current_level
        for s in list(level._sprites):
            level.remove_sprite(s)

        net = self._network
        cs = CELL
        sprite_idx = 0

        def _add(pixels, x, y, name=None, tags=None):
            nonlocal sprite_idx
            n = name or f"s_{sprite_idx}"
            sprite_idx += 1
            s = Sprite(pixels=pixels, name=n, visible=True, collidable=False,
                       tags=tags or [])
            s.set_position(x, y)
            level.add_sprite(s)

        # Draw track cells
        for (cx, cy) in net.track_cells:
            if (cx, cy) in net.switches:
                state = net.switch_states[(cx, cy)]
                color = SWITCH_A if state == 0 else SWITCH_B
                px = [
                    [color, TRACK, color],
                    [TRACK, color, TRACK],
                    [color, TRACK, color],
                ]
            elif (cx, cy) in net.stations:
                st_color = net.stations[(cx, cy)]
                px = [
                    [STATION_BORDER, STATION_BORDER, STATION_BORDER],
                    [STATION_BORDER, st_color, STATION_BORDER],
                    [STATION_BORDER, STATION_BORDER, STATION_BORDER],
                ]
            else:
                px = [
                    [BG, TRACK, BG],
                    [TRACK, TRACK, TRACK],
                    [BG, TRACK, BG],
                ]
            _add(px, cx * cs, cy * cs, tags=["track"])

        # Draw trains
        for i, t in enumerate(self._trains):
            if not t.active and not t.arrived:
                continue
            if t.arrived:
                # Show as green dot at station
                px = [[GREEN] * cs for _ in range(cs)]
            else:
                c = t.color
                px = [
                    [BG, c, BG],
                    [c, c, c],
                    [BG, c, BG],
                ]
            _add(px, t.pos[0] * cs, t.pos[1] * cs, name=f"train_{i}", tags=["train"])

        # HUD background
        hud_bg = [[HUD_BG] * GRID for _ in range(4)]
        _add(hud_bg, 0, 60, name="hud_bg")

        # Turn counter on row 62
        turn_text_width = min(self._turn, 50)
        if turn_text_width > 0:
            bar_px = [[TRACK_HI] * turn_text_width]
            _add(bar_px, 6, 62, name="turn_bar", tags=["hud"])

        # Pause indicator
        if self._paused:
            pause_px = [[PAUSE_IND] * 3 for _ in range(2)]
            _add(pause_px, 58, 62, name="pause_ind", tags=["hud"])

        # Train status dots on row 63
        for i, t in enumerate(self._trains):
            dot_color = GREEN if t.arrived else t.color
            dot_px = [[dot_color, dot_color]]
            _add(dot_px, 6 + i * 5, 63, name=f"status_{i}", tags=["hud"])

    # ------------------------------------------------------------------
    # Game loop
    # ------------------------------------------------------------------

    def step(self):
        action = self.action.id

        if action == GameAction.ACTION5:
            # Toggle pause
            self._paused = not self._paused
            self._render_full()
            self.complete_action()
            return

        if action == GameAction.ACTION6:
            # Click to toggle a switch
            x = self.action.data.get("x", 0)
            y = self.action.data.get("y", 0)
            coords = self.camera.display_to_grid(x, y)
            if coords:
                gx, gy = coords
                # Convert pixel coords to cell coords
                cx = gx // CELL
                cy = gy // CELL
                cell = (cx, cy)
                if cell in self._network.switches:
                    self._network.toggle_switch(cell)

        # Advance simulation if not paused
        if not self._paused:
            self._advance_trains()

            if self._level_failed:
                # Reset the level
                self.level_reset()
                self.complete_action()
                return

            if self._check_win():
                self._render_full()
                self.next_level()
                self.complete_action()
                return

        self._render_full()
        self.complete_action()
