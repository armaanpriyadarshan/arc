"""Perception-based exploration agent for ARC-AGI-3.

Instead of learning a policy via RL, this agent uses code-based perception
to understand the game grid and systematic exploration to solve levels.

Approach:
1. Calibration: Test each action to learn what it does (movement directions)
2. Player tracking: Identify the player by finding consistent change patterns
3. Map building: Track walls (no-op actions) and floor tiles
4. Object detection: Find colored entities that aren't walls or floor
5. Systematic exploration: BFS to visit all reachable cells
6. Goal discovery: Try interacting with objects to find key->door mechanics
"""

import logging
from collections import deque
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class PerceptionAgent:
    """Code-based perception agent that explores and solves grid games.

    Unlike the DreamerV3 agent, this uses no neural networks. It analyzes
    the grid directly with code to find the player, build a map, and
    navigate toward goals.
    """

    def __init__(self, game_id: str, config: Optional[dict] = None):
        from env_wrapper import ARCEnvWrapper
        self.game_id = game_id
        self.config = config or {}
        self.env = ARCEnvWrapper(game_id=game_id)
        self.max_actions = self.config.get("max_episode_steps", 300)

        # Perception state
        self.player_pos: Optional[tuple[int, int]] = None
        self.player_color: int = -1
        self.floor_color: int = -1
        self.bg_colors: set[int] = set()
        self.wall_cells: set[tuple[int, int]] = set()
        self.visited: set[tuple[int, int]] = set()
        self.objects: dict[tuple[int, int], int] = {}  # pos -> color
        self.action_effects: dict[int, tuple[int, int]] = {}  # action -> (dy, dx)

        # Game state
        self.action_count = 0
        self.levels_completed = 0

    def run(self):
        """Main agent loop."""
        logger.info("PerceptionAgent starting on game=%s", self.game_id)

        obs = self.env.reset()
        logger.info("Initial obs shape: %s, valid_actions: %s",
                     obs.shape, self.env.valid_action_indices)

        # Phase 1: Calibration — learn what each action does
        obs = self._calibrate(obs)

        # Phase 2: Systematic exploration
        self._explore(obs)

        self.env.close()
        logger.info(
            "PerceptionAgent done: %d actions used, %d levels completed",
            self.action_count, self.levels_completed,
        )

    def _calibrate(self, obs: np.ndarray) -> np.ndarray:
        """Test each action to learn movement directions and find the player.

        Returns the observation after calibration.
        """
        logger.info("Phase 1: Calibration")

        # Identify floor color: most common color in the grid
        unique, counts = np.unique(obs, return_counts=True)
        self.floor_color = int(unique[np.argmax(counts)])
        logger.info("Floor color (most common): %d", self.floor_color)

        move_actions = [a for a in self.env.valid_action_indices if a != 0]

        # Identify the second most common color (often walls/background)
        sorted_by_count = sorted(zip(unique, counts), key=lambda x: -x[1])
        self.bg_colors = {int(sorted_by_count[0][0])}
        if len(sorted_by_count) > 1:
            self.bg_colors.add(int(sorted_by_count[1][0]))
        logger.info("Background colors: %s", self.bg_colors)

        # Calibrate: move to center first, then test each action
        obs = self.env.reset()
        self.action_count += 1

        # Move toward center by trying each action a few times
        # This avoids being stuck against walls during calibration
        for _ in range(3):
            for a in move_actions:
                prev_obs = obs.copy()
                obs, _, terminated, _, _ = self.env.step(a)
                self.action_count += 1
                if terminated:
                    obs = self.env.reset()
                    self.action_count += 1

        # Now test each action: do 3 steps, measure position deltas
        for action in move_actions:
            positions = []
            for _ in range(3):
                prev_obs = obs.copy()
                obs, _, terminated, _, info = self.env.step(action)
                self.action_count += 1

                if terminated:
                    obs = self.env.reset()
                    self.action_count += 1
                    break

                diff_mask = prev_obs != obs
                if np.sum(diff_mask) == 0:
                    break  # Hit wall

                rows, cols = np.where(diff_mask)
                new_cells = [
                    (int(r), int(c)) for r, c in zip(rows, cols)
                    if int(obs[r, c]) not in self.bg_colors and int(prev_obs[r, c]) in self.bg_colors
                ]
                if new_cells:
                    cr = float(np.mean([r for r, c in new_cells]))
                    cc = float(np.mean([c for r, c in new_cells]))
                    positions.append((cr, cc))

            if len(positions) >= 2:
                deltas = [
                    (positions[i+1][0] - positions[i][0], positions[i+1][1] - positions[i][1])
                    for i in range(len(positions) - 1)
                ]
                avg_dy = np.mean([d[0] for d in deltas])
                avg_dx = np.mean([d[1] for d in deltas])
                step_dy = int(round(avg_dy))
                step_dx = int(round(avg_dx))
                self.action_effects[action] = (step_dy, step_dx)
                logger.info("Action %d: step (dy=%d, dx=%d)", action, step_dy, step_dx)
            else:
                logger.info("Action %d: could not determine direction (%d positions)", action, len(positions))

            # Undo by doing opposite actions to return to center
            for a in reversed(move_actions):
                if a != action:
                    obs, _, terminated, _, _ = self.env.step(a)
                    self.action_count += 1
                    if terminated:
                        obs = self.env.reset()
                        self.action_count += 1
                        break

        # Infer missing directions from known ones
        known_dirs = set()
        for dy, dx in self.action_effects.values():
            known_dirs.add((int(np.sign(dy)), int(np.sign(dx))))

        # If we have UP but not DOWN, assume the missing action is DOWN, etc.
        all_cardinal = {(-1, 0), (1, 0), (0, -1), (0, 1)}
        missing = all_cardinal - known_dirs
        unused_actions = set(move_actions) - set(self.action_effects.keys())
        if len(missing) == 1 and len(unused_actions) == 1:
            missing_dir = missing.pop()
            missing_action = unused_actions.pop()
            step_size = 5  # default
            if self.action_effects:
                step_size = abs(list(self.action_effects.values())[0][0]) or abs(list(self.action_effects.values())[0][1])
            self.action_effects[missing_action] = (missing_dir[0] * step_size, missing_dir[1] * step_size)
            logger.info("Inferred action %d: step (dy=%d, dx=%d)",
                       missing_action, *self.action_effects[missing_action])

        # Detect pixel step size
        step_sizes = [max(abs(dy), abs(dx)) for dy, dx in self.action_effects.values()]
        self.cell_size = int(np.median(step_sizes)) if step_sizes else 5
        logger.info("Detected cell size: %d pixels", self.cell_size)

        # Get a fresh start
        obs = self.env.reset()
        self.action_count += 1
        self._detect_player_position(obs)
        self._build_navigation()

        logger.info(
            "Calibration complete: %d action effects learned, player at %s",
            len(self.action_effects), self.player_pos,
        )
        return obs

    def _detect_player_position(self, obs: np.ndarray):
        """Detect player position by doing one test action and analyzing the diff."""
        move_actions = list(self.action_effects.keys())
        if not move_actions:
            return

        prev_obs = obs.copy()
        test_action = move_actions[0]
        obs2, _, terminated, _, _ = self.env.step(test_action)
        self.action_count += 1

        if terminated:
            return

        diff_mask = prev_obs != obs2
        if np.sum(diff_mask) > 0:
            rows, cols = np.where(diff_mask)
            new_cells = [
                (int(r), int(c)) for r, c in zip(rows, cols)
                if int(obs2[r, c]) not in self.bg_colors and int(prev_obs[r, c]) in self.bg_colors
            ]
            if new_cells:
                self.player_pos = (
                    int(np.mean([r for r, c in new_cells])),
                    int(np.mean([c for r, c in new_cells])),
                )
                logger.info("Player position after reset: (%d, %d)", *self.player_pos)

    def _build_navigation(self):
        """Build navigation helpers from learned action effects."""
        # Map normalized directions to actions
        self.direction_to_action: dict[tuple[int, int], int] = {}
        for action, (dy, dx) in self.action_effects.items():
            # Normalize to unit direction
            norm_dir = (int(np.sign(dy)), int(np.sign(dx)))
            self.direction_to_action[norm_dir] = action

        # Also build opposite actions for backtracking
        self.opposite_action: dict[int, int] = {}
        for action, (dy, dx) in self.action_effects.items():
            opp_dir = (-int(np.sign(dy)), -int(np.sign(dx)))
            if opp_dir in self.direction_to_action:
                self.opposite_action[action] = self.direction_to_action[opp_dir]

        logger.info("Navigation: directions=%s", self.direction_to_action)

    def _explore(self, obs: np.ndarray):
        """Systematic BFS exploration of the game grid.

        Tries to visit every reachable cell, identifying objects along the way.
        When objects are found, tries to interact with them.
        """
        logger.info("Phase 2: Systematic exploration")

        if self.player_pos is None:
            logger.warning("No player position detected — falling back to random exploration")
            self._random_fallback(obs)
            return

        self.visited.add(self.player_pos)
        frontier: deque[tuple[int, int]] = deque()
        self._add_neighbors_to_frontier(frontier)

        prev_levels = 0
        stuck_count = 0
        last_pos = self.player_pos

        while self.action_count < self.max_actions:
            # Check for level completion
            if self.levels_completed > prev_levels:
                logger.info("LEVEL COMPLETED! (%d total)", self.levels_completed)
                prev_levels = self.levels_completed
                # Reset exploration state for new level
                self.visited.clear()
                self.wall_cells.clear()
                self.objects.clear()
                frontier.clear()
                if self.player_pos:
                    self.visited.add(self.player_pos)
                    self._add_neighbors_to_frontier(frontier)

            # If frontier is empty, try to find unvisited cells
            if not frontier:
                logger.info("Frontier empty after %d actions, %d cells visited",
                           self.action_count, len(self.visited))
                # Try random exploration to find new areas
                obs = self._random_burst(obs, 20)
                self._add_neighbors_to_frontier(frontier)
                if not frontier:
                    logger.info("No more cells to explore — resetting level")
                    obs = self.env.reset()
                    self.action_count += 1
                    self._recalibrate_position(obs)
                    self.visited.clear()
                    self.wall_cells.clear()
                    frontier.clear()
                    if self.player_pos:
                        self.visited.add(self.player_pos)
                        self._add_neighbors_to_frontier(frontier)
                    continue

            # Pick next target from frontier (BFS order)
            target = frontier.popleft()
            if target in self.visited or target in self.wall_cells:
                continue

            # Navigate to target
            obs = self._navigate_to(obs, target)

            # Track stuck detection
            if self.player_pos == last_pos:
                stuck_count += 1
                if stuck_count > 10:
                    obs = self._random_burst(obs, 15)
                    stuck_count = 0
            else:
                stuck_count = 0
                last_pos = self.player_pos

            if self.player_pos:
                self.visited.add(self.player_pos)
                self._scan_surroundings(obs)
                self._add_neighbors_to_frontier(frontier)

        logger.info("Exploration complete: %d actions, %d cells visited, %d levels",
                    self.action_count, len(self.visited), self.levels_completed)

    def _add_neighbors_to_frontier(self, frontier: deque):
        """Add unvisited neighbors of current position to frontier."""
        if self.player_pos is None:
            return
        r, c = self.player_pos
        for dy, dx in self.action_effects.values():
            nr, nc = r + dy, c + dx
            if 0 <= nr < 64 and 0 <= nc < 64:
                # Round to nearest cell boundary
                pos = (nr, nc)
                if not self._is_visited_nearby(pos) and pos not in self.wall_cells:
                    frontier.append(pos)

    def _is_visited_nearby(self, pos: tuple[int, int]) -> bool:
        """Check if a position close to `pos` has been visited (within cell_size/2)."""
        threshold = max(self.cell_size // 2, 2)
        for vr, vc in self.visited:
            if abs(vr - pos[0]) <= threshold and abs(vc - pos[1]) <= threshold:
                return True
        return False

    def _navigate_to(self, obs: np.ndarray, target: tuple[int, int]) -> np.ndarray:
        """Try to move toward a target position. Returns updated observation."""
        if self.player_pos is None:
            return obs

        max_nav_steps = 30
        prev_pos = None
        stuck = 0

        for _ in range(max_nav_steps):
            if self.action_count >= self.max_actions:
                break

            dr = target[0] - self.player_pos[0]
            dc = target[1] - self.player_pos[1]

            # Close enough (within one cell size)
            if abs(dr) <= self.cell_size // 2 and abs(dc) <= self.cell_size // 2:
                break

            # Stuck detection
            if self.player_pos == prev_pos:
                stuck += 1
                if stuck > 3:
                    break  # Can't reach target
            else:
                stuck = 0
            prev_pos = self.player_pos

            # Choose direction: prefer the axis with larger gap
            if abs(dr) >= abs(dc) and dr != 0:
                desired_dir = (int(np.sign(dr)), 0)
            elif dc != 0:
                desired_dir = (0, int(np.sign(dc)))
            elif dr != 0:
                desired_dir = (int(np.sign(dr)), 0)
            else:
                break

            if desired_dir not in self.direction_to_action:
                # Try the other axis
                alt_dir = (0, int(np.sign(dc))) if dc != 0 else (int(np.sign(dr)), 0)
                if alt_dir in self.direction_to_action:
                    desired_dir = alt_dir
                else:
                    break

            action = self.direction_to_action[desired_dir]
            obs = self._step(obs, action)

        return obs

    def _step(self, obs: np.ndarray, action: int) -> np.ndarray:
        """Execute one action and update state."""
        prev_obs = obs.copy()
        prev_pos = self.player_pos

        obs, reward, terminated, truncated, info = self.env.step(action)
        self.action_count += 1

        # Update levels
        new_levels = info.get("levels_completed", 0)
        if new_levels > self.levels_completed:
            self.levels_completed = new_levels
            logger.info("LEVEL PROGRESS: %d -> %d", self.levels_completed - 1, new_levels)

        # Handle termination
        if terminated:
            state = info.get("state", "UNKNOWN")
            if state == "WIN":
                logger.info("GAME WON!")
            elif state == "GAME_OVER":
                logger.info("GAME OVER at action %d — resetting", self.action_count)
                obs = self.env.reset()
                self.action_count += 1
                self._recalibrate_position(obs)
            return obs

        # Update player position from diff
        diff_mask = prev_obs != obs
        if np.sum(diff_mask) == 0:
            # Wall hit — action had no effect
            if prev_pos and action in self.action_effects:
                dy, dx = self.action_effects[action]
                wall_pos = (prev_pos[0] + dy, prev_pos[1] + dx)
                self.wall_cells.add(wall_pos)
        else:
            # Movement succeeded — update position from centroid of new player cells
            changed_rows, changed_cols = np.where(diff_mask)
            new_cells = []
            for r, c in zip(changed_rows, changed_cols):
                if int(obs[r, c]) not in self.bg_colors and int(prev_obs[r, c]) in self.bg_colors:
                    new_cells.append((r, c))
            if new_cells:
                self.player_pos = (
                    int(np.mean([r for r, c in new_cells])),
                    int(np.mean([c for r, c in new_cells])),
                )
            elif action in self.action_effects and self.player_pos:
                dy, dx = self.action_effects[action]
                self.player_pos = (self.player_pos[0] + dy, self.player_pos[1] + dx)

        return obs

    def _scan_surroundings(self, obs: np.ndarray):
        """Look at the current observation for notable objects near the player."""
        if self.player_pos is None:
            return

        r, c = self.player_pos
        # Scan a small area around the player
        for dr in range(-5, 6):
            for dc in range(-5, 6):
                nr, nc = r + dr, c + dc
                if 0 <= nr < 64 and 0 <= nc < 64:
                    color = int(obs[nr, nc])
                    if color != self.floor_color and (nr, nc) not in self.objects:
                        self.objects[(nr, nc)] = color

    def _random_burst(self, obs: np.ndarray, steps: int) -> np.ndarray:
        """Do a burst of random actions to break out of stuck situations."""
        move_actions = [a for a in self.env.valid_action_indices if a != 0]
        for _ in range(steps):
            if self.action_count >= self.max_actions:
                break
            action = move_actions[np.random.randint(len(move_actions))]
            obs = self._step(obs, action)
        return obs

    def _random_fallback(self, obs: np.ndarray):
        """Fallback: just do random actions if perception failed."""
        logger.info("Random fallback mode")
        move_actions = [a for a in self.env.valid_action_indices if a != 0]
        while self.action_count < self.max_actions:
            action = move_actions[np.random.randint(len(move_actions))]
            obs = self._step(obs, action)

    def _recalibrate_position(self, obs: np.ndarray):
        """Re-detect player position after a reset."""
        # Find non-floor cells that could be the player
        # After a reset, try one action to find the player
        move_actions = [a for a in self.env.valid_action_indices if a != 0]
        if not move_actions:
            return

        prev_obs = obs.copy()
        obs, _, terminated, _, _ = self.env.step(move_actions[0])
        self.action_count += 1

        if terminated:
            return

        diff_mask = prev_obs != obs
        if np.sum(diff_mask) > 0:
            changed_rows, changed_cols = np.where(diff_mask)
            # New position is where non-floor colors appeared
            for r, c in zip(changed_rows, changed_cols):
                if obs[r, c] != self.floor_color and prev_obs[r, c] == self.floor_color:
                    self.player_pos = (int(r), int(c))
                    logger.info("Recalibrated player position: (%d, %d)", r, c)
                    return
