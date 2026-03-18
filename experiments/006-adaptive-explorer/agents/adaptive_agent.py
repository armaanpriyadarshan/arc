"""Adaptive Explorer Agent v7 for ARC-AGI-3.

Key insight: BFS is deterministic — each life explores the same path,
visiting modifiers in the same order. Different modifier combos require
different visit orders. Solution: RANDOMIZED exploration.

Strategy:
  - Calibrate movement
  - Each life: explore with randomized frontier (not BFS)
  - Different random seeds per life = different modifier visit sequences
  - Eventually one sequence hits the right combo

This is the simplest effective approach: no modifier analysis, no combo
search, just randomized exploration with different seeds.
"""

import logging
from collections import deque
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class AdaptiveExplorerAgent:

    def __init__(self, game_id: str, config: Optional[dict] = None):
        from env_wrapper import ARCEnvWrapper
        self.game_id = game_id
        self.config = config or {}
        self.env = ARCEnvWrapper(game_id=game_id)
        self.max_actions = self.config.get("max_episode_steps", 500)

        # Movement
        self.player_pos: Optional[tuple[int, int]] = None
        self.action_effects: dict[int, tuple[int, int]] = {}
        self.direction_to_action: dict[tuple[int, int], int] = {}
        self.cell_size: int = 5

        # Colors
        self.floor_color: int = -1
        self.bg_colors: set[int] = set()

        # Game state
        self.action_count = 0
        self.levels_completed = 0
        self.steps_this_life = 0
        self.life_count = 0
        self.died_flag = False

        # Persistent knowledge (walls survive across lives)
        self.walls: set[tuple[int, int]] = set()

    def run(self):
        logger.info("=== AdaptiveExplorer v7 on game=%s ===", self.game_id)

        obs = self.env.reset()
        self.action_count += 1

        obs = self._calibrate(obs)

        while self.action_count < self.max_actions:
            self.life_count += 1
            self.steps_this_life = 0
            self.died_flag = False
            obs = self._random_explore_life(obs)

        self.env.close()
        logger.info("=== Done: %d actions, %d levels, %d lives ===",
                     self.action_count, self.levels_completed, self.life_count)

    def _random_explore_life(self, obs: np.ndarray) -> np.ndarray:
        """One life of randomized exploration."""
        logger.info("--- Life %d (action %d/%d, levels=%d) ---",
                     self.life_count, self.action_count, self.max_actions,
                     self.levels_completed)

        if self.player_pos is None:
            self._detect_player(obs)
            if self.player_pos is None:
                return obs

        prev_levels = self.levels_completed
        rng = np.random.RandomState(42 + self.life_count * 17 + self.levels_completed * 1000)

        visited: set[tuple[int, int]] = set()
        start_c = self._coarse(self.player_pos)
        visited.add(start_c)

        # Build frontier as a list (for random access)
        frontier: list[tuple[int, int]] = []
        self._add_neighbors(frontier, start_c, visited)

        while frontier and self.action_count < self.max_actions and not self.died_flag:
            # Pick a random frontier cell (biased toward nearby ones)
            if len(frontier) > 1:
                # 50% chance: pick nearest, 50% chance: pick random
                if rng.random() < 0.5 and self.player_pos:
                    # Pick nearest
                    best_idx = 0
                    best_dist = float('inf')
                    for i, fc in enumerate(frontier):
                        px = self._coarse_to_px(fc)
                        d = abs(px[0] - self.player_pos[0]) + abs(px[1] - self.player_pos[1])
                        if d < best_dist:
                            best_dist = d
                            best_idx = i
                    idx = best_idx
                else:
                    idx = rng.randint(len(frontier))
            else:
                idx = 0

            tc = frontier.pop(idx)
            if tc in visited or tc in self.walls:
                continue

            obs = self._navigate_to(obs, self._coarse_to_px(tc))

            if self.died_flag:
                break

            if self.levels_completed > prev_levels:
                logger.info("LEVEL %d COMPLETED on life %d at action %d!",
                            self.levels_completed, self.life_count, self.action_count)
                self.walls.clear()  # reset walls for new level
                return obs

            if self.player_pos:
                ac = self._coarse(self.player_pos)
                visited.add(ac)
                self._add_neighbors(frontier, ac, visited)

        logger.info("Life %d: visited %d cells", self.life_count, len(visited))
        return obs

    def _add_neighbors(self, frontier: list, pos_c: tuple[int, int],
                       visited: set[tuple[int, int]]):
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = pos_c[0]+dr, pos_c[1]+dc
            if 0 <= nr*self.cell_size < 64 and 0 <= nc*self.cell_size < 64:
                if (nr, nc) not in visited and (nr, nc) not in self.walls:
                    if (nr, nc) not in frontier:
                        frontier.append((nr, nc))

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def _calibrate(self, obs: np.ndarray) -> np.ndarray:
        logger.info("--- Calibrating ---")
        unique, counts = np.unique(obs, return_counts=True)
        sorted_c = sorted(zip(unique, counts), key=lambda x: -x[1])
        self.floor_color = int(sorted_c[0][0])
        self.bg_colors = {int(sorted_c[0][0])}
        if len(sorted_c) > 1:
            self.bg_colors.add(int(sorted_c[1][0]))

        move_actions = [a for a in self.env.valid_action_indices if a != 0]
        for _ in range(2):
            for a in move_actions:
                obs, _, t, _, _ = self.env.step(a)
                self.action_count += 1
                if t:
                    obs = self.env.reset()
                    self.action_count += 1

        for action in move_actions:
            positions = []
            for _ in range(3):
                prev = obs.copy()
                obs, _, t, _, _ = self.env.step(action)
                self.action_count += 1
                if t:
                    obs = self.env.reset()
                    self.action_count += 1
                    break
                nc = self._new_cells(prev, obs)
                if nc:
                    positions.append((np.mean([r for r,c in nc]),
                                     np.mean([c for r,c in nc])))
            if len(positions) >= 2:
                deltas = [(positions[i+1][0]-positions[i][0],
                           positions[i+1][1]-positions[i][1]) for i in range(len(positions)-1)]
                dy = int(round(np.mean([d[0] for d in deltas])))
                dx = int(round(np.mean([d[1] for d in deltas])))
                self.action_effects[action] = (dy, dx)
                logger.info("Action %d -> dy=%d, dx=%d", action, dy, dx)

        known = {(int(np.sign(dy)), int(np.sign(dx))) for dy, dx in self.action_effects.values()}
        missing = {(-1,0),(1,0),(0,-1),(0,1)} - known
        unused = set(move_actions) - set(self.action_effects.keys())
        if len(missing) == 1 and len(unused) == 1:
            d = missing.pop()
            a = unused.pop()
            s = max(abs(v) for p in self.action_effects.values() for v in p) if self.action_effects else 5
            self.action_effects[a] = (d[0]*s, d[1]*s)
            logger.info("Inferred action %d -> dy=%d, dx=%d", a, *self.action_effects[a])

        for a, (dy, dx) in self.action_effects.items():
            self.direction_to_action[(int(np.sign(dy)), int(np.sign(dx)))] = a
        self.cell_size = max(abs(v) for p in self.action_effects.values() for v in p) if self.action_effects else 5

        obs = self.env.reset()
        self.action_count += 1
        self._detect_player(obs)
        logger.info("Calibrated: %d dirs, cell=%d, player=%s",
                     len(self.action_effects), self.cell_size, self.player_pos)
        return obs

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _navigate_to(self, obs: np.ndarray, target: tuple[int, int]) -> np.ndarray:
        if self.player_pos is None:
            return obs
        prev_pos = None
        stuck = 0
        for _ in range(50):
            if self.action_count >= self.max_actions or self.died_flag:
                break
            dr = target[0] - self.player_pos[0]
            dc = target[1] - self.player_pos[1]
            if abs(dr) <= self.cell_size//2 and abs(dc) <= self.cell_size//2:
                break
            if self.player_pos == prev_pos:
                stuck += 1
                if stuck > 3:
                    perps = [(0,1),(0,-1)] if abs(dr)>=abs(dc) else [(1,0),(-1,0)]
                    for d in perps:
                        if d in self.direction_to_action:
                            for _ in range(3):
                                if self.action_count >= self.max_actions or self.died_flag:
                                    return obs
                                obs = self._step(obs, self.direction_to_action[d])
                            break
                    if stuck > 10:
                        self.walls.add((target[0]//self.cell_size, target[1]//self.cell_size))
                        break
            else:
                stuck = 0
            prev_pos = self.player_pos
            if abs(dr)>=abs(dc) and dr!=0:
                desired = (int(np.sign(dr)), 0)
            elif dc!=0:
                desired = (0, int(np.sign(dc)))
            elif dr!=0:
                desired = (int(np.sign(dr)), 0)
            else:
                break
            if desired not in self.direction_to_action:
                alt = (0, int(np.sign(dc))) if dc!=0 else (int(np.sign(dr)), 0)
                if alt in self.direction_to_action:
                    desired = alt
                else:
                    break
            obs = self._step(obs, self.direction_to_action[desired])
        return obs

    # ------------------------------------------------------------------
    # Low-level
    # ------------------------------------------------------------------

    def _step(self, obs: np.ndarray, action: int) -> np.ndarray:
        prev_obs = obs.copy()
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.action_count += 1
        self.steps_this_life += 1

        new_levels = info.get("levels_completed", 0)
        if new_levels > self.levels_completed:
            logger.info("LEVEL %d -> %d at action %d",
                        self.levels_completed, new_levels, self.action_count)
            self.levels_completed = new_levels

        if terminated:
            state = info.get("state", "UNKNOWN")
            if state == "GAME_OVER":
                logger.info("GAME_OVER at action %d (steps=%d)",
                            self.action_count, self.steps_this_life)
                self.died_flag = True
                obs = self.env.reset()
                self.action_count += 1
                self.steps_this_life = 0
                self._detect_player(obs)
            return obs

        diff = prev_obs != obs
        if np.sum(diff) == 0:
            if self.player_pos and action in self.action_effects:
                dy, dx = self.action_effects[action]
                self.walls.add(((self.player_pos[0]+dy)//self.cell_size,
                                (self.player_pos[1]+dx)//self.cell_size))
        else:
            nc = self._new_cells(prev_obs, obs)
            if nc:
                self.player_pos = (int(np.mean([r for r,c in nc])),
                                   int(np.mean([c for r,c in nc])))
            elif action in self.action_effects and self.player_pos:
                dy, dx = self.action_effects[action]
                self.player_pos = (self.player_pos[0]+dy, self.player_pos[1]+dx)
        return obs

    def _detect_player(self, obs: np.ndarray):
        actions = list(self.action_effects.keys())
        if not actions:
            return
        prev = obs.copy()
        obs2, _, t, _, _ = self.env.step(actions[0])
        self.action_count += 1
        self.steps_this_life += 1
        if t:
            return
        nc = self._new_cells(prev, obs2)
        if nc:
            self.player_pos = (int(np.mean([r for r,c in nc])),
                               int(np.mean([c for r,c in nc])))
            logger.info("Player at (%d, %d)", *self.player_pos)

    def _new_cells(self, prev: np.ndarray, curr: np.ndarray) -> list[tuple[int,int]]:
        diff = prev != curr
        if not np.any(diff):
            return []
        rows, cols = np.where(diff)
        return [(int(r), int(c)) for r, c in zip(rows, cols)
                if int(curr[r,c]) not in self.bg_colors and int(prev[r,c]) in self.bg_colors]

    def _coarse(self, pos: tuple[int, int]) -> tuple[int, int]:
        return (pos[0] // self.cell_size, pos[1] // self.cell_size)

    def _coarse_to_px(self, c: tuple[int, int]) -> tuple[int, int]:
        return (c[0]*self.cell_size + self.cell_size//2,
                c[1]*self.cell_size + self.cell_size//2)
