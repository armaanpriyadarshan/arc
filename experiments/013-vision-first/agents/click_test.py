"""Minimal click-test agent. No LLM calls.

Tests 3 specific ACTION6 coordinates on the bottom-right enclosure
to diagnose coordinate system issues:
  1. (36, 36) — Looker's exact position for top-left enclosure block
  2. (38, 38) — Brain's estimated center of that same block
  3. (36, 25) — y-inverted version of (36, 36): y = 63 - 36 = 27...
               actually testing (36, 63-36) = (36, 27)

Records PNG diffs for each action. Zero LLM calls.
"""

import logging
import os
import time
from datetime import datetime

from arcengine import FrameData, GameAction, GameState

from .vision import grid_to_image, side_by_side

logger = logging.getLogger(__name__)


# Coordinates to test: (x, y) where x=col, y=row
TEST_CLICKS = [
    (36, 36, "Looker exact position [36,36]"),
    (38, 38, "Brain estimated center [38,38]"),
    (36, 27, "y-inverted: col=36, y=63-36=27"),
]


class ClickTestAgent:
    MAX_ACTIONS = 10

    def __init__(self, game_id: str, env=None) -> None:
        self.game_id = game_id
        if env is not None:
            self.arcade = None
            self.scorecard_id = None
            self.env = env
        else:
            from arc_agi import Arcade
            self.arcade = Arcade()
            self.scorecard_id = self.arcade.open_scorecard()
            self.env = self.arcade.make(game_id, scorecard_id=self.scorecard_id)
        self.frames: list[FrameData] = []
        self.action_counter = 0
        self.current_grid: list[list[int]] = []

    def _step(self, action: GameAction, reasoning: str = "") -> FrameData:
        try:
            data = None
            if action.is_complex() and hasattr(action, 'action_data'):
                ad = action.action_data
                data = {"x": ad.x, "y": ad.y}
            raw = self.env.step(action, data=data, reasoning=reasoning)
        except Exception as e:
            logger.warning(f"env.step({action.name}) exception: {e}")
            raw = None
        if raw is None:
            if self.frames:
                return self.frames[-1]
            return FrameData(frame=[], state=GameState.NOT_FINISHED,
                             levels_completed=0, available_actions=[])
        self.frames.append(raw)
        return raw

    def run(self) -> None:
        # --- Setup logging + run directory ---
        log_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(log_dir, f"{self.game_id}_clicktest_{ts}")
        os.makedirs(run_dir, exist_ok=True)
        frames_dir = os.path.join(run_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        # Set up file logging
        fh = logging.FileHandler(os.path.join(run_dir, "run.log"))
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(fh)
        logger.setLevel(logging.INFO)

        logger.info("=" * 60)
        logger.info(f"Click Test Agent on {self.game_id}")
        logger.info(f"Run directory: {run_dir}")
        logger.info(f"Test clicks: {TEST_CLICKS}")
        logger.info("=" * 60)

        # Reset
        frame = self._step(GameAction.RESET)
        self.current_grid = frame.frame[-1] if frame.frame else []
        self.action_counter += 1

        # Save initial frame
        if len(self.current_grid) > 0:
            img = grid_to_image(self.current_grid)
            img.save(os.path.join(frames_dir, "initial.png"), format="PNG", optimize=True)

        # Log board state at the click region
        if len(self.current_grid) > 0:
            logger.info("Board state around enclosure (rows 34-42, cols 34-42):")
            for r in range(34, min(42, len(self.current_grid))):
                row_hex = "".join(f"{self.current_grid[r][c]:X}" for c in range(34, min(42, len(self.current_grid[0]))))
                logger.info(f"  row {r}, cols 34-41: {row_hex}")

        # Execute test clicks
        for x, y, desc in TEST_CLICKS:
            grid_before = self.current_grid

            action = GameAction.from_id(6)
            action.set_data({"x": int(x), "y": int(y)})

            frame = self._step(action, reasoning=f"click test: {desc}")
            self.action_counter += 1
            self.current_grid = frame.frame[-1] if frame.frame else grid_before

            # Count changes
            changes = 0
            changed_cells = []
            if len(grid_before) > 0 and len(self.current_grid) > 0:
                for r in range(len(grid_before)):
                    for c in range(len(grid_before[0])):
                        if grid_before[r][c] != self.current_grid[r][c]:
                            changes += 1
                            if len(changed_cells) < 20:
                                changed_cells.append(
                                    f"  ({r},{c}): {grid_before[r][c]:X} -> {self.current_grid[r][c]:X}"
                                )

            # Log result
            logger.info(f"ACTION6({x},{y}) [{desc}]: {changes} cells changed")
            if changes > 0:
                logger.info(f"  Changed cells:")
                for cell in changed_cells:
                    logger.info(cell)
                if changes > 20:
                    logger.info(f"  ... and {changes - 20} more")
            else:
                # Log what color was at the clicked position
                if len(grid_before) > 0 and 0 <= y < len(grid_before) and 0 <= x < len(grid_before[0]):
                    color_val = grid_before[y][x]
                    logger.info(f"  Cell at grid[{y}][{x}] = {color_val:X} (color {color_val})")
                else:
                    logger.info(f"  Could not read grid[{y}][{x}]")

            # Save diff PNG
            if len(grid_before) > 0 and len(self.current_grid) > 0:
                if changes > 0:
                    img = side_by_side(grid_before, self.current_grid)
                else:
                    img = grid_to_image(self.current_grid)
                filename = f"click_{self.action_counter:02d}_ACTION6_{x}_{y}_{changes}changes.png"
                img.save(os.path.join(frames_dir, filename), format="PNG", optimize=True)

            logger.info(f"  State: {frame.state} | Score: {frame.levels_completed}")

        # Summary
        logger.info("=" * 60)
        logger.info("CLICK TEST COMPLETE")
        logger.info(f"  Actions used: {self.action_counter}")
        logger.info("=" * 60)

        # Cleanup
        if self.arcade and self.scorecard_id:
            try:
                self.arcade.close_scorecard(self.scorecard_id)
            except Exception:
                pass
        logger.removeHandler(fh)
        fh.close()
