"""Dummy navigation agent — scripted movement with LLM perception logging.

Executes scripted navigation toward the white plus in ls20. No LLM for
planning — GPT-5.4 is called purely as a describer at each step. Output
is a single human-readable text log revealing perception accuracy without
the noise of hypothesis-driven decision-making.

Usage:
    cd arc_agi_repo_collab/arc
    uv run python synthetic_games/visual_agent_play.py --game ls20 --standalone \
      --agent experiments/004-world-model-induction-v3/vision_testing/dummy_agent.py:DummyNavigationAgent
"""

import logging
import os
from datetime import datetime

from arcengine import GameAction, GameState

from agents.agent import ToolUseAgent
from agents.symbolic import grid_to_symbolic, diff_symbolic
from agents.vision import (
    diff_b64,
    grid_b64,
    grid_to_image,
    input_image_b64,
    input_text,
    side_by_side,
)

logger = logging.getLogger(__name__)

ACTION_NAMES = {
    GameAction.ACTION1: "ACTION1 (up)",
    GameAction.ACTION2: "ACTION2 (down)",
    GameAction.ACTION3: "ACTION3 (left)",
    GameAction.ACTION4: "ACTION4 (right)",
    GameAction.ACTION5: "ACTION5 (spacebar)",
}
ACTION_DIRS = {
    GameAction.ACTION1: (-1, 0),
    GameAction.ACTION2: (1, 0),
    GameAction.ACTION3: (0, -1),
    GameAction.ACTION4: (0, 1),
}
OPPOSITE = {
    GameAction.ACTION1: GameAction.ACTION2,
    GameAction.ACTION2: GameAction.ACTION1,
    GameAction.ACTION3: GameAction.ACTION4,
    GameAction.ACTION4: GameAction.ACTION3,
}


class DummyNavigationAgent(ToolUseAgent):
    """Scripted navigation with LLM perception logging.

    Identifies the player via movement, navigates toward the white plus
    using greedy Manhattan movement, and logs GPT-5.4 descriptions at
    every step.  No LLM is used for decision-making.
    """

    def __init__(self, game_id: str, env=None) -> None:
        super().__init__(game_id, env)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self._run_dir = os.path.join(base_dir, "vision_logs", f"run_{timestamp}")
        os.makedirs(self._run_dir, exist_ok=True)

        self._log_path = os.path.join(self._run_dir, "vision_log.txt")
        self._log_file = open(self._log_path, "w")

        self._target = None
        self._player_color_id = None
        self._player_obj = None
        self._turn = 0
        self._prev_symbolic = None
        self._prev_grid = None

        logger.info(f"[dummy-nav] Logging to {self._run_dir}")

    # ==================================================================
    # Main loop — completely overrides parent
    # ==================================================================

    def run(self) -> None:
        self._write_header()

        frame = self._step(GameAction.RESET)
        self.current_grid = frame.frame[-1] if frame.frame else []

        if frame.state == GameState.WIN:
            self._log("Game already won on reset.")
            self._cleanup()
            return

        # Phase 1 — identify player
        self._identify_player()

        # Phase 2 — find white plus target
        self._find_white_plus()
        if self._target is None:
            self._log("\nNo white plus target found. Aborting.")
            self._cleanup()
            return

        # Phase 3 — navigate toward target
        self._navigate(max_actions=30)

        # Phase 4 — interact with target
        self._interact()

        self._cleanup()

    # ==================================================================
    # Phase 1: Player identification
    # ==================================================================

    def _identify_player(self):
        self._log("\n--- PLAYER IDENTIFICATION ---")
        sym_before = grid_to_symbolic(self.current_grid)

        # Try ACTION3 (left) first
        moved_obj = self._try_move_detect(
            sym_before, GameAction.ACTION3, GameAction.ACTION4,
        )
        if moved_obj is None:
            # Retry with ACTION1 (up) / ACTION2 (down)
            sym_before = grid_to_symbolic(self.current_grid)
            moved_obj = self._try_move_detect(
                sym_before, GameAction.ACTION1, GameAction.ACTION2,
            )

        if moved_obj is not None:
            self._player_color_id = moved_obj["color_id"]
            self._player_obj = moved_obj
            self._log(f"Player color_id: {moved_obj['color_id']} ({moved_obj['color']})")
            self._log(f"Player shape: {moved_obj['shape']}, size: {moved_obj['size']}")
        else:
            self._log("WARNING: Could not identify player. Proceeding with best guess.")

        # Initialise symbolic tracking after identification
        self._prev_symbolic = grid_to_symbolic(self.current_grid)
        self._prev_grid = [row[:] for row in self.current_grid]

    def _try_move_detect(self, sym_before, action, undo_action):
        """Take *action*, find which object moved, then *undo_action*."""
        frame = self._step(action)
        self.current_grid = frame.frame[-1] if frame.frame else self.current_grid

        sym_after = grid_to_symbolic(self.current_grid)
        diff = diff_symbolic(sym_before, sym_after)

        moved = None
        was_center = None
        for change in diff:
            if change["type"] == "changed" and "center" in change:
                was_center = change["center"]["was"]
                now_center = change["center"]["now"]
                for obj in sym_after["objects"]:
                    if obj["color"] == change["color"] and obj["center"] == now_center:
                        moved = obj
                        break
                if moved:
                    break

        act_name = ACTION_NAMES.get(action, action.name)
        undo_name = ACTION_NAMES.get(undo_action, undo_action.name)

        if moved:
            self._log(f"Took {act_name}. Object that moved:")
            self._log(
                f"  {moved['color']} {moved['shape']} "
                f"(size {moved['size']}, color_id={moved['color_id']}) "
                f"at {was_center} \u2192 {moved['center']}"
            )
        else:
            self._log(f"Took {act_name}. Nothing moved.")

        # Undo the probing action
        frame = self._step(undo_action)
        self.current_grid = frame.frame[-1] if frame.frame else self.current_grid
        self._log(f"Took {undo_name} to restore position.")

        return moved

    # ==================================================================
    # Phase 2: Find target
    # ==================================================================

    def _find_white_plus(self):
        self._log("\n--- TARGET ---")
        symbolic = grid_to_symbolic(self.current_grid)
        player_center = self._player_obj["center"] if self._player_obj else None

        candidates = []
        for obj in symbolic["objects"]:
            if obj["color_id"] != 0:          # white only
                continue
            if obj.get("shape") == "background":
                continue
            if player_center and obj["center"] == player_center:
                continue                       # skip the player itself
            candidates.append(obj)

        if not candidates:
            self._log("No white non-player objects found.")
            self._target = None
            return

        # Prefer cross/plus subpattern, then smallest
        cross = [c for c in candidates if c.get("subpattern") in ("cross", "plus")]
        self._target = min(cross or candidates, key=lambda o: o["size"])

        t = self._target
        self._log(
            f"Target: {t['color']} {t['shape']} (size {t['size']}) "
            f"at {t['center']} (color_id={t['color_id']})"
        )
        if t.get("subpattern"):
            self._log(f"  Subpattern: {t['subpattern']}")
        if player_center:
            dr = t["center"][0] - player_center[0]
            dc = t["center"][1] - player_center[1]
            self._log(
                f"Distance from player: "
                f"{abs(dr)} {'down' if dr > 0 else 'up'}, "
                f"{abs(dc)} {'right' if dc > 0 else 'left'}"
            )

    # ==================================================================
    # Phase 3: Navigate
    # ==================================================================

    def _navigate(self, max_actions=30):
        self._log("\n" + "=" * 80)
        self._log("NAVIGATION PHASE")
        self._log("=" * 80)

        blocked_action = None

        for _ in range(max_actions):
            symbolic = grid_to_symbolic(self.current_grid)
            player = self._find_current_player(symbolic)
            if player is None:
                self._log(f"\nLost track of player at turn {self._turn + 1}. Stopping.")
                break

            dr = self._target["center"][0] - player["center"][0]
            dc = self._target["center"][1] - player["center"][1]

            if abs(dr) <= 1 and abs(dc) <= 1:
                self._log("\nAdjacent to target. Moving to interaction phase.")
                break

            action = self._pick_direction(dr, dc, blocked_action)
            action_name = ACTION_NAMES.get(action, action.name)

            grid_before = [row[:] for row in self.current_grid]
            frame = self._step(action)
            self.current_grid = frame.frame[-1] if frame.frame else self.current_grid

            changes = sum(
                1
                for r in range(len(grid_before))
                for c in range(len(grid_before[0]))
                if grid_before[r][c] != self.current_grid[r][c]
            )
            if changes == 0:
                blocked_action = action
                self._log(f"\nBLOCKED on {action_name} — will try orthogonal next.")
            else:
                blocked_action = None

            self._describe_turn(action_name)

            if frame.state == GameState.WIN:
                self._log("\nWIN during navigation!")
                break
            if frame.state == GameState.GAME_OVER:
                self._log("\nGAME OVER during navigation.")
                break

    def _pick_direction(self, dr, dc, blocked_action):
        """Greedy Manhattan: prefer the axis with the larger gap."""
        candidates = []
        if dr < 0:
            candidates.append((abs(dr), GameAction.ACTION1))
        elif dr > 0:
            candidates.append((abs(dr), GameAction.ACTION2))
        if dc < 0:
            candidates.append((abs(dc), GameAction.ACTION3))
        elif dc > 0:
            candidates.append((abs(dc), GameAction.ACTION4))

        candidates.sort(key=lambda x: x[0], reverse=True)

        for _, action in candidates:
            if action != blocked_action:
                return action
        return candidates[0][1] if candidates else GameAction.ACTION1

    def _find_current_player(self, symbolic):
        """Locate the player in *symbolic* by color_id + proximity."""
        if self._player_color_id is None:
            return None

        last_center = self._player_obj["center"] if self._player_obj else [32, 32]
        best, best_dist = None, 999

        for obj in symbolic["objects"]:
            if obj["color_id"] != self._player_color_id:
                continue
            if obj.get("shape") == "background":
                continue
            dist = (
                abs(obj["center"][0] - last_center[0])
                + abs(obj["center"][1] - last_center[1])
            )
            if dist < best_dist:
                best_dist = dist
                best = obj

        if best is not None:
            self._player_obj = best
        return best

    # ==================================================================
    # Phase 4: Interact
    # ==================================================================

    def _interact(self):
        self._log("\n" + "=" * 80)
        self._log("INTERACTION PHASE")
        self._log("=" * 80)

        if self._target is None:
            return

        symbolic = grid_to_symbolic(self.current_grid)
        player = self._find_current_player(symbolic)
        if player is None:
            self._log("Lost player. Cannot interact.")
            return

        # Walk into target
        dr = self._target["center"][0] - player["center"][0]
        dc = self._target["center"][1] - player["center"][1]
        if dr != 0 or dc != 0:
            if abs(dr) >= abs(dc):
                action = GameAction.ACTION1 if dr < 0 else GameAction.ACTION2
            else:
                action = GameAction.ACTION3 if dc < 0 else GameAction.ACTION4
            frame = self._step(action)
            self.current_grid = frame.frame[-1] if frame.frame else self.current_grid
            self._describe_turn(
                f"{ACTION_NAMES.get(action, action.name)} (walk into target)"
            )

        # Try ACTION5 (spacebar)
        frame = self._step(GameAction.ACTION5)
        self.current_grid = frame.frame[-1] if frame.frame else self.current_grid
        self._describe_turn("ACTION5 (spacebar)")

        # Try ACTION6 if available
        available = getattr(frame, "available_actions", None) or []
        if 6 in available:
            frame = self._step(GameAction.ACTION6)
            self.current_grid = frame.frame[-1] if frame.frame else self.current_grid
            self._describe_turn("ACTION6 (click)")

    # ==================================================================
    # LLM description call
    # ==================================================================

    def _describe_turn(self, action_name):
        """Call GPT-5.4 to describe what it sees.  Log everything."""
        self._turn += 1

        symbolic = grid_to_symbolic(self.current_grid)
        sym_diff = (
            diff_symbolic(self._prev_symbolic, symbolic)
            if self._prev_symbolic
            else []
        )

        # Build & save image
        if self._prev_grid is not None:
            img = side_by_side(self._prev_grid, self.current_grid)
            img_b64 = diff_b64(self._prev_grid, self.current_grid)
        else:
            img = grid_to_image(self.current_grid)
            img_b64 = grid_b64(self.current_grid)

        img_path = os.path.join(self._run_dir, f"turn_{self._turn:03d}.png")
        img.save(img_path)

        # Format text sections
        symbolic_text = _format_symbolic(symbolic)
        diff_text = _format_diff(sym_diff) if sym_diff else "No changes."

        prompt_text = (
            "You are observing a 64x64 grid-based game. You are given an image of the "
            "current game state (side-by-side with previous frame, red outlines = changed "
            "cells) and a symbolic analysis computed from the raw pixel grid.\n\n"
            "Answer each section:\n\n"
            "IMAGE PERCEPTION:\n"
            "1. What objects do you see in the image? (color, shape, approximate position, size)\n"
            "2. What changed since the previous frame? Be specific about movements, "
            "appearances, disappearances.\n"
            "3. Do any objects share visual similarities (same color, shape, pattern)?\n\n"
            "SYMBOLIC INTERPRETATION:\n"
            "The following symbolic analysis was computed from the pixel grid. For each "
            "object and each change listed, state whether it matches what you see in the "
            "image. Flag any discrepancies \u2014 objects the symbolic analysis found that you "
            "can\u2019t see, or things you see that the analysis missed.\n\n"
            f"SYMBOLIC STATE:\n{symbolic_text}\n\n"
            f"CHANGES DETECTED:\n{diff_text}\n\n"
            "SYNTHESIS:\n"
            "Combining the image and symbolic data, give a unified description of the "
            "current game state and what just happened. Note any contradictions between "
            "your visual perception and the symbolic analysis."
        )

        # Call GPT-5.4
        try:
            response = self.client.responses.create(
                model="gpt-5.4",
                input=[
                    {
                        "role": "user",
                        "content": [
                            input_image_b64(img_b64),
                            input_text(prompt_text),
                        ],
                    }
                ],
                temperature=0.1,
                max_output_tokens=1500,
            )
            description = (
                response.output_text
                if hasattr(response, "output_text")
                else str(response)
            )
        except Exception as e:
            description = f"[LLM ERROR: {e}]"
            logger.warning(f"[dummy-nav] LLM call failed turn {self._turn}: {e}")

        # Write to log
        self._log(f"\n{'=' * 80}")
        self._log(f"TURN {self._turn} | ACTION: {action_name}")
        self._log("=" * 80)
        self._log(f"[Image: turn_{self._turn:03d}.png]")
        self._log(f"\nSYMBOLIC STATE:\n{symbolic_text}")
        self._log(f"\nCHANGES:\n{diff_text}")
        self._log(f"\nGPT-5.4 DESCRIPTION:\n{description}")

        # Update tracking state
        self._prev_symbolic = symbolic
        self._prev_grid = [row[:] for row in self.current_grid]

    # ==================================================================
    # Logging helpers
    # ==================================================================

    def _write_header(self):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._log("=" * 80)
        self._log(f"VISION TEST RUN \u2014 {self.game_id} \u2014 {ts}")
        self._log("=" * 80)

    def _log(self, text):
        self._log_file.write(text + "\n")
        self._log_file.flush()
        logger.info(text)

    def _cleanup(self):
        self._log_file.close()
        self._close()
        logger.info(f"[dummy-nav] Run complete. Log: {self._log_path}")


# ======================================================================
# Module-level formatting helpers
# ======================================================================


def _format_symbolic(symbolic: dict) -> str:
    """Human-readable text from a symbolic state dict."""
    lines = []
    objects = symbolic.get("objects", [])
    lines.append(f"Objects ({len(objects)}):")

    id_to_num = {}
    for i, obj in enumerate(objects, 1):
        id_to_num[obj.get("id", i - 1)] = i
        parts = [
            f"#{i}  {obj['color']} {obj['shape']} "
            f"(size {obj['size']}) at center {obj['center']}"
        ]
        if obj.get("subpattern"):
            parts.append(f"subpattern: {obj['subpattern']}")
        if obj.get("holes") and obj["holes"]["count"] > 0:
            parts.append(f"holes: {obj['holes']['count']}")
        if obj.get("orientation"):
            parts.append(f"orientation: {obj['orientation']}")
        lines.append("  " + " \u2014 ".join(parts))

    bgs = symbolic.get("backgrounds", [])
    if bgs:
        bg_parts = [f"{b['color']} ({b['size']} cells)" for b in bgs]
        lines.append(f"Backgrounds: {', '.join(bg_parts)}")

    rels = symbolic.get("relations", [])
    if rels:
        rel_parts = [
            f"#{id_to_num.get(r['a'], r['a'])} {r['type']} "
            f"#{id_to_num.get(r['b'], r['b'])}"
            for r in rels
        ]
        lines.append(f"Relations: {', '.join(rel_parts)}")

    return "\n".join(lines)


def _format_diff(diff: list[dict]) -> str:
    """Human-readable text from a symbolic diff list."""
    if not diff:
        return "  (no changes)"

    lines = []
    for change in diff:
        color = change.get("color", "unknown")
        ctype = change.get("type", "unknown")

        if ctype == "changed":
            parts = [f"{color}:"]
            if "center" in change:
                was, now = change["center"]["was"], change["center"]["now"]
                dr, dc = now[0] - was[0], now[1] - was[1]
                dirs = []
                if dr < 0:
                    dirs.append("up")
                if dr > 0:
                    dirs.append("down")
                if dc < 0:
                    dirs.append("left")
                if dc > 0:
                    dirs.append("right")
                direction = "+".join(dirs) if dirs else "none"
                parts.append(f"center {was} -> {now} (moved {direction})")
            if "size" in change:
                ws, ns = change["size"]["was"], change["size"]["now"]
                word = "grew" if ns > ws else "shrunk"
                parts.append(f"size {ws} -> {ns} ({word} by {abs(ns - ws)})")
            if "shape" in change:
                parts.append(
                    f"shape {change['shape']['was']} -> {change['shape']['now']}"
                )
            lines.append("  " + " ".join(parts))

        elif ctype == "appeared":
            lines.append(
                f"  {color}: appeared at {change.get('at', '?')} "
                f"(size {change.get('size', '?')})"
            )
        elif ctype == "disappeared":
            lines.append(
                f"  {color}: disappeared from {change.get('was_at', '?')} "
                f"(was size {change.get('was_size', '?')})"
            )
        elif ctype == "background_size_changed":
            ws, ns = change["size"]["was"], change["size"]["now"]
            lines.append(f"  {color} background: {ws} -> {ns} cells")

    return "\n".join(lines)
