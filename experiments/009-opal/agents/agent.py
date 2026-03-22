"""OPAL Agent — Observe, Perceive, Act, Loop.

Minimizes LLM calls by using code for bookkeeping and only calling
the LLM when there's a real decision to make: initial perception,
surprises, and stagnation.

Key design choices (from experiments 001-008):
- Auto-probe with action CLASSIFICATION (not just raw diffs)
- LLM declares progress indicators upfront; code monitors them
- No DAG, no structured outputs — full reasoning ability
- Stagnation detection forces strategy revision
- Free-form JSON: interpretation, progress_indicators, actions, reasoning
"""

import json
import logging
import os
import time

from arcengine import FrameData, GameAction, GameState
from openai import OpenAI

from .sandbox import Sandbox
from .symbolic import grid_to_symbolic, diff_symbolic
from .vision import grid_b64, diff_b64, input_text, input_image_b64

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# System prompt
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """You are playing an unknown turn-based game on a 64x64 grid. Each cell is an integer 0-15 (colors).
Your score (levels_completed) increases when you complete a level. GAME_OVER means you died.

HUD (bottom of the grid, consistent across all games):
- The yellow bar along the bottom represents your REMAINING MOVES/ENERGY. It shrinks as you act. When it runs out, you die (GAME_OVER). Manage it carefully.
- Red dots near the bottom represent your REMAINING LIVES. Each death costs one. When all are gone, the run ends.
- These are NOT game objects — do not try to interact with them. They are status indicators only.

You will receive ESTABLISHED FACTS about what each action does, classified by effect type:
- movement: shifts cells consistently (directional control)
- toggle: causes large reversible changes (switch/trigger)
- transform: causes large irreversible changes
- parameterized: requires x,y coordinates (click — different targets may have different effects)
- null: no observable effect

CORE PRIORS:
- Objects are cohesive: connected cells of same color move as a unit
- Objects interact by CONTACT: walking into, stepping on, clicking
- If something changed far away after a nearby interaction, it's a remote trigger
- Visual similarity between objects usually means a gameplay relationship
- Count similar objects — the number often matters
- Proximity matters: nearby objects are more likely related

COLORS ARE KEY:
- You receive a COLOR CENSUS every turn showing all colors on the board and how counts change
- PAY ATTENTION to which colors are present — they define the game's visual language
- Colors that CHANGE in count are dynamic (player, interactables, HUD elements)
- Colors that STAY FIXED are structural (walls, background, borders)
- Colors that APPEAR or DISAPPEAR signal major state changes
- If two objects share a color, they are probably related mechanically
- A small colored object near a large same-colored structure often means "key fits lock"

You respond in JSON with these fields:
{
  "reasoning": "Your analysis of what's happening and what to do next",
  "interpretation": "What kind of game this is and how it works (updated each call)",
  "progress_indicators": [
    {"name": "indicator name", "description": "what to monitor", "metric": "how code should measure it"}
  ],
  "strategy": "Your current high-level strategy",
  "actions": ["ACTION1", "ACTION2", ...],
  "verified_rules": ["rule1", "rule2"]
}

IMPORTANT RULES:
1. On your FIRST call, you MUST define progress_indicators. These tell the system what progress
   looks like so it can detect when you're stuck. Be specific and measurable:
   - BAD: "making progress" (unmeasurable)
   - GOOD: "cells_changed_total increases" or "new grid regions reached" or "score increases"
2. Output MULTIPLE actions per call — enough to test your current hypothesis or make meaningful progress.
3. When you receive a STAGNATION warning, your current approach is NOT WORKING. You must:
   - Acknowledge which indicators failed to move
   - Explain why your previous strategy failed
   - Propose a fundamentally different approach (not a minor variation)
4. After interacting with something, study the symbolic diff. What SPECIFICALLY changed?
   Build causal models: "doing X causes Y."
5. Once you understand a mechanic, EXPLOIT it immediately. Don't re-test confirmed knowledge.
6. You have limited actions. Every action must gather information or make progress.
7. When you complete a level, layout changes but game MECHANICS carry over.
"""

# --------------------------------------------------------------------------
# Grid state hashing (game area only, excludes HUD rows 60-63)
# --------------------------------------------------------------------------

def game_area_hash(grid: list[list[int]]) -> int:
    """Hash the game area of the grid (rows 0-59), ignoring HUD."""
    return hash(tuple(
        grid[r][c] for r in range(min(len(grid), 60)) for c in range(0, len(grid[0]), 2)
    ))


# --------------------------------------------------------------------------
# Action classification
# --------------------------------------------------------------------------

def classify_action_effect(
    cells_changed: int,
    blocked: bool,
    sym_changes: list[dict],
    caused_death: bool,
    is_reversible: bool | None = None,
) -> str:
    """Classify an action's observed effect into a type."""
    if caused_death:
        return "death"
    if blocked or cells_changed == 0:
        return "null"
    if cells_changed < 10:
        return "null"
    if cells_changed > 200:
        if is_reversible:
            return "toggle"
        return "transform"
    # Normal movement range (typically 30-80 cells for player sprite moving)
    # Check if a single object moved consistently
    moved = [c for c in sym_changes if c.get("type") == "changed" and "center" in c]
    if moved:
        return "movement"
    return "movement"  # default for moderate cell changes


# --------------------------------------------------------------------------
# Progress indicator monitoring
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Color census — what colors exist and how they change
# --------------------------------------------------------------------------

def color_census(grid: list[list[int]]) -> dict[int, int]:
    """Count cells of each color in the grid."""
    counts: dict[int, int] = {}
    for row in grid:
        for val in row:
            counts[val] = counts.get(val, 0) + 1
    return counts


def color_census_summary(current: dict[int, int], previous: dict[int, int] | None = None) -> str:
    """Human-readable color census with optional delta from previous."""
    from .symbolic import COLOR_NAMES

    lines = []
    all_colors = sorted(set(list(current.keys()) + (list(previous.keys()) if previous else [])))

    for c in all_colors:
        name = COLOR_NAMES.get(c, f"color_{c}")
        count = current.get(c, 0)
        if count == 0 and (previous is None or previous.get(c, 0) == 0):
            continue

        if previous is not None:
            prev = previous.get(c, 0)
            delta = count - prev
            if delta != 0:
                sign = "+" if delta > 0 else ""
                lines.append(f"  {name}({c}): {count} cells ({sign}{delta})")
            else:
                lines.append(f"  {name}({c}): {count} cells")
        else:
            lines.append(f"  {name}({c}): {count} cells")

    return "\n".join(lines)


# --------------------------------------------------------------------------
# Progress monitoring — game-agnostic stagnation detection
# --------------------------------------------------------------------------

class ProgressMonitor:
    """Detects stagnation using game-agnostic signals.

    Three independent stagnation signals (any one triggers):
    1. Score hasn't changed AND player position hasn't meaningfully shifted
    2. Action diversity is very low (repeating 1-2 actions)
    3. Color distribution is oscillating (same pattern repeating = cycling)
    """

    def __init__(self) -> None:
        self.indicators: list[dict] = []
        self.snapshots: list[dict] = []
        self.last_progress_action: int = 0
        self.stagnation_threshold: int = 12
        self.all_visited_hashes: set[int] = set()  # every grid state ever seen

    def set_indicators(self, indicators: list[dict]) -> None:
        self.indicators = indicators
        # Don't reset snapshots — keep history for continuity
        self.last_progress_action = max(
            (s["action"] for s in self.snapshots), default=0
        )

    def record(self, action_num: int, grid: list[list[int]], frame: FrameData,
               cells_changed: int, blocked: bool, sym_state: dict,
               action_name: str = "") -> None:
        """Record a snapshot after an action."""
        # Full grid hash for exact state tracking
        grid_hash = hash(tuple(grid[r][c] for r in range(0, 64, 4) for c in range(0, 64, 4)))

        # Color distribution fingerprint — captures overall board composition
        census = color_census(grid)
        color_fingerprint = tuple(sorted(census.items()))

        is_new_state = grid_hash not in self.all_visited_hashes
        self.all_visited_hashes.add(grid_hash)

        snapshot = {
            "action": action_num,
            "action_name": action_name,
            "score": frame.levels_completed,
            "cells_changed": cells_changed,
            "blocked": blocked,
            "num_objects": sym_state.get("num_objects", 0),
            "grid_hash": grid_hash,
            "color_fingerprint": hash(color_fingerprint),
            "is_new_state": is_new_state,
        }
        self.snapshots.append(snapshot)

        if self._detect_progress(snapshot):
            self.last_progress_action = action_num

    def _detect_progress(self, snapshot: dict) -> bool:
        """Game-agnostic progress detection."""
        if len(self.snapshots) < 2:
            return True

        # Score increase — always progress
        if snapshot["score"] > self.snapshots[-2]["score"]:
            return True

        # Truly new grid state (never seen in entire run) — progress
        if snapshot["is_new_state"] and not snapshot["blocked"]:
            return True

        # Large change producing new color distribution — progress
        if snapshot["cells_changed"] > 100:
            recent_fingerprints = [s["color_fingerprint"] for s in self.snapshots[-6:-1]]
            if snapshot["color_fingerprint"] not in recent_fingerprints:
                return True

        return False

    def is_stagnating(self, current_action: int) -> bool:
        """Check if we've gone too long without progress."""
        return (current_action - self.last_progress_action) >= self.stagnation_threshold

    def stagnation_report(self, current_action: int) -> str:
        """Generate a detailed, game-agnostic report for the LLM."""
        actions_stuck = current_action - self.last_progress_action
        recent = self.snapshots[-self.stagnation_threshold:]
        if not recent:
            return "*** STAGNATION: No actions recorded ***"

        blocked_count = sum(1 for s in recent if s["blocked"])
        total_cells = sum(s["cells_changed"] for s in recent)
        new_states = sum(1 for s in recent if s["is_new_state"])
        score_change = recent[-1]["score"] - recent[0]["score"] if len(recent) >= 2 else 0

        # Action diversity — how many distinct actions used
        action_names = [s.get("action_name", "") for s in recent if s.get("action_name")]
        unique_actions = len(set(action_names))
        most_common = max(set(action_names), key=action_names.count) if action_names else "?"
        most_common_pct = round(action_names.count(most_common) / max(len(action_names), 1) * 100)

        # Color fingerprint cycling
        fingerprints = [s["color_fingerprint"] for s in recent]
        unique_fingerprints = len(set(fingerprints))
        is_color_cycling = unique_fingerprints <= max(2, len(fingerprints) // 4)

        # Grid state cycling
        hashes = [s["grid_hash"] for s in recent]
        unique_grid_states = len(set(hashes))
        is_grid_cycling = unique_grid_states <= max(2, len(hashes) // 3)

        lines = [
            f"*** STAGNATION DETECTED: {actions_stuck} actions with no meaningful progress ***",
            f"",
            f"WHAT HAPPENED in the last {len(recent)} actions:",
            f"  Score change: {score_change}",
            f"  Blocked: {blocked_count}/{len(recent)}",
            f"  New grid states (never seen before): {new_states}/{len(recent)}",
            f"  Total cells changed: {total_cells}",
            f"  Action diversity: {unique_actions} distinct actions ('{most_common}' used {most_common_pct}% of the time)",
            f"  Unique color distributions: {unique_fingerprints}/{len(fingerprints)}",
            f"  Unique grid states: {unique_grid_states}/{len(hashes)}",
        ]

        # Diagnosis
        lines.append("")
        lines.append("DIAGNOSIS:")
        if is_grid_cycling:
            lines.append("  - GRID CYCLING: You are revisiting the same board states. Your actions")
            lines.append("    are undoing each other (e.g., toggling back and forth, pacing in a loop).")
        if is_color_cycling:
            lines.append("  - COLOR CYCLING: The overall color distribution is oscillating between")
            lines.append("    the same few patterns. This means a mechanic you're using is reversible")
            lines.append("    and you're not making lasting changes to the board.")
        if unique_actions <= 2:
            lines.append(f"  - LOW DIVERSITY: You're only using {unique_actions} distinct action(s).")
            lines.append(f"    '{most_common}' accounts for {most_common_pct}% of actions. Try different actions.")
        if new_states == 0:
            lines.append("  - NO NEW STATES: Every grid state in this window was already visited earlier")
            lines.append("    in the run. You are retreading old ground entirely.")
        if blocked_count > len(recent) * 0.5:
            lines.append(f"  - MOSTLY BLOCKED: {blocked_count}/{len(recent)} actions were blocked.")
            lines.append("    You're hitting walls repeatedly. Move to a different area.")

        # Instructions
        lines.append("")
        lines.append("Your declared progress indicators:")
        for ind in self.indicators:
            lines.append(f"  - {ind.get('name', '?')}: {ind.get('description', '?')}")
        lines.append("")
        lines.append("NONE of these showed movement. You MUST change strategy:")
        lines.append("  1. STOP your current action pattern entirely")
        lines.append("  2. Move to a DIFFERENT area of the grid you haven't explored")
        lines.append("  3. Try actions or action COMBINATIONS you haven't used recently")
        lines.append("  4. If you used a switch/trigger, use it ONCE then immediately explore what changed")
        lines.append("  5. Reassess: what colors/objects have you NOT interacted with?")

        return "\n".join(lines)


# --------------------------------------------------------------------------
# OPAL Agent
# --------------------------------------------------------------------------

class OpalAgent:
    MAX_ACTIONS = 100
    STAGNATION_THRESHOLD = 12

    def __init__(self, game_id: str) -> None:
        from arc_agi import Arcade

        self.game_id = game_id
        self.arcade = Arcade()
        self.scorecard_id = self.arcade.open_scorecard()
        self.env = self.arcade.make(game_id, scorecard_id=self.scorecard_id)
        self.frames: list[FrameData] = []
        self.action_counter = 0
        self.total_deaths = 0

        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        self.current_grid: list[list[int]] = []
        self.prev_grid: list[list[int]] | None = None
        self.prev_symbolic: dict | None = None

        # OPAL state
        self.probe_facts: str = ""
        self.action_classifications: dict[str, str] = {}  # ACTION1 -> "movement"
        # Position-specific action blacklist: grid_hash -> set of action names that did nothing
        # "ACTION1 did nothing from THIS state, don't try it from here again"
        self.state_action_blacklist: dict[int, set[str]] = {}
        self.verified_rules: list[str] = []
        self.max_rules = 10
        self.llm_calls: int = 0
        self.last_llm_interpretation: str = ""
        self.last_llm_strategy: str = ""
        self.diff_log: list[dict] = []  # structured log of all action outcomes
        self.life_log: list[dict] = []  # action log for current life (reset on death)
        self.post_mortem: str = ""  # LLM analysis from previous life
        self.progress_monitor = ProgressMonitor()
        self.progress_monitor.stagnation_threshold = self.STAGNATION_THRESHOLD

        # Conversation history (sliding window)
        self.messages: list[dict] = []
        self.message_limit = 8

        # Sandbox for optional code
        self.sandbox = Sandbox()

    def run(self) -> None:
        timer = time.time()
        logger.info("=" * 60)
        logger.info(f"OPAL Agent on {self.game_id}")
        logger.info("=" * 60)

        # === OBSERVE ===
        frame = self._step(GameAction.RESET)
        self.current_grid = frame.frame[-1] if frame.frame else []

        if frame.state == GameState.WIN:
            self._close()
            return

        available = frame.available_actions or [1, 2, 3, 4]
        self.probe_facts, self.action_classifications = self._observe(frame, available)
        logger.info(f"[observe] {self.probe_facts}")
        logger.info(f"[classify] {self.action_classifications}")

        # === PERCEIVE (first LLM call) ===
        actions = self._perceive(frame, is_first=True)

        # === ACT + LOOP ===
        action_queue = list(actions)

        while self.action_counter < self.MAX_ACTIONS:
            if frame.state == GameState.WIN:
                logger.info("WIN!")
                break

            # Handle death — run post-mortem analysis on the life that just ended
            if frame.state == GameState.GAME_OVER:
                self.total_deaths += 1
                logger.info(f"DIED #{self.total_deaths}")

                # Post-mortem: analyze the full life before resetting
                self.post_mortem = self._post_mortem(frame)
                logger.info(f"[post-mortem] {self.post_mortem[:300]}")

                # Reset for next life
                frame = self._step(GameAction.RESET)
                self.action_counter += 1
                self.current_grid = frame.frame[-1] if frame.frame else []
                self.prev_grid = None
                self.prev_symbolic = None
                self.life_log = []  # fresh life

                # Perceive with post-mortem context
                action_queue = self._perceive(frame, death=True)
                continue

            # If no actions queued, or stagnation, call LLM
            if not action_queue:
                if self.progress_monitor.is_stagnating(self.action_counter):
                    action_queue = self._perceive(frame, stagnation=True)
                else:
                    action_queue = self._perceive(frame)

            if not action_queue:
                action_queue = [GameAction.ACTION1]

            # Filter out actions blacklisted from the current grid state
            current_hash = game_area_hash(self.current_grid)
            blocked_here = self.state_action_blacklist.get(current_hash, set())
            if blocked_here:
                action_queue = [a for a in action_queue if a.name not in blocked_here]
            if not action_queue:
                action_queue = [GameAction.ACTION1]

            # Execute next action
            action = action_queue.pop(0)
            grid_before = self.current_grid
            score_before = frame.levels_completed
            sym_before = grid_to_symbolic(self.current_grid)

            frame = self._step(action, reasoning="")
            self.action_counter += 1
            self.current_grid = frame.frame[-1] if frame.frame else grid_before

            cells_changed = sum(
                1 for r in range(len(grid_before))
                for c in range(len(grid_before[0]))
                if grid_before[r][c] != self.current_grid[r][c]
            )
            # Game-area-only change count (exclude bottom HUD rows 60-63)
            game_cells_changed = sum(
                1 for r in range(min(len(grid_before), 60))
                for c in range(len(grid_before[0]))
                if grid_before[r][c] != self.current_grid[r][c]
            )
            blocked = 0 < game_cells_changed < 10

            sym_after = grid_to_symbolic(self.current_grid)
            sym_changes = diff_symbolic(sym_before, sym_after) if sym_before else []

            # Record in diff log and life log
            log_entry = {
                "turn": self.action_counter,
                "action": action.name,
                "blocked": blocked,
                "cells": game_cells_changed,
                "trigger": game_cells_changed > 100,
                "sym_changes_count": len(sym_changes),
            }
            self.diff_log.append(log_entry)
            self.life_log.append(log_entry)

            # Record for progress monitoring
            self.progress_monitor.record(
                self.action_counter, self.current_grid, frame,
                cells_changed, blocked, sym_after,
                action_name=action.name,
            )

            self.prev_symbolic = sym_after

            status = "BLOCKED" if blocked else f"{cells_changed}ch"
            logger.info(f"#{self.action_counter} {action.name}: {status} score={frame.levels_completed}")

            # === LOOP: decide whether to call LLM ===

            # Zero change in game area — record that this action does nothing from
            # this specific grid state, so we never waste an action trying it here again.
            if game_cells_changed == 0:
                state_hash = game_area_hash(grid_before)
                if state_hash not in self.state_action_blacklist:
                    self.state_action_blacklist[state_hash] = set()
                self.state_action_blacklist[state_hash].add(action.name)
                logger.info(f"[no-effect] {action.name} blocked from this state — won't retry here")
                action_queue = []  # force LLM call
                continue

            # Win/death — break to outer loop
            if frame.state in (GameState.WIN, GameState.GAME_OVER):
                action_queue = []
                continue

            # Level up — re-observe and re-perceive
            if frame.levels_completed > score_before:
                logger.info(f"[level-up] Level {frame.levels_completed}!")
                self.prev_grid = None
                self.prev_symbolic = None
                self._prev_census = None
                self.state_action_blacklist = {}  # new level = fresh grid states
                self.life_log = []
                self.post_mortem = ""
                self.messages = []
                self.diff_log = []
                self.progress_monitor = ProgressMonitor()
                self.progress_monitor.stagnation_threshold = self.STAGNATION_THRESHOLD
                avail = frame.available_actions or [1, 2, 3, 4]
                self.probe_facts, self.action_classifications = self._observe(frame, avail)
                logger.info(f"[observe] {self.probe_facts}")
                action_queue = self._perceive(frame, is_first=True)
                continue

            # Surprise — large unexpected change, call LLM
            if cells_changed > 100:
                logger.info(f"[surprise] {cells_changed} cells changed — calling LLM")
                self.prev_grid = grid_before
                action_queue = self._perceive(frame, surprise=True, sym_changes=sym_changes)
                continue

            # Blocked — break current sequence UNLESS the queue contains
            # repeated same-action (LLM intentionally wants repeated blocked presses)
            if blocked:
                if action_queue and all(a == action for a in action_queue):
                    # LLM asked for repeated presses — keep going, it knows what it's doing
                    pass
                else:
                    action_queue = []  # will trigger LLM call on next iteration
                    continue

            # Stagnation check — mid-sequence
            if self.progress_monitor.is_stagnating(self.action_counter):
                logger.info(f"[stagnation] No progress for {self.STAGNATION_THRESHOLD} actions")
                action_queue = self._perceive(frame, stagnation=True)
                continue

        elapsed = round(time.time() - timer, 2)
        logger.info("=" * 60)
        logger.info(
            f"FINISHED: actions={self.action_counter} state={frame.state.name} "
            f"score={frame.levels_completed} deaths={self.total_deaths} "
            f"llm_calls={self.llm_calls} time={elapsed}s"
        )
        if self.verified_rules:
            logger.info(f"Verified rules: {self.verified_rules}")
        logger.info("=" * 60)
        self._close()

    # ------------------------------------------------------------------
    # OBSERVE: probe and classify all actions
    # ------------------------------------------------------------------

    def _observe(self, frame: FrameData, available: list[int]) -> tuple[str, dict]:
        """Probe each available action and classify its effect type."""
        facts = []
        classifications = {}

        simple_actions = [a for a in available if a != 6]
        has_action6 = 6 in available

        for action_id in simple_actions:
            try:
                action = GameAction.from_id(action_id)
            except (ValueError, KeyError):
                continue

            grid_before = self.current_grid
            sym_before = grid_to_symbolic(grid_before)

            result_frame = self._step(action)
            self.action_counter += 1
            self.current_grid = result_frame.frame[-1] if result_frame.frame else grid_before

            sym_after = grid_to_symbolic(self.current_grid)
            sym_changes = diff_symbolic(sym_before, sym_after)

            cells_changed = sum(
                1 for r in range(64) for c in range(64)
                if grid_before[r][c] != self.current_grid[r][c]
            )
            # Game-area-only (exclude HUD rows 60-63)
            game_cells_changed = sum(
                1 for r in range(min(64, 60)) for c in range(64)
                if grid_before[r][c] != self.current_grid[r][c]
            )
            blocked = 0 < game_cells_changed < 10
            caused_death = result_frame.state == GameState.GAME_OVER

            effect_type = classify_action_effect(
                game_cells_changed, blocked, sym_changes, caused_death
            )
            classifications[action.name] = effect_type

            # Record zero game-area-change actions at this probe state
            if game_cells_changed == 0 and not caused_death:
                state_hash = game_area_hash(grid_before)
                if state_hash not in self.state_action_blacklist:
                    self.state_action_blacklist[state_hash] = set()
                self.state_action_blacklist[state_hash].add(action.name)
                logger.info(f"[blacklist] {action.name} zero game-area change at probe state")

            # Build fact string with symbolic changes
            change_summary = []
            for c in sym_changes[:5]:
                if c.get("type") == "changed" and "center" in c:
                    change_summary.append(
                        f"{c.get('color','?')}: center {c['center']['was']}->{c['center']['now']}"
                    )
                elif c.get("type") == "background_size_changed":
                    change_summary.append(
                        f"background {c.get('color','?')}: size {c['size']['was']}->{c['size']['now']}"
                    )

            status = "BLOCKED" if blocked else f"{cells_changed} cells changed"
            fact = f"{action.name} [{effect_type}]: {status}"
            if change_summary:
                fact += "\n  " + "\n  ".join(change_summary)

            if caused_death:
                self.total_deaths += 1
                fact += "\n  (caused GAME_OVER — action is lethal or depletes resource)"
                result_frame = self._step(GameAction.RESET)
                self.action_counter += 1
                self.current_grid = result_frame.frame[-1] if result_frame.frame else []

            facts.append(fact)

        # ACTION6 (parameterized) probing
        if has_action6:
            symbolic = grid_to_symbolic(self.current_grid)
            fg = symbolic.get("objects", [])
            targets = [(32, 32, "grid center")]
            if fg:
                obj = fg[0]
                targets.append((obj["center"]["col"], obj["center"]["row"],
                               f"{obj['color']} object"))
            if len(fg) >= 2:
                obj = fg[-1]
                targets.append((obj["center"]["col"], obj["center"]["row"],
                               f"{obj['color']} object"))

            for x, y, desc in targets[:3]:
                try:
                    action = GameAction.from_id(6, x=int(x), y=int(y))
                except (ValueError, KeyError):
                    continue

                grid_before = self.current_grid
                sym_before = grid_to_symbolic(grid_before)

                result_frame = self._step(action)
                self.action_counter += 1
                self.current_grid = result_frame.frame[-1] if result_frame.frame else grid_before

                sym_after = grid_to_symbolic(self.current_grid)
                sym_changes = diff_symbolic(sym_before, sym_after)

                cells_changed = sum(
                    1 for r in range(64) for c in range(64)
                    if grid_before[r][c] != self.current_grid[r][c]
                )
                blocked = 0 < cells_changed < 10
                caused_death = result_frame.state == GameState.GAME_OVER

                effect_type = classify_action_effect(
                    cells_changed, blocked, sym_changes, caused_death
                )

                status = "BLOCKED" if blocked else f"{cells_changed} cells changed"
                fact = f"ACTION6({x},{y}) on {desc} [{effect_type}]: {status}"

                if caused_death:
                    self.total_deaths += 1
                    fact += "\n  (caused GAME_OVER)"
                    result_frame = self._step(GameAction.RESET)
                    self.action_counter += 1
                    self.current_grid = result_frame.frame[-1] if result_frame.frame else []

                facts.append(fact)
                classifications[f"ACTION6({x},{y})"] = effect_type

        return "\n".join(facts), classifications

    # ------------------------------------------------------------------
    # PERCEIVE: call the LLM
    # ------------------------------------------------------------------

    def _perceive(
        self,
        frame: FrameData,
        is_first: bool = False,
        stagnation: bool = False,
        surprise: bool = False,
        death: bool = False,
        sym_changes: list[dict] | None = None,
    ) -> list[GameAction]:
        """Call the LLM and return an action sequence."""
        self.llm_calls += 1

        available = frame.available_actions or [1, 2, 3, 4]
        avail_str = ", ".join(f"ACTION{i}" for i in available)

        # Build context
        parts = []
        parts.append(
            f"Score: {frame.levels_completed} | Deaths: {self.total_deaths} | "
            f"Actions: {self.action_counter}/{self.MAX_ACTIONS} | "
            f"Available: {avail_str}"
        )

        # Probe facts always included on first message
        if is_first or not self.messages:
            parts.append(f"\nESTABLISHED FACTS (from auto-probe):\n{self.probe_facts}")
            class_str = "\n".join(f"  {k}: {v}" for k, v in self.action_classifications.items())
            parts.append(f"\nACTION CLASSIFICATIONS:\n{class_str}")

        # Actions blocked from current position
        current_hash = game_area_hash(self.current_grid)
        blocked_here = self.state_action_blacklist.get(current_hash, set())
        if blocked_here:
            bl_str = ", ".join(sorted(blocked_here))
            parts.append(f"\nBLOCKED FROM HERE (already tried, zero effect): {bl_str}")

        # Verified rules
        if self.verified_rules:
            rules_text = "\n".join(f"- {r}" for r in self.verified_rules)
            parts.append(f"\nVERIFIED RULES:\n{rules_text}")

        # Recent action log (compact)
        if self.diff_log:
            recent = self.diff_log[-15:]
            log_lines = []
            for entry in recent:
                status = "BLOCKED" if entry["blocked"] else f"{entry['cells']}ch"
                trigger = " [TRIGGER]" if entry["trigger"] else ""
                log_lines.append(f"  #{entry['turn']} {entry['action']}: {status}{trigger}")
            parts.append(f"\nRECENT ACTIONS:\n" + "\n".join(log_lines))

        # Symbolic state
        symbolic = grid_to_symbolic(self.current_grid)
        sym_output = {
            "objects": symbolic.get("objects", []),
            "relations": symbolic.get("relations", []),
        }
        if "composites" in symbolic:
            sym_output["composites"] = symbolic["composites"]
        parts.append(f"\nSCENE:\n{json.dumps(sym_output, indent=1)}")

        # Color census — what colors are on the board and how they're changing
        current_census = color_census(self.current_grid)
        prev_census = getattr(self, '_prev_census', None)
        census_text = color_census_summary(current_census, prev_census)
        self._prev_census = current_census
        parts.append(f"\nCOLOR CENSUS (all colors on the board):\n{census_text}")

        # Symbolic diff if available
        if sym_changes:
            parts.append(f"\nCHANGES FROM LAST ACTION:\n{json.dumps(sym_changes, indent=1)}")
        elif self.prev_symbolic:
            changes = diff_symbolic(self.prev_symbolic, symbolic)
            if changes:
                parts.append(f"\nCHANGES SINCE LAST LLM CALL:\n{json.dumps(changes, indent=1)}")

        self.prev_symbolic = symbolic

        # Special context based on trigger
        if stagnation:
            report = self.progress_monitor.stagnation_report(self.action_counter)
            parts.append(f"\n{report}")
            # Reset stagnation counter so we don't immediately re-trigger
            self.progress_monitor.last_progress_action = self.action_counter

        if death:
            parts.append(
                f"\n*** YOU DIED (death #{self.total_deaths}). "
                f"The game reset. Use the post-mortem analysis below to do better this life. ***"
            )
            if self.post_mortem:
                parts.append(f"\nPOST-MORTEM FROM LAST LIFE:\n{self.post_mortem}")

        if surprise:
            parts.append(
                "\n*** SURPRISE: A large change just occurred (>100 cells). "
                "Study the symbolic diff carefully. What caused this? What opened/closed? ***"
            )

        text_content = "\n".join(parts)

        # Image: on first call, after surprise, after death
        send_image = is_first or surprise or death or not self.messages
        image_b64 = None
        if send_image:
            if self.prev_grid:
                image_b64 = diff_b64(self.prev_grid, self.current_grid)
            else:
                image_b64 = grid_b64(self.current_grid)
            self.prev_grid = self.current_grid

        # Build message
        user_content = [input_text(text_content)]
        if image_b64:
            label = "Current frame:" if is_first else "Side-by-side (PREVIOUS vs CURRENT, red=changed):"
            user_content.append(input_text(label))
            user_content.append(input_image_b64(image_b64))

        self.messages.append({"role": "user", "content": user_content})

        # Trim conversation window
        trimmed = self._trim_messages()

        # Call LLM — no structured outputs, free-form JSON
        try:
            response = self.client.responses.create(
                model="gpt-5.4",
                instructions=SYSTEM_PROMPT,
                input=trimmed,
                max_output_tokens=4000,
            )
            raw = response.output_text or "{}"
        except Exception as e:
            logger.warning(f"API error: {e}")
            time.sleep(2)
            return [GameAction.ACTION1]

        self.messages.append({"role": "assistant", "content": raw})

        # Parse response
        data = self._parse_json(raw)

        # Log reasoning
        reasoning = data.get("reasoning", "")
        interpretation = data.get("interpretation", "")
        strategy = data.get("strategy", "")

        if reasoning:
            logger.info(f"[reasoning] {reasoning[:200]}")
        if interpretation:
            self.last_llm_interpretation = interpretation
            logger.info(f"[interpretation] {interpretation[:200]}")
        if strategy:
            self.last_llm_strategy = strategy
            logger.info(f"[strategy] {strategy[:150]}")

        # Extract progress indicators
        indicators = data.get("progress_indicators", [])
        if indicators and isinstance(indicators, list):
            self.progress_monitor.set_indicators(indicators)
            for ind in indicators:
                logger.info(f"[indicator] {ind.get('name', '?')}: {ind.get('description', '?')}")

        # Extract verified rules
        new_rules = data.get("verified_rules", [])
        if new_rules and isinstance(new_rules, list):
            for rule in new_rules:
                if isinstance(rule, str) and rule:
                    self._add_rule(rule)

        # Extract actions
        action_names = data.get("actions", [])
        actions = []
        for name in action_names:
            if not isinstance(name, str):
                continue
            clean = name.strip().split("(")[0].split()[0]
            try:
                actions.append(GameAction.from_name(clean))
            except (ValueError, KeyError):
                continue

        if not actions:
            actions = [GameAction.ACTION1]

        logger.info(f"[actions] {len(actions)} actions: {[a.name for a in actions]}")
        return actions

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _post_mortem(self, frame: FrameData) -> str:
        """Analyze the life that just ended — what worked, what didn't, what to try next.

        Sends the full life action log to the LLM for analysis. Returns the
        LLM's assessment as a string to inject into the next life's context.
        """
        if not self.life_log:
            return "No actions taken this life."

        self.llm_calls += 1

        # Build compact life summary
        total_actions = len(self.life_log)
        blocked_count = sum(1 for e in self.life_log if e["blocked"])
        triggers = sum(1 for e in self.life_log if e["trigger"])

        # Full action sequence with results
        action_lines = []
        for entry in self.life_log:
            status = "BLOCKED" if entry["blocked"] else f"{entry['cells']}ch"
            trigger = " [TRIGGER]" if entry["trigger"] else ""
            action_lines.append(f"  #{entry['turn']} {entry['action']}: {status}{trigger}")

        life_summary = "\n".join([
            f"Life #{self.total_deaths} ended (GAME_OVER — energy depleted).",
            f"Total actions this life: {total_actions}",
            f"Blocked: {blocked_count}, Triggers: {triggers}",
            f"Score at death: {frame.levels_completed}",
            "",
            "Full action sequence:",
            *action_lines,
        ])

        # Add verified rules for context
        if self.verified_rules:
            life_summary += "\n\nVerified rules:\n" + "\n".join(f"- {r}" for r in self.verified_rules)

        prompt = (
            f"{life_summary}\n\n"
            "Analyze this life:\n"
            "1. What did you learn about the game mechanics?\n"
            "2. Which actions were WASTED (blocked, redundant, or unproductive)?\n"
            "3. What was the most promising direction or discovery?\n"
            "4. What specific strategy should the next life follow to be more efficient?\n"
            "   Be concrete: 'go left 3 times then up until blocked' not 'explore more'.\n"
            "5. What should the next life AVOID doing?\n\n"
            "Keep your analysis concise and actionable."
        )

        try:
            response = self.client.responses.create(
                model="gpt-5.4",
                instructions="You are analyzing a failed game life to plan the next attempt. Be concise and specific.",
                input=[{"role": "user", "content": prompt}],
                max_output_tokens=1500,
            )
            analysis = response.output_text or "No analysis generated."
        except Exception as e:
            logger.warning(f"Post-mortem API error: {e}")
            analysis = f"Post-mortem failed: {e}"

        return analysis

    def _trim_messages(self) -> list[dict]:
        """Sliding window keeping first message (probe facts) and recent tail."""
        if len(self.messages) <= self.message_limit:
            return list(self.messages)

        first = self.messages[0] if self.messages and self.messages[0]["role"] == "user" else None
        tail = self.messages[-(self.message_limit - (1 if first else 0)):]

        while tail and tail[0]["role"] == "assistant":
            tail = tail[1:]

        if first and (not tail or tail[0] is not first):
            return [first] + tail
        return tail

    def _add_rule(self, rule: str) -> None:
        """Add a verified rule with dedup."""
        import re
        if re.search(r'\[\d+,\s*\d+\]', rule) or re.search(r'at \(\d+', rule):
            return
        rule_lower = rule.lower().strip()
        for existing in self.verified_rules:
            existing_lower = existing.lower().strip()
            if rule_lower in existing_lower or existing_lower in rule_lower:
                return
            rule_words = set(rule_lower.split())
            existing_words = set(existing_lower.split())
            if rule_words and existing_words:
                overlap = len(rule_words & existing_words) / max(len(rule_words), len(existing_words))
                if overlap > 0.7:
                    return
        if len(self.verified_rules) >= self.max_rules:
            return
        self.verified_rules.append(rule)
        logger.info(f"[rule+] {rule}")

    def _parse_json(self, raw: str) -> dict:
        """Parse JSON from LLM response, with fallbacks."""
        # Try direct parse
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            pass

        # Try extracting from markdown code block
        cleaned = raw
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1]
            if "```" in cleaned:
                cleaned = cleaned.split("```", 1)[0]
        elif "```" in cleaned:
            cleaned = cleaned.split("```", 1)[1]
            if "```" in cleaned:
                cleaned = cleaned.split("```", 1)[0]
        try:
            return json.loads(cleaned.strip())
        except (json.JSONDecodeError, ValueError):
            pass

        # Try finding outermost braces
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except (json.JSONDecodeError, ValueError):
                pass

        logger.warning(f"[parse-error] Could not parse JSON from response ({len(raw)} chars)")
        return {}

    def _step(self, action: GameAction, reasoning: str = "") -> FrameData:
        """Take a step in the environment."""
        try:
            raw = self.env.step(action, reasoning=reasoning)
        except Exception as e:
            logger.warning(f"env.step({action.name}) exception: {e}")
            raw = None
        if raw is None:
            if self.frames:
                return self.frames[-1]
            return FrameData(levels_completed=0)
        frame = FrameData(
            game_id=raw.game_id,
            frame=[arr.tolist() for arr in raw.frame],
            state=raw.state,
            levels_completed=raw.levels_completed,
            win_levels=raw.win_levels,
            guid=raw.guid,
            full_reset=raw.full_reset,
            available_actions=raw.available_actions,
        )
        self.frames.append(frame)
        return frame

    def _close(self) -> None:
        if not self.scorecard_id:
            return
        scorecard = self.arcade.close_scorecard(self.scorecard_id)
        if scorecard:
            logger.info("--- SCORECARD ---")
            logger.info(json.dumps(scorecard.model_dump(), indent=2))
        self.scorecard_id = None
