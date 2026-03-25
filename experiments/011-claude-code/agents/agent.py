"""Claude Code agent: v5 game loop + Claude Code CLI as reasoning engine.

Combines:
- v5's game loop (arcengine), auto-probe, symbolic state, vision, run log
- RGB-Agent's batch action planning with queue draining
- Claude Code CLI as the reasoning engine (replaces OpenAI API calls)

The agent writes structured turn data to a run log file on disk. When the
action queue is empty, it calls Claude Code (`claude -p`) to analyze the
log and produce a new action plan. Claude Code uses its built-in Read/Grep/
Bash tools to parse the log.
"""

import json
import logging
import os
import re
import time
from collections import deque
from datetime import datetime

from arcengine import FrameData, GameAction, GameState

from .claude_code import ClaudeCodeAnalyzer
from .run_log import RunLog
from .symbolic import grid_to_symbolic, diff_symbolic
from .vision import (
    grid_to_image, side_by_side, image_to_b64, grid_b64, diff_b64,
)

logger = logging.getLogger(__name__)
vision_logger = logging.getLogger("agents.vision")


# ---------------------------------------------------------------------------
# Action queue (adapted from experiment 010)
# ---------------------------------------------------------------------------

_VALID_ACTIONS = {"ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7", "RESET"}


class ActionQueue:
    """Holds and serves a batch of parsed actions, with score-change flushing."""

    def __init__(self) -> None:
        self._queue: deque[dict] = deque()
        self.plan_total: int = 0
        self.plan_index: int = 0
        self._last_score: int = 0
        self.score_changed: bool = False

    def clear(self) -> None:
        self._queue.clear()
        self.plan_total = 0
        self.plan_index = 0

    def reset(self) -> None:
        self.clear()
        self._last_score = 0
        self.score_changed = False

    def __len__(self) -> int:
        return len(self._queue)

    def __bool__(self) -> bool:
        return bool(self._queue)

    def pop(self) -> dict:
        action = self._queue.popleft()
        self.plan_index += 1
        return action

    def check_score(self, score: int) -> None:
        """Flush the queue if the score changed."""
        if score != self._last_score:
            if self._queue:
                logger.info(
                    "score %d->%d: flushing %d queued actions",
                    self._last_score, score, len(self._queue),
                )
                self.clear()
            self.score_changed = True
            self._last_score = score

    def load(self, actions_text: str) -> bool:
        """Parse [ACTIONS] JSON and load the queue. Returns True on success."""
        clean = re.sub(r"```(?:json)?\s*", "", actions_text).strip()

        parsed = None
        decoder = json.JSONDecoder()
        for char in ("{", "["):
            idx = clean.find(char)
            if idx >= 0:
                try:
                    parsed, _ = decoder.raw_decode(clean, idx)
                    break
                except json.JSONDecodeError:
                    continue

        if parsed is None:
            logger.warning("ActionQueue.load: could not parse: %s", actions_text[:200])
            return False

        if isinstance(parsed, list):
            parsed = {"plan": parsed, "reasoning": ""}

        plan = parsed.get("plan", parsed.get("actions", []))
        if not isinstance(plan, list) or not plan:
            logger.warning("ActionQueue.load: empty or invalid plan")
            return False

        self._queue.clear()
        for step in plan:
            if isinstance(step, str):
                m = re.match(r"ACTION6\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", step)
                if m:
                    name = "ACTION6"
                    data = {"x": int(m.group(1)), "y": int(m.group(2))}
                else:
                    name, data = step, {}
            else:
                name = step.get("action")
                if not name:
                    logger.warning("skipping step with no action key: %s", step)
                    continue
                data = (
                    {"x": int(step.get("x", 0)), "y": int(step.get("y", 0))}
                    if name == "ACTION6" else {}
                )
            if name not in _VALID_ACTIONS:
                logger.warning("skipping unrecognized action: %s", name)
                continue
            self._queue.append({"name": name, "data": data})

        self.plan_total = len(self._queue)
        self.plan_index = 0
        reasoning = parsed.get("reasoning", "")
        logger.info(
            "loaded %d-step plan: %s — %s",
            self.plan_total,
            [s if isinstance(s, str) else s.get("action") for s in plan],
            reasoning[:100],
        )
        return True


# ---------------------------------------------------------------------------
# Retry nudge
# ---------------------------------------------------------------------------

_RETRY_NUDGE = (
    "CRITICAL: Your previous response was missing the [ACTIONS] section. "
    "You MUST end your response with EXACTLY this format:\n\n"
    "[ACTIONS]\n"
    '{"plan": [{"action": "ACTION1"}, {"action": "ACTION3"}], "reasoning": "why"}\n\n'
    "Do NOT skip this section. Do NOT write actions to a file. "
    "Output the [ACTIONS] section directly in your response text RIGHT NOW."
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _action_label(action: GameAction) -> str:
    """Return a human-readable label like 'ACTION6(12,8)' for complex actions."""
    if action.is_complex() and hasattr(action.action_data, "x"):
        return f"{action.name}({action.action_data.x},{action.action_data.y})"
    return action.name


def _dict_to_action(action_dict: dict) -> GameAction:
    """Convert an action dict from the queue into a GameAction."""
    name = action_dict["name"]
    action = GameAction.from_name(name)
    if name == "ACTION6":
        data = action_dict.get("data", {})
        x = max(0, min(63, int(data.get("x", 0))))
        y = max(0, min(63, int(data.get("y", 0))))
        action.set_data({"x": x, "y": y})
    return action


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------

class ClaudeCodeAgent:
    MAX_ACTIONS = 200
    PLAN_SIZE = 5
    ANALYZER_RETRIES = 5

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
        self.total_deaths = 0
        self.analyzer_calls = 0

        # Grid state
        self.current_grid: list[list[int]] = []
        self.prev_image_grid: list[list[int]] | None = None
        self.prev_symbolic: dict | None = None
        self.probe_facts = ""

        # Action queue (RGB-Agent style)
        self._queue = ActionQueue()

        # Run log — written to disk so Claude Code can read it
        self.run_log = RunLog()
        self._run_log_path = ""  # set in run()

        # Claude Code analyzer — initialized in run()
        self._analyzer: ClaudeCodeAnalyzer | None = None

        # File handlers added by _setup_logging, cleaned up in _close
        self._log_handlers: list[logging.Handler] = []

    def _setup_logging(self) -> None:
        """Set up file-based logging into the run directory."""
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

        agents_logger = logging.getLogger("agents")
        agents_logger.setLevel(logging.INFO)

        # Console handler
        has_console = any(
            isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
            for h in agents_logger.handlers + logging.getLogger().handlers
        )
        if not has_console:
            import sys
            sh = logging.StreamHandler(sys.stdout)
            sh.setLevel(logging.INFO)
            sh.setFormatter(fmt)
            agents_logger.addHandler(sh)
            self._log_handlers.append(sh)

        # run.log file handler
        run_log_path = os.path.join(self._run_dir, "run.log")
        fh = logging.FileHandler(run_log_path, mode="w")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        fh.stream.reconfigure(line_buffering=True)
        agents_logger.addHandler(fh)
        self._log_handlers.append(fh)

        # vision.log file handler
        vision_log_path = os.path.join(self._run_dir, "vision.log")
        vl = logging.getLogger("agents.vision")
        vl.setLevel(logging.INFO)
        vl.propagate = False
        vfh = logging.FileHandler(vision_log_path, mode="w")
        vfh.setLevel(logging.INFO)
        vfh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        vfh.stream.reconfigure(line_buffering=True)
        vl.addHandler(vfh)
        self._log_handlers.append(vfh)

        # Silence third-party noise
        for name in ("httpx", "httpcore", "arc_agi", "urllib3"):
            logging.getLogger(name).setLevel(logging.WARNING)

    def run(self) -> None:
        timer = time.time()

        # Create per-run directory
        log_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._run_dir = os.path.join(log_dir, f"{self.game_id}_{ts}")
        os.makedirs(self._run_dir, exist_ok=True)
        self._vision_frames_dir = os.path.join(self._run_dir, "frames")
        os.makedirs(self._vision_frames_dir, exist_ok=True)

        self._setup_logging()

        # Set up run log path and flush initial content
        self._run_log_path = os.path.join(self._run_dir, "agent_log.txt")

        # Initialize Claude Code analyzer
        self._analyzer = ClaudeCodeAnalyzer(
            log_path=self._run_log_path,
            plan_size=self.PLAN_SIZE,
        )

        logger.info("=" * 60)
        logger.info(f"Claude Code Agent on {self.game_id}")
        logger.info(f"Run directory: {self._run_dir}")
        logger.info(f"Run log: {self._run_log_path}")
        logger.info("=" * 60)
        self.run_log.write_header(self.game_id, self.MAX_ACTIONS)

        # Initial reset
        frame = self._step(GameAction.RESET)
        self.current_grid = frame.frame[-1] if frame.frame else []

        if frame.state == GameState.WIN:
            self._flush_log()
            self._close()
            return

        # Auto-probe
        available = frame.available_actions or [1, 2, 3, 4]
        self.probe_facts = self._auto_probe(frame, available)
        logger.info(f"[probe] {self.probe_facts}")
        self.run_log.write_probe(self.probe_facts)
        self._flush_log()

        arc_state = frame.state
        arc_score = frame.levels_completed

        # Main game loop
        while self.action_counter < self.MAX_ACTIONS:
            if arc_state == GameState.WIN:
                logger.info("WIN!")
                break

            if arc_state == GameState.GAME_OVER:
                self.total_deaths += 1
                logger.info(f"GAME_OVER #{self.total_deaths}")
                self.run_log.write_event("DEATH", f"Death #{self.total_deaths}")
                self._flush_log()
                frame = self._step(GameAction.RESET)
                self.action_counter += 1
                self.current_grid = frame.frame[-1] if frame.frame else []
                self.prev_image_grid = None
                self.prev_symbolic = None
                arc_state = frame.state
                arc_score = frame.levels_completed
                self._queue.clear()
                continue

            # Get next action: drain queue or fire analyzer
            action_dict = None
            if self._queue and not self._queue.score_changed:
                action_dict = self._queue.pop()
                plan_info = f" | Plan {self._queue.plan_index}/{self._queue.plan_total}"
                logger.info(
                    "queue drain -> %s (%s, %d remaining)",
                    action_dict.get("name"), f"step {self._queue.plan_index}/{self._queue.plan_total}",
                    len(self._queue),
                )
            else:
                self._queue.score_changed = False

                # Fire analyzer — write current state to log first
                self._write_current_state_to_log(frame)
                self._flush_log()

                loaded = False
                for attempt in range(self.ANALYZER_RETRIES):
                    nudge = _RETRY_NUDGE if attempt > 0 else ""
                    logger.info(
                        "analyzer attempt %d/%d action=%d",
                        attempt + 1, self.ANALYZER_RETRIES, self.action_counter,
                    )
                    if self._fire_analyzer(self.action_counter, retry_nudge=nudge):
                        loaded = True
                        break
                    logger.warning(
                        "analyzer attempt %d/%d failed",
                        attempt + 1, self.ANALYZER_RETRIES,
                    )

                if not loaded:
                    logger.error("all analyzer attempts failed — ending run")
                    break

                if self._queue:
                    action_dict = self._queue.pop()
                    plan_info = f" | Plan {self._queue.plan_index}/{self._queue.plan_total}"
                else:
                    logger.error("queue still empty after analyzer — ending run")
                    break

            # Execute the action
            action = _dict_to_action(action_dict)
            grid_before = self.current_grid
            score_before = arc_score

            frame = self._step(action, reasoning=f"plan step {self._queue.plan_index}/{self._queue.plan_total}")
            self.action_counter += 1
            self.current_grid = frame.frame[-1] if frame.frame else grid_before

            arc_state = frame.state
            arc_score = frame.levels_completed

            # Compute changes
            changes = sum(
                1 for r in range(64) for c in range(64)
                if grid_before[r][c] != self.current_grid[r][c]
            )
            blocked = 0 < changes < 10

            label = _action_label(action)
            result_str = f"{label}: {'BLOCKED' if blocked else f'{changes} cells changed'}"

            # Log vision
            self._log_vision(label, grid_before, changes, blocked, frame)

            # Write turn to run log
            symbolic = grid_to_symbolic(self.current_grid)
            sym_changes = diff_symbolic(self.prev_symbolic, symbolic) if self.prev_symbolic else []
            self.prev_symbolic = symbolic

            sym_diff_text = ""
            if sym_changes:
                for c in sym_changes[:5]:
                    if c.get("type") == "changed":
                        parts = []
                        if "center" in c:
                            parts.append(f"center {c['center']['was']}->{c['center']['now']}")
                        if "size" in c:
                            parts.append(f"size {c['size']['was']}->{c['size']['now']}")
                        sym_diff_text += f"  {c.get('color', '?')}: {', '.join(parts)}\n"
                    elif c.get("type") == "background_size_changed":
                        sym_diff_text += f"  bg {c.get('color', '?')}: size {c['size']['was']}->{c['size']['now']}\n"
                    elif c.get("type") in ("appeared", "disappeared"):
                        sym_diff_text += f"  {c.get('color', '?')}: {c.get('type')}\n"

            # Build objects summary
            objects_text = ""
            for obj in symbolic.get("objects", []):
                center = obj.get("center", {})
                r, c_col = center.get("row", "?"), center.get("col", "?")
                color = obj.get("color", "?")
                size = obj.get("size", "?")
                shape = obj.get("shape", "")
                objects_text += f"  {color} (size {size}, {shape}) at ({r}, {c_col})\n"

            self.run_log.write_turn(
                turn=self.action_counter,
                action_count=self.action_counter,
                score=arc_score,
                mode="QUEUE",
                goal="",
                plan_step=f"{self._queue.plan_index}/{self._queue.plan_total}",
                observation=result_str,
                hypotheses="",
                actions_taken=[label],
                action_results=[result_str],
                interactions=[],
                notes=f"Changes: {changes} cells\nSymbolic Diff:\n{sym_diff_text}Objects:\n{objects_text}",
            )
            # Flush after every action so Claude Code sees updates
            self._flush_log()

            logger.info(
                "#%d %s score=%d",
                self.action_counter, result_str, arc_score,
            )

            # Score change detection
            self._queue.check_score(arc_score)

            # Level-up: re-probe
            if arc_score > score_before and arc_state not in (GameState.WIN, GameState.GAME_OVER):
                logger.info(f"[level-up] Level {arc_score}! Re-probing.")
                self.run_log.write_event("LEVEL_UP", f"Level {arc_score}")
                avail = frame.available_actions or [1, 2, 3, 4]
                self.probe_facts = self._auto_probe(frame, avail)
                logger.info(f"[probe] {self.probe_facts}")
                self.run_log.write_probe(self.probe_facts)
                self.prev_symbolic = None
                self.prev_image_grid = None
                self._queue.clear()
                self._flush_log()

            # Large change: flush queue and re-analyze
            if changes > 100 and not blocked:
                logger.info(f"[large-change] {changes} cells — flushing queue for re-analysis")
                self._queue.clear()
                self.prev_image_grid = grid_before

        elapsed = round(time.time() - timer, 2)
        logger.info("=" * 60)
        logger.info(
            f"FINISHED: actions={self.action_counter} state={arc_state.name} "
            f"score={arc_score} deaths={self.total_deaths} "
            f"analyzer_calls={self.analyzer_calls} time={elapsed}s"
        )
        logger.info("=" * 60)

        self._flush_log()
        self._close()

    # --- Analyzer integration ---

    def _fire_analyzer(self, action_num: int, retry_nudge: str = "") -> bool:
        """Call Claude Code to analyze the log and load action plan into queue."""
        self.analyzer_calls += 1

        raw = self._analyzer.analyze(action_num, retry_nudge=retry_nudge)
        if not raw:
            return False

        # Write analyzer exchange to run log
        self.run_log.write_event(
            "ANALYZER",
            f"Call #{self.analyzer_calls} at action {action_num} ({len(raw)} chars response)",
        )

        # Parse response: extract [PLAN] and [ACTIONS]
        hint = "\n".join(line.rstrip() for line in raw.split("\n"))

        actions_text = None
        if "\n[ACTIONS]\n" in hint:
            hint, actions_text = hint.split("\n[ACTIONS]\n", 1)
            actions_text = actions_text.strip()

        plan_text = ""
        if "\n[PLAN]\n" in hint:
            _, plan_text = hint.split("\n[PLAN]\n", 1)
            plan_text = plan_text.strip()

        if plan_text:
            self.run_log.write_event("PLAN", plan_text[:200])

        if actions_text:
            if self._queue.load(actions_text):
                logger.info(
                    "analyzer at action %d: loaded action plan (%d chars)",
                    action_num, len(actions_text),
                )
                self._flush_log()
                return True
            logger.warning(
                "analyzer at action %d: load rejected the plan",
                action_num,
            )
            return False

        # Fallback: scan the entire response for any JSON action plan
        logger.warning(
            "analyzer at action %d: no [ACTIONS] section, trying fallback JSON extraction",
            action_num,
        )
        # Look for {"plan": [...]} or [{"action": "..."}] anywhere in the response
        import json as _json
        for start_char in ("{", "["):
            idx = raw.find(start_char)
            while idx >= 0:
                try:
                    decoder = _json.JSONDecoder()
                    parsed, end = decoder.raw_decode(raw, idx)
                    # Check if it looks like an action plan
                    if isinstance(parsed, dict) and "plan" in parsed:
                        if self._queue.load(raw[idx:idx+end]):
                            logger.info("analyzer at action %d: recovered plan via fallback extraction", action_num)
                            self._flush_log()
                            return True
                    elif isinstance(parsed, list) and parsed and isinstance(parsed[0], dict) and "action" in parsed[0]:
                        if self._queue.load(raw[idx:idx+end]):
                            logger.info("analyzer at action %d: recovered plan via fallback extraction", action_num)
                            self._flush_log()
                            return True
                except (_json.JSONDecodeError, ValueError):
                    pass
                idx = raw.find(start_char, idx + 1)

        logger.warning("analyzer at action %d: no action plan found anywhere in response", action_num)
        return False

    def _write_current_state_to_log(self, frame: FrameData) -> None:
        """Write a snapshot of the current game state for the analyzer."""
        symbolic = grid_to_symbolic(self.current_grid)
        objects_text = ""
        for obj in symbolic.get("objects", []):
            center = obj.get("center", {})
            r, c = center.get("row", "?"), center.get("col", "?")
            color = obj.get("color", "?")
            size = obj.get("size", "?")
            shape = obj.get("shape", "")
            objects_text += f"  {color} (size {size}, {shape}) at ({r}, {c})\n"

        self.run_log.write_event(
            "STATE_SNAPSHOT",
            f"Score={frame.levels_completed} | "
            f"Actions={self.action_counter}/{self.MAX_ACTIONS} | "
            f"Deaths={self.total_deaths}\n"
            f"Objects:\n{objects_text}",
        )

    # --- Auto-probe (from v5) ---

    def _auto_probe(self, frame: FrameData, available: list[int]) -> str:
        """Take one of each action to discover what they do."""
        facts = []

        simple_actions = [a for a in available if a != 6]
        has_action6 = 6 in available

        for action_id in simple_actions:
            try:
                action = GameAction.from_id(action_id)
            except (ValueError, KeyError):
                continue

            fact = self._probe_single_action(action)
            facts.append(fact)

            if self.frames and self.frames[-1].state == GameState.GAME_OVER:
                self.total_deaths += 1
                facts.append("  (caused GAME_OVER)")
                reset_frame = self._step(GameAction.RESET)
                self.action_counter += 1
                self.current_grid = reset_frame.frame[-1] if reset_frame.frame else []

        if has_action6:
            action6_facts = self._probe_action6()
            facts.extend(action6_facts)

        return "\n".join(facts)

    def _probe_single_action(self, action: GameAction) -> str:
        """Probe a single simple action and return a fact string."""
        grid_before = self.current_grid
        sym_before = grid_to_symbolic(grid_before)

        frame_result = self._step(action)
        self.action_counter += 1
        self.current_grid = frame_result.frame[-1] if frame_result.frame else grid_before

        sym_after = grid_to_symbolic(self.current_grid)
        sym_changes = diff_symbolic(sym_before, sym_after)

        changes = sum(
            1 for r in range(64) for c in range(64)
            if grid_before[r][c] != self.current_grid[r][c]
        )
        blocked = 0 < changes < 10

        change_summary = self._format_sym_changes(sym_changes)

        status = "BLOCKED" if blocked else f"{changes} cells changed"
        fact = f"{action.name}: {status}"
        if change_summary:
            fact += "\n  " + "\n  ".join(change_summary)

        return fact

    def _probe_action6(self) -> list[str]:
        """Probe ACTION6 at diverse coordinate targets."""
        facts = []
        symbolic = grid_to_symbolic(self.current_grid)
        fg_objects = symbolic.get("objects", [])

        targets: list[tuple[int, int, str]] = []

        # Target 1: center of the largest non-background object
        if fg_objects:
            largest = max(fg_objects, key=lambda o: o.get("size", 0))
            cr, cc = largest["center"]["row"], largest["center"]["col"]
            targets.append((cc, cr, f"largest object ({largest['color']}, size={largest['size']})"))

        # Target 2: center of a different-colored object
        if len(fg_objects) >= 2:
            first_color = fg_objects[0].get("color_id")
            for obj in fg_objects[1:]:
                if obj.get("color_id") != first_color:
                    cr, cc = obj["center"]["row"], obj["center"]["col"]
                    targets.append((cc, cr, f"different-colored object ({obj['color']}, size={obj['size']})"))
                    break

        # Target 3: background cell
        bg_info = symbolic.get("backgrounds", [])
        if bg_info:
            bg_color_name = bg_info[0]["color"]
            from .symbolic import COLOR_NAMES
            for dr in range(0, 32):
                found = False
                for r, c in [(32 + dr, 32), (32 - dr, 32), (32, 32 + dr), (32, 32 - dr)]:
                    if 0 <= r < 64 and 0 <= c < 64:
                        cell_color = self.current_grid[r][c]
                        if COLOR_NAMES.get(cell_color) == bg_color_name:
                            targets.append((c, r, f"background cell ({bg_color_name})"))
                            found = True
                            break
                if found:
                    break

        # Target 4: grid corner
        targets.append((0, 0, "grid corner (0,0)"))

        # Target 5: grid center
        targets.append((32, 32, "grid center"))

        # Deduplicate close targets
        unique_targets: list[tuple[int, int, str]] = []
        for x, y, desc in targets:
            too_close = False
            for ux, uy, _ in unique_targets:
                if abs(x - ux) + abs(y - uy) < 3:
                    too_close = True
                    break
            if not too_close:
                unique_targets.append((x, y, desc))

        for x, y, desc in unique_targets[:5]:
            grid_before = self.current_grid
            sym_before = grid_to_symbolic(grid_before)

            action = GameAction.from_id(6)
            action.set_data({"x": int(x), "y": int(y)})

            frame_result = self._step(action)
            self.action_counter += 1
            self.current_grid = frame_result.frame[-1] if frame_result.frame else grid_before

            sym_after = grid_to_symbolic(self.current_grid)
            sym_changes = diff_symbolic(sym_before, sym_after)

            changes = sum(
                1 for r in range(64) for c in range(64)
                if grid_before[r][c] != self.current_grid[r][c]
            )
            blocked = 0 < changes < 10

            change_summary = self._format_sym_changes(sym_changes)

            status = "BLOCKED" if blocked else f"{changes} cells changed"
            fact = f"ACTION6 at ({x},{y}) [{desc}]: {status}"
            if change_summary:
                fact += "\n  " + "\n  ".join(change_summary)

            facts.append(fact)

            if frame_result.state == GameState.GAME_OVER:
                self.total_deaths += 1
                facts.append("  (caused GAME_OVER)")
                reset_frame = self._step(GameAction.RESET)
                self.action_counter += 1
                self.current_grid = reset_frame.frame[-1] if reset_frame.frame else []

        return facts

    def _format_sym_changes(self, sym_changes: list[dict]) -> list[str]:
        """Format symbolic changes into human-readable summary lines."""
        change_summary = []
        for c in sym_changes[:5]:
            if c.get("type") == "changed":
                parts = []
                if "center" in c:
                    parts.append(f"center {c['center']['was']}->{c['center']['now']}")
                if "size" in c:
                    parts.append(f"size {c['size']['was']}->{c['size']['now']}")
                change_summary.append(f"{c.get('color', '?')}: {', '.join(parts)}")
            elif c.get("type") == "background_size_changed":
                change_summary.append(
                    f"background {c.get('color', '?')}: size {c['size']['was']}->{c['size']['now']}"
                )
        return change_summary

    # --- Vision logging (from v5) ---

    def _log_vision(
        self,
        action_label: str,
        grid_before: list[list[int]],
        changes: int,
        blocked: bool,
        frame: FrameData,
    ) -> None:
        """Log a structured description of what the agent sees after an action."""
        symbolic = grid_to_symbolic(self.current_grid)
        sym_changes = diff_symbolic(
            grid_to_symbolic(grid_before) if changes > 0 else None,
            symbolic,
        )

        lines = [
            f"=== ACTION #{self.action_counter}: {action_label} ===",
            f"Result: {'BLOCKED' if blocked else f'{changes} cells changed'}",
            f"Score: {frame.levels_completed} | State: {frame.state.name} | "
            f"Actions used: {self.action_counter}/{self.MAX_ACTIONS}",
        ]

        objects = symbolic.get("objects", [])
        if objects:
            lines.append("")
            lines.append("VISIBLE OBJECTS:")
            for obj in objects:
                center = obj.get("center", {})
                r, c = center.get("row", "?"), center.get("col", "?")
                color = obj.get("color", "?")
                size = obj.get("size", "?")
                shape = obj.get("shape", "")
                shape_str = f", {shape}" if shape else ""
                lines.append(f"  - {color} (size {size}{shape_str}) at ({r}, {c})")

        if sym_changes:
            lines.append("")
            lines.append("CHANGES:")
            for sc in sym_changes[:10]:
                ctype = sc.get("type", "?")
                color = sc.get("color", "?")
                if ctype == "changed":
                    parts = []
                    if "center" in sc:
                        parts.append(f"center {sc['center']['was']}->{sc['center']['now']}")
                    if "size" in sc:
                        parts.append(f"size {sc['size']['was']}->{sc['size']['now']}")
                    lines.append(f"  - {color} {ctype}: {', '.join(parts)}")
                elif ctype == "background_size_changed":
                    lines.append(f"  - bg {color}: size {sc['size']['was']}->{sc['size']['now']}")
                else:
                    lines.append(f"  - {color}: {ctype}")

        lines.append("")
        vision_logger.info("\n".join(lines))

        # Save frame image
        if changes > 0 and self.prev_image_grid is not None:
            img = side_by_side(grid_before, self.current_grid)
        else:
            img = grid_to_image(self.current_grid)

        if hasattr(self, "_vision_frames_dir"):
            safe_label = action_label.replace("(", "_").replace(")", "").replace(",", "_")
            filename = f"action_{self.action_counter:03d}_{safe_label}.png"
            filepath = os.path.join(self._vision_frames_dir, filename)
            img.save(filepath, format="PNG", optimize=True)

        self.prev_image_grid = self.current_grid

    # --- Log flushing ---

    def _flush_log(self) -> None:
        """Flush the run log to disk so Claude Code can read it."""
        if self._run_log_path:
            self.run_log.flush_to_disk(self._run_log_path)

    # --- Environment step ---

    def _step(self, action: GameAction, reasoning: str = "") -> FrameData:
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
            game_id=getattr(raw, "game_id", self.game_id),
            frame=[arr.tolist() if hasattr(arr, "tolist") else arr for arr in raw.frame],
            state=raw.state,
            levels_completed=raw.levels_completed,
            win_levels=getattr(raw, "win_levels", 0),
            guid=getattr(raw, "guid", ""),
            full_reset=getattr(raw, "full_reset", False),
            available_actions=raw.available_actions,
        )
        self.frames.append(frame)
        return frame

    # --- Cleanup ---

    def _close(self) -> None:
        """Clean up resources."""
        # Remove file handlers
        agents_logger = logging.getLogger("agents")
        vision_log = logging.getLogger("agents.vision")
        for h in self._log_handlers:
            agents_logger.removeHandler(h)
            vision_log.removeHandler(h)
            h.close()
        self._log_handlers.clear()

        if not self.scorecard_id or not self.arcade:
            return
        scorecard = self.arcade.close_scorecard(self.scorecard_id)
        if scorecard:
            logger.info("--- SCORECARD ---")
            logger.info(json.dumps(scorecard.model_dump(), indent=2))
        self.scorecard_id = None
