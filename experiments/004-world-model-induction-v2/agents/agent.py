"""Observe-act agent with auto-probe and structured hypotheses.

Same core as experiment 003 (GPT-5.4, symbolic state, symbolic diff, image).
Additions:
- Auto-probe: code takes one of each action on startup, identifies what changed
  operationally, and feeds this as established facts to the model
- Structured hypothesis: model outputs testable claims, not vague theories
- Probe results persist in notes so the model never has to re-discover primitives
"""

import json
import logging
import os
import time

from arcengine import FrameData, GameAction, GameState
from openai import OpenAI

from .symbolic import grid_to_symbolic, diff_symbolic
from .vision import grid_b64, diff_b64, input_text, input_image_b64

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are playing an unknown turn-based game on a 64x64 grid. Each cell is an integer 0-15 (colors).
Your score (levels_completed) increases when you complete a level. GAME_OVER means you failed.

You have actions (ACTION1-ACTION4+). Each maps to a keyboard/mouse input.
Not all actions may be available. Complex actions may require x,y coordinates.

Each turn you receive:
- Established facts from initial probing (what each action does)
- Symbolic state of objects on the grid
- What changed since the last action
- An image (side-by-side with previous frame, red = changed cells)
- Your own notes from previous turns

Output JSON:
- observation: what changed and what it means
- hypotheses: MULTIPLE testable claims. Each has a status and optionally test_actions.
  Maintain hypotheses about: what each object does, what the goal is, how game mechanics work.
  Example: [
    {"claim": "...", "status": "testing", "test_actions": ["ACTION3","ACTION3"], "evidence": "..."},
    {"claim": "...", "status": "confirmed", "evidence": "..."},
    {"claim": "...", "status": "untested"}
  ]
  The first hypothesis with status "testing" and test_actions will be executed.
- notes: anything to remember (gets carried to next turn)

The test_actions sequence executes automatically, stopping on BLOCKED or unexpected events.

IMPORTANT:
- When something unexpected happens (large change, object appears/disappears), INVESTIGATE.
  Look at the symbolic diff carefully. What objects changed? What appeared? What vanished?
  Formulate a hypothesis about WHY and test it.
- Don't just navigate. The game likely has mechanics beyond movement — objects may have
  functions you need to discover through interaction.
- If a hypothesis hasn't led to progress after several tests, REJECT it and try something new.
- You have limited actions. Don't re-explore areas you've already mapped."""


class ToolUseAgent:
    MAX_ACTIONS = 100

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
        self.prev_image_grid: list[list[int]] | None = None
        self.prev_symbolic: dict | None = None
        self.hypothesis = ""
        self.notes = ""
        self.probe_facts = ""  # established facts from auto-probe, never changes
        self.recent_actions: list[str] = []
        self.llm_calls = 0

    def run(self) -> None:
        timer = time.time()
        logger.info("=" * 60)
        logger.info(f"Probe+Observe Agent on {self.game_id}")
        logger.info("=" * 60)

        frame = self._step(GameAction.RESET)
        self.current_grid = frame.frame[-1] if frame.frame else []

        if frame.state == GameState.WIN:
            self._close()
            return

        # Auto-probe: take one of each available action, record what changed
        available = frame.available_actions or [1, 2, 3, 4]
        self.probe_facts = self._auto_probe(frame, available)
        logger.info(f"[probe] {self.probe_facts}")

        # Main loop
        while self.action_counter < self.MAX_ACTIONS:
            if frame.state == GameState.WIN:
                logger.info("WIN!")
                break

            if frame.state == GameState.GAME_OVER:
                self.total_deaths += 1
                logger.info(f"DIED #{self.total_deaths}")
                self.recent_actions.append("DIED")
                frame = self._step(GameAction.RESET)
                self.action_counter += 1
                self.current_grid = frame.frame[-1] if frame.frame else []
                self.prev_image_grid = None
                self.prev_symbolic = None
                continue

            actions = self._think_and_act(frame)

            # Execute the action sequence, stopping on events
            for action in actions:
                if self.action_counter >= self.MAX_ACTIONS:
                    break

                grid_before = self.current_grid
                score_before = frame.levels_completed
                frame = self._step(action, reasoning=self.hypothesis[:80])
                self.action_counter += 1
                self.current_grid = frame.frame[-1] if frame.frame else grid_before

                changes = sum(1 for r in range(64) for c in range(64)
                              if grid_before[r][c] != self.current_grid[r][c])
                blocked = 0 < changes < 10

                result = f"{action.name}: {'BLOCKED' if blocked else f'{changes} cells changed'}"

                # For unusual changes, include detailed symbolic diff
                if changes > 80 and not blocked:
                    sym_after = grid_to_symbolic(self.current_grid)
                    detail_changes = diff_symbolic(
                        grid_to_symbolic(grid_before) if grid_before else None,
                        sym_after
                    )
                    if detail_changes:
                        details = []
                        for c in detail_changes[:6]:
                            if c.get("type") == "changed":
                                parts = [c.get("color", "?")]
                                if "center" in c:
                                    parts.append(f"center {c['center']['was']}→{c['center']['now']}")
                                if "size" in c:
                                    parts.append(f"size {c['size']['was']}→{c['size']['now']}")
                                details.append(" ".join(parts))
                            elif c.get("type") in ("appeared", "disappeared"):
                                details.append(f"{c.get('color','?')} {c['type']} at {c.get('at', c.get('was_at', '?'))}")
                            elif c.get("type") == "background_size_changed":
                                details.append(f"bg {c.get('color','?')} size {c['size']['was']}→{c['size']['now']}")
                        result += " | UNUSUAL: " + "; ".join(details)

                self.recent_actions.append(result)
                if len(self.recent_actions) > 15:
                    self.recent_actions = self.recent_actions[-15:]

                logger.info(f"#{self.action_counter} {result} score={frame.levels_completed}")

                # Stop sequence on significant events
                if frame.state in (GameState.WIN, GameState.GAME_OVER):
                    break
                if blocked:
                    break
                if frame.levels_completed > score_before:
                    self.recent_actions.append(f"*** SCORE: {frame.levels_completed} ***")
                    self.prev_image_grid = None
                    break
                if changes > 100:
                    self.prev_image_grid = None
                    break  # stop and re-think on large changes

        elapsed = round(time.time() - timer, 2)
        logger.info("=" * 60)
        logger.info(
            f"FINISHED: actions={self.action_counter} state={frame.state.name} "
            f"score={frame.levels_completed} deaths={self.total_deaths} "
            f"llm_calls={self.llm_calls} time={elapsed}s"
        )
        logger.info(f"Hypothesis: {self.hypothesis}")
        logger.info(f"Notes: {self.notes}")
        logger.info("=" * 60)
        self._close()

    def _auto_probe(self, frame: FrameData, available: list[int]) -> str:
        """Take one of each action. Record what changed. Return as text facts."""
        facts = []

        for action_id in available:
            try:
                action = GameAction.from_id(action_id)
            except (ValueError, KeyError):
                continue

            grid_before = self.current_grid
            sym_before = grid_to_symbolic(grid_before)

            frame_result = self._step(action)
            self.action_counter += 1
            self.current_grid = frame_result.frame[-1] if frame_result.frame else grid_before

            sym_after = grid_to_symbolic(self.current_grid)
            sym_changes = diff_symbolic(sym_before, sym_after)

            changes = sum(1 for r in range(64) for c in range(64)
                          if grid_before[r][c] != self.current_grid[r][c])
            blocked = 0 < changes < 10

            # Summarize what changed
            change_summary = []
            for c in sym_changes[:5]:
                if c.get("type") == "changed":
                    parts = []
                    if "center" in c:
                        parts.append(f"center {c['center']['was']}→{c['center']['now']}")
                    if "size" in c:
                        parts.append(f"size {c['size']['was']}→{c['size']['now']}")
                    change_summary.append(f"{c.get('color','?')}: {', '.join(parts)}")
                elif c.get("type") == "background_size_changed":
                    change_summary.append(f"background {c.get('color','?')}: size {c['size']['was']}→{c['size']['now']}")

            status = "BLOCKED" if blocked else f"{changes} cells changed"
            fact = f"{action.name}: {status}"
            if change_summary:
                fact += "\n  " + "\n  ".join(change_summary)

            facts.append(fact)
            self.recent_actions.append(f"{action.name}: {status}")

            if frame_result.state == GameState.GAME_OVER:
                self.total_deaths += 1
                facts.append(f"  (caused GAME_OVER)")
                frame_result = self._step(GameAction.RESET)
                self.action_counter += 1
                self.current_grid = frame_result.frame[-1] if frame_result.frame else []

        return "\n".join(facts)

    def _think_and_act(self, frame: FrameData) -> list[GameAction]:
        self.llm_calls += 1

        available = frame.available_actions or [1, 2, 3, 4]
        avail_str = ", ".join(f"ACTION{i}" for i in available)
        recent = "\n".join(self.recent_actions[-10:]) if self.recent_actions else "(none)"

        symbolic = grid_to_symbolic(self.current_grid)
        sym_changes = diff_symbolic(self.prev_symbolic, symbolic) if self.prev_symbolic else []
        self.prev_symbolic = symbolic

        changes_text = ""
        if sym_changes:
            changes_text = f"CHANGES SINCE LAST ACTION:\n{json.dumps(sym_changes, indent=1)}\n\n"

        content = [
            input_text(SYSTEM_PROMPT),
            input_text(
                f"\nScore: {frame.levels_completed} | Deaths: {self.total_deaths} | "
                f"Actions: {self.action_counter}/{self.MAX_ACTIONS}\n"
                f"Available: {avail_str}\n\n"
                f"ESTABLISHED FACTS (from initial probing):\n{self.probe_facts}\n\n"
                + (f"YOUR NOTES:\n{self.notes}\n\n" if self.notes else "")
                + (f"CURRENT HYPOTHESES:\n{self.hypothesis}\n\n" if self.hypothesis else "")
                + f"RECENT:\n{recent}\n\n"
                + changes_text
                + f"OBJECTS:\n{json.dumps(symbolic.get('objects', []), indent=1)}\n"
            ),
        ]

        if self.prev_image_grid:
            content.append(input_text("Side-by-side (PREVIOUS vs CURRENT, red = changed):"))
            content.append(input_image_b64(diff_b64(self.prev_image_grid, self.current_grid)))
        else:
            content.append(input_text("Current frame:"))
            content.append(input_image_b64(grid_b64(self.current_grid)))

        self.prev_image_grid = self.current_grid

        content.append(input_text(
            "\nRespond with JSON:\n"
            '{"observation": "what changed and what it means",\n'
            ' "hypotheses": [\n'
            '   {"claim": "...", "status": "testing", "test_actions": ["ACTION3","ACTION3","ACTION3"], "evidence": "..."},\n'
            '   {"claim": "...", "status": "confirmed", "evidence": "..."}\n'
            ' ],\n'
            ' "notes": "persistent notes for next turn"}'
        ))

        try:
            response = self.client.responses.create(
                model="gpt-5.4",
                input=[{"role": "user", "content": content}],
                max_output_tokens=1500,
                temperature=0.2,
            )
            raw = response.output_text or "{}"
        except Exception as e:
            logger.warning(f"API error: {e}")
            time.sleep(2)
            return GameAction.ACTION1

        data = self._parse_json(raw)

        obs = data.get("observation", "")
        hypotheses = data.get("hypotheses", [])
        notes = data.get("notes", "")

        # Format hypotheses for next turn and extract test_actions from active one
        raw_actions = []
        active_claim = ""
        if hypotheses:
            hyp_lines = []
            for h in hypotheses:
                if isinstance(h, dict):
                    status = h.get("status", "?")
                    claim = h.get("claim", "")
                    evidence = h.get("evidence", "")
                    hyp_lines.append(f"[{status}] {claim} ({evidence})")

                    # Extract test_actions from the hypothesis being tested
                    if h.get("test_actions") and status in ("testing", "untested"):
                        raw_actions = h["test_actions"]
                        active_claim = claim
                else:
                    hyp_lines.append(str(h))
            self.hypothesis = "\n".join(hyp_lines)

        # Fallback: check for "actions" or "action" field
        if not raw_actions:
            raw_actions = data.get("actions", [])
        if not raw_actions:
            single = data.get("action", "")
            if single:
                raw_actions = [single]

        if notes:
            self.notes = notes

        logger.info(f"[observe] {obs[:200]}")
        if active_claim:
            logger.info(f"[testing] {active_claim}")
        if hypotheses:
            logger.info(f"[hypotheses] {len(hypotheses)} claims")

        # Resolve actions
        actions = []
        for action_name in raw_actions:
            if not isinstance(action_name, str):
                continue
            try:
                actions.append(GameAction.from_name(action_name))
            except (ValueError, KeyError):
                continue

        if not actions:
            actions = [GameAction.ACTION1]

        logger.info(f"[actions] {[a.name for a in actions]}")
        return actions

    def _parse_json(self, raw: str) -> dict:
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            pass
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
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except (json.JSONDecodeError, ValueError):
                pass
        return {}

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
