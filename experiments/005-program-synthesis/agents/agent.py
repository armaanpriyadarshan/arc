"""Observe-act agent with auto-probe, structured hypotheses, and program synthesis.

Same core as experiment 004 (GPT-5.4, symbolic state, symbolic diff, image).
Addition:
- Program synthesis: the model can write and execute Python code to analyze the
  grid however it wants. Variables persist across calls so the model can build
  up maps, tracking structures, pathfinding, etc. over the course of a game.
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

SYSTEM_PROMPT = """You are playing an unknown turn-based game on a 64x64 grid. Each cell is an integer 0-15 (colors).
Your score (levels_completed) increases when you complete a level. GAME_OVER means you failed.

You have actions (ACTION1-ACTION4+). Each maps to a keyboard/mouse input.
Not all actions may be available. Complex actions may require x,y coordinates.

Each turn you receive:
- Established facts from initial probing (what each action does)
- Symbolic state of objects on the grid
- What changed since your last plan
- An image (side-by-side with previous frame, red = changed cells)
- Your own notes from previous turns
- Results of your last action plan (every action and its outcome)
- Output from any code you ran last turn

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
- verified_rules: list of UNIVERSAL game rules you've confirmed (persist across levels, max 10).
  ONLY include rules about game MECHANICS — NOT positions, corridors, or level-specific layouts.
- notes: anything to remember (gets carried to next turn)
- code: optional Python code to execute. You have access to `grid` (64x64 list[list[int]]),
  `ROWS` (64), `COLS` (64). Set `result` to return data.
  All Python builtins available. Modules: collections, math, numpy (as np), json.
  Variables persist between turns — define functions and data structures that accumulate.

The test_actions sequence executes automatically, stopping on BLOCKED or unexpected events.

IMPORTANT:
- You can OBSERVE. After interacting with something, study the before/after image
  and the symbolic diff to understand EXACTLY what changed. What appeared, disappeared,
  or moved? Build a causal model: "doing X causes Y to happen."
- Be HUNGRY for understanding. Every object exists for a reason. When you find
  something interactive, figure out its FULL effect — not just "something changed"
  but specifically what changed, where, and how you can use that to make progress.
- Once you understand a mechanic, EXPLOIT it — don't re-test. Act on what you know.
- ACTIVELY FALSIFY your hypotheses. If you test a claim and the result contradicts it,
  mark it "rejected" immediately and try something completely different. Don't rationalize
  failures or keep testing minor variations of a broken theory.
- You have limited actions. Don't re-explore areas you've already mapped.
- When you complete a level, the layout changes but game MECHANICS carry over.
- If you're stuck (repeated BLOCKED), you're probably missing a game mechanic, not a path.
  Look for objects you haven't interacted with yet.
- You can write Python code to analyze the grid. Use this to build whatever representations
  help you — maps, path analysis, object tracking, etc. The code runs with the current
  grid state and your variables carry to the next turn."""


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
        self.probe_facts = ""
        self.verified_rules: list[str] = []
        self.max_rules = 10
        self.recent_actions: list[str] = []
        self.consecutive_blocked = 0
        self.llm_calls = 0

        # Program synthesis
        self.sandbox = Sandbox()
        self.program_output: str = ""

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
                self.consecutive_blocked = 0
                frame = self._step(GameAction.RESET)
                self.action_counter += 1
                self.current_grid = frame.frame[-1] if frame.frame else []
                self.prev_image_grid = None
                self.prev_symbolic = None
                continue

            # Circuit breaker: if stuck, inject warning
            if self.consecutive_blocked >= 5:
                self.recent_actions.append(
                    f"*** STUCK: {self.consecutive_blocked} consecutive BLOCKED actions. "
                    f"You MUST try a completely different direction or area. ***"
                )
                logger.info(f"[stuck] {self.consecutive_blocked} consecutive blocked")
                self.consecutive_blocked = 0
                self.prev_image_grid = None

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

                if blocked:
                    self.consecutive_blocked += 1
                else:
                    self.consecutive_blocked = 0

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
                    logger.info(f"[level-up] Level {frame.levels_completed}! Re-probing.")
                    self._promote_confirmed_hypotheses()
                    avail = frame.available_actions or [1, 2, 3, 4]
                    self.probe_facts = self._auto_probe(frame, avail)
                    logger.info(f"[probe] {self.probe_facts}")
                    self.hypothesis = ""
                    self.notes = ""
                    self.consecutive_blocked = 0
                    self.prev_symbolic = None
                    self.prev_image_grid = None
                    self.recent_actions = [f"*** LEVEL {frame.levels_completed} ***"]
                    self.program_output = ""
                    if self.verified_rules:
                        logger.info(f"[rules] {self.verified_rules}")
                    break
                if changes > 100:
                    self.prev_image_grid = grid_before
                    break

        elapsed = round(time.time() - timer, 2)
        logger.info("=" * 60)
        logger.info(
            f"FINISHED: actions={self.action_counter} state={frame.state.name} "
            f"score={frame.levels_completed} deaths={self.total_deaths} "
            f"llm_calls={self.llm_calls} time={elapsed}s"
        )
        logger.info(f"Hypothesis: {self.hypothesis}")
        logger.info(f"Notes: {self.notes}")
        if self.verified_rules:
            logger.info(f"Verified rules: {self.verified_rules}")
        logger.info("=" * 60)
        self._close()

    def _add_rule(self, rule: str) -> None:
        import re
        if re.search(r'\[\d+,\s*\d+\]', rule) or re.search(r'x[≈=]\d+', rule) or re.search(r'y[≈=]\d+', rule):
            return
        if re.search(r'at \(\d+', rule) or re.search(r'from the current', rule, re.IGNORECASE):
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

    def _promote_confirmed_hypotheses(self) -> None:
        if not self.hypothesis:
            return
        for line in self.hypothesis.split("\n"):
            if line.startswith("[confirmed]"):
                rule = line.replace("[confirmed]", "").strip()
                if "(" in rule:
                    rule = rule[:rule.rfind("(")].strip()
                if rule:
                    self._add_rule(rule)

    def _auto_probe(self, frame: FrameData, available: list[int]) -> str:
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
                f"ESTABLISHED FACTS (from probing this level):\n{self.probe_facts}\n\n"
                + (f"VERIFIED RULES (confirmed across levels — these are ground truth):\n"
                   + "\n".join(f"- {r}" for r in self.verified_rules) + "\n\n"
                   if self.verified_rules else "")
                + (f"YOUR NOTES:\n{self.notes}\n\n" if self.notes else "")
                + (f"CURRENT HYPOTHESES:\n{self.hypothesis}\n\n" if self.hypothesis else "")
                + (f"PROGRAM OUTPUT (from your last code):\n{self.program_output}\n\n"
                   if self.program_output else "")
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
            ' "verified_rules": ["universal game mechanic you confirmed"],\n'
            ' "notes": "persistent notes for next turn",\n'
            ' "code": "optional Python code to analyze the grid. You have `grid` (64x64 list[list[int]]), '
            'ROWS, COLS. Set `result` to return data. '
            'All Python builtins available. Imports: collections, math, numpy (as np), json. '
            'Variables persist between turns — build up data structures over time."}'
        ))

        try:
            response = self.client.responses.create(
                model="gpt-5.4",
                input=[{"role": "user", "content": content}],
                max_output_tokens=4000,
                temperature=0.2,
            )
            raw = response.output_text or "{}"
        except Exception as e:
            logger.warning(f"API error: {e}")
            time.sleep(2)
            return [GameAction.ACTION1]

        data = self._parse_json(raw)

        obs = data.get("observation", "")
        hypotheses = data.get("hypotheses", [])
        notes = data.get("notes", "")

        # Execute code if provided
        code = data.get("code", "")
        if code and isinstance(code, str) and code.strip():
            logger.info(f"[code] executing {len(code)} chars")
            self.program_output = self.sandbox.run(code, self.current_grid)
            logger.info(f"[code-result] {self.program_output[:300]}")
        else:
            self.program_output = ""

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

                    if h.get("test_actions") and status in ("testing", "untested"):
                        raw_actions = h["test_actions"]
                        active_claim = claim
                else:
                    hyp_lines.append(str(h))
            self.hypothesis = "\n".join(hyp_lines)

        if not raw_actions:
            raw_actions = data.get("actions", [])
        if not raw_actions:
            single = data.get("action", "")
            if single:
                raw_actions = [single]

        # Extract verified rules
        new_rules = data.get("verified_rules", [])
        if new_rules and isinstance(new_rules, list):
            for rule in new_rules:
                if isinstance(rule, str) and rule:
                    self._add_rule(rule)

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
