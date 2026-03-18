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
- verified_rules: list of UNIVERSAL game rules you've confirmed (persist across levels, max 10).
  ONLY include rules about game MECHANICS — NOT positions, corridors, or level-specific layouts.
  BAD: "The white object at [31,21] is a switch" (position-specific)
  BAD: "Moving right is blocked from x=40" (level-specific)
  GOOD: "Touching white objects triggers remote reconfiguration of other objects"
  GOOD: "When a trigger is activated, some colored objects MOVE/ROTATE to new positions"
- falsified: list of approaches/beliefs that turned out to be WRONG. These persist for this level
  so you don't repeat the same mistakes. Be specific about what failed and why.
- cause_effect: (include when a large change just happened) describe what you did, what changed
  remotely, and your best theory for the causal relationship. This helps you build a mental model.
- notes: anything to remember (gets carried to next turn)

The test_actions sequence executes automatically, stopping on BLOCKED or unexpected events.

IMPORTANT:
- When something unexpected happens (large change, object appears/disappears), INVESTIGATE.
  Look at the symbolic diff carefully. What objects changed? What appeared? What vanished?
  Formulate a hypothesis about WHY and test it.
- When a large change happens, you'll see a before/after image. Study it carefully —
  what objects moved? What appeared or disappeared? What does this tell you about the mechanic?
- Don't just navigate. The game likely has mechanics beyond movement — objects may have
  functions you need to discover through interaction.
- If a hypothesis hasn't led to progress after several tests, REJECT it and try something new.
- You have limited actions. Don't re-explore areas you've already mapped.
- When you complete a level, the layout changes but game MECHANICS carry over. Look for the
  same types of interactions (switches, gates, collectibles) in the new layout.
- If you're stuck (repeated BLOCKED), you're probably missing a game mechanic, not a path.
  Look for objects you haven't interacted with yet."""


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
        self.probe_facts = ""  # established facts from auto-probe, refreshed on level change
        self.verified_rules: list[str] = []  # universal game rules, persist across levels
        self.max_rules = 10  # cap to prevent token bloat
        self.falsified: list[str] = []  # approaches that failed, per-level
        self.blocked_tracker: dict[str, int] = {}  # "(region, action)" → count
        self.dead_ends: list[str] = []  # auto-detected from blocked_tracker
        self.recent_actions: list[str] = []
        self.llm_calls = 0
        self._large_change_pending = False  # flag for cause-effect prompt

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

                # Track blocked directions by region
                if blocked:
                    region = self._get_region()
                    key = f"{region}:{action.name}"
                    self.blocked_tracker[key] = self.blocked_tracker.get(key, 0) + 1
                    if self.blocked_tracker[key] >= 3:
                        dead_end = f"{action.name} from region {region}"
                        if dead_end not in self.dead_ends:
                            self.dead_ends.append(dead_end)
                            logger.info(f"[dead-end] {dead_end} (blocked {self.blocked_tracker[key]}x)")
                elif changes > 100:
                    # Large change may have altered the map — clear blocked tracker
                    self.blocked_tracker.clear()
                    self.dead_ends.clear()
                    self._large_change_pending = True

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
                    logger.info(f"[level-up] Level {frame.levels_completed}! Re-probing new layout.")
                    # Promote any confirmed hypotheses to verified rules
                    self._promote_confirmed_hypotheses()
                    # Re-probe new level layout
                    avail = frame.available_actions or [1, 2, 3, 4]
                    self.probe_facts = self._auto_probe(frame, avail)
                    logger.info(f"[probe] {self.probe_facts}")
                    # Clear per-level state but keep verified_rules
                    self.hypothesis = ""
                    self.notes = ""
                    self.falsified = []
                    self.blocked_tracker.clear()
                    self.dead_ends.clear()
                    self._large_change_pending = False
                    self.prev_symbolic = None
                    self.prev_image_grid = None
                    self.recent_actions = [f"*** LEVEL {frame.levels_completed} ***"]
                    if self.verified_rules:
                        logger.info(f"[rules] {self.verified_rules}")
                    break
                if changes > 100:
                    # Keep prev_image_grid so next LLM call shows before/after diff
                    self.prev_image_grid = grid_before
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
        if self.verified_rules:
            logger.info(f"Verified rules: {self.verified_rules}")
        logger.info("=" * 60)
        self._close()

    def _get_region(self) -> str:
        """Get approximate player region from current grid's symbolic state."""
        sym = grid_to_symbolic(self.current_grid)
        # Find the smallest non-background object (likely the player)
        objects = sym.get("objects", [])
        moving = [o for o in objects if o.get("size", 9999) < 50]
        if moving:
            moving.sort(key=lambda o: o.get("size", 9999))
            c = moving[0].get("center", [32, 32])
            # Bucket to 10x10 regions
            return f"({c[0]//10*10},{c[1]//10*10})"
        return "(?,?)"

    def _add_rule(self, rule: str) -> None:
        """Add a rule with deduplication and position filtering."""
        import re
        # Reject rules with specific coordinates — those are level-specific, not universal
        if re.search(r'\[\d+,\s*\d+\]', rule) or re.search(r'x[≈=]\d+', rule) or re.search(r'y[≈=]\d+', rule):
            return
        # Reject rules about specific positions/lanes
        if re.search(r'at \(\d+', rule) or re.search(r'from the current', rule, re.IGNORECASE):
            return
        # Reject if too similar to an existing rule (substring match)
        rule_lower = rule.lower().strip()
        for existing in self.verified_rules:
            existing_lower = existing.lower().strip()
            # Skip if one is a substring of the other
            if rule_lower in existing_lower or existing_lower in rule_lower:
                return
            # Skip if they share >70% of words
            rule_words = set(rule_lower.split())
            existing_words = set(existing_lower.split())
            if rule_words and existing_words:
                overlap = len(rule_words & existing_words) / max(len(rule_words), len(existing_words))
                if overlap > 0.7:
                    return
        # Cap total rules
        if len(self.verified_rules) >= self.max_rules:
            return
        self.verified_rules.append(rule)
        logger.info(f"[rule+] {rule}")

    def _promote_confirmed_hypotheses(self) -> None:
        """Extract confirmed hypotheses and add as verified rules on level-up."""
        if not self.hypothesis:
            return
        for line in self.hypothesis.split("\n"):
            if line.startswith("[confirmed]"):
                rule = line.replace("[confirmed]", "").strip()
                # Remove trailing evidence in parens
                if "(" in rule:
                    rule = rule[:rule.rfind("(")].strip()
                if rule:
                    self._add_rule(rule)

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
                f"ESTABLISHED FACTS (from probing this level):\n{self.probe_facts}\n\n"
                + (f"VERIFIED RULES (confirmed across levels — these are ground truth):\n"
                   + "\n".join(f"- {r}" for r in self.verified_rules) + "\n\n"
                   if self.verified_rules else "")
                + (f"DEAD ENDS (do NOT retry these — confirmed blocked multiple times):\n"
                   + "\n".join(f"- {d}" for d in self.dead_ends) + "\n\n"
                   if self.dead_ends else "")
                + (f"FALSIFIED (approaches that failed — do NOT repeat):\n"
                   + "\n".join(f"- {f}" for f in self.falsified) + "\n\n"
                   if self.falsified else "")
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

        json_format = (
            "\nRespond with JSON:\n"
            '{"observation": "what changed and what it means",\n'
            ' "hypotheses": [\n'
            '   {"claim": "...", "status": "testing", "test_actions": ["ACTION3","ACTION3","ACTION3"], "evidence": "..."},\n'
            '   {"claim": "...", "status": "confirmed", "evidence": "..."}\n'
            ' ],\n'
            ' "verified_rules": ["universal game mechanic you confirmed"],\n'
            ' "falsified": ["approach that failed and why"],\n'
        )
        use_reasoning = self._large_change_pending
        if use_reasoning:
            json_format += (
                ' "cause_effect": {"action": "what you did", "changes": "what changed remotely", '
                '"theory": "your best causal explanation"},\n'
            )
            self._large_change_pending = False
        json_format += ' "notes": "persistent notes for next turn"}'
        content.append(input_text(json_format))

        try:
            create_kwargs = {
                "model": "gpt-5.4",
                "input": [{"role": "user", "content": content}],
                "max_output_tokens": 2000 if use_reasoning else 1500,
            }
            if use_reasoning:
                create_kwargs["reasoning"] = {"effort": "medium"}
                logger.info("[reasoning] Using medium reasoning effort for post-trigger planning")
            else:
                create_kwargs["temperature"] = 0.2
            response = self.client.responses.create(**create_kwargs)
            raw = response.output_text or ""
            if not raw and hasattr(response, 'output'):
                # Reasoning mode may structure output differently — extract text from output items
                for item in (response.output or []):
                    if hasattr(item, 'content'):
                        for block in (item.content or []):
                            if hasattr(block, 'text') and block.text:
                                raw = block.text
                                break
                    if raw:
                        break
            if not raw:
                raw = "{}"
            if use_reasoning:
                logger.info(f"[reasoning-raw] {raw[:300]}")
        except Exception as e:
            logger.warning(f"API error: {e}")
            time.sleep(2)
            return [GameAction.ACTION1]

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

        # Extract verified rules from model output (only explicit verified_rules field)
        new_rules = data.get("verified_rules", [])
        if new_rules and isinstance(new_rules, list):
            for rule in new_rules:
                if isinstance(rule, str) and rule:
                    self._add_rule(rule)

        # Extract falsified approaches
        new_falsified = data.get("falsified", [])
        if new_falsified and isinstance(new_falsified, list):
            for f in new_falsified:
                if isinstance(f, str) and f and f not in self.falsified and len(self.falsified) < 10:
                    self.falsified.append(f)
                    logger.info(f"[falsified] {f}")

        # Log cause-effect analysis
        cause_effect = data.get("cause_effect")
        if cause_effect and isinstance(cause_effect, dict):
            logger.info(f"[cause-effect] {cause_effect.get('theory', '')}")

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
