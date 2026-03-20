"""Conversational sliding-window agent with auto-probe and program synthesis.

Same core as experiment 005 (GPT-5.4, symbolic state, symbolic diff, image,
program synthesis, hypothesis-driven actions). Key architectural change:

Instead of building one massive prompt every turn and making a single API call,
the agent maintains a CONVERSATION with the model using a sliding window of the
last 10 messages. The model can reference its own prior reasoning without us
re-injecting hypotheses/notes. System prompt goes via `instructions` parameter.
Uses reasoning effort "high" and no temperature.
"""

import json
import logging
import os
import re
import time

from arcengine import FrameData, GameAction, GameState
from openai import OpenAI

from .sandbox import Sandbox
from .symbolic import grid_to_symbolic, diff_symbolic
from .vision import grid_b64, diff_b64, input_text, input_image_b64

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are playing an unknown turn-based game on a 64x64 grid. Each cell is an integer 0-15 (colors).
Your score (levels_completed) increases when you complete a level. GAME_OVER means you failed.

AVAILABLE ACTIONS:
- ACTION1-ACTION4: Simple actions (typically directional movement)
- ACTION5: Simple action (interact, select, rotate, etc.)
- ACTION6: Complex action requiring x,y coordinates (0-63). This is a CLICK — different
  coordinates may have completely different effects. You must test diverse targets.
- ACTION7: Simple action (undo)
Not all actions may be available in every game. Check available_actions.

You maintain a conversation — your prior reasoning is visible in the message history.

CORE PRIORS — default assumptions about the game world:

OBJECTS AND PHYSICS:
- Objects are cohesive: connected cells of the same color move as a unit.
- Objects are persistent: they don't randomly appear/vanish. If something changed,
  something CAUSED it. Always ask "what caused this change?"
- Objects interact by CONTACT: walking into, clicking on, or standing adjacent.
  If something changed far away after you touched a nearby object, it's a remote trigger.
- Solid objects don't overlap: BLOCKED means you hit a wall or obstacle.

NUMBERS AND COUNTING:
- Count similar objects — the number often matters.
- Track quantities that change: if a bar shrinks by 2 per move, calculate moves remaining.

GEOMETRY:
- Proximity matters: nearby objects are more likely related than distant ones.
- Symmetry is meaningful: rotations (90°, 180°) and reflections are common mechanics.
- Visual similarity between objects almost always means a gameplay relationship.

BEFORE responding, mentally inventory the scene:
- What objects exist? Where are they? Have you interacted with each one?
- What directions are open/blocked from your current position?
- Do any objects share colors, shapes, or patterns? (These likely have a relationship)

You MUST reason through this DAG in order:

q0_code (root, MANDATORY): Write Python code to analyze the grid. You have `grid`
  (64x64 list[list[int]]), ROWS, COLS, `memory_bank` (persists across levels),
  `action_effects` (observed movement deltas from probe). Set `result` to return findings.
  Variables persist between turns — build up data structures.
  All builtins available. Modules: collections, math, numpy, json.
  USE THIS FOR SPATIAL REASONING. If you know you're in a grid-based game, compute
  which colors are traversable vs obstacles, trace connected regions, find paths between
  positions. Don't just list objects — compute actionable spatial information.
q1_objects (depends on q0): List every object — what it provides, what it requires.
q2_last_action (root): What was the last action and what happened?
q3_requirements (depends on q1): For each object, are interaction requirements met?
q4_action_result (depends on q2): Did the last action succeed? If not, why?
q5_subtasks (depends on q1, q3): Top 3 sub-tasks with priority.
q6_planning_code (depends on q5): OPTIONAL Python code for planning. Use this to
  compute routes, count steps needed, simulate action sequences, or verify your plan
  computationally. Set `result` to return the planned sequence. If not needed, empty string.
q7_candidate_actions (depends on q5, q6): Top 5 candidate actions with purpose.
q8_feasibility (depends on q7, q3): For each candidate, are requirements met?
qa_actions (depends on q7, q8): Sequence of actions to execute toward the top sub-task.
  Output MULTIPLE actions — enough to make meaningful progress.

Additional fields:
- verified_rules: UNIVERSAL game rules (persist across levels, max 10)

IMPORTANT:
- After interacting with something, study the before/after image and symbolic diff.
  What SPECIFICALLY changed? Build a causal model: "doing X causes Y."
- Once you understand a mechanic, EXPLOIT it immediately. Don't re-test.
- You have limited actions. Every action must gather information or make progress.
- When you complete a level, layout changes but game MECHANICS carry over.
- If repeated actions fail, your understanding of the game is wrong. Try something
  fundamentally different."""


RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "q0_code", "q1_objects", "q2_last_action", "q3_requirements",
        "q4_action_result", "q5_subtasks", "q6_planning_code",
        "q7_candidate_actions", "q8_feasibility", "qa_actions",
        "verified_rules",
    ],
    "properties": {
        "q0_code": {
            "type": "string",
            "description": "MANDATORY Python code for grid analysis. You have grid, ROWS, COLS, memory_bank, action_effects. Set result to return findings. Use this for spatial reasoning — compute traversable regions, find paths, track positions.",
        },
        "q1_objects": {
            "type": "array",
            "description": "List every object: what it provides and requires.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "position", "provides", "requires"],
                "properties": {
                    "name": {"type": "string"},
                    "position": {"type": "string"},
                    "provides": {"type": "string"},
                    "requires": {"type": "string"},
                },
            },
        },
        "q2_last_action": {
            "type": "string",
            "description": "What was the last action taken and what happened?",
        },
        "q3_requirements": {
            "type": "array",
            "description": "For each object from q1, are interaction requirements met?",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["object", "met", "reason"],
                "properties": {
                    "object": {"type": "string"},
                    "met": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
            },
        },
        "q4_action_result": {
            "type": "string",
            "description": "Did the last action succeed? If not, why?",
        },
        "q5_subtasks": {
            "type": "array",
            "description": "Top 3 sub-tasks with priority.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["priority", "task", "rationale"],
                "properties": {
                    "priority": {"type": "integer"},
                    "task": {"type": "string"},
                    "rationale": {"type": "string"},
                },
            },
        },
        "q6_planning_code": {
            "type": "string",
            "description": "OPTIONAL Python code for planning. Compute routes, count steps, simulate sequences. Set result to return the plan. Empty string if not needed.",
        },
        "q7_candidate_actions": {
            "type": "array",
            "description": "Top 5 candidate actions with purpose and requirement.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["action", "purpose", "requirement"],
                "properties": {
                    "action": {"type": "string"},
                    "purpose": {"type": "string"},
                    "requirement": {"type": "string"},
                },
            },
        },
        "q8_feasibility": {
            "type": "array",
            "description": "For each candidate action, are requirements met?",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["action", "feasible", "reason"],
                "properties": {
                    "action": {"type": "string"},
                    "feasible": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
            },
        },
        "qa_actions": {
            "type": "array",
            "description": "Sequence of actions to execute toward the top sub-task. Output MULTIPLE actions.",
            "items": {"type": "string"},
        },
        "verified_rules": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}


class ToolUseAgent:
    MAX_ACTIONS = 100
    MESSAGE_LIMIT = 10  # sliding window size

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
        self.messages: list[dict] = []  # conversation history
        self.current_grid: list[list[int]] = []
        self.prev_image_grid: list[list[int]] | None = None
        self.prev_symbolic: dict | None = None
        self.probe_facts: str = ""
        self.verified_rules: list[str] = []
        self.max_rules = 10
        self.consecutive_blocked: int = 0
        self.recent_actions: list[str] = []  # for circuit breaker only
        self.action_log: list[dict] = []  # full action history
        self.llm_calls: int = 0
        self._needs_deep_thought = False  # high reasoning on next call

        # Program synthesis
        self.sandbox = Sandbox()
        self.sandbox.globals["memory_bank"] = []  # persists across levels and deaths
        self.program_output: str = ""

    def run(self) -> None:
        timer = time.time()
        logger.info("=" * 60)
        logger.info(f"Conversational Agent on {self.game_id}")
        logger.info("=" * 60)

        frame = self._step(GameAction.RESET)
        self.current_grid = frame.frame[-1] if frame.frame else []

        if frame.state == GameState.WIN:
            self._close()
            return

        available = frame.available_actions or [1, 2, 3, 4]
        self.probe_facts = self._auto_probe(frame, available)
        logger.info(f"[probe] {self.probe_facts}")
        self._needs_deep_thought = True  # first observation deserves high effort

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
                # Notify the model about the death via conversation
                self._append_user_message(
                    f"DEATH #{self.total_deaths}. You hit GAME_OVER. "
                    f"The game has been reset. Reconsider your approach."
                )
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
                    f"Your current approach is wrong. Try something fundamentally different. ***"
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
                frame = self._step(action, reasoning="")
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

                # Track full action history
                self.action_log.append({
                    "turn": self.action_counter,
                    "action": action.name,
                    "blocked": blocked,
                    "cells": changes,
                    "trigger": changes > 100,
                })

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
                    self._promote_on_level_up()
                    avail = frame.available_actions or [1, 2, 3, 4]
                    self.probe_facts = self._auto_probe(frame, avail)
                    logger.info(f"[probe] {self.probe_facts}")
                    # Clear conversation on level change — start fresh
                    self.messages = []
                    self.consecutive_blocked = 0
                    self.action_log = []
                    self.prev_symbolic = None
                    self.prev_image_grid = None
                    self.recent_actions = [f"*** LEVEL {frame.levels_completed} ***"]
                    self.program_output = ""
                    if self.verified_rules:
                        logger.info(f"[rules] {self.verified_rules}")
                    break
                if changes > 100:
                    self.prev_image_grid = grid_before
                    self._needs_deep_thought = True
                    break

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

    def _append_user_message(self, text: str, image_b64: str | None = None) -> None:
        """Append a user message to the conversation history."""
        content = []
        content.append(input_text(text))
        if image_b64:
            content.append(input_image_b64(image_b64))
        self.messages.append({"role": "user", "content": content})

    def _append_assistant_message(self, text: str) -> None:
        """Append an assistant message to the conversation history."""
        self.messages.append({"role": "assistant", "content": text})

    def _trim_messages(self) -> list[dict]:
        """Return a trimmed message window for the API call.

        Rules:
        - Always keep the FIRST user message (has probe facts)
        - Keep the last MESSAGE_LIMIT messages
        - Never start the window with an assistant message
        """
        if len(self.messages) <= self.MESSAGE_LIMIT:
            return list(self.messages)

        # Always keep the first user message
        first_msg = self.messages[0] if self.messages and self.messages[0]["role"] == "user" else None

        # Take the tail
        tail = self.messages[-(self.MESSAGE_LIMIT - (1 if first_msg else 0)):]

        # Ensure we don't start the tail with an assistant message
        while tail and tail[0]["role"] == "assistant":
            tail = tail[1:]

        if first_msg and (not tail or tail[0] is not first_msg):
            return [first_msg] + tail
        return tail

    def _think_and_act(self, frame: FrameData) -> list[GameAction]:
        self.llm_calls += 1

        available = frame.available_actions or [1, 2, 3, 4]
        avail_str = ", ".join(f"ACTION{i}" for i in available)
        recent = "\n".join(self.recent_actions[-10:]) if self.recent_actions else "(none)"

        symbolic = grid_to_symbolic(self.current_grid)
        sym_changes = diff_symbolic(self.prev_symbolic, symbolic) if self.prev_symbolic else []
        self.prev_symbolic = symbolic

        # Build compact user message — no notes/hypotheses, those live in conversation
        parts = []
        parts.append(
            f"Score: {frame.levels_completed} | Deaths: {self.total_deaths} | "
            f"Actions: {self.action_counter}/{self.MAX_ACTIONS}\n"
            f"Available: {avail_str}"
        )

        # Probe facts are in the first message (always kept by _trim_messages)
        # Only include on the actual first turn
        if not self.messages:
            parts.append(f"\nESTABLISHED FACTS:\n{self.probe_facts}")

        if self.verified_rules:
            rules_text = "\n".join(f"- {r}" for r in self.verified_rules)
            parts.append(f"\nVERIFIED RULES:\n{rules_text}")

        parts.append(f"\nRECENT:\n{recent}")

        # Compact action history summary (beyond the sliding window)
        if self.action_log:
            total = len(self.action_log)
            blocked = sum(1 for a in self.action_log if a["blocked"])
            triggers = sum(1 for a in self.action_log if a["trigger"])
            # Count actions per direction
            action_counts = {}
            blocked_counts = {}
            for a in self.action_log:
                action_counts[a["action"]] = action_counts.get(a["action"], 0) + 1
                if a["blocked"]:
                    blocked_counts[a["action"]] = blocked_counts.get(a["action"], 0) + 1
            summary = f"Total: {total} actions, {blocked} blocked, {triggers} triggers"
            for act in sorted(action_counts):
                b = blocked_counts.get(act, 0)
                summary += f"\n  {act}: {action_counts[act]}x ({b} blocked)"
            parts.append(f"\nACTION HISTORY:\n{summary}")

        if self.program_output:
            parts.append(f"\nQ0 CODE OUTPUT:\n{self.program_output}")

        if sym_changes:
            parts.append(f"\nCHANGES SINCE LAST ACTION:\n{json.dumps(sym_changes, indent=1)}")

        parts.append(f"\nOBJECTS:\n{json.dumps(symbolic.get('objects', []), indent=1)}")

        text_content = "\n".join(parts)

        # Build image
        if self.prev_image_grid:
            image_b64 = diff_b64(self.prev_image_grid, self.current_grid)
        else:
            image_b64 = grid_b64(self.current_grid)

        self.prev_image_grid = self.current_grid

        # Append user message to conversation
        user_content = [
            input_text(text_content),
        ]
        if self.prev_image_grid is not None or True:  # always include image
            if self.messages and any(m["role"] == "user" for m in self.messages):
                user_content.append(input_text("Side-by-side (PREVIOUS vs CURRENT, red = changed):"))
            else:
                user_content.append(input_text("Current frame:"))
            user_content.append(input_image_b64(image_b64))

        self.messages.append({"role": "user", "content": user_content})

        # Get trimmed conversation window
        trimmed = self._trim_messages()

        effort = "none"  # reasoning effort breaks structured outputs
        try:
            response = self.client.responses.create(
                model="gpt-5.4",
                instructions=SYSTEM_PROMPT,
                input=trimmed,
                reasoning={"effort": effort},
                max_output_tokens=8000,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "agent_response",
                        "strict": True,
                        "schema": RESPONSE_SCHEMA,
                    }
                },
            )
            raw = response.output_text or "{}"
        except Exception as e:
            logger.warning(f"API error: {e}")
            time.sleep(2)
            return [GameAction.ACTION1]

        # Append assistant response to conversation
        self._append_assistant_message(raw)

        # Parse response — structured outputs should guarantee valid JSON,
        # but output may be truncated if it exceeds max_output_tokens
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            logger.warning(f"[parse-error] JSON decode failed, falling back to _parse_json. Raw length: {len(raw)}")
            data = self._parse_json(raw)

        # Execute q0 code (mandatory — grid analysis)
        code = data.get("q0_code", "")
        if code and isinstance(code, str) and code.strip():
            logger.info(f"[q0-code] executing {len(code)} chars")
            self.program_output = self.sandbox.run(code, self.current_grid)
            logger.info(f"[q0-result] {self.program_output[:300]}")
        else:
            self.program_output = ""

        # Execute q6 code (optional — planning)
        plan_code = data.get("q6_planning_code", "")
        if plan_code and isinstance(plan_code, str) and plan_code.strip():
            logger.info(f"[q6-code] executing {len(plan_code)} chars")
            plan_output = self.sandbox.run(plan_code, self.current_grid)
            logger.info(f"[q6-result] {plan_output[:300]}")
            if self.program_output:
                self.program_output += f"\n\nPLAN CODE OUTPUT:\n{plan_output}"
            else:
                self.program_output = f"PLAN CODE OUTPUT:\n{plan_output}"

        # Extract verified rules
        new_rules = data.get("verified_rules", [])
        if new_rules and isinstance(new_rules, list):
            for rule in new_rules:
                if isinstance(rule, str) and rule:
                    self._add_rule(rule)

        # Log DAG reasoning chain
        q4 = data.get("q4_action_result", "")
        subtasks = data.get("q5_subtasks", [])
        action_seq = data.get("qa_actions", [])

        objects_found = data.get("q1_objects", [])
        if objects_found:
            logger.info(f"[q1] {len(objects_found)} objects identified")
        if q4:
            logger.info(f"[q4] {q4[:150]}")
        if subtasks:
            top = subtasks[0] if subtasks else {}
            logger.info(f"[q5] Top task: {top.get('task', '?')[:100]}")

        # Resolve action sequence
        actions = []
        for action_name in action_seq:
            if not isinstance(action_name, str):
                continue
            name = action_name.strip().split("(")[0].split()[0]
            try:
                actions.append(GameAction.from_name(name))
            except (ValueError, KeyError):
                continue

        if not actions:
            actions = [GameAction.ACTION1]

        logger.info(f"[qa] {len(actions)} actions: {[a.name for a in actions]}")
        return actions

    def _add_rule(self, rule: str) -> None:
        if re.search(r'\[\d+,\s*\d+\]', rule) or re.search(r'x[=]\d+', rule) or re.search(r'y[=]\d+', rule):
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

    def _promote_on_level_up(self) -> None:
        """On level-up, promote high-confidence knowledge to verified rules."""
        # The DAG doesn't have explicit hypothesis confirmation, but verified_rules
        # are already extracted each turn. Nothing extra needed here.
        pass

    def _auto_probe(self, frame: FrameData, available: list[int]) -> str:
        """Probe each available action. Extract spatial data from observations."""
        facts = []
        simple_actions = [a for a in available if a != 6]
        has_action6 = 6 in available

        # Track positions before/after each probe for spatial extraction
        action_effects = {}
        player_positions = []

        # Get pre-probe symbolic state to identify player
        pre_sym = grid_to_symbolic(self.current_grid)
        pre_objects = {(o["color"], o["size"]): o["center"]
                       for o in pre_sym.get("objects", [])}

        for action_id in simple_actions:
            try:
                action = GameAction.from_id(action_id)
            except (ValueError, KeyError):
                continue

            sym_before = grid_to_symbolic(self.current_grid)
            before_centers = {(o["color"], o["size"]): o["center"]
                              for o in sym_before.get("objects", [])}

            fact = self._probe_one(action)
            facts.append(fact)

            sym_after = grid_to_symbolic(self.current_grid)
            after_centers = {(o["color"], o["size"]): o["center"]
                             for o in sym_after.get("objects", [])}

            # Find which object moved (that's the player)
            for key in before_centers:
                if key in after_centers and before_centers[key] != after_centers[key]:
                    bc = before_centers[key]
                    ac = after_centers[key]
                    dr = ac[0] - bc[0]
                    dc = ac[1] - bc[1]
                    if abs(dr) > 0 or abs(dc) > 0:
                        action_effects[action.name] = {"dr": dr, "dc": dc}
                        player_positions.append(ac)
                        break

        # Expanded ACTION6 probe at diverse coordinates
        if has_action6:
            symbolic = grid_to_symbolic(self.current_grid)
            fg = symbolic.get("objects", [])
            targets = [(32, 32, "grid center")]
            if fg:
                obj = fg[0]
                cr, cc = obj["center"]
                targets.append((cc, cr, f"{obj['color']} object"))
            if len(fg) >= 2:
                obj = fg[-1]
                cr, cc = obj["center"]
                targets.append((cc, cr, f"{obj['color']} object"))

            for x, y, desc in targets[:3]:
                action = GameAction.from_id(6, x=x, y=y)
                fact = self._probe_one(action, label=f"ACTION6({x},{y}) on {desc}")
                facts.append(fact)

        # Inject observed action effects into sandbox
        if action_effects:
            self.sandbox.globals["action_effects"] = action_effects
            effects_str = ", ".join(f"{k}: dr={v['dr']} dc={v['dc']}" for k, v in action_effects.items())
            facts.append(f"\nOBSERVED ACTION EFFECTS (available in sandbox as `action_effects`):")
            facts.append(f"  {effects_str}")

        return "\n".join(facts)

    def _probe_one(self, action: GameAction, label: str = "") -> str:
        """Probe a single action and return a fact string."""
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
                    parts.append(f"center {c['center']['was']}->{c['center']['now']}")
                if "size" in c:
                    parts.append(f"size {c['size']['was']}->{c['size']['now']}")
                change_summary.append(f"{c.get('color','?')}: {', '.join(parts)}")
            elif c.get("type") == "background_size_changed":
                change_summary.append(f"background {c.get('color','?')}: size {c['size']['was']}->{c['size']['now']}")

        name = label or action.name
        status = "BLOCKED" if blocked else f"{changes} cells changed"
        fact = f"{name}: {status}"
        if change_summary:
            fact += "\n  " + "\n  ".join(change_summary)

        if frame_result.state == GameState.GAME_OVER:
            self.total_deaths += 1
            fact += "\n  (caused GAME_OVER)"
            frame_result = self._step(GameAction.RESET)
            self.action_counter += 1
            self.current_grid = frame_result.frame[-1] if frame_result.frame else []

        return fact

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
