"""Lightweight conversational agent — no DAG, no structured outputs.

Tests whether a simple GPT-5.4 agent with low reasoning effort performs
comparably to experiment 007's heavy DAG architecture. Same infrastructure
(auto-probe, enhanced symbolic state, conversational sliding window,
action history, sandbox) but with a minimal output format.

If the bottleneck is strategic (wrong target) not tactical (bad routes),
then all the DAG overhead is wasted.
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

SYSTEM_PROMPT = """You are playing an unknown turn-based game on a 64x64 grid. Each cell is 0-15 (colors).
Score increases when you complete a level. GAME_OVER = failure.

You have actions ACTION1-ACTION7. Not all may be available.
ACTION6 requires x,y coordinates (0-63).

You maintain a conversation — your prior reasoning is in message history.

Output JSON:
- observation: what changed and why
- goal: what you're doing RIGHT NOW to increase score
- actions: list of actions to execute (e.g. ["ACTION1", "ACTION3", "ACTION3"])
- verified_rules: universal game rules you've confirmed (max 10)
- code: optional Python code. You have grid (64x64), ROWS, COLS, memory_bank, action_effects. Set result to return data.

IMPORTANT:
- Study what changed after each interaction. Build causal models.
- Once you understand a mechanic, exploit it immediately.
- If repeated actions fail, your understanding is wrong. Try something different.
- Every action must gather info or make progress."""


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
        self.recent_actions: list[str] = []
        self.action_log: list[dict] = []
        self.llm_calls: int = 0
        self._needs_image = False

        # Program synthesis (optional — model can use it but doesn't have to)
        self.sandbox = Sandbox()
        self.sandbox.globals["memory_bank"] = []
        self.program_output: str = ""

    def run(self) -> None:
        timer = time.time()
        logger.info("=" * 60)
        logger.info(f"Lightweight Agent on {self.game_id}")
        logger.info("=" * 60)

        frame = self._step(GameAction.RESET)
        self.current_grid = frame.frame[-1] if frame.frame else []

        if frame.state == GameState.WIN:
            self._close()
            return

        available = frame.available_actions or [1, 2, 3, 4]
        self.probe_facts = self._auto_probe(frame, available)
        logger.info(f"[probe] {self.probe_facts}")
        self._needs_image = True

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

            # Circuit breaker
            if self.consecutive_blocked >= 5:
                self.recent_actions.append(
                    f"*** STUCK: {self.consecutive_blocked} consecutive BLOCKED actions. "
                    f"Your current approach is wrong. Try something fundamentally different. ***"
                )
                logger.info(f"[stuck] {self.consecutive_blocked} consecutive blocked")
                self.consecutive_blocked = 0
                self.prev_image_grid = None

            actions = self._think_and_act(frame)

            # Execute actions, stopping on events
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

                # Stop on significant events
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
                    self._needs_image = True
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
        content = [input_text(text)]
        if image_b64:
            content.append(input_image_b64(image_b64))
        self.messages.append({"role": "user", "content": content})

    def _append_assistant_message(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})

    def _trim_messages(self) -> list[dict]:
        if len(self.messages) <= self.MESSAGE_LIMIT:
            return list(self.messages)

        first_msg = self.messages[0] if self.messages and self.messages[0]["role"] == "user" else None
        tail = self.messages[-(self.MESSAGE_LIMIT - (1 if first_msg else 0)):]

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

        # Build user message
        parts = []
        parts.append(
            f"Score: {frame.levels_completed} | Deaths: {self.total_deaths} | "
            f"Actions: {self.action_counter}/{self.MAX_ACTIONS}\n"
            f"Available: {avail_str}"
        )

        # Probe facts in first message only (preserved by _trim_messages)
        if not self.messages:
            parts.append(f"\nESTABLISHED FACTS:\n{self.probe_facts}")

        if self.verified_rules:
            rules_text = "\n".join(f"- {r}" for r in self.verified_rules)
            parts.append(f"\nVERIFIED RULES:\n{rules_text}")

        parts.append(f"\nRECENT:\n{recent}")

        # Action history summary
        if self.action_log:
            total = len(self.action_log)
            blocked = sum(1 for a in self.action_log if a["blocked"])
            triggers = sum(1 for a in self.action_log if a["trigger"])
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
            parts.append(f"\nCODE OUTPUT:\n{self.program_output}")

        if sym_changes:
            parts.append(f"\nCHANGES SINCE LAST ACTION:\n{json.dumps(sym_changes, indent=1)}")

        # Symbolic state
        sym_output = {
            "objects": symbolic.get("objects", []),
            "relations": symbolic.get("relations", []),
        }
        if "composites" in symbolic:
            sym_output["composites"] = symbolic["composites"]
        parts.append(f"\nSCENE:\n{json.dumps(sym_output, indent=1)}")

        text_content = "\n".join(parts)

        # Image on first turn or after triggers
        send_image = not self.messages or self._needs_image
        image_b64 = None
        if send_image:
            if self.prev_image_grid:
                image_b64 = diff_b64(self.prev_image_grid, self.current_grid)
            else:
                image_b64 = grid_b64(self.current_grid)
            self.prev_image_grid = self.current_grid
            self._needs_image = False

        # Build user content
        user_content = [input_text(text_content)]
        if send_image and image_b64:
            if len(self.messages) > 1:
                user_content.append(input_text("Side-by-side (PREVIOUS vs CURRENT, red = changed):"))
            else:
                user_content.append(input_text("Current frame:"))
            user_content.append(input_image_b64(image_b64))

        self.messages.append({"role": "user", "content": user_content})
        trimmed = self._trim_messages()

        # GPT-5.4 call — simple, no structured outputs, low reasoning
        try:
            response = self.client.responses.create(
                model="gpt-5.4",
                instructions=SYSTEM_PROMPT,
                input=trimmed,
                reasoning={"effort": "low"},
                max_output_tokens=2000,
            )
            raw = response.output_text or "{}"
        except Exception as e:
            logger.warning(f"API error: {e}")
            time.sleep(2)
            return [GameAction.ACTION1]

        self._append_assistant_message(raw)

        # Parse JSON from response
        data = self._parse_json(raw)

        # Execute optional code
        code = data.get("code", "")
        if code and isinstance(code, str) and code.strip():
            logger.info(f"[code] executing {len(code)} chars")
            self.program_output = self.sandbox.run(code, self.current_grid)
            logger.info(f"[code-result] {self.program_output[:300]}")
        else:
            self.program_output = ""

        # Extract verified rules
        new_rules = data.get("verified_rules", [])
        if new_rules and isinstance(new_rules, list):
            for rule in new_rules:
                if isinstance(rule, str) and rule:
                    self._add_rule(rule)

        # Log
        obs = data.get("observation", "")
        goal = data.get("goal", "")
        action_seq = data.get("actions", [])
        if obs:
            logger.info(f"[obs] {obs[:150]}")
        if goal:
            logger.info(f"[goal] {goal[:150]}")

        # Resolve actions
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

        logger.info(f"[actions] {len(actions)} actions: {[a.name for a in actions]}")
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
        """On level-up, nothing extra needed — verified_rules persist."""
        pass

    def _auto_probe(self, frame: FrameData, available: list[int]) -> str:
        """Probe each available action. Extract spatial data from observations."""
        facts = []
        simple_actions = [a for a in available if a != 6]
        has_action6 = 6 in available

        action_effects = {}

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

            for key in before_centers:
                if key in after_centers and before_centers[key] != after_centers[key]:
                    bc = before_centers[key]
                    ac = after_centers[key]
                    dr = ac["row"] - bc["row"]
                    dc = ac["col"] - bc["col"]
                    if abs(dr) > 0 or abs(dc) > 0:
                        action_effects[action.name] = {"dr": dr, "dc": dc}
                        break

        # ACTION6 probe
        if has_action6:
            symbolic = grid_to_symbolic(self.current_grid)
            fg = symbolic.get("objects", [])
            targets = [(32, 32, "grid center")]
            if fg:
                obj = fg[0]
                cr, cc = obj["center"]["row"], obj["center"]["col"]
                targets.append((cc, cr, f"{obj['color']} object"))
            if len(fg) >= 2:
                obj = fg[-1]
                cr, cc = obj["center"]["row"], obj["center"]["col"]
                targets.append((cc, cr, f"{obj['color']} object"))

            for x, y, desc in targets[:3]:
                action = GameAction.from_id(6, x=x, y=y)
                fact = self._probe_one(action, label=f"ACTION6({x},{y}) on {desc}")
                facts.append(fact)

        # Inject action effects into sandbox
        if action_effects:
            self.sandbox.globals["action_effects"] = action_effects
            effects_str = ", ".join(f"{k}: dr={v['dr']} dc={v['dc']}" for k, v in action_effects.items())
            facts.append(f"\nOBSERVED ACTION EFFECTS (available in sandbox as `action_effects`):")
            facts.append(f"  {effects_str}")

        return "\n".join(facts)

    def _probe_one(self, action: GameAction, label: str = "") -> str:
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
