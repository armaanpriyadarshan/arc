"""VLM Explorer — vision-first agent that sees the game as images.

Architecture (radically different from experiment 001):

1. DISCOVERY PHASE (~8 actions):
   - Reset, take each directional action once
   - Send the VLM: initial frame + 4 post-action frames + 4 diff images
   - VLM outputs: what is the player, what do actions do, what objects exist, what might the goal be

2. PLAY PHASE (remaining budget):
   - Each turn: send VLM the current frame image + game rules learned so far + action history
   - VLM picks one action at a time (not batched plans — the VLM needs to SEE each frame)
   - After every action, check for score changes, death, win
   - Every 10 actions: VLM does a "reflection" — what's working, what's not, what to try differently

3. SELF-CORRECTION:
   - If stuck (5+ actions with no grid change), VLM gets a special "you're stuck" prompt
   - If died, VLM gets death context and must explain what killed it
   - If score changed, VLM gets celebration + "what did you do right" prompt
   - Accumulated learnings persist across deaths/resets
"""

import json
import logging
import os
import time

from arcengine import FrameData, GameAction, GameState
from openai import OpenAI

from .vision import diff_to_b64, grid_to_b64, image_block, text_block

logger = logging.getLogger(__name__)


class VLMExplorerAgent:
    MAX_ACTIONS = 200
    MODEL = "gpt-4o"

    def __init__(self, game_id: str) -> None:
        from arc_agi import Arcade
        self.game_id = game_id
        self.arcade = Arcade()
        self.scorecard_id = self.arcade.open_scorecard()
        self.env = self.arcade.make(game_id, scorecard_id=self.scorecard_id)
        self.frames: list[FrameData] = []
        self.action_counter = 0
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        self.game_knowledge = ""  # accumulated understanding of the game
        self.action_history: list[str] = []  # for context
        self.total_deaths = 0
        self.highest_score = 0

    def run(self) -> None:
        timer = time.time()
        logger.info("=" * 60)
        logger.info(f"EXPERIMENT 002: VLM Explorer on {self.game_id}")
        logger.info("=" * 60)

        # RESET
        frame = self._step(GameAction.RESET)

        if frame.state == GameState.WIN:
            logger.info("Won on reset?!")
            self._close()
            return

        # === PHASE 1: DISCOVERY ===
        logger.info("=== PHASE 1: DISCOVERY ===")
        frame = self._discovery_phase(frame)

        if frame.state == GameState.WIN:
            logger.info("Won during discovery!")
            self._close()
            return

        # === PHASE 2: PLAY ===
        logger.info("=== PHASE 2: PLAY ===")
        frame = self._play_phase(frame)

        elapsed = round(time.time() - timer, 2)
        logger.info("=" * 60)
        logger.info(
            f"FINISHED: actions={self.action_counter} state={frame.state.name} "
            f"score={frame.levels_completed} deaths={self.total_deaths} time={elapsed}s"
        )
        logger.info(f"Game knowledge:\n{self.game_knowledge}")
        logger.info("=" * 60)
        self._close()

    def _discovery_phase(self, frame: FrameData) -> FrameData:
        """Take one of each directional action, show VLM the before/after/diff."""
        initial_grid = frame.frame[-1]
        initial_b64 = grid_to_b64(initial_grid)

        # Take 4 directional actions and collect results
        test_actions = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4]
        action_names = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]
        results = []

        for action, name in zip(test_actions, action_names):
            grid_before = frame.frame[-1]
            frame = self._step(action)
            grid_after = frame.frame[-1]

            results.append({
                "action": name,
                "after_b64": grid_to_b64(grid_after),
                "diff_b64": diff_to_b64(grid_before, grid_after),
                "num_changes": sum(1 for r in range(len(grid_before)) for c in range(len(grid_before[0])) if grid_before[r][c] != grid_after[r][c]),
            })

            self.action_history.append(f"{name}: {results[-1]['num_changes']} changes")

            if frame.state == GameState.GAME_OVER:
                frame = self._step(GameAction.RESET)
                self.total_deaths += 1

            if frame.state == GameState.WIN:
                return frame

        # Now ask VLM to analyze everything
        content = [
            text_block("You are analyzing an unknown video game displayed as a 64x64 pixel grid. "
                       "Each pixel is one of 16 colors. I will show you:\n"
                       "1. The initial game frame\n"
                       "2. The result of 4 different actions, each with the resulting frame and a diff image (red pixels = what changed)\n\n"
                       "Analyze carefully and tell me:\n"
                       "- What is the player? (describe its appearance, position, size)\n"
                       "- What does each action do? (direction of movement, step size)\n"
                       "- What objects/structures do you see? (walls, doors, items, UI elements)\n"
                       "- What might the goal be?\n"
                       "- What is the color coding? (what color = walls, floor, player, etc)\n"
                       "- Any UI elements? (score, health, inventory displayed on the grid)\n\n"
                       "Be very specific about positions and colors."),
            text_block("\n--- INITIAL FRAME ---"),
            image_block(initial_b64),
        ]

        for r in results:
            content.extend([
                text_block(f"\n--- After {r['action']} ({r['num_changes']} pixels changed) ---"),
                text_block("Resulting frame:"),
                image_block(r["after_b64"]),
                text_block("Diff (red = changed pixels):"),
                image_block(r["diff_b64"]),
            ])

        content.append(text_block(
            "\n\nBased on ALL of this, provide a comprehensive analysis. "
            "End your response with a section called GAME RULES that summarizes "
            "everything you've learned in a compact format I can reference later."
        ))

        response = self._vlm_call(content, "discovery")
        self.game_knowledge = response

        logger.info(f"[discovery] VLM analysis:\n{response[:500]}...")
        return frame

    def _play_phase(self, frame: FrameData) -> FrameData:
        """VLM-guided play loop. Each turn: see frame, pick action, observe result."""
        stale_count = 0
        actions_since_reflection = 0

        while self.action_counter < self.MAX_ACTIONS:
            if frame.state == GameState.WIN:
                logger.info("[play] WIN!")
                break

            if frame.state == GameState.GAME_OVER:
                self.total_deaths += 1
                logger.info(f"[play] DIED (death #{self.total_deaths}) — resetting")

                # Ask VLM what killed us
                death_analysis = self._vlm_call([
                    text_block(f"You just DIED in the game. Here's what you know:\n\n"
                               f"GAME KNOWLEDGE:\n{self.game_knowledge}\n\n"
                               f"RECENT ACTIONS:\n{chr(10).join(self.action_history[-15:])}\n\n"
                               f"What likely killed you? How should you avoid it? "
                               f"Update your understanding. Be specific."),
                    text_block("Last frame before death:"),
                    image_block(grid_to_b64(frame.frame[-1])),
                ], "death_analysis")

                self.game_knowledge += f"\n\nDEATH #{self.total_deaths} LESSON: {death_analysis}"
                logger.info(f"[play] death analysis: {death_analysis[:200]}...")

                frame = self._step(GameAction.RESET)
                stale_count = 0
                actions_since_reflection = 0
                continue

            grid_before = frame.frame[-1]

            # Every 10 actions, do a reflection
            if actions_since_reflection >= 10:
                frame, stale_count = self._reflect(frame, stale_count)
                actions_since_reflection = 0
                if frame.state in (GameState.WIN, GameState.GAME_OVER):
                    continue

            # Build the action prompt
            content = [
                text_block(f"GAME KNOWLEDGE:\n{self.game_knowledge}\n\n"
                           f"Score: {frame.levels_completed} | Deaths: {self.total_deaths} | "
                           f"Actions used: {self.action_counter}/{self.MAX_ACTIONS}\n\n"
                           f"Recent actions: {', '.join(self.action_history[-8:])}\n"),
            ]

            if stale_count >= 3:
                content.append(text_block(
                    f"\n⚠️ WARNING: Last {stale_count} actions had NO EFFECT. "
                    f"You're probably hitting a wall. Try a COMPLETELY different direction.\n"
                ))

            content.extend([
                text_block("Current frame:"),
                image_block(grid_to_b64(grid_before)),
                text_block(
                    "\nWhat is your next action? Choose ONE of: ACTION1, ACTION2, ACTION3, ACTION4, ACTION5\n"
                    "Respond with ONLY a JSON object: {\"action\": \"ACTION1\", \"reasoning\": \"brief why\"}"
                ),
            ])

            response = self._vlm_call(content, "action")

            # Parse action
            action_name, reasoning = self._parse_action(response)
            try:
                action = GameAction.from_name(action_name)
            except (ValueError, KeyError):
                action = GameAction.ACTION1
                reasoning = f"fallback (couldn't parse: {action_name})"

            # Execute
            frame = self._step(action)
            grid_after = frame.frame[-1]

            # Count changes
            changes = sum(1 for r in range(len(grid_before)) for c in range(len(grid_before[0]))
                          if grid_before[r][c] != grid_after[r][c])

            if changes == 0:
                stale_count += 1
            else:
                stale_count = 0

            # Check score change
            score_changed = frame.levels_completed > self.highest_score
            if score_changed:
                self.highest_score = frame.levels_completed
                logger.info(f"[play] *** SCORE UP: {frame.levels_completed} ***")
                self.game_knowledge += f"\n\nSCORE INCREASE after {action_name}! Score now {frame.levels_completed}."

            self.action_history.append(
                f"{action_name}: {changes}ch score={frame.levels_completed}"
                + (" SCORE!" if score_changed else "")
                + (f" ({reasoning[:50]})" if reasoning else "")
            )
            actions_since_reflection += 1

            logger.info(
                f"[play] #{self.action_counter} {action_name}: "
                f"{changes}ch score={frame.levels_completed} stale={stale_count} "
                f"| {reasoning[:60]}"
            )

        return frame

    def _reflect(self, frame: FrameData, stale_count: int) -> tuple[FrameData, int]:
        """Periodic reflection — VLM re-analyzes the situation."""
        content = [
            text_block(f"REFLECTION TIME. You've taken 10 actions. Step back and think.\n\n"
                       f"GAME KNOWLEDGE:\n{self.game_knowledge}\n\n"
                       f"Score: {frame.levels_completed} | Deaths: {self.total_deaths} | "
                       f"Actions: {self.action_counter}/{self.MAX_ACTIONS}\n\n"
                       f"Recent actions:\n{chr(10).join(self.action_history[-10:])}\n\n"
                       f"Current frame:"),
            image_block(grid_to_b64(frame.frame[-1])),
            text_block("\nQuestions to consider:\n"
                       "1. Are you making progress toward the goal? What evidence?\n"
                       "2. Are you stuck in a pattern? Repeating the same moves?\n"
                       "3. Is there something on the screen you haven't explored?\n"
                       "4. Should you change your strategy entirely?\n"
                       "5. What part of the game do you NOT understand yet?\n\n"
                       "Provide an updated GAME RULES section with everything you now know. "
                       "Be creative — consider possibilities you haven't tried."),
        ]

        reflection = self._vlm_call(content, "reflection")
        self.game_knowledge = reflection
        logger.info(f"[reflect] VLM reflection:\n{reflection[:300]}...")
        return frame, stale_count

    def _vlm_call(self, content: list[dict], label: str) -> str:
        """Make a VLM API call with retry logic."""
        for attempt in range(5):
            try:
                response = self.client.chat.completions.create(
                    model=self.MODEL,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=1500,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                wait = 2 ** attempt
                logger.warning(f"[{label}] API error (attempt {attempt+1}): {e}, retry in {wait}s")
                time.sleep(wait)
        logger.error(f"[{label}] All API attempts failed")
        return "API unavailable. Continue exploring with directional actions."

    def _parse_action(self, response: str) -> tuple[str, str]:
        """Extract action name and reasoning from VLM response."""
        # Try JSON parse
        try:
            # Find JSON in response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
                return data.get("action", "ACTION1"), data.get("reasoning", "")
        except (json.JSONDecodeError, KeyError):
            pass

        # Fallback: look for ACTION keywords
        for name in ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"]:
            if name in response.upper():
                return name, response[:100]

        return "ACTION1", "could not parse response"

    def _step(self, action: GameAction) -> FrameData:
        raw = self.env.step(action)
        if raw is None:
            logger.warning(f"env.step({action.name}) returned None")
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
        self.action_counter += 1
        return frame

    def _close(self) -> None:
        if not self.scorecard_id:
            return
        scorecard = self.arcade.close_scorecard(self.scorecard_id)
        if scorecard:
            logger.info("--- SCORECARD ---")
            logger.info(json.dumps(scorecard.model_dump(), indent=2))
        self.scorecard_id = None
