"""Observe-act agent with Bayesian hypothesis tracking (v3).

Builds on v2 (auto-probe + structured hypotheses) with three major changes:
1. Bayesian probabilities replace binary hypothesis status. Every hypothesis carries
   a probability 0.0-1.0 and a tests[] list documenting what was tried.
2. Expanded auto-probe for ACTION6: probes 3-5 diverse coordinates instead of one,
   preventing premature dismissal of click-based mechanics.
3. Stale hypothesis detection: warns when the top hypothesis hasn't changed for 5+
   turns or when 3+ consecutive turns produce 0 cell changes.
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

AVAILABLE ACTIONS:
- ACTION1: Simple action (typically mapped to UP)
- ACTION2: Simple action (typically mapped to DOWN)
- ACTION3: Simple action (typically mapped to LEFT)
- ACTION4: Simple action (typically mapped to RIGHT)
- ACTION5: Simple action (interact, select, rotate, attach/detach, etc.)
- ACTION6: Complex action requiring x,y coordinates (0-63 range). This is a CLICK — different coordinates may have completely different effects. You must explore the coordinate space, not just test one spot.
- ACTION7: Simple action (undo)

Not all actions may be available in every game. Check available_actions.

COORDINATE SYSTEM:
- All object positions use {"row": R, "col": C} format.
- Row increases DOWNWARD (row 0 = top edge, row 63 = bottom edge).
- Col increases RIGHTWARD (col 0 = left edge, col 63 = right edge).
- To go UP, DECREASE row. To go DOWN, INCREASE row. To go LEFT, DECREASE col. To go RIGHT, INCREASE col.
- ACTION6 takes (x, y) coordinates where x=col and y=row — note the order is REVERSED from object positions.
- Object "orientation" uses compass directions: N=up (decreasing row), E=right (increasing col), S=down, W=left.

GOAL-FIRST THINKING:

Every turn, your FIRST output must be a clear GOAL answering: "What specific thing am I
trying to do RIGHT NOW that will increase my score?"

GOOD goals: "Navigate to the white switch at [31,21] and interact with it"
BAD goals: "Explore the map" (too vague), "Test hypotheses" (not actionable)

Your PLAN is 2-5 concrete steps to achieve the goal. Hypotheses SUPPORT the goal —
they explain WHY you chose it. Goals and steps are the output; hypotheses are evidence.

Each turn you receive:
- Established facts from initial probing (what each action does)
- Symbolic state of objects on the grid
- What changed since the last action
- An image (side-by-side with previous frame, red = changed cells)
- Your own notes from previous turns

CORE PRIORS — use these as your default assumptions about the game world:

OBJECTS AND PHYSICS:
- Objects are cohesive: connected cells of the same color move as a unit. If part of
  an object moved but the rest didn't, it's actually two separate objects.
- Objects are persistent: they don't randomly appear or vanish. If an object appeared,
  something CAUSED it (you triggered a switch, crossed a threshold, etc.). If an object
  vanished, something removed it. Always ask: "what caused this change?"
- Objects interact by CONTACT: walking into, clicking on, or standing adjacent to an
  object is how you cause effects. If something changed far away after you touched
  a nearby object, the nearby object is a remote trigger.
- Solid objects don't overlap: BLOCKED means you hit a wall or obstacle. The boundary
  of your movement area IS the puzzle geometry.

AGENTS AND ENTITIES:
- You control one entity (typically the smallest moving object). It moves when you
  act; everything else is part of the environment.
- Environment objects don't move on their own unless a timer/automation is running.
  If something changes without your action, track it as a mechanic with a cycle.

NUMBERS AND COUNTING:
- Count objects that look similar — the number often matters (3 switches means 3
  activations, 4 keys for 4 locks).
- Track quantities that change: if a bar shrinks by 2 cells per move, calculate how
  many moves remain before it runs out. Plan accordingly.
- When you see a pattern that needs N steps, count the steps and commit to the plan.

GEOMETRY AND SPACE:
- Proximity matters: objects near each other are more likely to be related than
  distant objects.
- Symmetry is meaningful: if an object is symmetric, its orientation may matter.
  Rotations (90°, 180°) and reflections are common game mechanics.
- Inside/outside: if an object is inside another's boundary, they likely have a
  containment relationship (key in a room, switch behind a gate).
- Connectivity: can you reach an object from where you are? If not, you need to
  find a path or remove an obstacle first.

BEFORE generating hypotheses, you MUST fill in scene_inventory and untried.

scene_inventory requires you to:
- List EVERY distinct object visible in the symbolic state and image. Include position, color, and whether you've interacted with it.
- List which directions are currently open (no wall) and which are blocked from your current position.
- Look for VISUAL CORRESPONDENCES: objects that share colors, shapes, or patterns with other objects. Two things that look similar probably have a gameplay relationship (lock/key, switch/door, pattern matching).

untried requires you to:
- List every open direction you haven't explored from your current or any recent position.
- List every object you haven't directly interacted with (walked into, clicked on, used ACTION5 near).
- List any action types you haven't tested.

Your hypotheses MUST address the untried items. If you list "up" as an unexplored direction, you MUST have at least one hypothesis about what's up there. If you list an object as not interacted with, you MUST have a hypothesis about what interacting with it does. If you notice a visual correspondence, you MUST have a hypothesis about the relationship.

It is a LOGIC ERROR to list something as untried and not have a hypothesis about it.

Output JSON:
- scene_inventory: {
    "objects": [
      {"description": "black plus symbol", "position": [25, 38], "color": "black", "interacted": false}
    ],
    "open_directions": ["up", "left"],
    "blocked_directions": ["right", "down"],
    "visual_correspondences": [
      "The symbol inside the door (top) resembles the symbol in the bottom-left corner — same purple color, similar shape"
    ]
  }
- untried: {
    "directions_not_explored": ["up"],
    "objects_not_interacted": ["black plus symbol", "door with purple symbol"],
    "action_types_not_tested": ["ACTION5"]
  }
- observation: what changed and what it means
- hypotheses: MULTIPLE testable claims with Bayesian probabilities. Each hypothesis has:
  {
    "category": "OBJECT_INTERACTION",
    "claim": "ACTION6 clicking on objects toggles their state",
    "probability": 0.4,
    "tests": [
      {"action": "ACTION6(32,15)", "result": "0 changes", "interpretation": "clicked on gridline, not on object"},
      {"action": "ACTION6(12,8)", "result": "4 changes, red cell toggled", "interpretation": "clicking on grid cells toggles color"}
    ],
    "test_actions": ["ACTION6", 25, 36],
    "information_gain": "high — only tested empty space so far, never clicked an object"
  }

  HYPOTHESIS RULES:
  - Every hypothesis must have a `probability` between 0.0 and 1.0. Update probabilities each turn based on new evidence. Never use 0.0 or 1.0 — there is always some uncertainty.
  - Every hypothesis must have a `tests` list documenting what was specifically tried and what happened. This is how you track whether your evidence is strong or weak.
  - Maintain at least 3 competing hypotheses at all times. If you only have one theory, you're not exploring enough. Generate alternatives even if they seem unlikely.
  - Prioritize testing the hypothesis with the highest uncertainty (probability closest to 0.5) — that's where you gain the most information. If you already have a high-confidence hypothesis (>0.8), test something else.
  - For parameterized actions (ACTION6), evidence from one coordinate does NOT generalize. You must test diverse targets: click on different colored objects, empty cells, edges, corners. Track each coordinate tested in the `tests` list. You need at least 3-5 diverse coordinate tests before drawing any conclusion about ACTION6's behavior.
  - Re-evaluate old hypotheses when new evidence arrives. If you discover a new game mechanic, check whether it changes the probability of existing hypotheses.
  - The `test_actions` field drives execution. For simple actions: ["ACTION3","ACTION3"]. For ACTION6 with coordinates: ["ACTION6", x, y] where x and y are integers 0-63. The first hypothesis (sorted by descending probability) that has `test_actions` will be executed.

  HYPOTHESIS DIVERSITY RULES:

  You must maintain hypotheses in at least 3 of these 5 categories at all times:

  1. NAVIGATION: Where should I go? What's in each unexplored direction?
  2. OBJECT_INTERACTION: What does each object do when I interact with it? (walk into it, click it, use ACTION5 near it)
  3. VISUAL_PATTERN: What do visual similarities between objects mean? (matching symbols, same colors, similar shapes)
  4. GAME_MECHANIC: What are the rules? (Does interacting with X change Y? Do I need to match patterns? Is there a sequence?)
  5. GOAL: What is the win condition? (Reach a location? Match all patterns? Collect items? Clear obstacles?)

  If all your hypotheses are in the same category (e.g., all NAVIGATION), you are FAILING to explore the hypothesis space. Force yourself to generate at least one hypothesis in a different category.

  Each hypothesis must be tagged with its category, e.g.:
  {"category": "VISUAL_PATTERN", "claim": "The plus symbol rotates the door pattern; matching it to the corner symbol opens the door", "probability": 0.3, ...}

  When generating hypotheses, apply the CORE PRIORS above:
  - OBJECT_INTERACTION hypotheses should specify the CONTACT mechanism ("walking into X", "clicking X", "standing adjacent to X")
  - GAME_MECHANIC hypotheses should explain CAUSATION ("X changed because I touched Y" not just "X changed")
  - NAVIGATION hypotheses should consider CONNECTIVITY ("is there a path?") and PROXIMITY ("what's nearest?")
  - VISUAL_PATTERN hypotheses should consider SYMMETRY and ROTATION, not just color matching

PLANNING:

You have two modes: EXPLORE and EXECUTE.

EXPLORE mode: You're still figuring out the game. Generate hypotheses, test them, gather information. This is the default when you start a new level or when your plan just failed.

EXECUTE mode: You've figured out enough to act. You have a concrete plan and you're following it step by step.

Your JSON output must include a "plan" field:

{
  "goal": "Match the display pattern to the target pattern in the bottom-left, then reach the door",
  "plan": {
    "mode": "execute",
    "steps": [
      {"step": 1, "action": "Navigate to the plus symbol at [25, 38]", "status": "completed"},
      {"step": 2, "action": "Activate the plus (walk into it or ACTION5)", "status": "current"},
      {"step": 3, "action": "Check if display now matches target", "status": "pending"},
      {"step": 4, "action": "If not matched, activate plus again", "status": "pending"},
      {"step": 5, "action": "If matched, navigate to the door at [32, 12]", "status": "pending"}
    ],
    "abort_conditions": [
      "Display doesn't change after activating plus",
      "Door doesn't open after display matches target",
      "GAME_OVER"
    ],
    "supporting_hypotheses": ["plus controls orientation (0.7)", "goal is to match display to target (0.65)"]
  }
}

Or when still exploring:

{
  "goal": "Determine what the plus symbol does",
  "plan": {
    "mode": "explore",
    "steps": [
      {"step": 1, "action": "Navigate to plus symbol", "status": "current"},
      {"step": 2, "action": "Activate it and observe changes", "status": "pending"}
    ],
    "abort_conditions": ["Discover the plus is not interactable"],
    "supporting_hypotheses": ["plus is an interactable object (0.5)"]
  }
}

RULES FOR PLANNING:

1. WHEN TO SWITCH TO EXECUTE MODE:
   - Your top 2+ hypotheses are compatible (they don't contradict each other)
   - Your top hypothesis has probability >= 0.6
   - You can describe a concrete sequence of steps to test or achieve the goal
   When these conditions are met, you MUST switch to execute mode. Don't keep exploring when you have a workable theory.

2. WHEN TO STAY IN EXECUTE MODE:
   - Follow the plan step by step. Mark steps as "completed" as you finish them.
   - Your test_actions each turn should correspond to the CURRENT step in the plan.
   - Do NOT re-evaluate the plan from scratch each turn. Only update step statuses.
   - Stay in execute mode unless an abort condition is hit.

3. WHEN TO ABORT AND RETURN TO EXPLORE MODE:
   - An abort condition fires (something unexpected happened)
   - You've been stuck on the same step for 5+ turns with no progress
   - Your plan's key hypothesis drops below 0.3 probability due to new evidence
   - You completed all steps but didn't achieve the goal
   When aborting, log WHY the plan failed in the falsified list, then generate new hypotheses.

4. PLAN PERSISTENCE:
   - The plan carries forward in your notes between turns. Don't regenerate it from scratch unless aborting.
   - Each turn, just update which step is current and whether any steps completed.
   - Your test_actions should serve the current step, not be random exploration.

5. OVERLAPPING HYPOTHESES ARE FINE:
   - If multiple hypotheses point in the same direction, that's convergent evidence — it strengthens the plan.
   - Don't treat overlap as redundancy. Three hypotheses saying "interact with plus to change display" at 0.7, 0.6, 0.5 is STRONGER than one hypothesis at 0.7.
   - Use the supporting_hypotheses field to list which hypotheses back your plan.
   - If hypotheses disagree, pick the higher-probability one, commit to it, and test it. Only switch if you find direct contradictory evidence — not just because you're uncertain.

CONNECTING PLANS TO ACTIONS:

When in EXECUTE mode, your test_actions MUST correspond to your current plan step.
- If current step is "Navigate to plus symbol at [25, 38]", your test_actions should be movement actions toward that position.
- If current step is "Activate the plus", your test_actions should be ACTION5 or walking into it.
- If current step is "Check if display matches target", you don't need test_actions — just observe the symbolic state and image.

Do NOT output test_actions that are unrelated to your current plan step while in execute mode.
If you find yourself wanting to do something off-plan, either:
1. Add it as a new step to the plan, or
2. Abort the plan and return to explore mode.

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

CRITICAL — VISUAL CORRESPONDENCES:

Many games use visual pattern matching as a core mechanic. Two objects that share a color, shape, or symbol pattern almost always have a gameplay relationship. Common patterns:
- A "lock" shows a pattern, and you must make a "display" match that pattern
- Switches/buttons rotate, cycle, or toggle a displayed pattern
- The goal is to make two things match, or to arrange things in a specific configuration

When you see two objects with similar visual features:
1. Note the similarity explicitly in visual_correspondences
2. Describe exactly HOW they're similar and HOW they differ
3. Generate a hypothesis about the relationship (one changes the other, they need to match, etc.)
4. Generate a test: interact with one and observe whether the other changes

Do NOT dismiss visual similarities as coincidence. In puzzle games, visual similarity is almost always meaningful.

IMPORTANT:
- When something unexpected happens (large change, object appears/disappears), INVESTIGATE.
  Look at the symbolic diff carefully. What objects changed? What appeared? What vanished?
  Formulate a hypothesis about WHY and test it.
- When a large change happens, you'll see a before/after image. Study it carefully —
  what objects moved? What appeared or disappeared? What does this tell you about the mechanic?
- Don't just navigate. The game likely has mechanics beyond movement — objects may have
  functions you need to discover through interaction.
- If a hypothesis hasn't led to progress after several tests, lower its probability and try something new.
- You have limited actions. Don't re-explore areas you've already mapped.
- When you complete a level, the layout changes but game MECHANICS carry over. Look for the
  same types of interactions (switches, gates, collectibles) in the new layout.
- If you're stuck (repeated BLOCKED), you're probably missing a game mechanic, not a path.
  Look for objects you haven't interacted with yet.

INTERACTION JOURNAL:

Every time you interact with an object or try something new and observe a result, record it in the "interactions" field:

"interactions": [
  {
    "turn": 12,
    "action": "Walked into plus symbol at [25, 38]",
    "observed": "Purple shape inside the door rotated 90° clockwise. Display changed from L-facing-right to L-facing-down.",
    "objects_involved": ["plus symbol", "door display"],
    "reversible": "unknown",
    "useful": true
  }
]

Rules for the interaction journal:
- Record EVERY interaction where something non-trivial happened (cells changed beyond normal movement, objects appeared/disappeared/changed, score changed).
- Record interactions where NOTHING happened too, if you expected something — e.g., "Walked into door at [32,12], nothing happened. Door may require display to match target first."
- Be SPECIFIC about what changed. Not "something changed" but "the purple L-shape rotated 90° clockwise" or "the yellow bar decreased by 2 cells."
- Each entry is a FACT, not a hypothesis. Only record what you directly observed.
- The "useful" flag marks interactions that revealed a game mechanic. These get highlighted.

You don't need to repeat old entries — just add new ones each turn. The code accumulates them for you.

USING YOUR JOURNAL IN PLANS:

When forming a plan in EXECUTE mode, your plan steps should be directly informed by journal entries. For example:

- Journal says "Walking into plus rotated the display 90°"
- Target display shows a specific orientation
- Therefore plan: "Activate plus N times until display matches target, then go to door"

If your journal tells you HOW a mechanic works, your plan should USE that knowledge with specific expected outcomes. Don't just "go interact with the plus" — say "activate the plus 2 more times (I need 180° more rotation based on turn 12 observation)."

A plan that ignores your journal entries is a BAD plan. Before finalizing a plan, check: does this plan use everything I've learned?"""


def _action_label(action: GameAction) -> str:
    """Return a human-readable label like 'ACTION6(12,8)' for complex actions."""
    if action.is_complex() and hasattr(action.action_data, "x"):
        return f"{action.name}({action.action_data.x},{action.action_data.y})"
    return action.name


class ToolUseAgent:
    MAX_ACTIONS = 150

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

        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        self.current_grid: list[list[int]] = []
        self.prev_image_grid: list[list[int]] | None = None
        self.prev_symbolic: dict | None = None
        self.hypothesis = ""
        self.notes = ""
        self.probe_facts = ""
        self.verified_rules: list[str] = []
        self.max_rules = 10
        self.falsified: list[str] = []
        self.blocked_tracker: dict[str, int] = {}
        self.dead_ends: list[str] = []
        self.recent_actions: list[str] = []
        self.llm_calls = 0
        self._large_change_pending = False

        # v3: stale hypothesis detection
        self._top_hypothesis_claim = ""
        self._top_hypothesis_unchanged_turns = 0
        self._zero_change_turns = 0

        # v3: hypothesis category diversity tracking
        self._recent_categories: list[set[str]] = []

        # v3: goal tracking
        self.current_goal = ""

        # v3: planning layer
        self.current_plan = None  # dict from model output, persisted across turns
        self.plan_step_turns = 0  # how many turns we've been on the current step
        self.last_plan_step = None  # to detect when we're stuck on a step

        # v3: interaction journal
        self.interaction_journal: list[dict] = []  # accumulated across the entire game
        self.journal_by_object: dict[str, list[dict]] = {}  # indexed by object for quick lookup
        self._pending_journal_prompt: str | None = None

    def run(self) -> None:
        timer = time.time()
        logger.info("=" * 60)
        logger.info(f"Bayesian Hypothesis Agent (v3) on {self.game_id}")
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
                frame = self._step(GameAction.RESET)
                self.action_counter += 1
                self.current_grid = frame.frame[-1] if frame.frame else []
                self.prev_image_grid = None
                self.prev_symbolic = None
                continue

            actions = self._think_and_act(frame)

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
                no_effect = changes == 0

                label = _action_label(action)
                if no_effect:
                    result = f"{label}: NO EFFECT (0 cells changed)"
                elif blocked:
                    result = f"{label}: BLOCKED"
                else:
                    result = f"{label}: {changes} cells changed"

                # Track blocked directions by region
                if blocked or no_effect:
                    region = self._get_region()
                    key = f"{region}:{action.name}"
                    self.blocked_tracker[key] = self.blocked_tracker.get(key, 0) + 1
                    if self.blocked_tracker[key] >= 3:
                        dead_end = f"{action.name} from region {region}"
                        if dead_end not in self.dead_ends:
                            self.dead_ends.append(dead_end)
                            logger.info(f"[dead-end] {dead_end} (blocked {self.blocked_tracker[key]}x)")
                elif changes > 100:
                    self.blocked_tracker.clear()
                    self.dead_ends.clear()
                    self._large_change_pending = True

                # v3: auto-detect significant interactions for journal
                if changes > 80 and not blocked:
                    self._pending_journal_prompt = (
                        f"\nIMPORTANT: Your last action caused {changes} cell changes, which is "
                        "well beyond normal movement (~52 cells). Something significant happened. "
                        "You MUST record this in your interactions journal with specific details "
                        "about what changed. Look at the symbolic diff and the image carefully.\n"
                    )

                # v3: track zero-change turns for stale detection
                if changes == 0:
                    self._zero_change_turns += 1
                else:
                    self._zero_change_turns = 0

                self.recent_actions.append(result)
                if len(self.recent_actions) > 15:
                    self.recent_actions = self.recent_actions[-15:]

                mode_tag = ""
                if self.current_plan and isinstance(self.current_plan, dict):
                    m = self.current_plan.get("mode", "?").upper()
                    g = self.current_plan.get("goal", "")
                    goal_short = (g[:50] + "…") if len(g) > 50 else g
                    mode_tag = f" [{m}] {goal_short}"
                logger.info(f"#{self.action_counter} {result} score={frame.levels_completed}{mode_tag}")

                if frame.state in (GameState.WIN, GameState.GAME_OVER):
                    break
                if blocked or no_effect:
                    break
                if frame.levels_completed > score_before:
                    logger.info(f"[level-up] Level {frame.levels_completed}! Re-probing new layout.")
                    self._promote_high_probability_hypotheses()
                    avail = frame.available_actions or [1, 2, 3, 4]
                    self.probe_facts = self._auto_probe(frame, avail)
                    logger.info(f"[probe] {self.probe_facts}")
                    self.hypothesis = ""
                    self.notes = ""
                    self.falsified = []
                    self.blocked_tracker.clear()
                    self.dead_ends.clear()
                    self._large_change_pending = False
                    self._pending_journal_prompt = None
                    self.prev_symbolic = None
                    self.prev_image_grid = None
                    self.recent_actions = [f"*** LEVEL {frame.levels_completed} ***"]
                    self._top_hypothesis_claim = ""
                    self._top_hypothesis_unchanged_turns = 0
                    self._zero_change_turns = 0
                    self._recent_categories = []
                    self.current_goal = ""
                    self.current_plan = None
                    self.plan_step_turns = 0
                    self.last_plan_step = None
                    # v3: carry journal across levels — tag entries with current level
                    for entry in self.interaction_journal:
                        if "level" not in entry:
                            entry["level"] = score_before
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

    def _get_nearby_object_journal(self, symbolic_state: dict) -> str:
        """Check if the player is near any objects with journal entries."""
        if not self.journal_by_object:
            return ""

        reminders = []

        for obj_name, entries in self.journal_by_object.items():
            useful_entries = [e for e in entries if e.get("useful")]
            if useful_entries:
                latest = useful_entries[-1]
                reminders.append(
                    f"REMINDER — you already know about '{obj_name}': "
                    f"at turn {latest.get('turn', '?')} you found that: {latest.get('observed', '?')}"
                )

        if reminders:
            return "\n" + "\n".join(reminders) + "\n"
        return ""

    def _get_region(self) -> str:
        sym = grid_to_symbolic(self.current_grid)
        objects = sym.get("objects", [])
        moving = [o for o in objects if o.get("size", 9999) < 50]
        if moving:
            moving.sort(key=lambda o: o.get("size", 9999))
            c = moving[0].get("center", {"row": 32, "col": 32})
            return f"({c['row']//10*10},{c['col']//10*10})"
        return "(?,?)"

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

    def _promote_high_probability_hypotheses(self) -> None:
        """Extract high-probability hypotheses and add as verified rules on level-up.

        v3 change: instead of looking for [confirmed] tags, we parse the stored
        hypothesis JSON lines for probability >= 0.8.
        """
        if not self.hypothesis:
            return
        # Try to parse each line as a hypothesis with probability
        for line in self.hypothesis.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Format: [p=0.85] claim (evidence summary)
            if line.startswith("[p="):
                try:
                    prob_str = line[3:line.index("]")]
                    prob = float(prob_str)
                except (ValueError, IndexError):
                    continue
                if prob >= 0.8:
                    rule = line[line.index("]") + 1:].strip()
                    if "(" in rule:
                        rule = rule[:rule.rfind("(")].strip()
                    if rule:
                        self._add_rule(rule)

    def _auto_probe(self, frame: FrameData, available: list[int]) -> str:
        """Take one of each action, plus expanded ACTION6 probing at diverse coordinates.

        For simple actions (ACTION1-5, ACTION7): one test each.
        For ACTION6 (complex/click): probe 3-5 interesting coordinate targets identified
        from the symbolic state — object centers, empty cells, edges, grid center.
        """
        facts = []

        # Separate ACTION6 from simple actions
        simple_actions = [a for a in available if a != 6]
        has_action6 = 6 in available

        # Probe simple actions (one each)
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

        # Expanded ACTION6 probe: test at 3-5 diverse coordinates
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

        changes = sum(1 for r in range(64) for c in range(64)
                      if grid_before[r][c] != self.current_grid[r][c])
        blocked = 0 < changes < 10

        change_summary = self._format_sym_changes(sym_changes)

        status = "BLOCKED" if blocked else f"{changes} cells changed"
        fact = f"{action.name}: {status}"
        if change_summary:
            fact += "\n  " + "\n  ".join(change_summary)

        self.recent_actions.append(f"{action.name}: {status}")
        return fact

    def _probe_action6(self) -> list[str]:
        """Probe ACTION6 at 3-5 diverse coordinate targets.

        Targets are chosen from the symbolic state:
        1. Center of the largest non-background object
        2. Center of a different-colored object
        3. An empty/background cell
        4. A cell at the grid edge
        5. Center of the grid (32, 32)
        """
        facts = []
        symbolic = grid_to_symbolic(self.current_grid)
        fg_objects = symbolic.get("objects", [])

        targets: list[tuple[int, int, str]] = []  # (x, y, description)

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

        # Target 3: an empty/background cell — find a cell with the most common color
        bg_info = symbolic.get("backgrounds", [])
        if bg_info:
            bg_color_name = bg_info[0]["color"]
            # Find a cell of this background color near the center
            for dr in range(0, 32):
                found = False
                for r, c in [(32 + dr, 32), (32 - dr, 32), (32, 32 + dr), (32, 32 - dr)]:
                    if 0 <= r < 64 and 0 <= c < 64:
                        cell_color = self.current_grid[r][c]
                        from .symbolic import COLOR_NAMES
                        if COLOR_NAMES.get(cell_color) == bg_color_name:
                            targets.append((c, r, f"background cell ({bg_color_name})"))
                            found = True
                            break
                if found:
                    break

        # Target 4: grid edge
        targets.append((0, 0, "grid corner (0,0)"))

        # Target 5: grid center
        targets.append((32, 32, "grid center"))

        # Deduplicate targets that are too close together (within 3 cells)
        unique_targets: list[tuple[int, int, str]] = []
        for x, y, desc in targets:
            too_close = False
            for ux, uy, _ in unique_targets:
                if abs(x - ux) + abs(y - uy) < 3:
                    too_close = True
                    break
            if not too_close:
                unique_targets.append((x, y, desc))

        # Probe each target
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

            changes = sum(1 for r in range(64) for c in range(64)
                          if grid_before[r][c] != self.current_grid[r][c])
            blocked = 0 < changes < 10

            change_summary = self._format_sym_changes(sym_changes)

            status = "BLOCKED" if blocked else f"{changes} cells changed"
            fact = f"ACTION6 at ({x},{y}) [{desc}]: {status}"
            if change_summary:
                fact += "\n  " + "\n  ".join(change_summary)

            facts.append(fact)
            self.recent_actions.append(f"ACTION6({x},{y}): {status}")

            if frame_result.state == GameState.GAME_OVER:
                self.total_deaths += 1
                facts.append(f"  (caused GAME_OVER)")
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
                change_summary.append(f"background {c.get('color', '?')}: size {c['size']['was']}->{c['size']['now']}")
        return change_summary

    def _perceive(self, grid: list[list[int]]) -> str:
        """Perception step: have GPT-5.4 describe the current game frame."""
        self.llm_calls += 1

        content = [
            input_text(
                "Describe this 64x64 game grid image. Be exhaustive but concise.\n\n"
                "1. LAYOUT: What type of scene is this? (maze, open field, puzzle room, etc.)\n"
                "2. PLAYER: Identify the likely player object (usually the smallest distinct object). "
                "Where is it? What color?\n"
                "3. DIRECTIONS: From the player's position, what is in each cardinal direction "
                "(up/down/left/right)? For each: is it open (traversable), blocked (wall), or "
                "occupied by an object? How far until the next wall or object?\n"
                "4. OBJECTS: List every distinct non-wall, non-floor object. Color, approximate "
                "position (top-left, center, bottom-right, etc.), shape, size.\n"
                "5. PATHS: What corridors or routes are visible? Where do they lead?\n"
                "6. PATTERNS: Any visual symmetries, repeated elements, or notable features?\n\n"
                "Use grid coordinates where possible (row 0=top, col 0=left, max=63). "
                "Focus on spatial facts, not interpretation."
            ),
            input_image_b64(grid_b64(grid)),
        ]

        try:
            response = self.client.responses.create(
                model="gpt-5.4",
                input=[{"role": "user", "content": content}],
                max_output_tokens=600,
                temperature=0.2,
            )
            return response.output_text or ""
        except Exception as e:
            logger.warning(f"Perception API error: {e}")
            return ""

    def _think_and_act(self, frame: FrameData) -> list[GameAction]:
        self.llm_calls += 1

        available = frame.available_actions or [1, 2, 3, 4]
        avail_str = ", ".join(f"ACTION{i}" for i in available)
        recent = "\n".join(self.recent_actions[-10:]) if self.recent_actions else "(none)"

        symbolic = grid_to_symbolic(self.current_grid)
        sym_changes = diff_symbolic(self.prev_symbolic, symbolic) if self.prev_symbolic else []
        self.prev_symbolic = symbolic

        # --- Perception step ---
        scene_description = self._perceive(self.current_grid)
        if scene_description:
            logger.info(f"[perceive] {scene_description}")

        changes_text = ""
        if sym_changes:
            changes_text = f"CHANGES SINCE LAST ACTION:\n{json.dumps(sym_changes, indent=1)}\n\n"

        # v3: build stuck-on-step warning for planning layer
        plan_stuck_warning = ""
        if self.plan_step_turns >= 5:
            plan_stuck_warning = (
                f"\n\nWARNING: You've been on step {self.last_plan_step} of your plan "
                f"for {self.plan_step_turns} turns with no progress. Either:\n"
                "1. Break this step into smaller sub-steps\n"
                "2. Try a completely different approach to this step\n"
                "3. Abort the plan — your approach may be wrong\n"
                "Do NOT just repeat the same actions.\n"
            )
            logger.info(f"[plan-stuck] Step {self.last_plan_step} stuck for {self.plan_step_turns} turns")

        # v3: build auto-transition nudge (explore -> execute)
        commit_nudge = ""
        if (self.current_plan and self.current_plan.get("mode") == "explore"
                and self.action_counter > 20 and self.hypothesis):
            import re as _re
            probs = _re.findall(r'\[p=(\d+\.\d+)', self.hypothesis)
            high_prob_count = sum(1 for p in probs if float(p) >= 0.6)
            if high_prob_count >= 2:
                commit_nudge = (
                    f"\n\nYou have {high_prob_count} hypotheses at >= 0.6 probability "
                    "and you've been exploring for 20+ turns. You likely have enough information "
                    "to form a plan. Switch to EXECUTE mode: synthesize your hypotheses into "
                    "a concrete step-by-step plan and start following it.\n"
                )
                logger.info(f"[plan-nudge] Commit nudge injected ({high_prob_count} high-prob hypotheses)")

        # v3: build stale hypothesis warning
        stale_warning = ""
        if self._top_hypothesis_unchanged_turns >= 5:
            stale_warning = (
                f"\nWARNING: Your top hypothesis has been unchanged for "
                f"{self._top_hypothesis_unchanged_turns} turns with no score progress. "
                f"Your evidence may be insufficient or your approach may be wrong. Consider: "
                f"(1) What would FALSIFY your current top hypothesis? Test that. "
                f"(2) What alternative explanations have you not considered? "
                f"(3) For ACTION6, have you tested diverse coordinates?\n\n"
            )
        elif self._zero_change_turns >= 3:
            stale_warning = (
                f"\nWARNING: You have had {self._zero_change_turns} consecutive turns with "
                f"0 cell changes. You are doing nothing productive. Consider: "
                f"(1) What would FALSIFY your current top hypothesis? Test that. "
                f"(2) What alternative explanations have you not considered? "
                f"(3) For ACTION6, have you tested diverse coordinates?\n\n"
            )

        # v3: build diversity nudge from recent category tracking
        diversity_nudge = ""
        if len(self._recent_categories) >= 3:
            all_cats = [c for cats in self._recent_categories[-3:] for c in cats]
            if len(set(all_cats)) <= 1:
                stuck_cat = set(all_cats).pop() if all_cats else "unknown"
                diversity_nudge = (
                    f"\nWARNING: Your hypotheses have been in the same category "
                    f"({stuck_cat}) for 3+ turns. "
                    f"You are likely stuck in a local optimum. FORCE yourself to generate "
                    f"hypotheses in at least 2 OTHER categories from: NAVIGATION, "
                    f"OBJECT_INTERACTION, VISUAL_PATTERN, GAME_MECHANIC, GOAL.\n\n"
                )
                logger.info(f"[diversity] Nudge injected — stuck in category: {stuck_cat}")

        # v3: periodic forced reflection every 8 turns
        reflection_prompt = ""
        if self.action_counter > 0 and self.action_counter % 8 == 0:
            reflection_prompt = (
                "\n\n=== FORCED REFLECTION (every 8 turns) ===\n"
                "STOP and think about what you're MISSING.\n"
                "1. What directions have you NOT explored from positions you've visited?\n"
                "2. What objects have you NOT interacted with?\n"
                "3. What visual patterns or correspondences have you NOT investigated?\n"
                "4. What actions (ACTION5, ACTION6, ACTION7) have you NOT tested?\n"
                "5. Are there any objects that LOOK SIMILAR to each other? What might that mean?\n"
                "6. Have you been repeating the same strategy? What's a COMPLETELY DIFFERENT approach?\n\n"
                "Generate at least ONE new hypothesis about something you haven't tried. "
                "This hypothesis should have test_actions so it gets executed.\n"
                "=== END REFLECTION ===\n"
            )
            logger.info(f"[reflection] Forced reflection injected at action {self.action_counter}")

        # --- Prompt caching: content is ordered static-first so the OpenAI API
        # can cache the longest possible prefix across turns.
        # Layer 1 (static across all turns/levels/games): SYSTEM_PROMPT  ~4700 tok
        # Layer 2 (static within a level): probe facts + verified rules  ~200-300 tok
        # Layer 3 (changes every turn): score, warnings, dynamic context
        content = [
            input_text(SYSTEM_PROMPT),
            # --- semi-static: changes only on level-up / new game ---
            input_text(
                f"\nESTABLISHED FACTS (from probing this level):\n{self.probe_facts}\n\n"
                + (f"VERIFIED RULES (confirmed across levels — these are ground truth):\n"
                   + "\n".join(f"- {r}" for r in self.verified_rules) + "\n\n"
                   if self.verified_rules else "")
            ),
            # --- dynamic: changes every turn ---
            input_text(
                f"Score: {frame.levels_completed} | Deaths: {self.total_deaths} | "
                f"Actions: {self.action_counter}/{self.MAX_ACTIONS}\n"
                f"Available: {avail_str}\n\n"
                + (f"DEAD ENDS (do NOT retry these — confirmed blocked multiple times):\n"
                   + "\n".join(f"- {d}" for d in self.dead_ends[-10:]) + "\n\n"
                   if self.dead_ends else "")
                + (f"FALSIFIED (approaches that failed — do NOT repeat):\n"
                   + "\n".join(f"- {f}" for f in self.falsified[-8:]) + "\n\n"
                   if self.falsified else "")
                + stale_warning
                + diversity_nudge
                + plan_stuck_warning
                + commit_nudge
            ),
        ]

        # v3: inject interaction journal BEFORE hypotheses and plan (factual foundation)
        if self.interaction_journal:
            useful = [e for e in self.interaction_journal if e.get("useful")][-15:]
            recent_other = [e for e in self.interaction_journal if not e.get("useful")][-3:]

            journal_text = "=== INTERACTION JOURNAL (facts you've discovered) ===\n"

            if useful:
                journal_text += "\nKEY DISCOVERIES:\n"
                for entry in useful:
                    level_tag = f" [level {entry['level']}]" if "level" in entry else ""
                    journal_text += (
                        f"  Turn {entry.get('turn', '?')}{level_tag}: {entry.get('action', '?')}\n"
                        f"    → {entry.get('observed', '?')}\n"
                    )

            if recent_other:
                journal_text += "\nRECENT OBSERVATIONS:\n"
                for entry in recent_other:
                    level_tag = f" [level {entry['level']}]" if "level" in entry else ""
                    journal_text += (
                        f"  Turn {entry.get('turn', '?')}{level_tag}: {entry.get('action', '?')}\n"
                        f"    → {entry.get('observed', '?')}\n"
                    )

            journal_text += "\nUse these facts to inform your plan. If a discovery tells you how a mechanic works, ACT ON IT.\n"
            journal_text += "=== END JOURNAL ===\n"

            content.append(input_text(journal_text))

        # v3: surface relevant journal entries for known objects
        nearby_knowledge = self._get_nearby_object_journal(symbolic)
        if nearby_knowledge:
            content.append(input_text(nearby_knowledge))

        # v3: inject pending journal prompt (auto-detected large change)
        if self._pending_journal_prompt:
            content.append(input_text(self._pending_journal_prompt))
            self._pending_journal_prompt = None

        # Perception: inject scene description before hypotheses/goals
        if scene_description:
            content.append(input_text(
                f"VISUAL SCENE DESCRIPTION (from dedicated perception analysis):\n"
                f"{scene_description}\n"
            ))

        content.append(input_text(
            (f"YOUR CURRENT GOAL:\n{self.current_goal}\n\n" if self.current_goal and not self.current_plan else "")
            + (f"YOUR NOTES:\n{self.notes}\n\n" if self.notes else "")
            + (f"CURRENT HYPOTHESES:\n{self.hypothesis}\n\n" if self.hypothesis else "")
            + f"RECENT:\n{recent}\n\n"
            + changes_text
            + f"OBJECTS:\n{json.dumps(symbolic.get('objects', []), indent=1)}\n"
            + f"SPATIAL RELATIONS:\n{json.dumps(symbolic.get('relations', []), indent=1)}\n"
        ))

        # v3: inject current plan into prompt (goal is visible via the plan)
        if self.current_plan:
            plan_text = json.dumps(self.current_plan, indent=2)
            content.append(input_text(f"\nYOUR CURRENT PLAN:\n{plan_text}\n"))

        if self.prev_image_grid:
            content.append(input_text("Side-by-side (PREVIOUS vs CURRENT, red = changed):"))
            content.append(input_image_b64(diff_b64(self.prev_image_grid, self.current_grid)))
        else:
            content.append(input_text("Current frame:"))
            content.append(input_image_b64(grid_b64(self.current_grid)))

        self.prev_image_grid = self.current_grid

        if reflection_prompt:
            content.append(input_text(reflection_prompt))

        is_large_change = self._large_change_pending
        json_format = (
            "\nRespond with COMPACT JSON (no pretty-printing, no newlines). You MUST fill in ALL fields, especially hypotheses with test_actions.\n"
            '{"goal": "What you are trying to achieve RIGHT NOW to win",\n'
            ' "plan": {"mode": "explore/execute", '
            '"steps": [{"step": 1, "action": "...", "status": "current/completed/pending"}], '
            '"abort_conditions": [...], "supporting_hypotheses": [...]},\n'
            ' "observation": "what changed and what it means",\n'
            ' "scene_inventory": {"objects": [...], "open_directions": [...], '
            '"blocked_directions": [...], "visual_correspondences": [...]},\n'
            ' "untried": {"directions_not_explored": [...], '
            '"objects_not_interacted": [...], "action_types_not_tested": [...]},\n'
            ' "interactions": [{"turn": N, "action": "what you did", '
            '"observed": "specific result", "objects_involved": ["obj1"], '
            '"reversible": "unknown/yes/no", "useful": true/false}],\n'
            ' "hypotheses": [\n'
            '   {"category": "GAME_MECHANIC", "claim": "...", "probability": 0.6, '
            '"tests": [{"action": "...", "result": "...", "interpretation": "..."}], '
            '"test_actions": ["ACTION3","ACTION3","ACTION3"], "information_gain": "..."},\n'
            '   {"category": "VISUAL_PATTERN", "claim": "...", "probability": 0.3, '
            '"tests": [...], "information_gain": "..."}\n'
            ' ],\n'
            ' "verified_rules": ["universal game mechanic you confirmed"],\n'
            ' "falsified": ["approach that failed and why"],\n'
        )
        if is_large_change:
            json_format += (
                ' "cause_effect": {"action": "what you did", "changes": "what changed remotely", '
                '"theory": "your best causal explanation"},\n'
            )
            self._large_change_pending = False
            logger.info("[large-change] Including cause_effect field for large change analysis")
        json_format += ' "notes": "persistent notes for next turn"}'
        content.append(input_text(json_format))

        try:
            create_kwargs = {
                "model": "gpt-5.4",
                "input": [{"role": "user", "content": content}],
                "temperature": 0.2,
                "max_output_tokens": 5000 if is_large_change else 4096,
                "prompt_cache_retention": "24h",
            }
            response = self.client.responses.create(**create_kwargs)

            # Log prompt cache hit rate
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                prompt_tok = getattr(usage, 'prompt_tokens', 0)
                cached_tok = 0
                details = getattr(usage, 'prompt_tokens_details', None)
                if details:
                    cached_tok = getattr(details, 'cached_tokens', 0)
                pct = cached_tok * 100 // max(prompt_tok, 1)
                logger.info(f"[cache] prompt={prompt_tok} cached={cached_tok} ({pct}%)")

            raw = response.output_text or ""
            if not raw and hasattr(response, 'output'):
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
            logger.debug(f"[raw-response] {raw[:500]}")
        except Exception as e:
            logger.warning(f"API error: {e}")
            time.sleep(2)
            return [GameAction.ACTION1]

        data = self._parse_json(raw)
        if not data or data == {}:
            logger.warning(f"[empty-response] Model returned empty/unparseable JSON. Raw (first 300 chars): {raw[:300]}")

        # v3: extract and accumulate interaction journal entries
        new_interactions = data.get("interactions", [])
        if new_interactions and isinstance(new_interactions, list):
            for entry in new_interactions:
                if isinstance(entry, dict) and entry.get("action"):
                    # Add turn number if not present
                    if "turn" not in entry:
                        entry["turn"] = self.action_counter

                    self.interaction_journal.append(entry)

                    # Index by objects involved
                    for obj in entry.get("objects_involved", []):
                        if obj not in self.journal_by_object:
                            self.journal_by_object[obj] = []
                        self.journal_by_object[obj].append(entry)

                    logger.info(
                        f"[journal] Turn {entry.get('turn')}: {entry.get('action')} -> {entry.get('observed')}"
                    )

        obs = data.get("observation", "")
        hypotheses = data.get("hypotheses", [])
        notes = data.get("notes", "")

        # v3: extract goal (top-level field)
        goal = data.get("goal", "")
        if not goal and hypotheses:
            # Fallback: synthesize from top hypothesis
            sorted_for_goal = sorted(
                [h for h in hypotheses if isinstance(h, dict)],
                key=lambda h: h.get("probability", 0.5),
                reverse=True,
            )
            if sorted_for_goal:
                goal = f"Testing: {sorted_for_goal[0].get('claim', 'unknown')}"
            logger.debug("[goal-fallback] No goal field from model, synthesized from top hypothesis")
        if goal:
            self.current_goal = goal

        # v3: sort hypotheses by probability descending, find first with test_actions
        raw_actions = []
        active_claim = ""
        if hypotheses:
            # Sort by probability descending
            sorted_hyps = sorted(
                [h for h in hypotheses if isinstance(h, dict)],
                key=lambda h: h.get("probability", 0.5),
                reverse=True,
            )

            hyp_lines = []
            current_categories: set[str] = set()
            for h in sorted_hyps:
                prob = h.get("probability", 0.5)
                claim = h.get("claim", "")
                category = h.get("category", "unknown")
                tests = h.get("tests", [])
                info_gain = h.get("information_gain", "")
                test_count = len(tests) if isinstance(tests, list) else 0
                hyp_lines.append(f"[p={prob:.2f}|{category}] {claim} ({test_count} tests, info_gain={info_gain})")
                current_categories.add(category)

                # Find the first hypothesis (by probability) that has test_actions
                if h.get("test_actions") and not raw_actions:
                    raw_actions = h["test_actions"]
                    active_claim = claim

            # Also include any non-dict entries
            for h in hypotheses:
                if not isinstance(h, dict):
                    hyp_lines.append(str(h))

            self.hypothesis = "\n".join(hyp_lines)

            # v3: track hypothesis categories for diversity detection
            self._recent_categories.append(current_categories)
            if len(self._recent_categories) > 5:
                self._recent_categories = self._recent_categories[-5:]

            # v3: stale hypothesis tracking
            top_claim = sorted_hyps[0].get("claim", "") if sorted_hyps else ""
            if top_claim and top_claim == self._top_hypothesis_claim:
                self._top_hypothesis_unchanged_turns += 1
            else:
                self._top_hypothesis_claim = top_claim
                self._top_hypothesis_unchanged_turns = 0

        # v3: extract and persist plan
        new_plan = data.get("plan", None)
        if new_plan and isinstance(new_plan, dict):
            # Inject top-level goal into plan for backward compat (mode_tag uses plan.goal)
            if self.current_goal and "goal" not in new_plan:
                new_plan["goal"] = self.current_goal

            # Detect if we're stuck on the same step
            current_step = None
            for step in new_plan.get("steps", []):
                if step.get("status") == "current":
                    current_step = step.get("step")
                    break

            if current_step == self.last_plan_step:
                self.plan_step_turns += 1
            else:
                self.plan_step_turns = 0
                self.last_plan_step = current_step

            self.current_plan = new_plan

        # Fallback: check for "actions" or "action" field
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

        # === GOAL-FIRST LOGGING (console ~9 lines per turn) ===
        # Turn header with goal
        new_plan = self.current_plan
        mode_str = new_plan.get("mode", "?").upper() if new_plan and isinstance(new_plan, dict) else "?"
        hyp_count = len(hypotheses) if hypotheses else 0
        logger.info("=" * 50)
        logger.info(f"TURN {self.llm_calls} | GOAL: {self.current_goal or '(none)'}")

        # Current step from plan
        current_step_desc = ""
        if new_plan and isinstance(new_plan, dict):
            steps = new_plan.get("steps", [])
            current = next((s for s in steps if s.get("status") == "current"), None)
            current_step_desc = current.get("action", "?") if current else "none"
        logger.info(f"  STEP: {current_step_desc or '(no plan)'}")
        logger.info(f"  MODE: {mode_str} | {hyp_count} hypotheses")

        # Observation in full
        logger.info(f"[observe] {obs}")

        # Top 3 hypotheses at INFO with full claim text
        if hypotheses:
            sorted_hyps_for_log = sorted(
                [h for h in hypotheses if isinstance(h, dict)],
                key=lambda h: h.get("probability", 0.5),
                reverse=True,
            )
            for h in sorted_hyps_for_log[:3]:
                prob = h.get("probability", "?")
                category = h.get("category", "?")
                claim = h.get("claim", "")
                logger.info(f"  [p={prob}|{category}] {claim}")
            # Remaining hypotheses at DEBUG
            for h in sorted_hyps_for_log[3:]:
                prob = h.get("probability", "?")
                category = h.get("category", "?")
                claim = h.get("claim", "")
                tests = h.get("tests", [])
                test_count = len(tests) if isinstance(tests, list) else 0
                logger.debug(f"  [p={prob}|{category}] {claim} | {test_count} tests")

        # Visual correspondences, untried → DEBUG level
        scene_inv = data.get("scene_inventory")
        if scene_inv and isinstance(scene_inv, dict):
            corrs = scene_inv.get("visual_correspondences", [])
            if corrs:
                for corr in corrs:
                    logger.debug(f"[visual-corr] {corr}")
            untried_objs = data.get("untried", {}).get("objects_not_interacted", [])
            untried_dirs = data.get("untried", {}).get("directions_not_explored", [])
            if untried_dirs:
                logger.debug(f"[untried-dirs] {untried_dirs}")
            if untried_objs:
                logger.debug(f"[untried-objs] {untried_objs}")

        if stale_warning:
            logger.info(f"[stale] Warning injected (unchanged={self._top_hypothesis_unchanged_turns}, zero_changes={self._zero_change_turns})")

        # v3: resolve actions, handling ACTION6 coordinate format
        available_set = set(available)
        actions = self._resolve_actions(raw_actions, available_set, avail_str)

        if not actions:
            fallback = GameAction.from_id(available[0])
            actions = [fallback]
            logger.info(f"[fallback] No valid actions from LLM, using {fallback.name}")

        logger.info(f"[actions] {[_action_label(a) for a in actions]}")
        return actions

    def _resolve_actions(
        self, raw_actions: list, available_set: set[int], avail_str: str
    ) -> list[GameAction]:
        """Parse raw_actions list into GameAction objects.

        Handles two formats:
        - Simple actions: ["ACTION3", "ACTION3", "ACTION1"]
        - ACTION6 with coordinates: ["ACTION6", x, y] where x, y are ints 0-63
          The triplet ["ACTION6", x, y] is consumed as one action.
        """
        actions = []
        i = 0
        while i < len(raw_actions):
            item = raw_actions[i]

            if isinstance(item, str) and item.upper() == "ACTION6":
                # Check if next two items are x, y coordinates
                if i + 2 < len(raw_actions):
                    x_val = raw_actions[i + 1]
                    y_val = raw_actions[i + 2]
                    if isinstance(x_val, (int, float)) and isinstance(y_val, (int, float)):
                        x = int(x_val)
                        y = int(y_val)
                        if 0 <= x <= 63 and 0 <= y <= 63 and 6 in available_set:
                            action = GameAction.from_id(6)
                            action.set_data({"x": x, "y": y})
                            actions.append(action)
                            i += 3
                            continue
                        else:
                            logger.warning(f"[filter] ACTION6({x},{y}) coords out of range or ACTION6 not available")
                            i += 3
                            continue
                # ACTION6 without valid coordinates — skip
                logger.warning(f"[filter] ACTION6 without valid coordinates, skipping")
                i += 1
                continue

            if isinstance(item, str):
                try:
                    ga = GameAction.from_name(item)
                except (ValueError, KeyError):
                    i += 1
                    continue
                if ga.value in available_set:
                    actions.append(ga)
                else:
                    logger.warning(f"[filter] {item} not in available actions {avail_str}, skipping")
            i += 1

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
        logger.warning(
            f"[parse-fail] All JSON parse strategies failed. "
            f"Raw length={len(raw)}, first 200 chars: {raw[:200]!r}"
        )
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

    def _close(self) -> None:
        if not self.scorecard_id or not self.arcade:
            return
        scorecard = self.arcade.close_scorecard(self.scorecard_id)
        if scorecard:
            logger.info("--- SCORECARD ---")
            logger.info(json.dumps(scorecard.model_dump(), indent=2))
        self.scorecard_id = None
