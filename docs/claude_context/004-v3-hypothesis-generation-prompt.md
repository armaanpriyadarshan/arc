# Supplementary Task: Improve Hypothesis Generation in 004-v3

This is an addition to the main v3 implementation prompt. Apply these changes ON TOP of the Bayesian probability, evidence tracking, and action description changes already described there. All changes are in `experiments/004-world-model-induction-v3/agents/agent.py` (the system prompt and the main loop).

## Problem

The agent generates hypotheses from LLM priors ("explore right", "navigate corridors") rather than from systematic observation of the actual game state. It ignores open directions it hasn't tried, objects it hasn't interacted with, and visual correspondences between objects — even when these are clearly visible on screen. The Bayesian probability system is useless if the correct hypothesis is never generated in the first place.

## Changes

### 1. Add a structured observation phase to the output schema

Before the model generates hypotheses, it must first produce a scene inventory. Add these required fields to the JSON output schema in `SYSTEM_PROMPT`:

```json
{
  "scene_inventory": {
    "objects": [
      {"description": "black plus symbol", "position": [25, 38], "color": "black", "interacted": false},
      {"description": "door with purple symbol", "position": [32, 12], "color": "green/black/purple", "interacted": false},
      {"description": "purple symbol in corner", "position": [5, 55], "color": "purple", "interacted": false}
    ],
    "open_directions": ["up", "left"],
    "blocked_directions": ["right", "down"],
    "visual_correspondences": [
      "The symbol inside the door (top) resembles the symbol in the bottom-left corner — same purple color, similar shape"
    ]
  },
  "untried": {
    "directions_not_explored": ["up"],
    "objects_not_interacted": ["black plus symbol", "door with purple symbol"],
    "action_types_not_tested": ["ACTION5"]
  },
  ...rest of existing fields (observation, hypotheses, verified_rules, etc.)
}
```

Add this instruction to the system prompt:

```
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
```

### 2. Add hypothesis category requirements

Add this to the system prompt, right after the hypothesis format description:

```
HYPOTHESIS DIVERSITY RULES:

You must maintain hypotheses in at least 3 of these 5 categories at all times:

1. NAVIGATION: Where should I go? What's in each unexplored direction?
2. OBJECT INTERACTION: What does each object do when I interact with it? (walk into it, click it, use ACTION5 near it)
3. VISUAL PATTERN: What do visual similarities between objects mean? (matching symbols, same colors, similar shapes)
4. GAME MECHANIC: What are the rules? (Does interacting with X change Y? Do I need to match patterns? Is there a sequence?)
5. GOAL: What is the win condition? (Reach a location? Match all patterns? Collect items? Clear obstacles?)

If all your hypotheses are in the same category (e.g., all NAVIGATION), you are FAILING to explore the hypothesis space. Force yourself to generate at least one hypothesis in a different category.

Each hypothesis must be tagged with its category, e.g.:
{"category": "VISUAL_PATTERN", "claim": "The plus symbol rotates the door pattern; matching it to the corner symbol opens the door", "probability": 0.3, ...}
```

### 3. Add a periodic "what haven't I tried" forced reflection

In the main agent loop, add a code-level check. Every 8 turns (adjustable), inject an extra block into the prompt content:

```python
# In the main loop, before building the LLM prompt:
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
    # Append this to the content list before the JSON format instruction
```

### 4. Add a "novel hypothesis" bonus nudge when stuck

Separately from the stale hypothesis detection in the main prompt (described in the previous implementation prompt), add detection for the model repeatedly generating the same types of hypotheses. In the main loop:

```python
# Track hypothesis categories over time
if hasattr(self, '_recent_categories'):
    # Extract categories from current hypotheses
    current_categories = set()
    for h in hypotheses:
        if isinstance(h, dict):
            current_categories.add(h.get("category", "unknown"))
    self._recent_categories.append(current_categories)
    if len(self._recent_categories) > 5:
        self._recent_categories = self._recent_categories[-5:]
    
    # Check if categories are monotonous (same single category for 3+ turns)
    if len(self._recent_categories) >= 3:
        all_cats = [c for cats in self._recent_categories[-3:] for c in cats]
        if len(set(all_cats)) <= 1:
            # Inject diversity nudge
            diversity_nudge = (
                "\n\nWARNING: Your hypotheses have been in the same category "
                f"({set(all_cats).pop() if all_cats else 'unknown'}) for 3+ turns. "
                "You are likely stuck in a local optimum. FORCE yourself to generate "
                "hypotheses in at least 2 OTHER categories from: NAVIGATION, "
                "OBJECT_INTERACTION, VISUAL_PATTERN, GAME_MECHANIC, GOAL.\n"
            )
            # Append to prompt content
else:
    self._recent_categories = []
```

### 5. Strengthen visual correspondence detection in the system prompt

Add this specific instruction, because visual pattern matching is the mechanic the model most consistently misses:

```
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
```

## What NOT to change

- Don't modify `symbolic.py` or `vision.py`
- Don't change the Bayesian probability system from the main prompt — these changes layer on top
- Don't change the auto-probe logic — these changes are about the ongoing reasoning loop, not the initial probe
- Don't change the action description block — keep it as specified in the main prompt

## How to verify

Run against ls20 or any game with visual pattern mechanics. Check logs for:
- `scene_inventory` appearing in model output with objects, directions, and correspondences listed
- `untried` fields populated with things the model hasn't tested
- Hypotheses tagged with diverse categories (not all NAVIGATION)
- Forced reflection messages appearing every 8 turns
- Diversity nudge appearing when categories are monotonous
- Visual correspondences being noted when similar-looking objects exist
- The model generating hypotheses about going in directions it hasn't explored, even if its "instinct" is to keep going the current direction
