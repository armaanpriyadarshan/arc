# Experiment 003: Observe-Act with GPT-5.4

**Date:** 2026-03-17
**Game:** ls20 (LockSmith)
**Architecture:** Observe-then-act loop using GPT-5.4 Responses API
**Best result:** 1/7 levels in one run, 0/7 in subsequent runs

## Architecture

Single GPT-5.4 call per turn. The model receives:
- Symbolic state (connected components with colors, shapes, positions, relations)
- Symbolic diff (what changed since last turn — no interpretation, just facts)
- Side-by-side image (previous frame vs current with red outlines on changed cells)
- Its own persistent notes and hypothesis from the previous turn
- Recent action results

It outputs JSON with: observation, hypothesis, notes, and one action.

Earlier iterations in this experiment tried: pure tool-use (o4-mini with grid query tools), program synthesis (model writes Python), code-driven navigation with LLM strategy, internal-model architecture. The observe-act pattern with GPT-5.4 + symbolic state was the final and most successful iteration.

## What worked

### Best run: Level 1 completed in 34 actions

GPT-5.4 produced the best reasoning of any model across all experiments:
- **Turn 1:** Correctly identified maze layout, player sprite, blue marker, orange block from image + symbolic state
- **Turns 2-4:** Systematically tested each action, correctly deduced ACTION1=left, ACTION2=right, ACTION3=up, ACTION4=down
- **Turn 9:** Hit a wall, correctly noted "blocked" and changed direction
- **Turn 25:** Passed through the white cross, noticed it disappeared and a new one appeared — identified it as a collectible
- **Turn 34:** Completed level 1 (1589 cells changed = level transition)

The symbolic diff was crucial. When the model saw `{"color": "blue", "type": "changed", "center": {"was": [48,41], "now": [44,41]}}`, it immediately understood "the blue/orange block shifted position" rather than hallucinating "nothing moved" like all previous experiments.

### Observation quality

GPT-5.4's observations were consistently accurate and specific:
- "The orange/blue player block successfully moved 5 cells left, now at orange bbox [40,39]-[41,43]"
- "The right-side yellow vertical strip retracted upward by 1 cell"
- "Moving RIGHT advanced the blue player from about (28,21) to (33,21), passing through/over the white cross"

Compare with o4-mini which said things like "52 cells changed, likely just a counter" or gpt-4o-mini which said "ACTION2 is underexplored."

## What failed

### Inconsistency across runs

The model's interpretation of the same game varies wildly between runs:
- **Run A (good):** "The orange/blue block is the controllable player moving in 5-cell steps" → completed level 1
- **Run B (bad):** "The white cross is the player and the actions control a remote block/platform" → score 0
- **Run C (bad):** "This is a configuration puzzle where actions toggle a machine state" → score 0

Same game, same symbolic state, same images. The model picks a different interpretation each time and commits to it for the entire run. Once it decides "the white cross is the player" (wrong), it never revises despite the white cross never moving.

### No error correction

When the model's hypothesis is wrong, it doesn't self-correct. In the bad runs:
- The model saw the white cross stay at (31,21) for 10+ turns and concluded "the player is stationary, this must be a control panel puzzle"
- It never asked "what if the MOVING thing is what I control?"
- The symbolic diff clearly shows the blue/orange block changing position every turn, but the model ignores this because it already decided the white cross is the player

### Level 2 exploration without goal pursuit

Even in the successful run, after completing level 1 the model spent 22 actions (35-56) systematically exploring level 2 without heading toward the goal. It knew the white cluster at (46,51) was "probably the goal" but kept testing every direction at every node instead of planning a path.

### Energy/timer blindness

The model tracks the yellow bar shrinking and correctly identifies it as "a timer or progress indicator" but never connects "timer depleting" with "I need to hurry." It explored at the same pace whether it had 80 actions left or 20.

## Architecture evolution within experiment 003

| Iteration | Architecture | Result | Key issue |
|-----------|-------------|--------|-----------|
| 1 | Tool-use (o4-mini + grid query tools) | 0/7, 100 actions, 1396s | Over-inspected, never acted purposefully |
| 2 | Hybrid (context + tools) | 0/7, took forever | Tool call truncation, context overflow |
| 3 | Code-driven navigator + LLM strategy | 1/7 (luck), 83s | Navigator oscillated, LLM gave wrong coordinates |
| 4 | Internal-model (LLM maintains text map) | 0/7, 336s | Model thinks "52ch" means "collected 52 coins" |
| 5 | Program synthesis (model writes Python) | 0/7, crashed | Brilliant code (BFS, tile grid) but couldn't convert paths to actions |
| 6 | Observe-act with gpt-5.4-mini | 0/7 | Can't see images (text-only model) |
| 7 | Observe-act with GPT-5.4 + symbolic state | 1/7 (best), inconsistent | Best reasoning but wrong initial interpretation kills the run |

## Key insights

1. **GPT-5.4 + symbolic state is the right combination.** The model can reason precisely when it has structured data (object positions, colors, spatial relations) alongside visual context. The symbolic diff is essential — it tells the model exactly what changed without requiring pixel comparison.

2. **The bottleneck is not perception or reasoning — it's commitment to wrong hypotheses.** The model picks an interpretation on turn 1 and never revises it. If it guesses wrong (white cross = player), the entire run is wasted. There's no mechanism for the model to say "my hypothesis has been wrong for 10 turns, let me reconsider from scratch."

3. **Exploration vs exploitation is unsolved.** The model explores systematically but never switches to goal-directed behavior. It needs to learn when it has "enough" information and start executing.

4. **Single-call-per-turn with persistent notes works well.** No conversation accumulation, no trimming bugs, no context overflow. The notes carry forward the model's understanding cleanly.

5. **Speed is acceptable.** ~6 seconds per turn with GPT-5.4. 100 actions in ~10 minutes. Fast enough for experimentation.

## What experiment 004 should address

1. **Hypothesis revision mechanism.** If the model's hypothesis hasn't led to score progress after N actions, force it to reconsider. Perhaps: "Your hypothesis has been unchanged for 15 actions with no score increase. Propose an alternative interpretation."

2. **Multiple hypothesis tracking.** Instead of committing to one theory, maintain 2-3 alternative interpretations and test the most promising one. If it fails, switch to the next.

3. **Urgency from depleting resources.** The model should be told when its resource bar is low, not just that it exists. The symbolic diff shows the yellow bar shrinking — this should trigger more aggressive goal-pursuit.

4. **Consistency across runs.** The same game should produce the same level of reasoning. The model's first-turn interpretation is too sensitive to the random initial observation. Perhaps: run the first 4 actions (one of each direction) as a fixed probe before the model starts reasoning, so it has consistent initial data.

## Scorecard references

- Best run: `d5362cf3` — 1/7 levels, 100 actions, 83s (code-driven navigator iteration)
- GPT-5.4 run: scored 1 level in 34 actions before running out of budget on level 2
