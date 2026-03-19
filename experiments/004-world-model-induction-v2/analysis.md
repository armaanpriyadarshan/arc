# Experiment 004: World Model Induction v2

**Date:** 2026-03-18 to 2026-03-19
**Game:** ls20 (LockSmith)
**Architecture:** Observe-act loop with auto-probe, symbolic state, hypothesis-driven action sequences (GPT-5.4)
**Best result:** Score 1 (level 1 completed at action 62), then exhausted budget on level 2
**Runs:** ~10

## Motivation

Experiment 003 identified two critical problems: (1) the model committed to wrong hypotheses on turn 1 and never revised, and (2) inconsistent initial interpretations across runs. It proposed four fixes: hypothesis revision, multiple hypothesis tracking, urgency from depleting resources, and consistency via fixed initial probing. This experiment implements all four.

## Architecture

### Overview

Single GPT-5.4 call per turn via the Responses API. The model receives symbolic state, symbolic diff, a side-by-side diff image, probe facts, verified rules, its own notes, and recent action history. It outputs structured JSON with observations, multiple hypotheses (each with status and test_actions), verified rules, and notes.

### Auto-probe (new in 004)

On startup and after each level-up, the agent takes one of each available action (typically ACTION1-4) with zero LLM calls. For each action, it records:
- Whether the action was BLOCKED or how many cells changed
- The symbolic diff (which objects moved, changed size, appeared, disappeared)

These results are injected as ESTABLISHED FACTS into every subsequent LLM call. The model never has to guess what actions do — it starts with ground truth about movement directions and their observable effects.

This directly addresses experiment 003's consistency problem. In 003, the model spent its first 2-4 actions testing directions and sometimes got them wrong ("ACTION1 might trigger interaction with nearby elements"). With auto-probe, every run starts with the same factual basis.

**Cost:** 4 actions per probe (8 if a probe causes GAME_OVER and requires a RESET). Across levels, this is 4-8 actions per level.

### Symbolic state (`symbolic.py`)

Connected component analysis on the 64x64 grid. Each component becomes an object with:
- Color (mapped to human-readable names), shape classification (rectangle, square, line, sparse_cluster, irregular)
- Size, position (top-left), bounding box, center
- Objects >1000 cells classified as "background" and reported separately

Spatial relations computed between foreground objects: above, below, left_of, right_of, adjacent. Objects sorted small-first so interesting game elements appear before large structural regions.

### Symbolic diff (`diff_symbolic`)

Matches objects across frames by color + closest center (with size tolerance). Reports:
- **changed:** center moved, size changed, bbox shifted, shape changed
- **disappeared:** object present before but not after
- **appeared:** new object in current frame
- **background_size_changed:** background region grew or shrank

No interpretation — just structural facts. The model decides what the changes mean.

### Hypothesis-driven action sequences (new in 004)

In experiment 003, the model output a single action per turn. Here, it outputs multiple hypotheses, each with a `status` field:
- `testing` — actively being tested, includes `test_actions` (a sequence of actions to execute)
- `confirmed` — evidence supports this claim
- `untested` — needs investigation
- `falsified` — evidence contradicts this claim

The first hypothesis with status `testing` drives execution. Its `test_actions` sequence runs automatically, stopping early on:
- BLOCKED (immediate course correction)
- GAME_OVER or WIN
- Level-up (triggers re-probe)
- Large changes (>100 cells beyond normal movement)

This batches 2-5 actions per LLM call instead of 1, reducing total LLM calls by roughly 3x compared to experiment 003.

### Verified rules (new in 004)

The model can output `verified_rules` — universal game mechanics confirmed through testing. These persist across levels and are presented as ground truth on every turn. Rules are filtered aggressively:
- Position-specific rules rejected (regex filters for coordinates, "at (x,y)", etc.)
- Substring deduplication (if new rule is a substring of an existing rule, skip)
- Word overlap deduplication (>70% word overlap with existing rule, skip)
- Hard cap at 10 rules

On level-up, confirmed hypotheses are promoted to verified rules before the hypothesis slate is cleared.

### Re-probe on level change

When `levels_completed` increases, the agent:
1. Promotes confirmed hypotheses to verified rules
2. Runs auto-probe again on the new level layout
3. Clears hypothesis and notes (level-specific state)
4. Keeps verified rules (cross-level mechanics)

### Circuit breaker

5 consecutive BLOCKED actions injects a warning into the action history: "You MUST try a completely different direction or area." Resets the blocked counter so the model gets another 5 tries before the next warning.

## Results

| Run | Score | Level 1 completed? | Actions used | Key behavior |
|-----|-------|-------------------|--------------|--------------|
| Best | 1 | Yes (action 62) | 100 | Stumbled into solution, collapsed on level 2 |
| Typical | 0 | No | 100 | Found switch, toggled repeatedly, never progressed |
| Worst | 0 | No | 100 | Misidentified player as "remote mechanism" |

**Best run:** Level 1 completed at action 62 (baseline is 29 actions, so 2.1x baseline). The model found the white switch and toggled it, but the completion came from stumbling into the right sequence rather than understanding the full mechanic. On level 2, the model couldn't navigate the new layout efficiently and exhausted the remaining 38 actions without scoring.

**Typical run:** The model identifies the white switch and discovers that interacting with it causes remote changes (gate rotation, cells reconfiguring). It then toggles the switch repeatedly — sometimes 5-10 times in a row — without ever connecting "switch toggled" to "this specific corridor is now passable, go through it now." It treats the switch as an object to study rather than a tool to use.

**Worst runs:** The model builds a fundamentally wrong world model early and never recovers. In one run, it decided the player sprite was a "remote mechanism" being operated by the actions, rather than a character it was controlling. This is a regression from experiment 003's best run but is consistent with 003's worst runs.

## Key findings

### 1. Reactive beats planned

A plan-based variant was tested: the model outputs 10-15 actions per LLM call and execution does not break on BLOCKED. This was substantially worse. The model's spatial reasoning is not accurate enough to plan multi-step routes through corridors. It would plan "go right 5 then down 3" and hit a wall on step 2, wasting the remaining 13 actions.

The reactive variant (break on BLOCKED, LLM decides next move) wastes fewer actions because every wall collision triggers immediate reconsideration. The tradeoff is more LLM calls, but each call produces a better action.

### 2. The corridor problem

The symbolic state describes OBJECTS (colored blobs with positions and shapes) but says nothing about CORRIDORS or WALKABLE SPACE. The model knows "there's a green object at [30, 45]" but has no information about whether a path exists from the player's current position to that object, or where that path goes.

The side-by-side diff image shows corridors visually, but GPT-5.4 cannot reliably extract precise spatial information from it — it can see "there's a corridor going right" but can't tell if it's 3 cells wide or 8, or exactly where it turns. This forces navigation by trial-and-error: try a direction, see if BLOCKED, try another.

This is the single biggest bottleneck. A human player can glance at the maze and plan a route. The model has to discover the route one step at a time.

### 3. The causal chain gap

The model consistently identifies that the white switch is a toggle and that toggling it causes "remote reconfiguration" — objects elsewhere on the grid change position or size. But it never completes the causal chain: **switch toggled -> gate rotates -> specific corridor opens -> go through it now.**

It sees "remote reconfiguration" as an abstract fact rather than an actionable opportunity. After toggling the switch, it should immediately check which paths changed and try to traverse them. Instead, it either toggles again (more data!) or goes back to exploring areas it already mapped.

This is the exploration-vs-exploitation problem from experiment 003, now manifesting at the mechanic level rather than the navigation level.

### 4. Simplicity wins

Multiple additions were tested to improve reasoning:

| Addition | Effect |
|----------|--------|
| Falsified hypothesis list | Model ignored it, bloated prompt |
| Dead-end tracking | Model ignored it, bloated prompt |
| Reasoning mode (`reasoning={"effort":"medium/high"}`) | Returned empty JSON, wasted 30-40s per call |
| Trigger probe (auto-probe 4 directions after each trigger) | Always returned "all OPEN" from player position, wasted 8 actions per trigger |
| Cause-effect fields in hypothesis schema | Model filled them generically, no improvement |
| Categorized unusual-change reporting (TRIGGERED/REMOTE) | Too game-specific, violated design principles |

The simplest version of the architecture — probe facts, symbolic state, symbolic diff, image, hypothesis list — performed best. Every addition either bloated the prompt (reducing reasoning quality on the core task), wasted actions on low-value probes, or introduced game-specific assumptions.

### 5. Rule deduplication is necessary

Without filtering, verified rules accumulate aggressively. One run had 65 duplicate rules, mostly paraphrases of "ACTION1 moves the player left." The deduplication system (substring matching + word overlap >70% + position-coordinate filtering + cap at 10) solved this completely.

### 6. Memory window causes re-exploration

The sliding window of the 10-15 most recent actions means the model loses information about earlier exploration. It rediscovers dead ends it already found 30 actions ago. The notes field partially compensates — the model can write "corridor blocked at [30, 20]" — but it rarely writes precise enough spatial notes to prevent revisits.

### 7. Energy depletion cascades

When the in-game energy/timer depletes, the level resets. This wastes 15-20 actions re-navigating from the start position to wherever the model had reached. Combined with the re-probe cost (4 actions), a single death can cost 20+ actions — a fifth of the total budget. The model notices the energy bar shrinking but never adjusts its pace.

## Comparison with prior experiments

### vs. Experiment 001

Experiment 001 failed entirely at perception: wrong background inference meant the player was invisible. Experiment 004's symbolic state and auto-probe completely solve this class of problem. The model always knows what objects exist and what each action does. The failure mode has shifted from "can't see" to "can see but can't plan."

### vs. Experiment 002

Experiment 002 tried increasingly complex architectures (three-model pipelines, grid memory layers, skill libraries) that all suffered from game theory hallucination — the model saw shapes resembling Sokoban and invented Sokoban mechanics. Experiment 004's auto-probe partially prevents this by giving the model empirical action-effect data on turn 1 rather than letting it theorize from images. However, hallucinated world models still occur in the worst runs.

### vs. Experiment 003

Experiment 003's best run completed level 1 in 34 actions. Experiment 004's best run completed it in 62 actions — slower, but with a more principled architecture. The key improvements from 003's recommendations:

- **Hypothesis revision:** Implemented via multiple hypothesis tracking with status fields. Partially works — the model does sometimes falsify and replace hypotheses. But it also falls into "switch toggle addiction" where it keeps testing a confirmed hypothesis instead of exploiting it.
- **Multiple hypothesis tracking:** Implemented. The model maintains 3-5 hypotheses per turn. But the first `testing` hypothesis dominates behavior, and the model rarely switches to alternatives.
- **Consistency via probing:** Implemented and effective. Auto-probe eliminates the "wrong first impression" problem for action mappings. The model always knows ACTION1=left, etc. However, higher-level interpretation (what is the player? what is the goal?) still varies between runs.
- **Urgency:** Not effectively implemented. Action count is shown but the model doesn't adjust strategy as budget depletes.

## What the next experiment should address

### 1. Spatial representation

The fundamental bottleneck is that the model has no persistent spatial map. It knows where objects are but not where paths go. Two approaches worth testing:

**Program synthesis for map building.** Let the model write Python code that processes the grid to extract walkable corridors, walls, and path connectivity. The model in experiment 003 iteration 5 wrote brilliant BFS code but couldn't convert paths to actions — this time, the code would output a text map (not action sequences) that the model reads for planning.

**Tile-type classification.** Code classifies each cell as wall/floor/object based on the grid, then generates a simplified text map showing corridors. This is close to hardcoding, but if the classification is purely based on connectivity (not color semantics), it stays game-agnostic.

### 2. Exploitation mode

The model needs a mechanism to shift from "discover mechanics" to "use mechanics." Once a switch-gate relationship is confirmed, the model should:
1. Toggle the switch
2. Immediately check which corridors changed (diff the symbolic state)
3. Navigate through the opened corridor before toggling again

This could be a prompt-level intervention ("once a mechanic is confirmed, your next action MUST exploit it") or a code-level one (after detecting a trigger event, inject "now navigate through the change" into the context).

### 3. Persistent spatial memory

Replace the sliding window with a structured spatial log. Instead of "ACTION3: 52 cells changed" disappearing after 15 actions, maintain a set of known blocked positions and known traversable paths. This doesn't need to be a full map — even "ACTION1 was BLOCKED when at position ~[30, 20]" would prevent re-exploration.

### 4. Budget-aware behavior

Inject urgency when the action budget is below 30%: "You have N actions remaining. Prioritize reaching the goal over gathering information." Currently the model explores at the same pace with 90 actions left as with 10.

## Conclusion

Experiment 004 validates the auto-probe + symbolic state + hypothesis-driven architecture as the strongest approach so far. It solves the perception and action-mapping problems from experiments 001-003 and provides a clean, game-agnostic framework for mechanic discovery. But it hits a hard ceiling at spatial reasoning: the model can discover what game objects do but cannot navigate efficiently between them or plan multi-step routes through corridors it can see in the image but cannot represent in its working memory.

The path forward is not more prompt engineering or additional LLM fields — experiment 004 proved that simplicity outperforms complexity in the prompt. The path forward is giving the model the ability to build its own spatial representations, likely through program synthesis: let the model write code that extracts corridor structure from the grid, producing a text-based map it can reason over precisely.
