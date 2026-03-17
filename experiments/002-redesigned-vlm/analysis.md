# Experiment 002: Redesigned VLM Agent

**Date:** 2026-03-15 to 2026-03-16
**Game:** ls20 (LockSmith)
**Result:** 0/7 levels across ~12 iterations
**Total runs:** ~12 with architectural changes between each

## What was tried

### Iteration 1: Pure VLM per-action (gpt-4o)
- One VLM call per action, multi-turn conversation
- VLM couldn't determine movement direction from 256×256 images
- Obsessively repeated ACTION2 near a "white cross" for 100+ actions
- 200 API calls, 504s runtime

### Iteration 2: VLM with side-by-side images + computed directions
- Added side-by-side before/after images during discovery
- Code attempted to compute movement direction from diff centroids
- VLM still couldn't determine ACTION3/4 mapping
- Plans were random ("go right and down")

### Iteration 3: Zero game knowledge prompts
- Stripped all game-specific language from prompts
- VLM identified visual elements but didn't understand movement
- "ACTION1 might trigger interaction with nearby elements"

### Iteration 4: Eyes + Brain + Hands (three models)
- gpt-4o (eyes) per-action, o4-mini (brain) periodic, gpt-4o-mini (hands) per-action
- Brain correctly confirmed ACTION1=up, ACTION2=down but couldn't confirm ACTION3/4
- 385 eye calls, brain stuck in direction-mapping loop for 10+ calls
- Brain eventually hallucinated (contradictory confirmed knowledge)

### Iteration 5: Brain outputs action sequences
- Removed per-action hands model
- Brain outputs 15-30 action sequences, executed without LLM calls
- Brain still stuck mapping ACTION3/4, fell back to systematic exploration 10/16 times

### Iteration 6: Code-based player tracking
- Attempted to find player by scanning for color clusters
- floor_color inference wrong (picked player body color as floor)
- Tracker found nothing, all directions "unknown", brain hallucinated shooter game

### Iteration 7: Raw diffs to reasoning model
- Sent raw numerical cell changes to o4-mini
- o4-mini returned empty (JSON parsing issue with response_format)
- Brain hallucinated "5×5 wavefront propagation" game from image alone

### Iteration 8: Hardcoded API input mappings + compositional architecture
- Used known API mappings (ACTION1=up, etc.) — not game-specific, it's the controller
- Pydantic memory DSL, skill library, hypothesis tracking, transition table
- Model correctly identified game elements but couldn't navigate
- Plans all "go to the colored block" but hit walls after 1-2 moves
- 200 actions, 1 death, 0 score

### Iteration 9: Larger images (12x scale, 768×768)
- **First time the model correctly identified and tracked the player**
- Confirmed "white plus sign is the player" and saw it move
- BUT: can't build a mental map, keeps hitting walls
- Stuck detection (3 consecutive similar outcomes) fires on successful movement too
- ~20 replans in 40 actions, each with 2 LLM calls = 40+ calls for 40 actions
- Model tries same blocked paths repeatedly from different angles

## Root cause analysis

### What works
- Larger images (768×768) let the VLM see the player and confirm movement
- Known API input mappings eliminate the direction-discovery problem
- Hypothesis tracking provides a structured reasoning framework
- Side-by-side comparisons help the model see what changed
- Stuck detection catches blocked-move loops
- Failed plan tracking prevents some repetition

### What doesn't work

**1. VLMs cannot do precise spatial reasoning from pixel images.**
The model sees a maze but can't extract: where are the walls? How wide are corridors? Where exactly is the player relative to the goal? It says "U-shaped wall" and "bottom-right colored block" but these descriptions are too imprecise to plan navigation. Every plan is a guess about how many steps to take in each direction.

**2. No persistent spatial model across plans.**
A human remembers "I went right 2 and hit a wall." The model doesn't retain this. Each plan is based on a fresh image that it can't accurately parse, so it repeats failed paths. The failed_plans list helps but the model doesn't connect "failed at step 10" with "I'm in the same place at step 34."

**3. Stuck detection threshold is wrong.**
3 consecutive 52ch moves triggers replan — but 52ch IS successful movement. The detector should only fire on BLOCKED moves (2ch), not on normal movement. This burns half the action budget on false re-plans.

**4. Too many LLM calls per cycle.**
Analyze (gpt-4o) + plan (o4-mini) = 2 calls per cycle. With stuck detection firing every 3-4 actions, that's 2 calls per 3-4 actions ≈ per-action calling. Need longer plan execution without interruption.

**5. The model theorizes before it has data.**
It immediately hypothesizes "the colored block is the goal" and tries to reach it, before understanding the map layout. A human would first explore systematically to build a map.

## Prioritized improvements for next iteration

### Implemented: Fix stuck detection (Priority 1)
Changed from "3 consecutive similar total_changes" to "5+ consecutive BLOCKED moves." Plans that are actually working no longer get interrupted.

### Implemented: Compressed grid as text (Priority 2)
Run-length encoded grid sent alongside image. Row 20 = `4×18 8×28 4×18` means "18 cells of color 4, 28 cells of color 8, 18 cells of color 4."

### Implemented: Three-layer memory
- **Layer 1 (GridMemory):** Per-cell change counts, color history, activity heatmap. Shows rows 45-49 are the movement corridor, rows 61-62 are the HUD.
- **Layer 2 (EpisodicBuffer):** Rolling window clustered into episodes. Correctly flags stuck loops and consecutive blocked moves.
- **Layer 3 (GameKnowledge):** Rules, hypotheses, skills. Updated by LLM on significant events.

### Result of iteration 11 (three-layer memory + grid text + fixed stuck detection)
100 actions, 0/7 levels, 0 deaths, 28 model calls (3 gpt-4o, 25 o4-mini), 653s.

**What worked:**
- Grid memory correctly identified the active area (rows 45-49), HUD (rows 61-62), and hotspots
- Episodic buffer correctly detected "12 consecutive blocked moves" and stuck loops
- Stuck detection no longer fires on successful movement
- The model correctly identified the player (white cross) in iteration 9-10

**What failed catastrophically:**
The model invented a completely wrong game theory. It thinks it's playing **Sokoban** (block-pushing puzzle) instead of a maze navigation game. Its entire strategy is:
- "Push the blue-orange block up the narrow central shaft into the goal box"
- "Rotate the block horizontal so it fits through the shaft"
- Skills like `tumble_block_down`, `rotate_block_horizontal`, `push_block_upshaft`

This is 100% hallucinated. The blue/orange sprite IS the player (confirmed in iteration 9-10), but the model now treats it as a pushable block. The white cross IS the player (also confirmed), but after a 138ch "large change" event (likely the energy bar refilling), the model's understanding reset.

**The model spent 100 actions trying to reach an object it misidentified, using mechanics it invented.**

### Contradictory knowledge problem
The rules section contains contradictions:
- "The start chamber only connects via a single tile opening at its bottom‐left corner" (conf=0.9)
- "The start chamber does not open at its bottom-left corner but instead at its bottom-right" (conf=0.6)
- "The starting chamber opens at its bottom‐middle tile" (conf=0.7)
- "The start chamber connects via a one‐tile opening in its eastern wall" (conf=0.6)

Four contradictory rules about the SAME thing, all with medium-high confidence. The dedup catches exact duplicates but not semantic contradictions. The model can't resolve these because it never actually tested them — it theorized from images and never systematically walked to each wall to find the opening.

### Systematic probe fallback loop
The first 12 actions were all "systematic probe" fallback (ACTION1, ACTION2, ACTION3, ACTION4 repeated 3 times). The o4-mini plan calls returned empty/unparseable 3 times in a row. The compressed grid text may be overwhelming the context — the full run-length grid is large.

## Open problems for experiment 003

### 1. Game theory hallucination
The model invents game mechanics from visual similarity to training data (sees rectangles → "Sokoban"). It needs to be forced to test theories before building on them. Currently hypotheses accumulate without testing because the model can't navigate to the objects it hypothesizes about.

### 2. Exploration before theorizing
The model immediately theorizes ("push block into goal") instead of first exploring ("what can I reach? where are the corridors?"). A human would walk around the starting area first. The model needs to establish what's reachable before planning paths to distant objects.

### 3. Grid text may be too large
The full run-length encoded grid is potentially thousands of tokens. It may be overwhelming the model's context or causing parse failures. Consider sending only the interesting rows (non-uniform rows near the active area) rather than the full grid.

### 4. Semantic contradiction resolution
The knowledge base accumulates contradictory rules with similar confidence. Need a mechanism to detect and resolve contradictions — either by testing, or by requiring new rules to explicitly supersede old ones.

### 5. The 138ch "large change" problem
Several times during the run, an action produces 138ch instead of the usual 52ch. This triggers scene re-description and replan, but the model doesn't understand what happened. In LockSmith this is likely the energy bar refilling or the player crossing into a new area. The model should investigate large changes rather than just replanning.

### 6. Skill pollution
The skill library accumulated 16 skills, most with placeholder actions like `navigate_to(north_of_block)` or `ACTION4×8` (not valid action names). These pollute the context without being usable. Skills should be validated before storage — only sequences of valid ACTION names.

## Scorecard references

- `832a07d8` — Experiment 001 (staged explorer, 0/7)
- `e82c315b` — Iteration 1 (VLM per-action, 0/7)
- `f592c0dc` — Iteration 2 (side-by-side + directions, 0/7)
- `812277eb` — Iteration 3 (zero game knowledge, 0/7)
- `a3c71835` — Iteration 4 (eyes+brain+hands, 0/7)
- `82b9baa0` — Iteration 5 (code tracker, 0/7)
- `cf62fee5` — Iteration 6 (raw diffs, 0/7)
- `df10633a` — Iteration 7 (compositional, 0/7)
- `02706eee` — Iteration 8 (known inputs, 0/7)
- `f1f39ba2` — Iteration 9 (larger images, 0/7)
- `08570829` — Iteration 11 (three-layer memory, 0/7)

## Key insights

1. **The spatial reasoning bottleneck is NOT just about images vs text.** Even with compressed grid text, the model can't navigate because it builds wrong game theories and never tests them. The bottleneck is the reasoning loop, not the perception format.

2. **The model conflates visual similarity with mechanical similarity.** It sees shapes that look like Sokoban and assumes Sokoban mechanics. It needs to be prompted to distinguish "what I see" from "what I assume" and test assumptions before acting on them.

3. **Grid memory (Layer 1) produces the most useful data.** Rows 45-49 being the most active, with hotspots showing colors 3 and 12 alternating, is exactly the player movement corridor. This is precise, game-agnostic, and came from zero LLM calls. The model should be taught to read this data.

4. **Episodic buffer (Layer 2) correctly identifies problems** (stuck loops, blocked streaks) but the model doesn't act on the warnings. "12 consecutive blocked moves" should trigger a fundamental strategy change, not just another "try going right."

5. **The baseline is 29 actions for level 1.** After 12 iterations and ~2000 total actions across runs, we haven't completed a single level. The architecture keeps getting more sophisticated but the fundamental action — "walk through a maze and interact with objects" — remains unachieved because the model can't build an accurate spatial model from its observations.
