# Experiment 002: Redesigned VLM Agent

**Date:** 2026-03-15 to 2026-03-16
**Game:** ls20 (LockSmith)
**Result:** 0/7 levels across ~10 iterations
**Total runs:** ~10 with architectural changes between each

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

### Priority 1: Fix stuck detection
The stuck detector should only fire on BLOCKED moves, not successful movement. Change from "3 consecutive similar total_changes" to "5+ consecutive moves where total_changes < 10 (blocked)."

**Impact:** Immediately doubles effective plan execution length. Plans that are actually working won't be interrupted.

### Priority 2: Send compressed grid as text alongside image
The 64×64 grid is structured data. Encode it compactly (run-length per row) and send as text. The model can then reason numerically about walls and corridors instead of guessing from pixels.

Example: Row 20 = `4×18, 8×28, 4×18` means "wall for 18 cells, corridor for 28 cells, wall for 18 cells."

**Impact:** Gives the model precise spatial truth. It can count cells to plan paths, identify corridor widths, locate objects by coordinate. This is how the upstream GuidedLLM works — it sends the full grid as text.

### Priority 3: Directed exploration before theorizing
Prompt the model to do systematic exploration first: "Walk in each direction until blocked to map the boundaries of your current area." This is game-agnostic (works for any grid game) and builds a spatial model before the agent commits to a strategy.

**Impact:** The model builds an accurate map from direct observation instead of guessing from images. Subsequent plans are grounded in actual wall positions.

### Priority 4: Grid query tools
Let the model REQUEST specific spatial information: "What values are in row 25, columns 20-40?" or "Is there a path from (25,30) to (35,40)?" The model decides what it needs to know, and code provides precise answers.

**Impact:** The model controls its own perception. It can verify hypotheses without burning game actions (querying the current frame is free).

### Priority 5: Better models for spatial reasoning
Claude (Sonnet/Opus) is significantly better than GPT-4o at structured spatial reasoning from text/numbers. If we send the grid as text, the reasoning model doesn't have to be OpenAI. The constraint is that `env.step()` goes through the arc-agi SDK which is model-agnostic — only the agent's brain needs an LLM.

**Impact:** Better spatial reasoning → better plans → fewer wasted actions.

### Priority 6: Reduce analyze+plan to single call
Merge the analyze and plan phases back into a single LLM call. The two-call pattern was supposed to improve quality but at the cost of 2x API calls. With better spatial data (Priority 2), a single call should be sufficient.

**Impact:** Halves LLM calls per cycle. Budget goes further.

### Priority 7: Accumulate a wall map
Every time a move is BLOCKED, record the direction and approximate position. Feed this "known walls" list to the model so it doesn't try blocked paths again. This is game-agnostic (any game with obstacles benefits).

**Impact:** The model stops repeating failed navigation attempts. Plans avoid known walls.

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

## Key insight

The game is fundamentally a spatial navigation problem. Level 1 baseline is 29 actions. The model needs to: find the player, understand the maze layout, identify the objective, and navigate there. We've solved "find the player" (iteration 9) but "understand the maze layout" remains unsolved because VLMs can't extract precise spatial structure from images. The most promising path forward is supplementing images with structured grid data (text) so the model can reason numerically about positions and paths.
