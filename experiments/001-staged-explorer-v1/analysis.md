# Experiment 001: Staged Explorer v1

**Date:** 2026-03-15
**Agent:** ExplorerAgent (staged: sensorimotor → object discovery → causal → goal inference → planning)
**Game:** ls20 (LockSmith)
**Result:** 0/7 levels, 196 actions, 1 death, 7 LLM calls, 122s

## Scorecard

- Scorecard ID: `832a07d8-e387-47e6-9eaf-a2f2cf1099d3`
- Score: 0.0
- Levels completed: 0/7
- Actions used: 196 (baseline for level 1 is 29)
- Resets: 1 (died from energy depletion)
- Final state: GAME_OVER

## What happened

### Stage 1: Sensorimotor (12 actions)
- ACTION1-4 all produce exactly 52 cell changes
- ACTION5 produces 0 changes
- ACTION6 errors (500 from API — LockSmith doesn't support click actions)
- **Movement direction: None for all actions.** The `_estimate_movement()` function failed to determine any directions

### Stage 2: Object Discovery (8 actions)
- Inferred background colors: `[3, 4]` — **WRONG**
  - Color 3 = neutral gray, color 4 = off-black (player body / dark elements)
  - Real background: color 0 (white/transparent) and color 8 (floor/orange)
- Found 14 objects, 3 marked "controllable"
- The player body (color 4) was invisible because it was classified as background
- "Controllable" objects were actually the energy bar (changes every turn) and other noise

### Stage 3: Causal Testing (20 actions)
- Every movement action: 52 changes, score stays 0
- One wall collision detected: ACTION3 → 2 changes (good!)
- 20 transitions recorded, but all say the same thing: "moderate change, no score change"
- No useful causal relationships discovered

### Stage 4: Goal Inference (1 LLM call)
- Hypothesis: "collect or interact with objects of color 12 or 13" — **hallucinated**
- The LLM got a list of 14 objects with no spatial context, no player identification, no movement model
- It had nothing to work with and made something up

### Stage 5: Planning (159 actions, 7 LLM calls)
- Plans were all "go right and down" variants, targeting color 12/13 objects at wrong coordinates
- Died once from energy depletion around action 100
- After reset, continued wandering until budget exhausted
- No score progress at any point

## Root cause analysis

**The entire failure cascades from perception.** Two specific bugs:

1. **Background inference used `>15%` frequency threshold.** This incorrectly classified colors 3 and 4 as background. Color 4 is the player's body. The player became invisible to the object detector.

2. **Movement estimation returns None on every action.** The `_estimate_movement()` function tries to find cells where a non-background color "appeared" vs "disappeared" — but since background colors are wrong, the heuristic fails. Result: the agent doesn't know ACTION1=up, ACTION2=down, etc.

Without knowing where the player is or what direction actions move it, no downstream stage can function:
- Object discovery can't identify the player
- Causal testing can't test meaningful interactions (it doesn't know it's near objects or walls)
- Goal inference gets garbage input and hallucinates
- Planning has no world model and wanders randomly

## What experiment 002 should fix

1. **Don't infer background from frequency.** Instead, use the sensorimotor diffs: cells that NEVER change across any of the 12 actions are structural (walls, borders). The rest is foreground. This is data-driven and game-agnostic.

2. **Compute movement direction from diff centroids.** The 52 changed cells per action ARE the movement signal. Split them into "cells that gained the player color" vs "cells that lost the player color" to get direction. Or simpler: compute the centroid of all changed cells in the before vs after frame.

3. **The player IS the moving entity.** After sensorimotor, the cells that move consistently with every ACTION1-4 are the player. No need for connected-component detection — just union the diff masks.

4. **Pass actual spatial context to the LLM.** The goal inference and planning prompts should include: where the player is, what's nearby, what direction each action moves, what objects are reachable. Not just a flat object list.
