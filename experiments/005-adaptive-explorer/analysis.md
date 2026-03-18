# Experiment 005: Adaptive Explorer Agent

## Hypothesis
An agent that discovers game mechanics through interaction testing and systematic combination search can solve more levels than pure BFS exploration (exp 004: 1/7 levels).

## Approaches Tested

### v1-v2: Interaction Discovery + Display Tracking
- Scan grid for objects, navigate to each, record before/after changes
- Detect "display region" (cells that change when stepping on modifiers)
- **Result**: 0 levels. Too many false positives (player movement detected as interactions). Analysis consumed all action budget.

### v3: Object-Directed Exploration with Randomized Ordering
- Grid scan for objects, visit each in shuffled order per life
- **Result**: 0 levels. Objects visited once each — game requires multiple visits to cycle modifier states.

### v4: Step-On-Step-Off Display Tracking + Modifier Characterization
- Detect display changes by stepping on/off objects
- Found 5 modifier types, detected cycles (length 4, 5)
- **Result**: 0 levels. All 500 actions consumed by analysis phases, none left for exploration.

### v5: BFS with Piggybacked Learning (best)
- Minimal calibration (~20 actions), then pure BFS exploration
- Detect interesting positions during BFS (zero extra actions)
- After GAME_OVER, restart BFS from reset position
- **Result**: **1/7 levels** at action 348 (3 lives). Level 2 failed after 640+ actions.

### v6: Exploration + Systematic Combination Search
- BFS exploration to find modifier groups, then odometer-style combo search
- Each combo: reset → visit modifiers N times → BFS to trigger locks
- **Result**: 0 levels with 1000 actions. Combo search too slow — only tested 7 of 343 possible combinations.

### v7: Randomized Frontier Exploration
- Random frontier selection (50% nearest, 50% random) instead of BFS
- Different random seeds per life for exploration variety
- **Result**: 0 levels in 2000 actions (16 lives). Random selection causes inefficient navigation between distant cells.

## Key Findings

### What Works
1. **BFS exploration** is the most efficient coverage strategy (~35-45 cells per 128-step life)
2. **Piggybacked learning** (detecting interactions during movement) costs zero extra actions
3. **Calibration** (learning movement directions) is reliable and fast (~20 actions)
4. **Level 1** (baseline 29 actions) is consistently solvable via BFS in ~350 actions

### What Doesn't Work
1. **Separate analysis phases** consume too many actions — by the time analysis is done, no budget for exploration
2. **Combination search** is exponentially expensive — 3 modifier types × 6 cycle lengths = 216+ combinations at ~90 actions each
3. **Randomized exploration** is worse than BFS — random jumps waste steps on navigation overhead
4. **Display region detection** is unreliable — player movement changes many cells, hard to isolate display-only changes

### Root Cause of Level 2+ Failure
ARC-AGI-3 puzzle games require **multi-step planning**:
1. Observe which modifiers affect which aspects of the display
2. Read what each lock requires (visual pattern matching)
3. Calculate the modifier visit sequence to match each lock
4. Execute the sequence: visit modifiers → walk to lock → repeat for next lock

Pure exploration (BFS, random) can accidentally solve simple levels (1 lock) but fails on multi-lock levels because the probability of hitting the right modifier sequence AND approaching the right lock decreases exponentially with level complexity.

## Comparison with Experiment 004

| Approach | Best Result | Actions | Key Insight |
|----------|------------|---------|-------------|
| DreamerV3 (exp 004) | 0/7 | 500 | RL has no reward gradient |
| Perception BFS (exp 004) | 1/7 | 500 | BFS + calibration works for level 1 |
| Adaptive Explorer v5 | **1/7** | 500 | BFS w/ piggybacked learning, same result |
| Adaptive Explorer v6 | 0/7 | 1000 | Combo search too slow |
| Adaptive Explorer v7 | 0/7 | 2000 | Random exploration worse than BFS |

## What Would Actually Help (Next Experiment)

1. **Visual pattern matching**: Compare the current display state to each lock's visual pattern. This requires extracting the display sprite and lock sprites as normalized patterns, then matching them.

2. **Modifier-lock association**: When a modifier changes the display, record how it changed. When approaching a lock, compare the display to the lock pattern. If they match, step on the lock. If not, backtrack and visit different modifiers.

3. **Incremental lock solving**: Solve one lock at a time. After each lock is removed, re-scan for remaining locks and adjust strategy.

4. **LLM-assisted reasoning**: Use an LLM to look at the grid as an image, identify modifiers and locks, and plan the visit sequence. The LLM provides the "game understanding" that algorithmic approaches lack.

5. **State hashing + replay**: Hash the grid state after each action. When a state matches a previous level-completing state, replay the same actions.

## Technical Lessons
- `died_flag` pattern needed for GAME_OVER detection (env_wrapper resets player position, masking death)
- BFS frontier deque gives FIFO order which is good for coverage efficiency
- Random frontier selection causes O(n) navigation overhead per cell
- Grid scan via flood fill finds ~14 objects on level 1, many are wall segments
- Energy limit is consistently ~128 steps per life on level 1
- The game resets modifier state on death (3 lives per level attempt)
