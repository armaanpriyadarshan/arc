# Experiment 004: DreamerV3 Agent for ARC-AGI-3

## Hypothesis
Model-based RL (DreamerV3) can learn a world model of the environment dynamics, then train an actor-critic on imagined trajectories to solve LockSmith (ls20).

## Architecture
- **World Model**: CNN encoder (16,64,64 -> 1024) + RSSM (h=256, z=8x8=64 categorical) + CNN decoder + reward/continue heads
- **Actor-Critic**: Actor (5 actions: RESET + 4 directional) + symlog critic, trained on imagined rollouts
- **Reward Shaping**: Position-based novelty + state novelty + direction diversity + level completion bonus
- **RND** (run 5): Random Network Distillation for intrinsic curiosity reward

## Runs

### Run 1 (200 iters, default config)
- **Bug found**: KL loss stuck at 10.4 (free_bits=1.0 too high -> stochastic state completely unused)
- Reconstruction loss: 2.77 -> 0.05 (world model learns well)
- **Result**: 0 levels completed

### Run 2 (500 iters, free_bits=0.1, novelty reward)
- KL dropped to 1.04 (correct free_bits floor)
- Returns variable (13.4 -> -1.1 -> 11.0) due to novelty exhaustion
- **Result**: 0 levels completed

### Run 3 (500 iters, position-tracking reward)
- Position tracking via change centroid gives better exploration signal
- Returns: 18.3 early (lots of new positions) -> declining as novelty exhausts
- **Result**: 0 levels completed

### Run 4 (500 iters, systematic wall-following exploration)
- **Discovery**: Episodes terminate at ~129 steps (game has energy limit)
- Mixed systematic/random exploration seeds buffer with better coverage
- Reward shaper attached before exploration (was missing in earlier runs)
- **Result**: 0 levels completed

### Run 5 (500 iters, DreamerV3 + RND intrinsic motivation)
- RND predictor loss drops to 0 very quickly — not enough state diversity for RND to be useful
- Mean imagined return ~1.0 (from RND curiosity) but doesn't translate to real task progress
- **Result**: 0 levels completed

## Key Findings

### What Works
1. **World model reconstruction**: recon_loss reliably drops from 2.77 to ~0.03. The CNN encoder/RSSM/decoder accurately models 64x64 grid dynamics.
2. **Training stability**: All components train stably across 500+ iterations with no NaN or divergence (after fixing RND normalization).
3. **Local environment**: Successfully runs against local ls20 environment (fast, no API calls).
4. **Systematic exploration**: Wall-following explores more of the map than random actions.

### What Doesn't Work
1. **Reward too sparse for RL**: LockSmith gives 0 reward except at level completion. No amount of shaped rewards (novelty, position, RND) creates a gradient toward the goal.
2. **Exploration never finds the goal**: With 4 directional actions in a 64x64 maze, random/systematic exploration doesn't discover key→door sequences within the ~129-step energy limit.
3. **KL collapse**: Stochastic state stays at free_bits floor because environment transitions are deterministic. The RSSM's stochastic component adds nothing.
4. **Imagination without reward signal**: DreamerV3's advantage (dreaming) is useless when reward is near-zero in all imagined trajectories.
5. **RND too fast to converge**: The predictor learns the small observation space quickly, providing no lasting exploration bonus.

## Root Cause

**This is a reward discovery problem, not an architecture problem.** DreamerV3 assumes environments with semi-frequent reward signals (Atari gives score updates). ARC-AGI-3 requires:
- Discovering game mechanics through experimentation (no instructions)
- Multi-step goal planning (find key → navigate to door)
- All within ~129 actions (energy limit)
- With zero intermediate reward

No amount of world-model quality fixes the fact that the policy has nothing to optimize toward.

## What Would Actually Help

Based on this experiment and prior experiments (001-003):

1. **Perception-first approach** (experiments 001-003 direction): Code-based object detection to identify player, walls, key, door. Then pathfinding to navigate. RL is overkill when the problem is perception + planning.

2. **LLM-in-the-loop**: Use an LLM to look at the grid (as an image), hypothesize what objects exist and what the goal might be, then generate an exploration strategy. The LLM provides the "game understanding" that RL can't discover.

3. **Test-time training** (NVARC approach): Train a small model at test time specifically for the game being played. Use the first N actions to learn game dynamics, then exploit.

4. **Action-effect prediction** (1st place StochasticGoose insight): Instead of a full world model, predict which actions cause state changes and which don't. This is much simpler and more directly useful.

5. **Hierarchical exploration**: First explore to build a map, then plan paths to interesting objects, then try interacting. The hierarchy is: map → objects → interactions → goal.

## Technical Lessons

- `OPERATION_MODE=local` is essential for fast iteration
- LockSmith only uses ACTION1-4 (directional), no ACTION5-7
- Game has ~129-step energy limit causing GAME_OVER
- `available_actions` from the SDK returns int values (1-4), not GameAction enums
- KL free_bits = 1.0 is too high for this domain; 0.1 works but KL still collapses to the floor because transitions are deterministic
- RND reward normalization must be simple (tanh, not running stats) to avoid NaN

## Perception Agent (Runs 6-9)

After 5 DreamerV3 runs yielded 0 levels, pivoted to a code-based perception agent.

### Architecture
- **Calibration**: Test each action to learn movement directions (UP/DOWN/LEFT/RIGHT = ±5 pixels)
- **Player tracking**: Track player position via diff centroid of non-background cells
- **Map building**: BFS exploration, wall detection via no-op actions
- **No neural networks**: Pure algorithmic exploration

### Key Results
- **Run 6 (v1)**: Calibration failed — couldn't detect player (single-step diff too noisy)
- **Run 7 (v2)**: Player detected at (42,41), 3/4 directions learned, 15 cells visited
- **Run 8 (v3)**: 3 directions + inference for 4th, 42 cells visited before GAME_OVER
- **Run 9 (v4)**: **1 LEVEL COMPLETED** in 500 actions! All 4 directions calibrated, 77 cells visited

### Why It Works
1. **Direct perception** beats learned representations for this domain
2. **Systematic BFS** covers more ground than policy-based exploration
3. **No training needed** — immediate exploitation of game structure
4. **Robust to resets** — the agent reconstructs its state after GAME_OVER

## Conclusion

| Approach | Levels Completed | Actions | Key Insight |
|----------|-----------------|---------|-------------|
| DreamerV3 (5 runs) | 0/7 | 500 each | World model learns but policy has no reward gradient |
| DreamerV3 + RND | 0/7 | 500 | Intrinsic curiosity exhausts too quickly |
| Perception Agent | **1/7** | 500 | Code-based perception + BFS succeeds where RL fails |

**The fundamental lesson**: ARC-AGI-3 is a perception + planning problem, not an RL problem. The game's sparse rewards, deterministic transitions, and multi-step goal structure make it hostile to policy gradient methods. Code-based perception that directly analyzes the grid is far more effective than learning representations from scratch.

## Next Steps
- Optimize perception agent to complete more levels within the action budget
- Add object interaction logic (key pickup, door opening)
- Improve navigation efficiency (A* instead of BFS, avoid revisiting)
- Test on other games (ft09, vc33) to verify generalization

## References
- Experiments 001-003: Prior approaches with code-based perception + LLM reasoning
- StochasticGoose (1st place ARC-AGI-3 Preview): CNN + RL, action-effect prediction, 12.58%
- DreamerV3 paper: Hafner et al., "Mastering Diverse Domains through World Models" (2023)
- RND paper: Burda et al., "Exploration by Random Network Distillation" (2018)
