# Experiment 007: DAG Reasoning

**Date:** 2026-03-20
**Game:** ls20 (LockSmith)
**Architecture:** Conversational sliding window (10 messages) with GPT-5.4 structured outputs, DAG reasoning chain (q0-qa), mandatory code execution (q0, q6), auto-probe with action_effects extraction, memory bank, enhanced symbolic state
**Best result:** Score 1 (level 1 completed), 34 LLM calls, 100 actions
**Runs:** Multiple

## Motivation

Experiments 004-005 converged on the same finding: the model understands game mechanics but cannot maintain or act on spatial knowledge. Experiment 004's conclusion called for program synthesis as mandatory infrastructure rather than an optional tool, and experiment 005 confirmed that the model will not spontaneously write strategic code. Two specific gaps persisted: (1) the model navigates by trial-and-error because it has no route verification, and (2) direction confusion (ACTION1=left vs. right) caused wasted actions in early experiments.

This experiment tests two interventions. First, a DAG reasoning chain with 10 nodes (q0 through qa) that forces systematic analysis before action selection — including mandatory code nodes for perception (q0) and route planning (q6). Second, structured outputs via GPT-5.4's JSON schema mode, which should eliminate the JSON parsing failures that occasionally corrupted responses in experiments 004-005.

Additionally, the symbolic state is enhanced from experiment 004's version with sub-pattern classification, orientation detection, symmetry analysis, hole detection, and multi-color composite grouping. Action effects (movement deltas) are extracted during auto-probe and injected into the sandbox as `action_effects`, and dict-style coordinates (row/col labeled) replace the ambiguous list format from prior experiments.

## Architecture

### Conversational sliding window

Same as the conversational approach tested in experiment 006: the agent maintains a rolling conversation with GPT-5.4 using the Responses API's `input` parameter. A sliding window of 10 messages is sent per call, always preserving the first user message (which contains probe facts). The model can reference its own prior reasoning in the message history without explicit note injection.

### DAG reasoning chain

The response schema enforces a 10-node directed acyclic graph:

- **q0_code** (root, mandatory): Python code for grid analysis. Has access to `grid`, `ROWS`, `COLS`, `memory_bank`, `action_effects`. Must set `result`.
- **q1_objects** (depends on q0): Object inventory — name, position, provides, requires.
- **q2_last_action** (root): Last action analysis.
- **q3_requirements** (depends on q1): Per-object interaction requirements check.
- **q4_action_result** (depends on q2): Success/failure analysis of last action.
- **q5_subtasks** (depends on q1, q3): Top 3 prioritized sub-tasks.
- **q6_planning_code** (depends on q5, mandatory): Python code to plan and verify actions — simulate moves against the grid, check traversability.
- **q7_candidate_actions** (depends on q5, q6): Top 5 candidate actions with purpose.
- **q8_feasibility** (depends on q7, q3): Feasibility check for each candidate.
- **qa_actions** (depends on q7, q8): Final action sequence to execute.

Plus `verified_rules` (universal game rules, max 10).

### Structured outputs

GPT-5.4's `text.format` parameter with `type: "json_schema"` and `strict: True`. This guarantees the response conforms to the schema — no JSON parsing failures, no missing fields, no hallucinated field names.

### Enhanced symbolic state

The symbolic state from experiment 004 is extended with:
- **Sub-pattern classification:** Template matching against known shapes (L, T, S, Z, cross, etc.) with all 8 orientations, plus topological fallback for small unmatched shapes.
- **Orientation detection:** PCA-based dominant axis with compass direction (N, NE, E, etc.).
- **Symmetry detection:** Horizontal mirror, vertical mirror, 90-degree and 180-degree rotation symmetry.
- **Hole detection:** Flood-fill from bounding box border identifies enclosed voids.
- **Canonical rotation ID:** MD5 hash invariant to rotation for cross-frame matching.
- **Multi-color composites:** Adjacent objects of different colors grouped via dilation-based union-find.
- **Containment relations:** Strict bounding-box containment between objects.

### Other components carried forward

- Auto-probe with action_effects extraction (from 004)
- Memory bank persisting across levels in sandbox (from 005)
- Action history summary injected every turn
- Image only on first turn and after triggers (>100 cell changes)
- Break on BLOCKED (reactive loop from 004)
- Circuit breaker after 5 consecutive blocked
- Re-probe on level change

## Results

| Metric | Value |
|--------|-------|
| Best score | 1 (level 1 completed) |
| LLM calls (best run) | 34 |
| Actions used | 100 |
| BLOCKED actions (best run) | 11 out of 100 |
| Time (best run) | ~18 minutes |
| Time (worst config) | ~43 minutes |
| Tokens per response | ~3000 |
| Response latency | ~30 seconds per call |
| Actions per LLM call | 3-5 (batched) |

### What worked

**q6 verification is the most impactful single addition.** In experiments 004-005, roughly 31 out of 100 actions were BLOCKED — the model planned routes through walls it couldn't see. With q6 mandatory verification code, BLOCKED actions dropped to 11 out of 100. The model writes 2000-3000 character Python programs that check whether planned moves land on traversable cells (color 3) vs. walls (color 4). This catches most bad routes before execution.

**Structured outputs eliminated JSON parsing.** Zero parse failures across all runs when reasoning effort is set to "none." In experiments 004-005, approximately 5-10% of responses had JSON issues requiring fallback parsing. This is a clean solve.

**Action batching improved efficiency.** The DAG's qa_actions node outputs 3-5 actions per call. Combined with the sliding window, total LLM calls dropped from ~88 (experiment 004, one action per call in many cases) to 34. Each call is slower (~30s vs. ~10s) due to the 3000-token response, but wall-clock time is comparable.

**Dict-style coordinates fixed direction confusion.** Prior experiments used list-format coordinates `[row, col]` which the model frequently confused with `[x, y]`. Switching to `{"row": R, "col": C}` eliminated direction confusion entirely. ACTION1-4 mappings are correct in every run.

**Code execution is reliable when mandatory.** Both q0 and q6 execute every turn. The model writes substantial analysis code (2000-3000 chars) that correctly identifies traversable regions, computes distances, and verifies routes. This validates experiment 005's conclusion that code must be mandatory, not optional.

### What didn't work

**Strategic target selection remains wrong.** Across multiple runs, the model fixates on the top-right blue object instead of navigating to the lower-left goal. The q6 verification catches bad routes to the wrong destination but cannot catch wrong destinations. The model's strategic decisions (which target to pursue) are the bottleneck, not its tactical execution (how to get there).

**Reasoning effort above "none" breaks structured outputs.** Setting reasoning to "medium" or "high" causes GPT-5.4 to violate the JSON schema — missing fields, wrong types, or truncated output. Only `reasoning={"effort": "none"}` reliably produces schema-compliant responses. This is a significant limitation: the model cannot use its internal reasoning chain when constrained to structured output.

**The full DAG is expensive.** 10 nodes at ~3000 tokens per response means ~30 seconds per call. Over 34 calls, that is 17 minutes of pure API time. With overhead, total runtime is 18-43 minutes depending on configuration. For a 100-action budget, this is too slow for competition use.

**Enhanced symbolic features are unused.** Sub-patterns, orientation, symmetry, composites, and containment are computed every turn and included in the scene description. The model rarely references them. It uses object color, position, and size — the same features available in experiment 004. The enhanced features add prompt size without adding reasoning value.

**Optional code is never written.** When code fields are optional (as in the hypothesis format from 005), the model never uses them. q0 and q6 are mandatory and produce valuable code. Any field not forced by the schema remains empty.

## Key findings

### 1. Structured outputs solve JSON reliability (when reasoning=none)

GPT-5.4's strict JSON schema mode produces valid, complete JSON on every call — but only when reasoning effort is "none." Any reasoning effort ("low", "medium", "high") causes schema violations. This is a hard tradeoff: schema compliance or internal reasoning, not both. For agents that need reliable structured output, reasoning must be disabled.

### 2. The DAG forces systematic reasoning but adds massive overhead

The 10-node DAG ensures the model considers objects, requirements, subtasks, feasibility, and verification before selecting actions. This produces measurably better tactical decisions (fewer BLOCKED actions). But each response is ~3000 tokens and takes ~30 seconds, versus ~800 tokens and ~10 seconds for the simple format in experiment 004. The question is whether the quality improvement justifies the 3x overhead.

### 3. q6 verification is the single most impactful addition

BLOCKED actions dropped from 31% to 11% — a 65% reduction. The model writes Python that simulates its planned route against the actual grid, checking cell colors against known traversable/obstacle classifications. This is exactly the "mandatory code for spatial reasoning" that experiments 004-005 called for. It works.

### 4. Conversational memory helps but 10 messages is insufficient

The sliding window preserves the first message (probe facts) and the last 9 messages. This gives the model ~5 turns of context. It successfully references its own prior observations within this window, but older context is lost. The model rediscovers dead ends and re-analyzes objects it characterized 20 actions ago. A larger window would help but increase token costs.

### 5. Strategic decisions are the bottleneck, not tactical execution

The model navigates to its chosen target efficiently (few BLOCKED actions, good route verification). But it chooses the wrong target. Across multiple runs, it fixates on a visually salient blue object in the top-right instead of pursuing the actual goal in the lower-left. No amount of route verification can fix wrong destination selection. This is a higher-level reasoning problem that the DAG's structure doesn't address.

### 6. Code synthesis works when mandatory, never when optional

q0 (perception) and q6 (planning) are mandatory schema fields. The model writes 2000-3000 character programs that do genuine spatial analysis — BFS for reachable regions, traversability classification, path verification. When code is an optional field (as in experiment 005), the model never uses it. The conclusion from experiment 005 is confirmed: if you want the model to write code, force it.

### 7. Speed is a competition concern

At 18-43 minutes per 100-action run, this architecture is too slow for the ARC competition format. The DAG response generation (~30s per call) is the primary bottleneck. A simpler output format with fewer mandatory fields could cut response time significantly while retaining the most impactful components.

## Comparison with prior experiments

### vs. Experiment 004

004's best run also scored 1 (level 1 at action 62). 007's best run scores 1 with fewer LLM calls (34 vs. ~55) and fewer BLOCKED actions (11 vs. ~31). The q6 verification is the clear improvement — it delivers the spatial route checking that 004 identified as the core bottleneck. However, 007 is 2-3x slower per run due to the DAG overhead, and the strategic failure (wrong target) persists from 004.

### vs. Experiment 005

005 never scored (0 across all runs) despite introducing program synthesis. The key difference: 005 made code optional and the model wrote trivial code. 007 makes code mandatory (q0, q6) and gets substantial, useful code every turn. This validates 005's recommendation #1 ("not optional — every action result should be automatically recorded") and recommendation #3 ("give up on optional code generation").

### vs. Experiments 001-003

The progression is clear:
- 001: Cannot see (wrong background inference)
- 002: Sees wrong things (game theory hallucination)
- 003: Sees correctly but plans poorly (single hypothesis, no code)
- 004: Plans better but navigates badly (no route verification)
- 005: Has tools but doesn't use them (optional code unused)
- 006: Conversational context helps memory
- 007: Navigates well but picks wrong destinations (strategic failure)

Each experiment has narrowed the failure mode. The remaining gap — strategic target selection — is qualitatively different from the tactical gaps that dominated 001-006.

## Failed improvements (cumulative 004-007)

| Intervention | Experiment | Result |
|---|---|---|
| Reasoning effort "medium"/"high" | 004, 007 | Breaks structured outputs; empty JSON or schema violations |
| Enhanced symbolic features | 007 | Computed but rarely referenced by model |
| Optional code fields | 005 | Model never writes code unless forced |
| Competing hypotheses with if_true/if_false | 005 | Better mechanic ID, no navigation improvement |
| Falsified hypothesis list | 004 | Ignored, bloated prompt |
| Dead-end tracking | 004 | Ignored, bloated prompt |
| Trigger probes | 004 | Wasted actions |
| Cause-effect fields | 004 | Filled generically |

## What the next experiment should address

### 1. Strip the DAG — test whether simple is enough

The strategic failure (wrong target) is the bottleneck, and the DAG doesn't address it. A stripped-down agent with no DAG, no structured outputs, and no mandatory code would test whether the overhead is justified. If a simple agent with low reasoning effort performs comparably (same score, same strategic failure), then the DAG is wasted complexity.

Keep: auto-probe, enhanced symbolic state, conversational window, action history, break-on-BLOCKED.
Remove: DAG, structured outputs, mandatory code.

### 2. Strategic reasoning intervention

The model needs help choosing WHAT to pursue, not HOW to get there. Possible approaches:
- Explicit goal-ranking in the prompt: "List all potential goals. Rank by evidence of being the level objective."
- Reward signal amplification: when the model interacts with the actual goal object, highlight this more prominently than other interactions.
- Exploration diversity: penalize revisiting the same target area, force the model to try different regions of the grid.

### 3. Speed optimization

For competition viability, runs must complete in under 5 minutes. The DAG's 30-second response time is the bottleneck. Options:
- Smaller output format (~800 tokens vs. ~3000)
- Lower reasoning effort (already "none" — this is maximally fast for GPT-5.4)
- Fewer LLM calls via larger action batches (risky — 004 showed plan-based fails)

## Conclusion

Experiment 007 delivers the most mechanically sound agent so far. Mandatory code execution for perception (q0) and route verification (q6) solves the spatial reasoning gap that plagued experiments 004-005 — BLOCKED actions drop by 65%. Structured outputs eliminate JSON parsing failures. The DAG forces systematic analysis that produces better tactical decisions.

But the strategic bottleneck remains. The model chooses wrong targets and no amount of tactical improvement helps. The DAG adds massive token and latency overhead without addressing this core problem. The next experiment should test whether a simpler, faster agent performs comparably — if the bottleneck is strategic, not tactical, then all the DAG overhead is wasted.
