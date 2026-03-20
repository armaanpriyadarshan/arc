# Experiment 005: Program Synthesis

**Date:** 2026-03-19
**Game:** ls20 (LockSmith)
**Architecture:** Observe-act loop with auto-probe, symbolic state, hypothesis-driven actions, sandbox for model-generated Python, competing hypothesis format (GPT-5.4)
**Best result:** Score 0 across all runs. Never completed level 1.
**Runs:** 3

## Motivation

Experiment 004's conclusion identified the core bottleneck: the model can discover game mechanics but cannot navigate efficiently because it has no persistent spatial representation. It proposed program synthesis as the path forward — let the model write Python code that extracts corridor structure from the grid, producing a text-based map it can reason over precisely.

This experiment implements that proposal: a persistent sandbox where the model can write and execute arbitrary Python against the current grid. Variables survive across turns, so the model can build up tracking structures, analysis tools, and spatial representations over the course of a game.

Additionally, the hypothesis format is upgraded to require competing explanations with explicit `if_true`/`if_false` predictions, forcing the model to design discriminating tests rather than confirmatory ones.

## Architecture

### What carried over from 004

The core loop is identical: single GPT-5.4 call per turn via the Responses API, auto-probe on startup and level-up, symbolic state and diff, side-by-side diff images with red change highlights, hypothesis-driven action sequences that stop on BLOCKED or unexpected events, verified rules that persist across levels, circuit breaker after 5 consecutive BLOCKEDs.

### New: Program synthesis sandbox

The model can output an optional `code` field containing Python. The code executes in a persistent sandbox (`sandbox.py`) with access to:

- `grid` — the current 64x64 grid as `list[list[int]]`, updated before each execution
- `ROWS`, `COLS` — grid dimensions (64)
- All Python builtins via `dict(vars(builtins))`, minus dangerous ones (`open`, `exec`, `eval`, `compile`, `exit`, `quit`, `breakpoint`, `input`)
- Pre-imported modules: `collections` (plus `defaultdict`, `deque`, `Counter` at top level), `math`, `functools`, `json`, `numpy`
- Custom `safe_import` that restricts `__import__` to the allowed module set

Variables persist across calls — define a function or data structure in turn 5, reference it in turn 20. The sandbox captures stdout and the `result` variable; output is truncated to 2000 chars and injected as PROGRAM OUTPUT in the next turn's context.

Execution is time-limited to 5 seconds via `signal.SIGALRM`. Errors return a truncated traceback.

### New: Competing hypothesis format

Each hypothesis now requires:
- `claim` — what the model believes
- `status` — testing / confirmed / rejected / untested
- `test_actions` — actions that would distinguish this hypothesis from alternatives
- `if_true` — what the model expects if correct
- `if_false` — what would disprove this hypothesis

The prompt explicitly instructs: "Design tests that would CONFIRM one hypothesis while REJECTING another." The intent is to prevent the 004 failure mode where the model tests one hypothesis at a time without considering alternatives.

## Results

| Run | Score | Actions | LLM calls | Time (s) | Key behavior |
|-----|-------|---------|-----------|----------|--------------|
| 1 (pre-sandbox-fix) | 0 | 100 | ~50 | ~700 | Most code errored; misidentified action directions |
| 2 (post-sandbox-fix) | 0 | 100 | ~55 | ~800 | Code worked; trivial usage only; toggled switch ~6 times |
| 3 (hypothesis competition) | 0 | 100 | 72 | 1067 | Maintained 3 competing hypotheses; rejected 7; toggled switch ~10 times |

### Run 1: Pre-sandbox-fix

The initial sandbox implementation was too restrictive. It used a custom builtins dict that omitted common names the model expected: `globals`, `locals`, `NameError`, `Exception`, `isinstance`, `type`. Nearly every code block the model wrote failed with a `NameError` or similar. The model attempted position tracking, color counting, and neighbor analysis, but nothing executed.

Separately, the model misidentified action directions — it said "ACTION1 moves up" when probing showed ACTION1 moves left. Without working code to compute position deltas, there was no self-correction mechanism. The model navigated in wrong directions and wasted its budget.

### Run 2: Post-sandbox-fix

After switching to `dict(vars(builtins))` (full builtins minus explicitly dangerous ones), code executed successfully. The model used it for:

- **Position tracking:** Finding the player sprite by scanning the grid for specific colors
- **Delta computation:** Comparing grid values between turns to count what changed
- **Color counting:** Tallying how many cells of each color exist
- **One BFS attempt:** Computing reachable cells from the player's position

All of these are useful but trivial — they replicate information already available in the symbolic state and diff. The BFS attempt was the most promising (this is exactly the kind of code 004 predicted would help), but the model ran it once, got the result, and never used it to inform navigation.

The core behavior was unchanged: find the white switch, toggle it repeatedly (~6 times), observe "remote reconfiguration," never navigate through the opened path.

### Run 3: Hypothesis competition

With the competing hypothesis format, the model maintained 2-3 hypotheses every turn with explicit `if_true`/`if_false` predictions. Over 100 actions, it rejected 7 hypotheses total. Example:

- Hypothesis A: "The white object is a collectible that disappears when touched" — rejected after object reappeared
- Hypothesis B: "The white object is a reversible pressure plate that toggles remote barriers" — confirmed

The model correctly identified the switch mechanic faster and more precisely than in any 004 run. But this understanding did not translate to action: it toggled the switch ~10 times, each time confirming the same already-confirmed hypothesis. The `if_false` predictions were never surprising enough to trigger strategic pivots.

This was the slowest run at 1067s — the hypothesis format and code output add significant token overhead to both prompt and response.

## Key findings

### 1. The sandbox works but the model writes trivial code

After the builtins fix, every code block executed successfully. But the model only wrote code that replicated existing context: position lookups (already in symbolic state), color counts (already in symbolic state), delta computations (already in symbolic diff). It never spontaneously wrote strategic code — no persistent visited-cell map, no trigger-effect recorder, no reachable-area computation that accumulates across turns.

The single BFS attempt in run 2 was promising but was treated as a one-shot query, not a persistent tool. The model has the capability to build spatial representations in code — it just doesn't occur to the model to do so.

What the sandbox should have enabled but didn't:
- A visited-cell map that accumulates across turns, preventing re-exploration
- A trigger-effect log recording what cells changed each time the switch was toggled
- Reachable-area computation after each toggle, identifying newly opened corridors
- A wall map tracking which directions are blocked from which positions

### 2. Hypothesis competition works structurally but at the wrong level

The competing hypothesis format produced genuinely better mechanic identification. The model maintained multiple explanations and rejected falsified ones cleanly. Seven hypotheses were rejected across run 3 — real scientific reasoning about game mechanics.

But the hypotheses operate at the wrong level of abstraction. They ask "what IS this object?" (collectible vs. switch vs. decoration) rather than "what should I DO next?" (navigate through the opened area vs. find a different path vs. look for a second switch). The model resolves the mechanics question correctly and then has no strategic hypotheses to guide its next move.

This is a restatement of 004's exploration-vs-exploitation gap, now with more precise language: the model excels at **ontological hypotheses** (what are things?) but never formulates **strategic hypotheses** (what should I do with them?).

### 3. Code-computed ground truth can self-correct model errors

In run 1, the model misidentified action directions (ACTION1 as "up" instead of "left"). In run 2, the model wrote code to compute its own position delta after each action and used the result to correct its direction mapping. This demonstrates a genuine value of the sandbox: the model can check its own beliefs against computed ground truth.

However, this happened only when the error was obvious and the code was trivial. The model never used code to check more subtle beliefs — for example, it never computed whether a specific corridor was actually passable after toggling the switch.

### 4. The prompt is not the bottleneck

Across approximately 15 total prompt variations between experiments 004 and 005 — competing hypotheses, falsification instructions, observation guidance, urgency injection, dead-end tracking, cause-effect fields, exploitation prompts — the core behavior is identical: find the switch, toggle repeatedly, never navigate through the opened path.

Prompt additions that were tested and failed in 004 remain failed here:
- Falsified hypothesis lists (ignored, bloated prompt)
- Dead-end tracking (ignored, bloated prompt)
- Reasoning mode (returned empty JSON)
- Trigger probes (wasted actions)
- Cause-effect fields (filled generically)

The competing hypothesis format from this experiment is the strongest prompt intervention tested so far — it genuinely improves mechanic identification. But it doesn't change navigation behavior.

### 5. Persistent spatial memory is the fundamental gap

Each LLM call sees the current frame, symbolic state, symbolic diff, 10-15 recent actions, and free-form notes. The model reconstructs its spatial understanding from scratch every call. It has no accumulated map, no record of where it has been, no structured memory of what changed when triggers fired.

The notes field is the model's only persistent memory, and it writes notes like "the white object toggles remote barriers" (mechanic understanding) rather than "after toggling, cells [30,20]-[30,25] became color 0 (passable)" (actionable spatial data). Even when the model has correct mechanic understanding, it cannot act on it because it doesn't remember the spatial details.

This is the same finding as 004's finding #6 (memory window causes re-exploration) and finding #2 (the corridor problem), now confirmed to persist even when the model has the ability to build persistent memory structures in code.

## Comparison with prior experiments

### vs. Experiment 004

004's best run scored 1 (completed level 1 at action 62). 005 never scored. This is a regression, though with only 3 runs vs. 004's ~10, it may be within variance. The important comparison:

| | 004 | 005 |
|---|-----|-----|
| Mechanic identification | Correct in ~60% of runs | Correct in 100% of runs (hypothesis competition helps) |
| Navigation after identification | Poor (toggle addiction) | Poor (toggle addiction) |
| Token overhead per call | Moderate | High (code + hypothesis format) |
| Average time per run | ~600-700s | ~850s |
| Score | 0-1 | 0 |

005 is better at understanding and worse at acting. The additional overhead from code output and expanded hypothesis format adds latency without improving the behavior that matters.

### vs. Experiment 003

003's best run completed level 1 in 34 actions — still the fastest level completion across all experiments. 003 used a simpler single-hypothesis format and had no sandbox or competing hypothesis system. The model in 003 sometimes "stumbled into" correct behavior through aggressive exploration, which 005's more deliberate hypothesis-testing approach prevents.

### vs. Experiments 001-002

001 failed at perception (wrong background inference). 002 hallucinated game mechanics from visual patterns. Both problems are fully solved by the auto-probe + symbolic state architecture shared across 004 and 005. The failure mode has progressively narrowed from "can't see" (001) to "sees wrong things" (002) to "sees correctly but can't plan" (003-005).

## Failed improvements (cumulative across 004-005)

| Intervention | Experiments | Result |
|---|---|---|
| Prompt: falsified hypothesis list | 004 | Ignored, bloated prompt |
| Prompt: dead-end tracking | 004 | Ignored, bloated prompt |
| Prompt: cause-effect fields | 004 | Filled generically |
| Prompt: exploitation urgency | 004, 005 | No behavior change |
| Prompt: competing hypotheses with if_true/if_false | 005 | Better mechanic ID, no navigation improvement |
| Architecture: reasoning mode | 004 | Empty JSON output |
| Architecture: plan-based (10+ actions per call) | 004 | Worse — spatial reasoning too imprecise for multi-step plans |
| Architecture: trigger probe (auto-test 4 directions after trigger) | 004 | Wasted 8 actions per trigger |
| Architecture: optional sandbox for model-generated code | 005 | Model writes trivial code only |

## What the next experiment should address

### 1. Auto-populated structured game state (not optional sandbox)

The sandbox failed because the model doesn't spontaneously write strategic code. The fix is to not make it optional. Every action result should be automatically recorded in a persistent, code-maintained data structure:

- **Wall map:** After every BLOCKED action, record the position and direction. After every successful move, record the position as traversable. This accumulates into a partial map without any LLM involvement.
- **Trigger log:** After every high-change event (>50 cells), record a before/after diff keyed by the action that caused it. The model can query "what happened last time I toggled the switch?" and get precise cell-level data.
- **Position history:** Track the player's position after every action (computed by code, not by the model). Inject a simple text map showing visited vs. unvisited areas.

The model's job shifts from "remember everything in notes" to "query the structured log to make decisions." This is a code-level architectural change, not a prompt change.

### 2. Force strategic hypotheses

The competing hypothesis format works for mechanic identification but needs to also require strategic hypotheses. The prompt should distinguish:

- **Mechanic hypotheses:** "What does this object do?" (current format, works well)
- **Strategic hypotheses:** "What should I do next to score?" (not currently prompted)

After a mechanic is confirmed, the model should be required to formulate and test strategic hypotheses: "I should navigate through the corridor that opened when I toggled the switch" vs. "I should look for a second switch before proceeding."

### 3. Give up on optional code generation

Three runs showed the model will not spontaneously build useful persistent data structures. If code-maintained state is valuable (it is — experiment 004 identified this), it must be built into the architecture, not left as an optional model output.

## Conclusion

Program synthesis as an optional tool doesn't work. The model has the ability to write Python code and a persistent sandbox to run it in, but it only writes trivial code that replicates existing context. It never builds the spatial representations that 004 identified as the core bottleneck. Meanwhile, the competing hypothesis format improves mechanic identification — the model resolves "what is this object?" faster and more accurately than ever — but this doesn't translate to better navigation or scoring.

The cumulative evidence from 004 and 005 is clear: the model understands game mechanics but cannot maintain or act on spatial knowledge. Prompt changes don't help. Optional tools don't help. The next experiment must make the architectural change: auto-populate structured spatial state in code, so the model receives a persistent map and query interface rather than reconstructing spatial understanding from scratch every turn.
