# Supplementary Task Part 3: Planning Layer and Hypothesis Convergence

This is an addition to the main v3 implementation prompt and the Part 2 hypothesis generation prompt. Apply these changes ON TOP of everything already described. All changes are in `experiments/004-world-model-induction-v3/agents/agent.py`.

## Problem

By turn ~30, the agent often has correct, converging hypotheses — e.g., "the plus controls orientation", "the goal is to match the display to the target", "I need to interact with the plus to rotate." These hypotheses agree with each other. But the agent doesn't synthesize them into a plan. Instead, each turn it re-reads its notes, re-evaluates hypotheses from scratch, and picks a single action. It drifts aimlessly because it has no persistent intent across turns.

The architecture is: hypotheses → pick one → execute its test_actions → repeat. What's missing is a planning layer: when hypotheses converge, collapse them into a multi-step plan and execute it until something goes wrong.

## Changes

### 1. Add a `plan` field to the JSON output schema

Add a new top-level field to the model's JSON output. Update `SYSTEM_PROMPT`:

```
PLANNING:

You have two modes: EXPLORE and EXECUTE.

EXPLORE mode: You're still figuring out the game. Generate hypotheses, test them, gather information. This is the default when you start a new level or when your plan just failed.

EXECUTE mode: You've figured out enough to act. You have a concrete plan and you're following it step by step.

Your JSON output must include a "plan" field:

{
  "plan": {
    "mode": "execute",
    "goal": "Match the display pattern to the target pattern in the bottom-left, then reach the door",
    "steps": [
      {"step": 1, "action": "Navigate to the plus symbol at [25, 38]", "status": "completed"},
      {"step": 2, "action": "Activate the plus (walk into it or ACTION5)", "status": "current"},
      {"step": 3, "action": "Check if display now matches target", "status": "pending"},
      {"step": 4, "action": "If not matched, activate plus again", "status": "pending"},
      {"step": 5, "action": "If matched, navigate to the door at [32, 12]", "status": "pending"}
    ],
    "abort_conditions": [
      "Display doesn't change after activating plus",
      "Door doesn't open after display matches target",
      "GAME_OVER"
    ],
    "supporting_hypotheses": ["plus controls orientation (0.7)", "goal is to match display to target (0.65)"]
  }
}

Or when still exploring:

{
  "plan": {
    "mode": "explore",
    "goal": "Determine what the plus symbol does",
    "steps": [
      {"step": 1, "action": "Navigate to plus symbol", "status": "current"},
      {"step": 2, "action": "Activate it and observe changes", "status": "pending"}
    ],
    "abort_conditions": ["Discover the plus is not interactable"],
    "supporting_hypotheses": ["plus is an interactable object (0.5)"]
  }
}

RULES FOR PLANNING:

1. WHEN TO SWITCH TO EXECUTE MODE:
   - Your top 2+ hypotheses are compatible (they don't contradict each other)
   - Your top hypothesis has probability >= 0.6
   - You can describe a concrete sequence of steps to test or achieve the goal
   When these conditions are met, you MUST switch to execute mode. Don't keep exploring when you have a workable theory.

2. WHEN TO STAY IN EXECUTE MODE:
   - Follow the plan step by step. Mark steps as "completed" as you finish them.
   - Your test_actions each turn should correspond to the CURRENT step in the plan.
   - Do NOT re-evaluate the plan from scratch each turn. Only update step statuses.
   - Stay in execute mode unless an abort condition is hit.

3. WHEN TO ABORT AND RETURN TO EXPLORE MODE:
   - An abort condition fires (something unexpected happened)
   - You've been stuck on the same step for 5+ turns with no progress
   - Your plan's key hypothesis drops below 0.3 probability due to new evidence
   - You completed all steps but didn't achieve the goal
   When aborting, log WHY the plan failed in the falsified list, then generate new hypotheses.

4. PLAN PERSISTENCE:
   - The plan carries forward in your notes between turns. Don't regenerate it from scratch unless aborting.
   - Each turn, just update which step is current and whether any steps completed.
   - Your test_actions should serve the current step, not be random exploration.

5. OVERLAPPING HYPOTHESES ARE FINE:
   - If multiple hypotheses point in the same direction, that's convergent evidence — it strengthens the plan.
   - Don't treat overlap as redundancy. Three hypotheses saying "interact with plus to change display" at 0.7, 0.6, 0.5 is STRONGER than one hypothesis at 0.7.
   - Use the supporting_hypotheses field to list which hypotheses back your plan.
   - If hypotheses disagree, pick the higher-probability one, commit to it, and test it. Only switch if you find direct contradictory evidence — not just because you're uncertain.
```

### 2. Add plan persistence in the agent code

The plan needs to survive across turns. In the agent's `__init__`, add:

```python
self.current_plan = None  # dict from model output, persisted across turns
self.plan_step_turns = 0  # how many turns we've been on the current step
self.last_plan_step = None  # to detect when we're stuck on a step
```

When building the prompt content each turn, include the current plan:

```python
if self.current_plan:
    plan_text = json.dumps(self.current_plan, indent=2)
    content.append(input_text(f"\nYOUR CURRENT PLAN:\n{plan_text}\n"))
```

When parsing the model's response, extract and persist the plan:

```python
new_plan = data.get("plan", None)
if new_plan and isinstance(new_plan, dict):
    # Detect if we're stuck on the same step
    current_step = None
    for step in new_plan.get("steps", []):
        if step.get("status") == "current":
            current_step = step.get("step")
            break
    
    if current_step == self.last_plan_step:
        self.plan_step_turns += 1
    else:
        self.plan_step_turns = 0
        self.last_plan_step = current_step
    
    self.current_plan = new_plan
```

### 3. Add stuck-on-step detection

If the agent has been on the same plan step for 5+ turns, inject a warning:

```python
if self.plan_step_turns >= 5:
    stuck_msg = (
        f"\n\nWARNING: You've been on step {self.last_plan_step} of your plan "
        f"for {self.plan_step_turns} turns with no progress. Either:\n"
        "1. Break this step into smaller sub-steps\n"
        "2. Try a completely different approach to this step\n"
        "3. Abort the plan — your approach may be wrong\n"
        "Do NOT just repeat the same actions.\n"
    )
    content.append(input_text(stuck_msg))
```

### 4. Add auto-transition to execute mode

If the model stays in explore mode for too long despite having high-probability hypotheses, nudge it:

```python
if self.current_plan and self.current_plan.get("mode") == "explore":
    # Check if hypotheses are strong enough to commit
    hypotheses = data.get("hypotheses", [])
    high_prob = [h for h in hypotheses if isinstance(h, dict) and h.get("probability", 0) >= 0.6]
    
    if len(high_prob) >= 2 and self.action_counter > 20:
        commit_msg = (
            f"\n\nYou have {len(high_prob)} hypotheses at >= 0.6 probability "
            "and you've been exploring for 20+ turns. You likely have enough information "
            "to form a plan. Switch to EXECUTE mode: synthesize your hypotheses into "
            "a concrete step-by-step plan and start following it.\n"
        )
        content.append(input_text(commit_msg))
```

### 5. Tie test_actions to plan steps

Update the action selection logic so that when in execute mode, the model's `test_actions` are expected to serve the current plan step. Add to the system prompt:

```
CONNECTING PLANS TO ACTIONS:

When in EXECUTE mode, your test_actions MUST correspond to your current plan step.
- If current step is "Navigate to plus symbol at [25, 38]", your test_actions should be movement actions toward that position.
- If current step is "Activate the plus", your test_actions should be ACTION5 or walking into it.
- If current step is "Check if display matches target", you don't need test_actions — just observe the symbolic state and image.

Do NOT output test_actions that are unrelated to your current plan step while in execute mode.
If you find yourself wanting to do something off-plan, either:
1. Add it as a new step to the plan, or
2. Abort the plan and return to explore mode.
```

### 6. Add plan summary to logging

When the model outputs a plan, log it clearly:

```python
if new_plan:
    mode = new_plan.get("mode", "unknown")
    goal = new_plan.get("goal", "")
    steps = new_plan.get("steps", [])
    current = next((s for s in steps if s.get("status") == "current"), None)
    current_desc = current.get("action", "?") if current else "none"
    logger.info(f"[plan] mode={mode} goal='{goal}' current_step='{current_desc}' ({len(steps)} steps total)")
```

## What NOT to change

- Don't modify `symbolic.py` or `vision.py` (orientation detection is a separate issue for a future prompt)
- Don't change the Bayesian probability system — the plan layer sits on top of it
- Don't change the hypothesis generation improvements from Part 2 — the scene inventory and category diversity still apply during explore mode
- Don't change the auto-probe logic

## How to verify

Run against ls20 or a synthetic game. Check logs for:
- `[plan] mode=explore` appearing in early turns while the agent is still learning
- `[plan] mode=execute` appearing once the agent has formed a theory, with concrete steps listed
- Plan steps transitioning from "pending" to "current" to "completed" across turns
- The commit nudge appearing if the agent stays in explore mode past turn 20 with strong hypotheses
- The stuck warning appearing if the agent is on the same step for 5+ turns
- When a plan is aborted, a clear entry in the falsified list explaining why
- test_actions corresponding to the current plan step, not random exploration
- The agent NOT re-evaluating its entire worldview every turn during execute mode — it should be following the plan
