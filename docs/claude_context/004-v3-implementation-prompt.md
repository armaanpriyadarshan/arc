# Task: Create Experiment 004-world-model-induction-v3

Create a new experiment folder at `experiments/004-world-model-induction-v3/` based on `experiments/004-world-model-induction-v2/`. This is a v3 of the observe-act agent with three major changes to how hypotheses work. Copy the full v2 codebase as a starting point, then modify as described below.

## Context

The v2 agent (in `experiments/004-world-model-induction-v2/`) uses GPT-5.4 in a per-turn observe-act loop. It auto-probes actions at startup, feeds symbolic state + diffs + images to the model each turn, and the model outputs structured JSON with hypotheses, observations, and actions. It works well for simple movement games but has three critical failure modes:

1. **Binary hypothesis states kill exploration.** Hypotheses have a `status` field (`testing`/`confirmed`/`untested`/`rejected`). Once something is marked `confirmed` or `rejected`, the model never revisits it — even when the evidence was weak (e.g., testing ACTION6 at one coordinate and concluding it's a global no-op).

2. **No evidence tracking.** Hypotheses carry a single `evidence` string but don't track *what specific tests were run*. The model can't see that it only tested one case and overgeneralized.

3. **Parameterized actions are not understood.** ACTION6 requires (x, y) coordinates but the auto-probe only tries it once. The model doesn't know ACTION6 is parameterized and treats one failed click as proof that clicking does nothing.

## Changes to make

### 1. Add action descriptions to `SYSTEM_PROMPT`

Add this block to the system prompt so the model knows the action space upfront:

```
AVAILABLE ACTIONS:
- ACTION1: Simple action (typically mapped to UP)
- ACTION2: Simple action (typically mapped to DOWN)
- ACTION3: Simple action (typically mapped to LEFT)
- ACTION4: Simple action (typically mapped to RIGHT)
- ACTION5: Simple action (interact, select, rotate, attach/detach, etc.)
- ACTION6: Complex action requiring x,y coordinates (0-63 range). This is a CLICK — different coordinates may have completely different effects. You must explore the coordinate space, not just test one spot.
- ACTION7: Simple action (undo)

Not all actions may be available in every game. Check available_actions.
```

### 2. Replace binary hypothesis status with Bayesian probabilities

Change the hypothesis JSON schema in the system prompt. Old format:
```json
{"claim": "...", "status": "testing", "test_actions": [...], "evidence": "..."}
```

New format:
```json
{
  "claim": "ACTION6 clicking on objects toggles their state",
  "probability": 0.4,
  "tests": [
    {"action": "ACTION6(32,15)", "result": "0 changes", "interpretation": "clicked on gridline, not on object"},
    {"action": "ACTION6(12,8)", "result": "4 changes, red cell toggled", "interpretation": "clicking on grid cells toggles color"}
  ],
  "test_actions": ["ACTION6", 25, 36],
  "information_gain": "high — only tested empty space so far, never clicked an object"
}
```

Key rules to add to the system prompt about hypothesis management:

- **Every hypothesis must have a `probability` between 0.0 and 1.0.** Update probabilities each turn based on new evidence. Never use 0.0 or 1.0 — there is always some uncertainty.
- **Every hypothesis must have a `tests` list** documenting what was specifically tried and what happened. This is how you track whether your evidence is strong or weak.
- **Maintain at least 3 competing hypotheses at all times.** If you only have one theory, you're not exploring enough. Generate alternatives even if they seem unlikely.
- **Prioritize testing the hypothesis with the highest uncertainty** (probability closest to 0.5) — that's where you gain the most information. If you already have a high-confidence hypothesis (>0.8), test something else.
- **For parameterized actions (ACTION6), evidence from one coordinate does NOT generalize.** You must test diverse targets: click on different colored objects, empty cells, edges, corners. Track each coordinate tested in the `tests` list. You need at least 3-5 diverse coordinate tests before drawing any conclusion about ACTION6's behavior.
- **Re-evaluate old hypotheses when new evidence arrives.** If you discover a new game mechanic, check whether it changes the probability of existing hypotheses.
- **The `test_actions` field drives execution.** For simple actions: `["ACTION3", "ACTION3"]`. For ACTION6 with coordinates: `["ACTION6", x, y]` where x and y are integers 0-63. The first hypothesis (sorted by descending probability) that has `test_actions` will be executed.

### 3. Update `_auto_probe` for parameterized actions

The current `_auto_probe` takes each action once. For ACTION6 (and any complex action requiring coordinates), expand the probe:

- After probing ACTION1-5 and ACTION7 normally (one test each), check if ACTION6 is in the available actions
- If ACTION6 is available, use the symbolic state to identify 3-5 interesting click targets:
  - The center of the largest non-background object
  - The center of a different-colored object
  - An empty/background cell
  - A cell at the grid edge
  - The center of the grid
- Probe ACTION6 at each of these coordinates, recording what changed
- Include all ACTION6 probe results in the established facts with their coordinates, so the model sees "ACTION6 at (12, 8): 4 cells changed. ACTION6 at (32, 32): 0 cells changed" rather than just "ACTION6: 0 cells changed"

The total probe budget goes from ~4 actions to ~8-12 actions. This is worth it because it prevents the model from writing off ACTION6 entirely.

### 4. Update action selection logic in the main loop

The current code finds the first hypothesis with `status == "testing"` and executes its `test_actions`. Change this to:

- Sort hypotheses by probability descending
- Find the first hypothesis that has a `test_actions` field
- Execute those actions
- When parsing `test_actions`, handle the ACTION6 coordinate format: if the list contains `["ACTION6", x, y]`, construct the appropriate complex action with those coordinates

Make sure the code correctly passes x, y coordinates to `env.step()` for ACTION6. Check how the ARC SDK handles complex actions — the `GameAction` enum likely has a way to specify coordinates.

### 5. Add stale hypothesis detection

Add a code-level check in the main loop. Track how many consecutive turns the top hypothesis (highest probability) has been the same claim. If it's been the same for 5+ turns with no score change:

- Inject an extra message into the prompt: "WARNING: Your top hypothesis has been unchanged for N turns with no score progress. Your evidence may be insufficient or your approach may be wrong. Consider: (1) What would FALSIFY your current top hypothesis? Test that. (2) What alternative explanations have you not considered? (3) For ACTION6, have you tested diverse coordinates?"
- Also inject it if the model has had 3+ consecutive turns with 0 cell changes (it's doing nothing productive)

### 6. Update `config.py`

Update the docstring and metadata to reflect v3. Increase `MAX_ACTIONS` from 100 to 150 to account for the expanded probe phase and richer exploration.

### 7. Don't change these files

- `symbolic.py` — keep as-is, the connected component analysis works fine
- `vision.py` — keep as-is, the image rendering works fine
- `run.py` — keep as-is, just copy from v2
- `pyproject.toml` — copy from v2, update the project name

### File structure

```
experiments/004-world-model-induction-v3/
├── pyproject.toml          # copied from v2, updated name
├── run.py                  # copied from v2 unchanged
├── config.py               # updated docstring + MAX_ACTIONS=150
├── agents/
│   ├── __init__.py         # same as v2
│   ├── agent.py            # MAIN CHANGES HERE — system prompt, hypothesis format, probe, action selection, stale detection
│   ├── symbolic.py         # copied from v2 unchanged
│   └── vision.py           # copied from v2 unchanged
```

## Testing

After implementing, run against the ls20 game:
```bash
cd experiments/004-world-model-induction-v3
uv run python run.py --game ls20
```

Check the logs for:
- Auto-probe should show multiple ACTION6 tests at different coordinates (if ACTION6 is available)
- Hypotheses should have `probability` fields, not `status` fields
- Hypotheses should have `tests` lists showing what was tried
- After 5+ stale turns, the warning message should appear
- The model should maintain 3+ competing hypotheses

Also test with a game that uses ACTION6 (click-based) if one is available, to verify the parameterized action handling works.
