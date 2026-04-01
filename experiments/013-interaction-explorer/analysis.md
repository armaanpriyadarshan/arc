# Experiment 013: Interaction Explorer

## Hypothesis

Experiment 004's agent navigates well but doesn't investigate game objects deeply enough. Hypotheses tend to be about paths ("go up to reach X") rather than mechanics ("interacting with X does Y"). Adding an explicit interaction-testing phase should produce better world models and more effective goal pursuit.

## Design

Three-phase architecture building on experiment 004:

1. **Auto-probe** (12 actions): Same as 004. Systematically test each action, record transitions, identify controllable object operationally.

2. **Interaction testing** (40 actions): NEW. Navigate toward each discovered object type and test contact. Classify results as: collected, blocked, died, score_up, transformed, pass_through. Build an InteractionRule for each tested object.

3. **Goal-directed play** (remaining actions): Use the interaction map to make informed decisions. Avoid objects classified as hazards, seek collectibles, navigate toward goal objects.

### Key differences from 004

- `InteractionRule` dataclass tracks what happens on object contact
- `InteractionPlanner` manages approach navigation and test prioritization
- `classify_interaction()` in predictor.py categorizes contact outcomes
- `_choose_action()` uses interaction knowledge: avoids hazards, seeks collectibles
- Refinement prompts include the interaction map so GPT reasons about mechanics
- `proximity_to()` in symbolic.py enables distance-based contact detection

## References

- Experiment 004: World-model induction baseline
- Experiment 004 findings: "navigates well but doesn't investigate game objects deeply enough"

## Results

*Not yet run.*

## Analysis

*Pending results.*
