# Supplementary Task Part 4: Interaction Journal

This is an addition to the main v3 prompt, Part 2 (hypothesis generation), and Part 3 (planning layer). Apply on top of all previous changes. Changes are in `experiments/004-world-model-induction-v3/agents/agent.py`.

## Problem

The agent discovers things and then forgets them. At turn 12 it walks into the plus symbol, the display rotates, and the model says "interesting, the plus controls orientation." By turn 20 it's wandering right. By turn 30 it rediscovers the plus. The observations are there in the notes somewhere, but they're buried under hypothesis updates, plan fragments, and general commentary. The model doesn't have a clean, prominent record of "here is exactly what I learned from each interaction."

## Solution: Interaction Journal

Add a structured, code-managed log of cause-effect observations. This is NOT part of the model's free-form notes — it's a separate data structure that the code maintains and injects prominently into every prompt. The model proposes entries; the code stores and formats them.

### 1. Add the interaction journal to the JSON output schema

Add this to `SYSTEM_PROMPT`:

```
INTERACTION JOURNAL:

Every time you interact with an object or try something new and observe a result, record it in the "interactions" field:

"interactions": [
  {
    "turn": 12,
    "action": "Walked into plus symbol at [25, 38]",
    "observed": "Purple shape inside the door rotated 90° clockwise. Display changed from L-facing-right to L-facing-down.",
    "objects_involved": ["plus symbol", "door display"],
    "reversible": "unknown",
    "useful": true
  }
]

Rules for the interaction journal:
- Record EVERY interaction where something non-trivial happened (cells changed beyond normal movement, objects appeared/disappeared/changed, score changed).
- Record interactions where NOTHING happened too, if you expected something — e.g., "Walked into door at [32,12], nothing happened. Door may require display to match target first."
- Be SPECIFIC about what changed. Not "something changed" but "the purple L-shape rotated 90° clockwise" or "the yellow bar decreased by 2 cells."
- Each entry is a FACT, not a hypothesis. Only record what you directly observed.
- The "useful" flag marks interactions that revealed a game mechanic. These get highlighted.

You don't need to repeat old entries — just add new ones each turn. The code accumulates them for you.
```

### 2. Code-side journal management

In `__init__`, add:

```python
self.interaction_journal: list[dict] = []  # accumulated across the entire game
self.journal_by_object: dict[str, list[dict]] = {}  # indexed by object for quick lookup
```

When parsing the model's response, extract and accumulate journal entries:

```python
new_interactions = data.get("interactions", [])
if new_interactions and isinstance(new_interactions, list):
    for entry in new_interactions:
        if isinstance(entry, dict) and entry.get("action"):
            # Add turn number if not present
            if "turn" not in entry:
                entry["turn"] = self.action_counter
            
            self.interaction_journal.append(entry)
            
            # Index by objects involved
            for obj in entry.get("objects_involved", []):
                if obj not in self.journal_by_object:
                    self.journal_by_object[obj] = []
                self.journal_by_object[obj].append(entry)
            
            logger.info(
                f"[journal] Turn {entry.get('turn')}: {entry.get('action')} -> {entry.get('observed')}"
            )
```

### 3. Inject the journal prominently into every prompt

The journal should appear BEFORE hypotheses and BEFORE the plan — it's the factual foundation that everything else is built on. When building prompt content:

```python
if self.interaction_journal:
    # Show all useful interactions, plus the last 3 non-useful ones
    useful = [e for e in self.interaction_journal if e.get("useful")]
    recent_other = [e for e in self.interaction_journal if not e.get("useful")][-3:]
    
    journal_text = "=== INTERACTION JOURNAL (facts you've discovered) ===\n"
    
    if useful:
        journal_text += "\nKEY DISCOVERIES:\n"
        for entry in useful:
            journal_text += (
                f"  Turn {entry.get('turn', '?')}: {entry.get('action', '?')}\n"
                f"    → {entry.get('observed', '?')}\n"
            )
    
    if recent_other:
        journal_text += "\nRECENT OBSERVATIONS:\n"
        for entry in recent_other:
            journal_text += (
                f"  Turn {entry.get('turn', '?')}: {entry.get('action', '?')}\n"
                f"    → {entry.get('observed', '?')}\n"
            )
    
    journal_text += "\nUse these facts to inform your plan. If a discovery tells you how a mechanic works, ACT ON IT.\n"
    journal_text += "=== END JOURNAL ===\n"
    
    content.append(input_text(journal_text))
```

Place this BEFORE the hypothesis and plan sections in the prompt content, so the model reads its experimental results before deciding what to do.

### 4. Auto-detect interactions the model missed

Sometimes the model does something significant but doesn't record it. The code should detect large state changes and prompt the model to journal them. After executing an action:

```python
# After executing an action and computing the diff
if changes > 80 and not blocked:
    # Something big happened — make sure the model journals it
    auto_journal_prompt = (
        f"\nIMPORTANT: Your last action caused {changes} cell changes, which is "
        "well beyond normal movement (~52 cells). Something significant happened. "
        "You MUST record this in your interactions journal with specific details "
        "about what changed. Look at the symbolic diff and the image carefully.\n"
    )
    # Store this to inject into the next prompt
    self._pending_journal_prompt = auto_journal_prompt
```

Then when building the next prompt:

```python
if hasattr(self, '_pending_journal_prompt') and self._pending_journal_prompt:
    content.append(input_text(self._pending_journal_prompt))
    self._pending_journal_prompt = None
```

### 5. Surface relevant journal entries when near known objects

When the symbolic state shows the player near an object that has journal entries, remind the model what it already knows about that object:

```python
def _get_nearby_object_journal(self, symbolic_state: dict) -> str:
    """Check if the player is near any objects with journal entries."""
    if not self.journal_by_object:
        return ""
    
    # Find player position (the controllable object from probe facts)
    # This is approximate — use the object that moved most recently
    reminders = []
    
    for obj_name, entries in self.journal_by_object.items():
        useful_entries = [e for e in entries if e.get("useful")]
        if useful_entries:
            latest = useful_entries[-1]
            reminders.append(
                f"REMINDER — you already know about '{obj_name}': "
                f"at turn {latest.get('turn', '?')} you found that: {latest.get('observed', '?')}"
            )
    
    if reminders:
        return "\n" + "\n".join(reminders) + "\n"
    return ""
```

Inject this into the prompt:

```python
nearby_knowledge = self._get_nearby_object_journal(symbolic)
if nearby_knowledge:
    content.append(input_text(nearby_knowledge))
```

### 6. Connect journal to planning

Add this to the system prompt, in the planning section:

```
USING YOUR JOURNAL IN PLANS:

When forming a plan in EXECUTE mode, your plan steps should be directly informed by journal entries. For example:

- Journal says "Walking into plus rotated the display 90°"
- Target display shows a specific orientation
- Therefore plan: "Activate plus N times until display matches target, then go to door"

If your journal tells you HOW a mechanic works, your plan should USE that knowledge with specific expected outcomes. Don't just "go interact with the plus" — say "activate the plus 2 more times (I need 180° more rotation based on turn 12 observation)."

A plan that ignores your journal entries is a BAD plan. Before finalizing a plan, check: does this plan use everything I've learned?
```

### 7. Carry journal across levels

When a level is completed, keep the journal but mark which entries are from which level:

```python
# When level completion is detected:
for entry in self.interaction_journal:
    if "level" not in entry:
        entry["level"] = self.current_level

# When injecting journal into prompt, note which entries are from previous levels:
# "These interactions are from level 1 — mechanics may carry over but positions won't."
```

## What NOT to change

- Don't modify `symbolic.py` or `vision.py`
- Don't change the Bayesian probability system
- Don't change hypothesis generation (Part 2) or planning (Part 3)
- The journal is a NEW parallel data structure, not a replacement for notes/hypotheses/rules

## Relationship to existing fields

- **notes**: Free-form, model-managed. Can contain anything. Often messy and verbose.
- **verified_rules**: Abstract universal mechanics ("touching white objects triggers reconfiguration"). Persist across levels.
- **falsified**: Things that didn't work. Prevents repeating mistakes.
- **interaction journal (NEW)**: Specific, concrete cause-effect observations with turn numbers. The raw experimental data. Code-managed and formatted for prominence.

The journal is the EVIDENCE layer. Hypotheses are interpretations of journal entries. Rules are generalizations across journal entries. Plans are action sequences derived from journal entries. The journal is the ground truth.

## How to verify

Run against ls20 or a synthetic game. Check logs for:
- `[journal]` entries appearing when the agent interacts with objects
- The "KEY DISCOVERIES" section appearing in prompts after meaningful interactions
- The auto-detect prompt firing when >80 cells change
- Journal entries being specific ("purple L rotated 90° clockwise") not vague ("something changed")
- The model's plans referencing journal discoveries ("based on turn 12, I need to activate plus 2 more times")
- Journal surviving across levels with level tags
- The model NOT rediscovering things it already has journal entries for
