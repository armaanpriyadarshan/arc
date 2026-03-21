"""Tests for optimization 5: repeat clean navigation batches without LLM calls."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from arcengine import GameAction


def _make_agent_state():
    """Create a minimal mock of the agent's repeat-related state."""
    class AgentState:
        def __init__(self):
            self._repeat_actions = None
            self._repeat_count = 0
            self._max_repeats = 1
            self.current_plan = None
    return AgentState()


def _should_repeat(state, actions, batch_clean):
    """Replicate the repeat decision logic from the agent's main loop."""
    if batch_clean and len(actions) > 1:
        directional_only = all(a.value in (1, 2, 3, 4) for a in actions)
        in_execute = (
            state.current_plan
            and isinstance(state.current_plan, dict)
            and state.current_plan.get("mode", "").lower() == "execute"
        )
        if directional_only and in_execute and state._repeat_actions is None:
            state._repeat_actions = actions
            state._repeat_count = 0
            return True
    state._repeat_actions = None
    state._repeat_count = 0
    return False


def test_repeat_on_clean_navigation():
    """Clean batch of directional actions in EXECUTE mode -> repeat triggered."""
    state = _make_agent_state()
    state.current_plan = {"mode": "execute", "steps": []}
    actions = [GameAction.ACTION2, GameAction.ACTION2, GameAction.ACTION2]
    assert _should_repeat(state, actions, batch_clean=True)
    assert state._repeat_actions == actions


def test_no_repeat_when_blocked():
    """Batch that hit BLOCKED -> no repeat."""
    state = _make_agent_state()
    state.current_plan = {"mode": "execute", "steps": []}
    actions = [GameAction.ACTION2, GameAction.ACTION2]
    assert not _should_repeat(state, actions, batch_clean=False)
    assert state._repeat_actions is None


def test_no_repeat_in_explore_mode():
    """Explore mode -> no repeat even if batch is clean."""
    state = _make_agent_state()
    state.current_plan = {"mode": "explore", "steps": []}
    actions = [GameAction.ACTION1, GameAction.ACTION1]
    assert not _should_repeat(state, actions, batch_clean=True)


def test_no_repeat_with_action5():
    """Batch containing ACTION5 -> no repeat (not directional-only)."""
    state = _make_agent_state()
    state.current_plan = {"mode": "execute", "steps": []}
    actions = [GameAction.ACTION2, GameAction.ACTION5]
    assert not _should_repeat(state, actions, batch_clean=True)


def test_no_repeat_single_action():
    """Batch with only 1 action -> no repeat."""
    state = _make_agent_state()
    state.current_plan = {"mode": "execute", "steps": []}
    actions = [GameAction.ACTION2]
    assert not _should_repeat(state, actions, batch_clean=True)


def test_max_one_repeat():
    """After 1 repeat, the repeat state should not allow another."""
    state = _make_agent_state()
    state.current_plan = {"mode": "execute", "steps": []}
    actions = [GameAction.ACTION2, GameAction.ACTION2]

    # First: trigger repeat
    _should_repeat(state, actions, batch_clean=True)
    assert state._repeat_actions is not None
    assert state._repeat_count == 0

    # Simulate one repeat
    state._repeat_count = 1

    # Check: repeat_count >= max_repeats, so next iteration should call LLM
    can_repeat = (state._repeat_actions is not None
                  and state._repeat_count < state._max_repeats)
    assert not can_repeat, "Should not allow second repeat"


def test_case_insensitive_mode():
    """Mode 'EXECUTE' and 'Execute' should both trigger repeat."""
    for mode in ["execute", "EXECUTE", "Execute", "eXeCuTe"]:
        state = _make_agent_state()
        state.current_plan = {"mode": mode, "steps": []}
        actions = [GameAction.ACTION1, GameAction.ACTION1]
        assert _should_repeat(state, actions, batch_clean=True), (
            f"Mode '{mode}' should trigger repeat"
        )


def test_no_repeat_without_plan():
    """No plan at all -> no repeat."""
    state = _make_agent_state()
    state.current_plan = None
    actions = [GameAction.ACTION2, GameAction.ACTION2]
    assert not _should_repeat(state, actions, batch_clean=True)
