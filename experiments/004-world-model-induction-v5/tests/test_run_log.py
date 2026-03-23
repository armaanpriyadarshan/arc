"""Tests for the RunLog class (structured agent-readable log)."""

import os
import tempfile

from agents.run_log import RunLog


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_log_with_turns(n: int = 3) -> RunLog:
    """Create a RunLog pre-populated with header, probe, and *n* turns."""
    log = RunLog()
    log.write_header("test_game", 150)
    log.write_probe("ACTION1: 52 cells changed\nACTION2: 0 cells changed (BLOCKED)")

    for i in range(1, n + 1):
        log.write_turn(
            turn=i,
            action_count=i * 5,
            score=0,
            mode="EXPLORE" if i <= 2 else "EXECUTE",
            goal=f"Test goal {i}",
            plan_step=f"Step {i}" if i > 1 else "",
            observation=f"Observation for turn {i}",
            hypotheses=f"  [p=0.{5 + i}|NAV] Hyp turn {i}",
            actions_taken=[f"ACTION{i}", "ACTION1"],
            action_results=[f"ACTION{i}: 52 cells changed", "ACTION1: BLOCKED"],
            interactions=[{"action": f"walk into obj_{i}", "observed": f"effect_{i}"}]
            if i == 2
            else [],
            notes=f"Notes for turn {i}",
        )
    return log


# ---------------------------------------------------------------------------
# write_* tests
# ---------------------------------------------------------------------------

class TestWriteHeader:
    def test_header_line(self):
        log = RunLog()
        log.write_header("ls20", 150)
        assert log.line_count() == 2  # header + blank
        assert "ls20" in log._lines[0]
        assert "MAX_ACTIONS=150" in log._lines[0]


class TestWriteProbe:
    def test_probe_markers(self):
        log = RunLog()
        log.write_probe("ACTION1: moved\nACTION2: blocked")
        text = "\n".join(log._lines)
        assert "=== PROBE RESULTS ===" in text
        assert "=== END PROBE ===" in text
        assert "ACTION1: moved" in text


class TestWriteTurn:
    def test_turn_markers(self):
        log = _make_log_with_turns(1)
        text = "\n".join(log._lines)
        assert "=== TURN 1 |" in text
        assert "=== END TURN 1 ===" in text

    def test_turn_fields(self):
        log = _make_log_with_turns(1)
        text = "\n".join(log._lines)
        assert "GOAL: Test goal 1" in text
        assert "OBSERVATION: Observation for turn 1" in text
        assert "NOTES: Notes for turn 1" in text

    def test_interactions_logged(self):
        log = _make_log_with_turns(2)
        text = "\n".join(log._lines)
        assert "walk into obj_2" in text
        assert "effect_2" in text

    def test_mode_in_header(self):
        log = _make_log_with_turns(3)
        text = "\n".join(log._lines)
        assert "MODE=EXPLORE" in text
        assert "MODE=EXECUTE" in text


class TestWriteEvent:
    def test_event_format(self):
        log = RunLog()
        log.write_event("DEATH", "Death #1")
        assert "*** DEATH: Death #1 ***" in log._lines[0]


# ---------------------------------------------------------------------------
# read_lines tests
# ---------------------------------------------------------------------------

class TestReadLines:
    def test_basic_read(self):
        log = _make_log_with_turns(1)
        result = log.read_lines(offset=1, limit=5)
        assert "1:" in result
        assert "=== GAME:" in result

    def test_offset_beyond_end(self):
        log = _make_log_with_turns(1)
        result = log.read_lines(offset=9999, limit=10)
        assert "no lines at offset 9999" in result

    def test_limit_zero_clamps_to_one(self):
        log = _make_log_with_turns(1)
        result = log.read_lines(offset=1, limit=0)
        # limit=0 clamped to 1, so we get 1 line
        assert "1:" in result

    def test_limit_clamps_to_200(self):
        log = _make_log_with_turns(1)
        # Should not crash with huge limit
        result = log.read_lines(offset=1, limit=9999)
        assert "1:" in result

    def test_negative_offset_clamps(self):
        log = _make_log_with_turns(1)
        result = log.read_lines(offset=-5, limit=3)
        assert "1:" in result

    def test_truncation(self):
        """A very large log should get truncated at the char limit."""
        log = RunLog()
        for i in range(200):
            log.write_event("FILLER", "x" * 100)
        result = log.read_lines(offset=1, limit=200)
        assert "truncated" in result


# ---------------------------------------------------------------------------
# grep tests
# ---------------------------------------------------------------------------

class TestGrep:
    def test_basic_match(self):
        log = _make_log_with_turns(3)
        result = log.grep("TURN 2")
        assert "TURN 2" in result

    def test_no_match(self):
        log = _make_log_with_turns(1)
        result = log.grep("NONEXISTENT_PATTERN_XYZ")
        assert "no matches" in result

    def test_regex_match(self):
        log = _make_log_with_turns(3)
        result = log.grep(r"TURN \d")
        assert "TURN 1" in result
        assert "TURN 2" in result

    def test_invalid_regex(self):
        log = _make_log_with_turns(1)
        result = log.grep("[invalid")
        assert "invalid regex" in result

    def test_context_lines(self):
        log = _make_log_with_turns(3)
        result_0 = log.grep("GOAL: Test goal 2", context_lines=0)
        result_3 = log.grep("GOAL: Test goal 2", context_lines=3)
        # More context = more lines
        assert len(result_3) > len(result_0)

    def test_context_clamps_at_5(self):
        log = _make_log_with_turns(1)
        # Should not crash with huge context
        result = log.grep("TURN 1", context_lines=999)
        assert "TURN 1" in result

    def test_truncation(self):
        log = RunLog()
        for i in range(200):
            log.write_event("MATCH_ME", "x" * 100)
        result = log.grep("MATCH_ME", context_lines=0)
        assert "truncated" in result

    def test_separator_between_groups(self):
        log = _make_log_with_turns(3)
        result = log.grep("END TURN", context_lines=0)
        assert "---" in result  # separators between non-adjacent matches


# ---------------------------------------------------------------------------
# flush_to_disk tests
# ---------------------------------------------------------------------------

class TestFlushToDisk:
    def test_creates_file(self):
        log = _make_log_with_turns(1)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            path = f.name
        try:
            log.flush_to_disk(path)
            with open(path) as f:
                content = f.read()
            assert "=== GAME:" in content
            assert "=== TURN 1 |" in content
        finally:
            os.unlink(path)
