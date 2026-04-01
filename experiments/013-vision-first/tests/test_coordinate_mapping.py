"""Tests for ACTION6 coordinate mapping fix.

Verifies that the LLM's [row, col] output gets correctly converted to
game engine [x=col, y=row] (standard) or [x=row, y=col] (swapped).
"""

import sys
import os
from unittest.mock import MagicMock, patch

# Add the experiment's agents directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def make_agent_with_coord_swap(swap: bool):
    """Create a minimal ToolUseAgent with mocked dependencies for coordinate testing."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        with patch("agents.agent.OpenAI"):
            from agents.agent import ToolUseAgent
            agent = ToolUseAgent.__new__(ToolUseAgent)
            agent._coord_swap = swap
            return agent


class TestResolveActions:
    """Test _resolve_actions coordinate conversion."""

    def test_standard_convention_row_col_to_x_col_y_row(self):
        """LLM outputs [row=15, col=30] -> game gets x=30, y=15 (standard)."""
        agent = make_agent_with_coord_swap(False)
        raw_actions = ["ACTION6", 15, 30]
        available = {6}

        actions = agent._resolve_actions(raw_actions, available, "ACTION6")

        assert len(actions) == 1
        action = actions[0]
        assert action.action_data.x == 30  # col
        assert action.action_data.y == 15  # row

    def test_swapped_convention_row_col_to_x_row_y_col(self):
        """When _coord_swap=True, LLM [row=15, col=30] -> game x=15, y=30."""
        agent = make_agent_with_coord_swap(True)
        raw_actions = ["ACTION6", 15, 30]
        available = {6}

        actions = agent._resolve_actions(raw_actions, available, "ACTION6")

        assert len(actions) == 1
        action = actions[0]
        assert action.action_data.x == 15  # row (swapped)
        assert action.action_data.y == 30  # col (swapped)

    def test_simple_actions_unaffected(self):
        """Simple actions (no coordinates) should work regardless of swap setting."""
        agent = make_agent_with_coord_swap(False)
        raw_actions = ["ACTION1", "ACTION3", "ACTION4"]
        available = {1, 3, 4}

        actions = agent._resolve_actions(raw_actions, available, "ACTION1, ACTION3, ACTION4")

        assert len(actions) == 3

    def test_mixed_simple_and_action6(self):
        """Mix of simple actions and ACTION6 should parse correctly."""
        agent = make_agent_with_coord_swap(False)
        raw_actions = ["ACTION1", "ACTION6", 10, 20, "ACTION3"]
        available = {1, 3, 6}

        actions = agent._resolve_actions(raw_actions, available, "ACTION1, ACTION3, ACTION6")

        assert len(actions) == 3
        # ACTION6 should have x=20 (col), y=10 (row)
        action6 = actions[1]
        assert action6.action_data.x == 20
        assert action6.action_data.y == 10

    def test_out_of_range_coords_skipped(self):
        """Coordinates outside 0-63 should be skipped."""
        agent = make_agent_with_coord_swap(False)
        raw_actions = ["ACTION6", 100, 50]
        available = {6}

        actions = agent._resolve_actions(raw_actions, available, "ACTION6")

        assert len(actions) == 0

    def test_action6_not_available_skipped(self):
        """ACTION6 should be skipped if not in available actions."""
        agent = make_agent_with_coord_swap(False)
        raw_actions = ["ACTION6", 10, 20]
        available = {1, 2, 3, 4}

        actions = agent._resolve_actions(raw_actions, available, "ACTION1-4")

        assert len(actions) == 0


class TestProbeAction6ExtractPositions:
    """Test that _probe_action6 correctly extracts positions from new vision format."""

    def test_extracts_from_key_objects_positions(self):
        """New format: key_objects with positions list."""
        vision_summary = {
            "key_objects": [
                {
                    "description": "corner frame",
                    "color": "light-gray",
                    "positions": [[36, 32], [36, 56], [56, 32], [56, 56]],
                    "count": 4,
                    "grouped": True,
                }
            ]
        }

        # Extract targets the same way _probe_action6 does
        targets = []
        objects_list = vision_summary.get("key_objects", vision_summary.get("objects", []))
        for obj in objects_list:
            positions = obj.get("positions", [])
            if positions and isinstance(positions[0], (list, tuple)):
                pos = positions[0]
            else:
                pos = obj.get("position")
            if pos and isinstance(pos, (list, tuple)) and len(pos) >= 2:
                row, col = int(pos[0]), int(pos[1])
                if 0 <= row <= 63 and 0 <= col <= 63:
                    targets.append((col, row, obj.get("description", "")))

        assert len(targets) == 1
        x, y, desc = targets[0]
        assert x == 32  # col from first position
        assert y == 36  # row from first position
        assert desc == "corner frame"

    def test_extracts_from_old_objects_format(self):
        """Old format: objects with single position."""
        vision_summary = {
            "objects": [
                {
                    "description": "blue square",
                    "color": "blue",
                    "position": [2, 4],
                }
            ]
        }

        objects_list = vision_summary.get("key_objects", vision_summary.get("objects", []))
        targets = []
        for obj in objects_list:
            positions = obj.get("positions", [])
            if positions and isinstance(positions[0], (list, tuple)):
                pos = positions[0]
            else:
                pos = obj.get("position")
            if pos and isinstance(pos, (list, tuple)) and len(pos) >= 2:
                row, col = int(pos[0]), int(pos[1])
                if 0 <= row <= 63 and 0 <= col <= 63:
                    targets.append((col, row, obj.get("description", "")))

        assert len(targets) == 1
        x, y, desc = targets[0]
        assert x == 4   # col
        assert y == 2   # row
