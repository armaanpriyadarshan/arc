from .agent import ToolUseAgent
from .click_test import ClickTestAgent

AGENTS = {
    "vision_agent": ToolUseAgent,
    "explorer": ToolUseAgent,
    "click_test": ClickTestAgent,
}
