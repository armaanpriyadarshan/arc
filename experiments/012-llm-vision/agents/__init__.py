from .agent import ToolUseAgent
from .click_test import ClickTestAgent

AGENTS = {
    "explorer": ToolUseAgent,
    "click_test": ClickTestAgent,
}
