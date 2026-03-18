"""Agent registry for experiment 004.

Following the repo convention each experiment exports an ``AGENTS`` dict
mapping string keys to agent classes. The entry point (``run.py``) uses
this dict to instantiate the chosen agent.
"""

from .agent import DreamerAgent
from .perception_agent import PerceptionAgent
from .dqn_agent import DQNAgent

AGENTS: dict[str, type] = {
    "dreamer": DreamerAgent,
    "perception": PerceptionAgent,
    "dqn": DQNAgent,
}

__all__ = ["AGENTS", "DreamerAgent", "PerceptionAgent", "DQNAgent"]
