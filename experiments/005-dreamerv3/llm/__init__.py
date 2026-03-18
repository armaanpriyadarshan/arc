"""LLM integration for DreamerV3 agent.

Provides three entry points:

* ``generate_reward_shaping_function`` — ask Claude to write a reward shaping
  function from initial gameplay frames.
* ``analyze_action_effects`` — ask Claude which actions seem useful after a
  short random exploration phase.
* ``diagnose_stuck_agent`` — ask Claude for suggestions when the agent stops
  improving.
* ``default_shaped_reward`` — fallback reward shaping that requires no API call.
"""

from .reward_shaping import generate_reward_shaping_function, default_shaped_reward
from .action_analysis import analyze_action_effects
from .diagnosis import diagnose_stuck_agent

__all__ = [
    "generate_reward_shaping_function",
    "default_shaped_reward",
    "analyze_action_effects",
    "diagnose_stuck_agent",
]
