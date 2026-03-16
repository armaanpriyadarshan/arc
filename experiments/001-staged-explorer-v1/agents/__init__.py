from .base import Agent
from .explorer import ExplorerAgent

AGENTS: dict[str, type[Agent]] = {
    "explorer": ExplorerAgent,
}
