"""DreamerV3 model components for ARC-AGI-3 RL agent.

Exports all model classes and utility functions used by the training loop
and agent.

Models:
    Encoder          : CNN observation encoder (B, 16, 64, 64) -> (B, embed_dim)
    Decoder          : Transposed CNN reconstruction decoder
    RSSM             : Recurrent State-Space Model (world model core)
    HierarchicalActor: Policy with click-action spatial head
    Critic           : MLP value estimator (symlog output)
    SlowCritic       : EMA target critic for stable bootstrapping
    WorldModel       : Full world model combining all components

Utilities:
    symlog           : Symmetric log transform for reward compression
    symexp           : Inverse of symlog
"""

from .encoder import Encoder
from .decoder import Decoder
from .rssm import RSSM
from .actor import HierarchicalActor
from .critic import Critic, SlowCritic
from .world_model import WorldModel, symlog, symexp

__all__ = [
    "Encoder",
    "Decoder",
    "RSSM",
    "HierarchicalActor",
    "Critic",
    "SlowCritic",
    "WorldModel",
    "symlog",
    "symexp",
]
