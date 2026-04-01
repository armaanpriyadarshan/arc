"""Experiment 013: Vision-First Agent — structural scene analysis + coordinate fix.

Replaces the flat object-inventory vision summarizer with a structural analyst
that identifies board regions, walls, corridors, and interactive zones.
Also fixes the ACTION6 coordinate mapping bug (LLM [row,col] → game [x=col,y=row]).

Base: experiment 012 (LLM vision + tool-use reasoning).
"""

AGENT = "vision_agent"
GAME = "ls20"
MAX_ACTIONS = 150
