"""Experiment 004-v3: Bayesian hypothesis observe-act agent

Builds on v2 (auto-probe + structured hypotheses) with three major changes:
1. Bayesian probabilities replace binary hypothesis status (testing/confirmed/rejected).
   Every hypothesis now carries a probability 0.0-1.0 and a tests[] list documenting
   exactly what evidence was gathered.
2. Expanded auto-probe for parameterized actions — ACTION6 (click) is probed at 3-5
   diverse coordinates instead of once, preventing premature dismissal.
3. Stale hypothesis detection — if the top hypothesis is unchanged for 5+ turns or
   the agent has 3+ turns with 0 cell changes, a warning is injected into the prompt.

References experiment 004-v2 analysis: the core failure modes were (a) binary hypothesis
states killing exploration once something was marked confirmed/rejected, (b) no evidence
tracking so weak evidence led to overgeneralization, and (c) ACTION6 being dismissed
after a single failed click test.
"""

AGENT = "explorer"
GAME = "ls20"
MAX_ACTIONS = 150
