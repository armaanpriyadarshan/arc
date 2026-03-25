"""Claude Code CLI integration — replaces OpenAI API calls.

Calls `claude -p` (non-interactive print mode) with a prompt that references
the run log file on disk. Claude Code uses its built-in Read/Grep/Bash tools
to analyze the log, then outputs a strategic briefing + action plan.
"""

import logging
import os
import subprocess

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts — adapted from RGB-Agent (experiment 010) for Claude Code
# ---------------------------------------------------------------------------

INITIAL_PROMPT = """\
You are a strategic advisor for an AI agent playing an unknown turn-based game
on a 64x64 grid. Each cell is an integer 0-15 (colors).

The agent's full run log is at this ABSOLUTE path: {log_path}

Read the log file to understand the current game state. The log uses section
markers you can search for:
  === GAME: ... ===           — game header
  === PROBE RESULTS ===       — initial action probing results
  === TURN N | ... ===        — each turn's data
  === END TURN N ===          — end of turn data

Key facts:
- Score (levels_completed) increases when the agent completes a level.
- GAME_OVER means the agent failed and was reset.
- Available actions: ACTION1-4 (directional moves), ACTION5 (interact),
  ACTION6(x,y) (click at coordinates 0-63), ACTION7 (undo), RESET.
- The agent must discover game mechanics through observation — nothing is
  hardcoded about any specific game.

Deeply analyze the log to understand:
1. What actions have been tried and what happened
2. What objects exist on the grid and how they behave
3. What game mechanics have been discovered
4. What the win condition might be

Your response MUST contain ALL sections below — the agent cannot act without [ACTIONS]:
1. A detailed strategic briefing (explain your reasoning, be specific)
2. Followed by exactly this separator and a 2-3 sentence action plan:

[PLAN]
<concise action plan the agent should follow until the next analysis>
"""

RESUME_PROMPT = """\
The run log has grown since your last analysis. The log file is at: {log_path}

Re-read the latest turns and update your strategic briefing.
Focus on what changed: new actions, their results, score transitions, and
whether the previous plan succeeded or failed.

Your response MUST contain ALL sections below — the agent cannot act without [ACTIONS]:
1. A detailed strategic briefing (explain your reasoning, be specific)
2. Followed by exactly this separator and a 2-3 sentence action plan:

[PLAN]
<concise action plan the agent should follow until the next analysis>
"""

PYTHON_ADDENDUM = """

Bash (and therefore Python) is available to you. Use Python to parse the game
log programmatically — do NOT try to visually read large grid dumps.

The log file is at: {log_path}

To extract data from the log:
```python
import re
data = open('{log_path}').read()
# Find all turn sections
turns = re.findall(r'=== TURN (\\d+).*?=== END TURN \\1 ===', data, re.DOTALL)
# Parse the latest turn
if turns:
    latest = turns[-1]
    print(latest)
```
Run Python inline with Bash to analyze patterns, count objects, etc.
"""

ACTIONS_ADDENDUM = """
3. Followed by exactly this separator and a JSON action plan (REQUIRED — the agent cannot act without this):

[ACTIONS]
{{"plan": [{{"action": "ACTION1"}}, {{"action": "ACTION6", "x": 3, "y": 7}}, ...], "reasoning": "why these steps"}}

Available actions: ACTION1-4 (moves), ACTION5 (interact), ACTION6 (click at x,y), ACTION7 (undo), RESET.
Each action MUST be a JSON object: {{"action": "ACTION6", "x": <col>, "y": <row>}} for clicks, {{"action": "ACTION1"}} for moves.
Plan 1-{plan_size} actions. Shorter plans (3-5 steps) are strongly preferred
because the agent can re-evaluate sooner.

CRITICAL INTERACTION RULE: In most grid games, you interact with objects by
MOVING ONTO them (ACTION1-4), not by clicking (ACTION6) or using ACTION5.
If you see an interesting object, try walking into it first before trying other actions.

YOU MUST END YOUR RESPONSE WITH THE [ACTIONS] SECTION. If you omit it, the agent
cannot take any actions and the turn is wasted. Always include [ACTIONS] even if
you are uncertain — a best-guess plan is better than no plan.
"""


class ClaudeCodeAnalyzer:
    """Calls Claude Code CLI to analyze the game log and produce action plans."""

    def __init__(self, log_path: str, plan_size: int = 5) -> None:
        self.log_path = log_path
        self.plan_size = plan_size
        self._is_first = True

    def analyze(self, action_num: int, retry_nudge: str = "") -> str | None:
        """Call Claude Code to analyze the game log and return response text.

        Returns the raw text output from Claude Code, or None on failure.
        """
        # Build the prompt
        if self._is_first:
            prompt = INITIAL_PROMPT.format(log_path=self.log_path)
            self._is_first = False
        else:
            prompt = RESUME_PROMPT.format(log_path=self.log_path)

        prompt += PYTHON_ADDENDUM.format(log_path=self.log_path)
        prompt += ACTIONS_ADDENDUM.format(plan_size=self.plan_size)

        if retry_nudge:
            prompt += f"\n\n{retry_nudge}"

        # Call claude CLI in non-interactive print mode
        cmd = [
            "claude",
            "--dangerously-skip-permissions",
            "-p", prompt,
            "--allowedTools", "Read,Grep,Bash",
            "--max-turns", "20",
            "--output-format", "text",
        ]

        logger.info(
            "[claude-code] Calling analyzer (action=%d, prompt=%d chars)",
            action_num, len(prompt),
        )

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=os.path.dirname(self.log_path) or ".",
            )

            if result.returncode != 0:
                stderr = result.stderr.strip()[:500] if result.stderr else "(no stderr)"
                logger.warning(
                    "[claude-code] Non-zero exit code %d: %s",
                    result.returncode, stderr,
                )
                # Still try to use stdout if there's output
                if result.stdout and result.stdout.strip():
                    logger.info("[claude-code] Got output despite non-zero exit, using it")
                    return result.stdout.strip()
                return None

            output = result.stdout.strip()
            if not output:
                logger.warning("[claude-code] Empty output from claude CLI")
                return None

            logger.info(
                "[claude-code] Got response (%d chars): %s",
                len(output), output[:500],
            )
            if result.stderr:
                logger.info("[claude-code] stderr: %s", result.stderr.strip()[:300])
            return output

        except subprocess.TimeoutExpired:
            logger.warning("[claude-code] Timed out after 300s")
            return None
        except FileNotFoundError:
            logger.error(
                "[claude-code] 'claude' CLI not found. "
                "Install it: npm install -g @anthropic-ai/claude-code"
            )
            return None
        except Exception as e:
            logger.warning("[claude-code] Unexpected error: %s", e)
            return None
