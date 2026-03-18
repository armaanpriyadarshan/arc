"""Experiment 004: DreamerV3 agent for ARC-AGI-3.

Entry point. Run from the experiment directory:

    uv run python run.py --agent dreamer --game ls20

Optional overrides:
    --max-iterations N   Override max training iterations
    --seed N             Override random seed
"""

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

# Load .env from multiple locations — experiment dir first, then project root,
# then ARC-AGI-3-Agents (upstream framework .env as final fallback).
_here = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_here, ".env"))
load_dotenv(os.path.join(_here, "../../.env"))
load_dotenv(os.path.join(_here, "../../ARC-AGI-3-Agents/.env"))

from agents import AGENTS
from config import CONFIG


def _setup_logging(log_file: str) -> None:
    """Configure logging to stdout and a file.

    Only configures the ``agents`` logger (and sub-loggers) to avoid
    double-printing from the arc-agi SDK's own root-logger setup.

    Args:
        log_file: Absolute or relative path for the log file.
    """
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(fmt)

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    # Configure the top-level logger for this experiment's code
    for namespace in ("agents", "training", "evaluation", "llm", "env_wrapper", "replay_buffer"):
        ns_logger = logging.getLogger(namespace)
        ns_logger.setLevel(logging.DEBUG)
        ns_logger.addHandler(stdout_handler)
        ns_logger.addHandler(file_handler)
        ns_logger.propagate = False

    # Silence noisy SDK and HTTP loggers
    for noisy in ("openai", "httpx", "httpcore", "arc_agi", "urllib3", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 004: DreamerV3 agent for ARC-AGI-3"
    )
    parser.add_argument(
        "-a", "--agent",
        choices=list(AGENTS.keys()),
        default="dreamer",
        help="Agent to run (default: dreamer)",
    )
    parser.add_argument(
        "-g", "--game",
        default="ls20",
        help="Game ID (default: ls20)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Override max training iterations from config",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed from config",
    )
    args = parser.parse_args()

    log_file = os.path.join(_here, "experiment.log")
    _setup_logging(log_file)

    # Build config, applying any CLI overrides
    config = CONFIG.copy()
    if args.max_iterations is not None:
        config["max_iterations"] = args.max_iterations
    if args.seed is not None:
        config["seed"] = args.seed

    agent_cls = AGENTS[args.agent]
    agent = agent_cls(game_id=args.game, config=config)
    agent.run()


if __name__ == "__main__":
    main()
