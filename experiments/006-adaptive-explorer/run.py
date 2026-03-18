"""Experiment 005: Adaptive Explorer agent for ARC-AGI-3.

Entry point:
    uv run python run.py --agent adaptive --game ls20
"""

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

_here = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_here, ".env"))
load_dotenv(os.path.join(_here, "../../.env"))
load_dotenv(os.path.join(_here, "../../ARC-AGI-3-Agents/.env"))

from agents import AGENTS
from config import CONFIG


def _setup_logging(log_file: str) -> None:
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(fmt)

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    for namespace in ("agents", "env_wrapper"):
        ns_logger = logging.getLogger(namespace)
        ns_logger.setLevel(logging.DEBUG)
        ns_logger.addHandler(stdout_handler)
        ns_logger.addHandler(file_handler)
        ns_logger.propagate = False

    for noisy in ("openai", "httpx", "httpcore", "arc_agi", "urllib3", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 005: Adaptive Explorer")
    parser.add_argument("-a", "--agent", choices=list(AGENTS.keys()), default="adaptive")
    parser.add_argument("-g", "--game", default="ls20")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--log-suffix", default="")
    args = parser.parse_args()

    suffix = f"_{args.log_suffix}" if args.log_suffix else ""
    log_file = os.path.join(_here, f"experiment{suffix}.log")
    _setup_logging(log_file)

    config = CONFIG.copy()
    if args.max_steps is not None:
        config["max_episode_steps"] = args.max_steps

    agent_cls = AGENTS[args.agent]
    agent = agent_cls(game_id=args.game, config=config)
    agent.run()


if __name__ == "__main__":
    main()
