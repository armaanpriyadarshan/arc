"""CLI entry point for running experiments."""

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()
load_dotenv("ARC-AGI-3-Agents/.env")

from agents import AGENTS


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an ARC-AGI-3 experiment")
    parser.add_argument(
        "-a", "--agent",
        choices=AGENTS.keys(),
        required=True,
        help="Agent to run",
    )
    parser.add_argument(
        "-g", "--game",
        default="ls20",
        help="Game ID (default: ls20)",
    )
    parser.add_argument(
        "-e", "--experiment",
        default=None,
        help="Experiment directory (e.g. experiments/001-staged-explorer-v1)",
    )
    args = parser.parse_args()

    # Set up experiment output directory
    if args.experiment:
        os.makedirs(args.experiment, exist_ok=True)
        log_file = os.path.join(args.experiment, "experiment.log")
    else:
        log_file = "experiment.log"

    # Configure logging — only OUR loggers, not the root logger
    # This prevents arc-agi SDK from double-printing everything
    our_logger = logging.getLogger("agents")
    our_logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    our_logger.addHandler(stdout_handler)
    our_logger.addHandler(file_handler)

    # Silence noisy third-party loggers
    for name in ("openai", "httpx", "httpcore", "arc_agi", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)

    agent_cls = AGENTS[args.agent]
    agent = agent_cls(game_id=args.game)
    agent.run()


if __name__ == "__main__":
    main()
