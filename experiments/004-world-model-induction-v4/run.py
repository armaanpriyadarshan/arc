"""Run experiment 002."""

import argparse
import logging
import os
import sys
from datetime import datetime

from dotenv import load_dotenv

# Load env from multiple locations
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))
load_dotenv(os.path.join(os.path.dirname(__file__), "../../ARC-AGI-3-Agents/.env"))

from agents import AGENTS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--game", default="ls20")
    parser.add_argument("-a", "--agent", default="explorer", choices=AGENTS.keys())
    args = parser.parse_args()

    # Create logs directory and timestamped log file
    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"{args.game}_{timestamp}.log")

    # Only configure our logger
    agent_logger = logging.getLogger("agents")
    agent_logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    fh.stream.reconfigure(line_buffering=True)

    agent_logger.addHandler(sh)
    agent_logger.addHandler(fh)

    agent_logger.info(f"Logging to {log_file}")

    # Silence third-party noise
    for name in ("openai", "httpx", "httpcore", "arc_agi", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)

    agent_cls = AGENTS[args.agent]
    agent = agent_cls(game_id=args.game)
    agent.run()


if __name__ == "__main__":
    main()
