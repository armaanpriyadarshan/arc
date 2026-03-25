"""Run experiment 004-v5."""

import argparse
import logging
import os

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

    agent_cls = AGENTS[args.agent]
    agent = agent_cls(game_id=args.game)
    agent.run()


if __name__ == "__main__":
    main()
