"""Vision testing harness — intercepts LLM calls to dump full prompts, images, and responses.

Replaces self.client with a thin proxy so that every call to
client.responses.create() writes the complete prompt (text + decoded images),
symbolic state, and raw LLM response to disk.  Zero duplication of prompt-building
logic — if the parent _think_and_act changes, logging still works.
"""

import base64
import json
import logging
import os
import time
from datetime import datetime

from agents.agent import ToolUseAgent
from agents.symbolic import grid_to_symbolic, diff_symbolic

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Proxy layer — intercepts client.responses.create()
# ---------------------------------------------------------------------------

class _LoggingResponses:
    """Wraps the real ``client.responses`` object.

    Intercepts ``create(**kwargs)`` to dump the full prompt, decoded images,
    LLM config, and response to a per-turn directory on disk.
    """

    def __init__(self, real_responses, log_dir: str):
        self._real = real_responses
        self._log_dir = log_dir
        self._turn = 0
        # Set by VisionTestAgent before each call
        self.symbolic_state: dict | None = None
        self.symbolic_diff: list[dict] | None = None

    def create(self, **kwargs):
        self._turn += 1
        turn_dir = os.path.join(self._log_dir, f"turn_{self._turn:03d}")
        os.makedirs(turn_dir, exist_ok=True)

        try:
            self._dump_input(kwargs, turn_dir)
            self._dump_symbolic(turn_dir)
            self._dump_config(kwargs, turn_dir)
        except Exception as exc:
            logger.warning(f"[vision-test] Failed to dump input for turn {self._turn}: {exc}")

        # Call the real API
        try:
            response = self._real.create(**kwargs)
        except Exception as exc:
            error_path = os.path.join(turn_dir, "error.txt")
            with open(error_path, "w") as f:
                f.write(f"{type(exc).__name__}: {exc}\n")
            raise

        try:
            self._dump_response(response, turn_dir)
        except Exception as exc:
            logger.warning(f"[vision-test] Failed to dump response for turn {self._turn}: {exc}")

        return response

    def __getattr__(self, name):
        return getattr(self._real, name)

    # -- private helpers --------------------------------------------------

    def _dump_input(self, kwargs, turn_dir: str):
        """Extract content blocks from the API kwargs and write prompt.json + images."""
        input_messages = kwargs.get("input", [])
        if not input_messages:
            return

        # The agent sends a single user message: input_messages[0]["content"]
        content_blocks = []
        for msg in input_messages:
            if isinstance(msg, dict) and "content" in msg:
                content_blocks = msg["content"]
                break

        prompt_entries = []
        image_idx = 0

        for block in content_blocks:
            if not isinstance(block, dict):
                continue

            block_type = block.get("type", "")

            if block_type == "input_text":
                prompt_entries.append({
                    "type": "text",
                    "text": block.get("text", ""),
                })

            elif block_type == "input_image":
                image_url = block.get("image_url", "")
                image_file = f"image_{image_idx}.png"

                # Decode base64 image data and write to disk
                if image_url.startswith("data:image/png;base64,"):
                    b64_data = image_url[len("data:image/png;base64,"):]
                    try:
                        raw_bytes = base64.b64decode(b64_data)
                        image_path = os.path.join(turn_dir, image_file)
                        with open(image_path, "wb") as f:
                            f.write(raw_bytes)
                    except Exception as exc:
                        logger.warning(f"[vision-test] Failed to decode image_{image_idx}: {exc}")

                prompt_entries.append({
                    "type": "image",
                    "file": image_file,
                })
                image_idx += 1

        prompt_path = os.path.join(turn_dir, "prompt.json")
        with open(prompt_path, "w") as f:
            json.dump(prompt_entries, f, indent=2)

    def _dump_symbolic(self, turn_dir: str):
        """Write symbolic_state.json and symbolic_diff.json."""
        if self.symbolic_state is not None:
            path = os.path.join(turn_dir, "symbolic_state.json")
            with open(path, "w") as f:
                json.dump(self.symbolic_state, f, indent=2)

        if self.symbolic_diff is not None:
            path = os.path.join(turn_dir, "symbolic_diff.json")
            with open(path, "w") as f:
                json.dump(self.symbolic_diff, f, indent=2)

    def _dump_config(self, kwargs, turn_dir: str):
        """Write llm_config.json with model, temperature, etc."""
        config = {
            "model": kwargs.get("model", "unknown"),
            "temperature": kwargs.get("temperature"),
            "max_output_tokens": kwargs.get("max_output_tokens"),
        }
        # Include reasoning settings if present
        for key in ("reasoning", "reasoning_effort"):
            if key in kwargs:
                config[key] = kwargs[key]

        path = os.path.join(turn_dir, "llm_config.json")
        with open(path, "w") as f:
            json.dump(config, f, indent=2)

    def _dump_response(self, response, turn_dir: str):
        """Write response.json with the raw LLM output text."""
        raw = ""
        if hasattr(response, "output_text") and response.output_text:
            raw = response.output_text
        elif hasattr(response, "output"):
            for item in (response.output or []):
                if hasattr(item, "content"):
                    for block in (item.content or []):
                        if hasattr(block, "text") and block.text:
                            raw = block.text
                            break
                if raw:
                    break

        path = os.path.join(turn_dir, "response.json")
        with open(path, "w") as f:
            json.dump({"raw_output": raw}, f, indent=2)


# ---------------------------------------------------------------------------
# Drop-in client replacement
# ---------------------------------------------------------------------------

class _LoggingClient:
    """Drop-in replacement for ``self.client`` (OpenAI client).

    Proxies ``client.responses`` through ``_LoggingResponses``; all other
    attribute access is forwarded to the real client.
    """

    def __init__(self, real_client, log_dir: str):
        self._real = real_client
        self.responses = _LoggingResponses(real_client.responses, log_dir)

    def __getattr__(self, name):
        return getattr(self._real, name)


# ---------------------------------------------------------------------------
# VisionTestAgent — minimal ToolUseAgent subclass
# ---------------------------------------------------------------------------

class VisionTestAgent(ToolUseAgent):
    """Subclass that logs every LLM call's full prompt, images, and response.

    Usage:
        uv run python synthetic_games/visual_agent_play.py --game wiring --standalone \\
          --agent experiments/004-world-model-induction-v3/vision_testing/vision_test_agent.py:VisionTestAgent
    """

    def __init__(self, game_id: str, env=None) -> None:
        super().__init__(game_id, env)

        # Create run directory: vision_logs/run_<timestamp>/
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self._run_dir = os.path.join(base_dir, "vision_logs", f"run_{timestamp}")
        os.makedirs(self._run_dir, exist_ok=True)

        # Replace self.client with the logging proxy
        self.client = _LoggingClient(self.client, self._run_dir)

        logger.info(f"[vision-test] Logging to {self._run_dir}")

    def _think_and_act(self, frame) -> list:
        """Compute symbolic state read-only, attach to proxy, then delegate to parent."""
        # Compute symbolic state + diff WITHOUT mutating self.prev_symbolic
        # (the parent's _think_and_act handles that mutation at line 777)
        symbolic = grid_to_symbolic(self.current_grid)
        sym_diff = diff_symbolic(self.prev_symbolic, symbolic) if self.prev_symbolic else []

        # Attach to the proxy so _LoggingResponses can write them
        self.client.responses.symbolic_state = symbolic
        self.client.responses.symbolic_diff = sym_diff

        # Delegate entirely to parent — no duplication of prompt logic
        return super()._think_and_act(frame)
