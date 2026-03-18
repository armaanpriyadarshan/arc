"""Model router — picks the right model for each task.

Handles different API quirks per model family:
- gpt-5 family: max_completion_tokens, NO response_format support
- gpt-4.1 family: max_tokens, supports response_format
- o-series: max_completion_tokens, supports response_format
"""

import json
import logging
import os
import time

from openai import OpenAI

logger = logging.getLogger(__name__)

# Model assignments
VISION_MODEL = "gpt-4o"
REASONING_MODEL = "o4-mini"
FAST_MODEL = "gpt-4o-mini"

# Models that use max_completion_tokens instead of max_tokens
COMPLETION_TOKEN_MODELS = {
    "o3", "o4-mini", "o3-mini", "o1", "o1-mini",
    "gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5.4",
}

# Models that do NOT support response_format: json_object
NO_JSON_MODE_MODELS = {
    "gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5.4", "gpt-5.4-pro",
}


class ModelRouter:
    def __init__(self) -> None:
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        self.call_count = 0
        self.calls_by_model: dict[str, int] = {}

    def call(
        self,
        model: str,
        content: list[dict],
        max_tokens: int = 2000,
        json_mode: bool = True,
    ) -> str:
        """Make an LLM call with retry logic."""
        self.call_count += 1
        self.calls_by_model[model] = self.calls_by_model.get(model, 0) + 1

        # Token parameter name varies by model
        is_completion_token_model = any(model.startswith(prefix) for prefix in COMPLETION_TOKEN_MODELS)
        token_key = "max_completion_tokens" if is_completion_token_model else "max_tokens"

        kwargs: dict = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            token_key: max_tokens,
        }

        # Only add response_format for models that support it
        supports_json_mode = not any(model.startswith(prefix) for prefix in NO_JSON_MODE_MODELS)
        if json_mode and supports_json_mode:
            kwargs["response_format"] = {"type": "json_object"}
            # API requires the word "json" somewhere in the messages
            msgs = kwargs["messages"]
            has_json_word = any(
                "json" in (m.get("content", "") if isinstance(m.get("content"), str)
                           else str(m.get("content", "")))
                .lower()
                for m in msgs
            )
            if not has_json_word:
                msgs.append({"role": "user", "content": "Respond with JSON."})

        for attempt in range(5):
            try:
                response = self.client.chat.completions.create(**kwargs)
                result = response.choices[0].message.content or ""
                logger.info(f"[model #{self.call_count} {model}] {result[:500]}...")
                return result
            except Exception as e:
                wait = 2 ** attempt
                logger.warning(f"[model {model}] attempt {attempt + 1} failed: {e}, retry in {wait}s")
                time.sleep(wait)

        logger.error(f"[model {model}] all attempts failed")
        return "{}"

    def stats(self) -> str:
        parts = [f"{m}:{c}" for m, c in sorted(self.calls_by_model.items())]
        return f"Total: {self.call_count} ({', '.join(parts)})"
