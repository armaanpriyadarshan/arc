"""CodexAgent: runs OpenAI Codex CLI natively to produce action plans."""
from __future__ import annotations

import atexit
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import IO, Optional

from rgb_agent.agent.prompts import (
    INITIAL_PROMPT,
    RESUME_PROMPT,
    ACTIONS_ADDENDUM,
    PYTHON_ADDENDUM,
)

log = logging.getLogger(__name__)


class _CodexEventStreamParser:
    """Parses JSONL events from `codex exec --json` and writes to an analyzer log."""

    def __init__(self, f: IO[str]):
        self._f = f
        self.accumulated_text = ""
        self.session_id: str | None = None

    def _write(self, label: str, content: str) -> None:
        if content:
            self._f.write(f"[{label}]\n{content}\n\n")
            self._f.flush()

    def handle(self, event: dict) -> None:
        etype = event.get("type", "")

        # Try to capture session ID from various possible locations
        for key in ("session_id", "sessionId", "id"):
            sid = event.get(key)
            if isinstance(sid, str) and len(sid) > 8 and not self.session_id:
                self.session_id = sid

        if etype == "message":
            role = event.get("role", "")
            content = event.get("content", [])
            if role == "assistant":
                if isinstance(content, str):
                    self.accumulated_text += content
                    self._write("ASSISTANT", content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            btype = block.get("type", "")
                            if btype == "text":
                                text = block.get("text", "")
                                self.accumulated_text += text
                                self._write("ASSISTANT", text)
                            elif btype in ("thinking", "reasoning"):
                                self._write("THINKING", block.get("text", ""))
                        elif isinstance(block, str):
                            self.accumulated_text += block
                            self._write("ASSISTANT", block)

        elif etype == "function_call":
            name = event.get("name", "?")
            args = event.get("arguments", event.get("call_id", ""))
            self._write(f"TOOL CALL: {name}", str(args)[:4000])

        elif etype == "function_call_output":
            output = event.get("output", "")
            is_error = event.get("status", "") == "incomplete"
            label = "TOOL RESULT ERROR" if is_error else "TOOL RESULT"
            self._write(label, str(output)[:4000])

        elif etype == "error":
            msg = event.get("message", event.get("error", str(event)))
            self._write("ERROR", str(msg))
            log.error("Codex error: %s", msg)
            if any(kw in str(msg).lower() for kw in ("overflow", "too long", "context")):
                self.session_id = None

        elif etype in ("text", "text_delta"):
            text = event.get("text", event.get("delta", ""))
            if text:
                self.accumulated_text += text
                self._write("ASSISTANT", text)

        elif etype in ("reasoning", "thinking"):
            text = event.get("text", "")
            self._write("THINKING", text)

        elif etype == "tool_use":
            part = event.get("part", event)
            name = part.get("tool", part.get("name", "?"))
            state = part.get("state", {})
            status = state.get("status", "") if isinstance(state, dict) else str(state)
            if status in ("running", "completed", "done", ""):
                input_data = (state.get("input", {}) if isinstance(state, dict)
                              else part.get("input", part.get("arguments", {})))
                input_str = json.dumps(input_data, indent=2) if isinstance(input_data, dict) else str(input_data)
                self._write(f"TOOL CALL: {name}", input_str)
            if status in ("completed", "done"):
                output = (state.get("output", state.get("result", ""))
                          if isinstance(state, dict) else "")
                is_error = (state.get("is_error", False) if isinstance(state, dict) else False)
                label = "TOOL RESULT ERROR" if is_error else "TOOL RESULT"
                self._write(label, str(output)[:4000])

        elif etype == "step_start":
            sid = event.get("sessionID", event.get("session_id"))
            if sid and not self.session_id:
                self.session_id = sid

        elif etype in ("step_finish", "result"):
            cost = event.get("cost", event.get("total_cost_usd",
                             event.get("part", {}).get("cost") if isinstance(event.get("part"), dict) else None))
            if cost is not None:
                self._write("RESULT", f"cost=${cost}")
            result_text = event.get("result", "").strip() if isinstance(event.get("result"), str) else ""
            if result_text and not self.accumulated_text.strip():
                self.accumulated_text = result_text

        elif etype == "assistant":
            for block in event.get("message", {}).get("content", []):
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "thinking":
                    self._write("THINKING", block.get("thinking", block.get("text", "")))
                elif btype == "text":
                    text = block["text"]
                    self.accumulated_text += text
                    self._write("ASSISTANT", text)
                elif btype == "tool_use":
                    self._write(f"TOOL CALL: {block.get('name', '?')}",
                                json.dumps(block.get("input", {}), indent=2))

        elif etype == "user":
            for block in event.get("message", {}).get("content", []):
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_result":
                    content = block.get("content", "")
                    if isinstance(content, list):
                        text = "\n".join(
                            c.get("text", "") for c in content
                            if isinstance(c, dict) and c.get("type") == "text"
                        )
                    elif isinstance(content, str):
                        text = content
                    else:
                        text = str(content)
                    is_error = block.get("is_error", False)
                    label = "TOOL RESULT ERROR" if is_error else "TOOL RESULT"
                    self._write(label, text[:4000])

        else:
            self._f.write(f"[RAW:{etype}] {json.dumps(event)[:500]}\n")
            self._f.flush()


class CodexAgent:
    """Runs OpenAI Codex CLI natively to analyze game logs and produce action plans.

    Drop-in replacement for OpenCodeAgent — same analyze() interface, no Docker required.
    """

    def __init__(
        self,
        *,
        model: str = "gpt-5.4",
        plan_size: int = 5,
        timeout: Optional[int] = None,
        resume_session: bool = True,
    ) -> None:
        if not shutil.which("codex"):
            raise FileNotFoundError(
                "'codex' CLI not found. Install with:\n"
                "  npm install -g @openai/codex\n"
                "  # or: brew install --cask codex"
            )
        log.info("using Codex CLI (native, no Docker)")

        self._model = self._normalize_model(model)
        self._plan_size = plan_size
        self._timeout = timeout
        self._resume_session = resume_session

        self._workspaces: dict[str, Path] = {}
        self._workspace_lock = threading.Lock()

        self._call_count: dict[str, int] = {}
        self._call_count_lock = threading.Lock()

        atexit.register(self._cleanup)

    @staticmethod
    def _normalize_model(model: str) -> str:
        if "/" in model:
            return model.split("/", 1)[1]
        return model

    def _get_workspace(self, log_path: Path) -> Path:
        key = str(log_path)
        with self._workspace_lock:
            if key not in self._workspaces:
                workspace = Path(tempfile.mkdtemp(prefix="codex_workspace_"))
                os.chmod(workspace, 0o755)
                self._workspaces[key] = workspace
                log.info("created workspace: %s", workspace)
            return self._workspaces[key]

    def _build_prompt(self, log_name: str, is_first: bool) -> str:
        if self._resume_session and not is_first:
            prompt = RESUME_PROMPT.format(log_path=log_name)
        else:
            prompt = INITIAL_PROMPT.format(log_path=log_name)
        prompt += PYTHON_ADDENDUM.format(log_path=log_name)
        prompt += ACTIONS_ADDENDUM.format(plan_size=self._plan_size)
        return prompt

    def _cleanup(self) -> None:
        with self._workspace_lock:
            for workspace in self._workspaces.values():
                try:
                    shutil.rmtree(workspace, ignore_errors=True)
                except Exception as e:
                    log.warning("failed to cleanup workspace %s: %s", workspace, e)
            self._workspaces.clear()

    def analyze(self, log_path: Path, action_num: int, retry_nudge: str = "") -> Optional[str]:
        """Analyze the game log and return the agent's response text, or None on failure."""
        if not log_path.exists():
            return None

        analyzer_log = log_path.parent / (log_path.stem + "_analyzer.txt")
        path_key = str(log_path)

        # Track call count for session resumption
        is_first = True
        with self._call_count_lock:
            count = self._call_count.get(path_key, 0)
            if count > 0 and self._resume_session:
                is_first = False
            self._call_count[path_key] = count + 1

        workspace = self._get_workspace(log_path)

        try:
            # Copy latest game log into workspace
            shutil.copy2(log_path, workspace / log_path.name)

            # Clear previous output file
            output_file = workspace / "_codex_output.txt"
            if output_file.exists():
                output_file.unlink()

            prompt = self._build_prompt(log_path.name, is_first)
            if retry_nudge:
                prompt += f"\n\n{retry_nudge}"

            # Build command
            cmd = ["codex", "exec"]

            if self._resume_session and not is_first:
                cmd.extend(["resume", "--last"])

            cmd.extend([
                "--json",
                "-o", str(output_file),
                "--full-auto",
                "--sandbox", "workspace-write",
                "-C", str(workspace),
                "--model", self._model,
                "--skip-git-repo-check",
                prompt,
            ])

            log.info("exec codex model=%s is_first=%s action=%d",
                     self._model, is_first, action_num)

            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            stderr_lines: list[str] = []

            def drain_stderr():
                for line in proc.stderr:
                    stderr_lines.append(line.rstrip("\n"))
                    log.debug("STDERR: %s", line[:300].rstrip())

            stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
            stderr_thread.start()

            with open(analyzer_log, "a", encoding="utf-8") as f:
                f.write(f"\n--- action={action_num} | {datetime.now().strftime('%H:%M:%S')} | codex ---\n")
                if is_first or not self._resume_session:
                    f.write(f"[SYSTEM PROMPT]\n{prompt}\n\n")
                f.flush()

                parser = _CodexEventStreamParser(f)
                deadline = time.monotonic() + self._timeout if self._timeout is not None else None

                while True:
                    line = proc.stdout.readline()
                    if not line:
                        break
                    if deadline is not None and time.monotonic() > deadline:
                        proc.kill()
                        f.write("[TIMEOUT]\n")
                        log.warning("timed out at action %d", action_num)
                        return None

                    line = line.rstrip("\n")
                    if not line.strip():
                        continue
                    try:
                        parser.handle(json.loads(line))
                    except json.JSONDecodeError:
                        f.write(f"[RAW] {line}\n")
                        f.flush()

                proc.wait()
                stderr_thread.join(timeout=5)
                if stderr_lines:
                    f.write(f"\n--- STDERR ---\n{''.join(l + chr(10) for l in stderr_lines)}")
                    f.flush()

            # Extract response: primary from -o file, fallback from accumulated text
            response_text = ""
            if output_file.exists():
                response_text = output_file.read_text(encoding="utf-8").strip()
                if response_text:
                    log.info("action=%d: got %d chars from -o file", action_num, len(response_text))

            if not response_text:
                response_text = parser.accumulated_text.strip()
                if response_text:
                    log.info("action=%d: got %d chars from JSONL stream (fallback)", action_num, len(response_text))

            if proc.returncode != 0 or not response_text:
                log.warning("action=%d failed: rc=%d, response_len=%d",
                            action_num, proc.returncode, len(response_text) if response_text else 0)
                # Clear call count so next call starts fresh
                if self._resume_session:
                    with self._call_count_lock:
                        self._call_count.pop(path_key, None)
                return None

            log.info("action=%d OK (%d chars)", action_num, len(response_text))
            return response_text

        except Exception as e:
            log.error("unexpected error: %s", e, exc_info=True)
            return None
