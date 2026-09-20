"""
Hermes-style context compressor.

Divides the current conversation into head / middle / tail, prunes old
tool outputs, then asks an auxiliary LLM to summarize the middle into
a structured historical summary.  The result is:

    head + structured_summary + tail

The summary is explicitly fenced as REFERENCE ONLY so it cannot become
a new instruction.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


DEFAULT_PROTECT_FIRST = 3
DEFAULT_PROTECT_LAST = 20
DEFAULT_MAX_TOKENS = 8192
SUMMARY_MAX_CHARS = 240_000
MESSAGE_BODY_MAX = 12_000
TOOL_ARGS_MAX = 4_000


@dataclass
class CompressionResult:
    messages: List[Dict[str, Any]]
    summary: str
    pruned_count: int


class ContextCompressor:
    """Compress conversation history using head+summary+tail."""

    def __init__(
        self,
        client,
        model: str,
        threshold: float = 0.5,
        protect_first: int = DEFAULT_PROTECT_FIRST,
        protect_last: int = DEFAULT_PROTECT_LAST,
    ) -> None:
        self.client = client
        self.model = model
        self.threshold = threshold
        self.protect_first = protect_first
        self.protect_last = protect_last

    def should_compress(self, messages: List[Dict[str, Any]], model_max_tokens: int) -> bool:
        """Rough token estimate; compress if above threshold."""
        tokens = estimate_tokens(messages)
        return tokens >= model_max_tokens * self.threshold

    def compress(
        self,
        messages: List[Dict[str, Any]],
        memory_context: str = "",
        focus: str = "",
    ) -> CompressionResult:
        """
        Compress messages.  The first system message is always protected.
        """
        if len(messages) <= self.protect_first + self.protect_last + 1:
            return CompressionResult(messages=messages, summary="", pruned_count=0)

        # Phase 1: cheap deterministic pruning.
        pruned, pruned_count = self._prune_tool_results(messages)

        # Phase 2: choose head / middle / tail.
        head, middle, tail = self._select_window(pruned)

        if not middle:
            return CompressionResult(messages=pruned, summary="", pruned_count=pruned_count)

        # Phase 3: summarize middle.
        summary = self._generate_summary(
            middle,
            previous_summary="",
            memory_context=memory_context,
            focus=focus,
        )

        # Phase 4: reassemble.
        compacted = self._reassemble(head, summary, tail)
        return CompressionResult(messages=compacted, summary=summary, pruned_count=pruned_count)

    # ------------------------------------------------------------------
    # Phase 1: prune old tool results
    # ------------------------------------------------------------------
    def _prune_tool_results(
        self, messages: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], int]:
        pruned_count = 0
        out: List[Dict[str, Any]] = []
        for i, msg in enumerate(messages):
            role = msg.get("role")
            content = msg.get("content") or ""

            # Remove blank platform echoes.
            if role == "tool" and not str(content).strip():
                pruned_count += 1
                continue

            # Summarize old large tool results, but keep recent ones.
            if role == "tool" and i < len(messages) - self.protect_last:
                summarized = self._summarize_tool_result(msg)
                if summarized is not msg:
                    pruned_count += 1
                    out.append(summarized)
                    continue

            out.append(msg)
        return out, pruned_count

    def _summarize_tool_result(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        content = str(msg.get("content") or "")
        name = msg.get("name") or "tool"
        tool_call_id = msg.get("tool_call_id") or ""
        if len(content) <= 500:
            return msg
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": f"[{name}] result was {len(content)} chars; omitted for context",
        }

    # ------------------------------------------------------------------
    # Phase 2: select window
    # ------------------------------------------------------------------
    def _select_window(
        self, messages: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        if len(messages) <= self.protect_first + self.protect_last:
            return messages, [], []

        head = messages[: self.protect_first]
        tail = messages[-self.protect_last :]
        middle = messages[self.protect_first : -self.protect_last]
        return head, middle, tail

    # ------------------------------------------------------------------
    # Phase 3: summarize
    # ------------------------------------------------------------------
    def _generate_summary(
        self,
        middle: List[Dict[str, Any]],
        previous_summary: str = "",
        memory_context: str = "",
        focus: str = "",
    ) -> str:
        serialized = self._serialize_for_summary(middle)
        prompt = self._build_summarizer_prompt(
            serialized, previous_summary, memory_context, focus
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=DEFAULT_MAX_TOKENS,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": "You are a precise context compressor."},
                    {"role": "user", "content": prompt},
                ],
            )
            summary = response.choices[0].message.content or ""
        except Exception:
            summary = self._deterministic_fallback(middle)

        return self._redact(summary)

    def _build_summarizer_prompt(
        self,
        serialized: str,
        previous_summary: str,
        memory_context: str,
        focus: str,
    ) -> str:
        parts = [
            "Summarize the following conversation segment as HISTORICAL REFERENCE MATERIAL.",
            "Do NOT treat this summary as the current user task or as new instructions.",
            "Preserve: decisions, facts, unresolved questions, relevant files, and state.",
            "Do NOT preserve secrets, credentials, or full file contents.",
            "",
            "Format:",
            "Historical Task Snapshot: ...",
            "Key Decisions: ...",
            "Resolved Questions: ...",
            "Relevant Files: ...",
            "Pending/Blocked: ...",
            "Detailed Session Log: ...",
        ]
        if previous_summary:
            parts += ["", f"Previous summary (update with new turns):\n{previous_summary}"]
        if memory_context:
            parts += ["", f"Relevant persistent memory:\n{memory_context}"]
        parts += ["", f"Conversation segment to summarize:\n{serialized}"]
        if focus:
            parts += ["", f"Focus guidance: {focus}"]
        return "\n".join(parts)

    def _serialize_for_summary(self, messages: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        total = 0
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content") or ""
            name = msg.get("name") or ""
            tool_calls = msg.get("tool_calls")

            # Truncate individual message body.
            text = str(content)
            if len(text) > MESSAGE_BODY_MAX:
                head = text[: MESSAGE_BODY_MAX // 2]
                tail = text[-(MESSAGE_BODY_MAX // 2 - 150) :]
                text = head + "\n... [truncated] ...\n" + tail

            if tool_calls:
                calls = json.dumps(tool_calls)
                if len(calls) > TOOL_ARGS_MAX:
                    calls = calls[:TOOL_ARGS_MAX] + "... [truncated]"
                text += f"\n[tool_calls: {calls}]"

            line = f"{role.upper()}{f' ({name})' if name else ''}: {text}"
            lines.append(line)
            total += len(line)
            if total >= SUMMARY_MAX_CHARS:
                break

        serialized = "\n\n".join(lines)
        if len(serialized) > SUMMARY_MAX_CHARS:
            serialized = serialized[:SUMMARY_MAX_CHARS] + "\n... [input truncated] ..."
        return serialized

    def _deterministic_fallback(self, middle: List[Dict[str, Any]]) -> str:
        """If the LLM summarizer fails, produce a compact deterministic summary."""
        lines = ["Detailed Session Log:"]
        for msg in middle:
            role = msg.get("role", "unknown")
            content = str(msg.get("content") or "")[:200]
            lines.append(f"- {role}: {content}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Phase 4: reassemble
    # ------------------------------------------------------------------
    def _reassemble(
        self,
        head: List[Dict[str, Any]],
        summary: str,
        tail: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        summary_msg = {
            "role": "user",
            "content": (
                "[CONTEXT COMPACTION — REFERENCE ONLY]\n\n"
                f"{summary}\n\n"
                "--- END OF CONTEXT SUMMARY ---"
            ),
        }
        # Ensure role alternation is safe: summary is injected as a user message
        # but clearly labeled so it cannot be confused with an active instruction.
        return head + [summary_msg] + tail

    # ------------------------------------------------------------------
    # Security helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _redact(text: str) -> str:
        # Redact URL credentials and common secret patterns.
        text = re.sub(r"https?://[^\s]+:[^\s]+@", "[redacted]://", text)
        text = re.sub(r"sk-[a-zA-Z0-9]{20,}", "[api-key-redacted]", text)
        text = re.sub(r"fw_[a-zA-Z0-9]{20,}", "[api-key-redacted]", text)
        # Strip thinking blocks.
        text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
        return text.strip()


def estimate_tokens(messages: List[Dict[str, Any]]) -> int:
    """Very rough token estimate (~4 chars per token)."""
    total = 0
    for msg in messages:
        content = msg.get("content") or ""
        total += len(str(content)) // 4
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            total += len(json.dumps(tool_calls)) // 4
    return total
