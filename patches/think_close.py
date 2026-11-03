"""Loop guard for dfranzen's harness: close a reasoning trace that hit the output cap and let
the model act on it ("budget forcing").

In dfranzen's 4x25 public run, 89 of 9,374 requests stopped at LOCAL_ANALYZER_MAX_OUTPUT
(12,288 tokens): 6% of all generated tokens. The replies read in s5i5's request log were
all reasoning-only. They were not repetition loops (zlib ratio 0.34 vs 0.39-0.48 for
normal long traces): they were long deliberation that kept circling the same hypotheses.
A repetition detector would not fire on them. The harness drops such a reply's turn
without an action, and with preserve_thinking the ~38k-character dead trace is replayed
in later requests.

Here a reasoning-only reply with finish_reason=length is continued instead.
- The request is re-sent with the trace as a final assistant message ending in a short
  closing sentence and "</think>", with continue_final_message=true.
- SGLang (pennyroyal 0.5.19) handles that request as follows:
  - It renders the earlier messages with the generation prompt and appends the prefix
    tokens. The prompt is identical to the original request, and the trace was just
    generated, so the radix cache covers it.
  - Its qwen3 reasoning parser sees "</think>" in the prefix, so the continuation is
    parsed as content and tool calls.
- The two replies are merged into one result:
  - reasoning = the trace plus the closing sentence, i.e. exactly what the model saw;
  - content, tool calls and finish_reason come from the continuation;
  - completion tokens are summed.
  The harness then handles the merged result like a normal reply.

Only the main turn request is touched: tools present, no explicit max_tokens and no
thinking override. The rolling-summary call passes both, so it is left alone.

Knobs:
- ARC3_THINK_CLOSE_MAX_TOKENS: cap on the continuation (default 4096).
- LOCAL_ANALYZER_MAX_OUTPUT: the first cap, which then acts as a thinking budget.

install(tool_agent_module) wraps ToolAgent._chat_completion and the module's
build_chat_payload. It raises if the names it relies on are missing.
"""
from __future__ import annotations

import logging
import os
import threading

log = logging.getLogger(__name__)

CLOSE_MAX_TOKENS = int(os.environ.get("ARC3_THINK_CLOSE_MAX_TOKENS", "4096"))
CLOSE_TEXT = (
    "\n\nI have deliberated long enough. I will now act on my best current hypothesis "
    "with one informative python tool call, and check the result next turn."
)
THINK_END = "\n</think>\n\n"
FIRED = [0]      # truncated reasoning-only replies that were continued
RECOVERED = [0]  # continuations that produced at least one tool call
FAILED = [0]     # continuations that raised; the truncated reply is kept

_local = threading.local()


def install(tool_agent) -> None:
    cls = tool_agent.ToolAgent
    result_cls = tool_agent._ChatCompletionResult
    extract_reasoning = tool_agent._extract_reasoning_text
    normalize = tool_agent._normalize_message_content
    original = cls._chat_completion
    original_payload = tool_agent.build_chat_payload
    if getattr(original, "_think_close", False):
        return

    def build_chat_payload(*args, **kwargs):
        payload = original_payload(*args, **kwargs)
        if getattr(_local, "continue_final", False):
            payload["continue_final_message"] = True
        return payload

    def _chat_completion(self, messages, *, tools=None, request_timeout_seconds=None,
                         max_tokens=None, enable_thinking=None):
        result = original(self, messages, tools=tools,
                          request_timeout_seconds=request_timeout_seconds,
                          max_tokens=max_tokens, enable_thinking=enable_thinking)
        if not tools or max_tokens is not None or enable_thinking is not None:
            return result
        if str(result.finish_reason or "").strip() != "length":
            return result
        message = result.message or {}
        reasoning = extract_reasoning(message)
        if (not reasoning.strip() or message.get("tool_calls")
                or normalize(message.get("content", "")).strip()):
            return result
        trace = reasoning.rstrip() + CLOSE_TEXT
        FIRED[0] += 1
        _local.continue_final = True
        try:
            cont = original(self, [*messages, {"role": "assistant", "content": trace + THINK_END}],
                            tools=tools, request_timeout_seconds=request_timeout_seconds,
                            max_tokens=CLOSE_MAX_TOKENS)
        except Exception as exc:  # the truncated reply is still a valid (empty) turn
            FAILED[0] += 1
            log.warning("think-close continuation failed, keeping truncated reply: %s", exc)
            return result
        finally:
            _local.continue_final = False
        merged = dict(cont.message or {})
        merged.pop("reasoning", None)
        merged["reasoning_content"] = trace
        if merged.get("tool_calls"):
            RECOVERED[0] += 1
        usage = dict(result.usage or {})
        cont_usage = cont.usage or {}
        for key in ("completion_tokens", "reasoning_tokens"):
            if key in usage or key in cont_usage:
                usage[key] = int(usage.get(key) or 0) + int(cont_usage.get(key) or 0)
        if "total_tokens" in usage:
            usage["total_tokens"] = int(usage.get("prompt_tokens") or 0) + usage["completion_tokens"]
        log.warning("think-close: continued a %d-char truncated trace -> finish=%s, %d tool call(s)",
                    len(reasoning), cont.finish_reason, len(merged.get("tool_calls") or []))
        return result_cls(message=merged, finish_reason=cont.finish_reason, usage=usage,
                          served_by=cont.served_by or result.served_by)

    _chat_completion._think_close = True
    cls._chat_completion = _chat_completion
    tool_agent.build_chat_payload = build_chat_payload
