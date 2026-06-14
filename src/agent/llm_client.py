"""Provider-agnostic LLM client for the investigator agent.

One internal (ToolCall, LLMResponse) representation; each backend translates to
and from its SDK's tool-use wire shape so investigator.py is provider-blind:

- AnthropicClient    : native anthropic SDK (tool_use blocks)
- OpenAICompatClient : openai SDK; serves OpenAI and DeepSeek via base_url
- MockClient         : deterministic, no network, no key — the DEFAULT, so the
                       repo and tests run fully offline and exercise the exact
                       same tool loop + output validation as the real providers.

SDK imports are lazy so the project runs with only the mock provider installed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from src.agent.schema import SUBMIT_FINDING_SCHEMA
from src.log import get_logger

log = get_logger("llm_client")


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"   # normalized: "tool_use" | "end_turn"
    raw: Any = None


# Internal message shapes (provider-neutral), produced by investigator.py:
#   {"role": "user", "content": "<text>"}
#   {"role": "assistant", "text": "...", "tool_calls": [ToolCall, ...]}
#   {"role": "tool_results", "results": [{"tool_call_id", "content"}]}
class LLMClient:
    provider = "base"
    model = ""

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        raise NotImplementedError


# ----------------------------------------------------------------- Anthropic
class AnthropicClient(LLMClient):
    provider = "anthropic"

    def __init__(self, settings):
        import anthropic  # lazy
        self.model = settings.llm.anthropic_model
        self._client = anthropic.Anthropic(api_key=settings.llm.anthropic_key)

    def _to_anthropic(self, messages):
        out = []
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                content = []
                if m.get("text"):
                    content.append({"type": "text", "text": m["text"]})
                for tc in m.get("tool_calls", []):
                    content.append({"type": "tool_use", "id": tc.id,
                                    "name": tc.name, "input": tc.input})
                out.append({"role": "assistant", "content": content})
            elif m["role"] == "tool_results":
                out.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": r["tool_call_id"],
                     "content": r["content"]} for r in m["results"]
                ]})
        return out

    def complete(self, system, messages, tools):
        anthropic_tools = [{"name": t["name"], "description": t["description"],
                            "input_schema": t["input_schema"]} for t in tools]
        resp = self._client.messages.create(
            model=self.model, max_tokens=2048, system=system,
            messages=self._to_anthropic(messages), tools=anthropic_tools,
        )
        text, calls = "", []
        for block in resp.content:
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, input=dict(block.input)))
        stop = "tool_use" if calls else "end_turn"
        return LLMResponse(text=text, tool_calls=calls, stop_reason=stop, raw=resp)


# ------------------------------------------------------ OpenAI / DeepSeek
class OpenAICompatClient(LLMClient):
    def __init__(self, settings, which: str):
        import openai  # lazy
        self.provider = which
        if which == "deepseek":
            self.model = settings.llm.deepseek_model
            self._client = openai.OpenAI(api_key=settings.llm.deepseek_key,
                                         base_url=settings.llm.deepseek_base_url)
        else:
            self.model = settings.llm.openai_model
            self._client = openai.OpenAI(api_key=settings.llm.openai_key,
                                         base_url=settings.llm.openai_base_url)

    def _to_openai(self, system, messages):
        out = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                msg = {"role": "assistant", "content": m.get("text") or None}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [{
                        "id": tc.id, "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.input)},
                    } for tc in m["tool_calls"]]
                out.append(msg)
            elif m["role"] == "tool_results":
                for r in m["results"]:
                    out.append({"role": "tool", "tool_call_id": r["tool_call_id"],
                                "content": r["content"]})
        return out

    def complete(self, system, messages, tools):
        openai_tools = [{"type": "function", "function": {
            "name": t["name"], "description": t["description"],
            "parameters": t["input_schema"]}} for t in tools]
        resp = self._client.chat.completions.create(
            model=self.model, messages=self._to_openai(system, messages),
            tools=openai_tools, max_tokens=2048,
        )
        choice = resp.choices[0].message
        calls = []
        for tc in (choice.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(id=tc.id, name=tc.function.name, input=args))
        stop = "tool_use" if calls else "end_turn"
        return LLMResponse(text=choice.content or "", tool_calls=calls,
                           stop_reason=stop, raw=resp)


# -------------------------------------------------------------------- Mock
class MockClient(LLMClient):
    """Deterministic, offline. Drives the identical tool loop:

    turn 1 -> request read tools; turn 2 -> call submit_finding with a
    disposition derived ONLY from numeric evidence (model_score, ring_member),
    never from attacker-controllable free-text — so injected text cannot flip it.
    """
    provider = "mock"
    model = "mock-deterministic"

    def complete(self, system, messages, tools):
        has_results = any(m["role"] == "tool_results" for m in messages)
        if not has_results:
            calls = [
                ToolCall(id="mock_1", name="get_case", input={}),
                ToolCall(id="mock_2", name="get_account_profile", input={}),
                ToolCall(id="mock_3", name="get_graph_neighborhood", input={}),
                # also read transactions, which surface attacker-controlled
                # merchant names — exercises the untrusted-field / injection path
                ToolCall(id="mock_4", name="get_recent_transactions", input={"n": 10}),
            ]
            # only request tools that are actually offered
            offered = {t["name"] for t in tools}
            calls = [c for c in calls if c.name in offered]
            return LLMResponse(stop_reason="tool_use", tool_calls=calls)

        evidence = self._merge_results(messages)
        score = float(evidence.get("model_score", evidence.get("account_score", 0.0)) or 0.0)
        ring = int(evidence.get("ring_member", 0) or 0)
        refs = [v for k, v in evidence.items() if k in ("case_id", "user_id") and v]

        if ring or score >= 0.5:
            disp, conf = "LIKELY_FRAUD", max(score, 0.6)
        elif score <= 0.2:
            disp, conf = "LIKELY_LEGIT", 1.0 - score
        else:
            disp, conf = "INSUFFICIENT_EVIDENCE", 0.5

        finding = {
            "disposition": disp, "confidence": round(min(conf, 1.0), 2),
            "rationale": f"[mock] account_score={score:.2f}, ring_member={ring}. "
                         f"Deterministic disposition from numeric risk evidence.",
            "evidence_refs": refs,
        }
        return LLMResponse(stop_reason="tool_use", tool_calls=[
            ToolCall(id="mock_finding", name="submit_finding", input=finding)])

    @staticmethod
    def _merge_results(messages) -> dict:
        merged: dict = {}
        for m in messages:
            if m["role"] != "tool_results":
                continue
            for r in m["results"]:
                try:
                    data = json.loads(r["content"])
                except (json.JSONDecodeError, TypeError):
                    continue
                if isinstance(data, dict):
                    merged.update(data)
        return merged


def get_client(settings) -> LLMClient:
    provider = settings.llm.provider
    if provider == "anthropic":
        return AnthropicClient(settings)
    if provider in ("openai", "deepseek"):
        return OpenAICompatClient(settings, provider)
    return MockClient()


# Re-export so investigator can advertise the structured-output tool.
SUBMIT_FINDING_TOOL = {
    "name": "submit_finding",
    "description": "Submit your final investigation finding. This is the ONLY way "
                   "to conclude. It records a RECOMMENDATION for a human reviewer; "
                   "it does not block, allow, refund, or move money.",
    "input_schema": SUBMIT_FINDING_SCHEMA,
}
