"""OpenAI-compatible chat client.

Point it at a Bifrost gateway (default http://localhost:8080/v1) or straight at any
OpenAI-compatible provider. Only the standard library is used.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field


class ModelUnavailable(Exception):
    """Raised when every model in the fallback chain fails."""


@dataclass
class LLMResponse:
    content: str
    tool_calls: list
    prompt_tokens: int
    completion_tokens: int
    model: str
    raw: dict = field(default_factory=dict)


class ChatClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 timeout: float = 60, min_interval: float = 0.0):
        self.base_url = (base_url or os.environ.get("AGENTBENCH_BASE_URL")
                         or "http://localhost:8080/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("AGENTBENCH_API_KEY", "")
        self.timeout = timeout
        self.min_interval = min_interval
        self._last = 0.0

    def _post(self, payload: dict) -> dict:
        wait = self.min_interval - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}",
                     "User-Agent": "agentbench/0.1"},
        )
        for attempt in range(4):
            try:
                self._last = time.time()
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")[:300]
                if e.code == 429 and attempt < 3:
                    time.sleep(2 ** attempt * 5)
                    continue
                raise RuntimeError(f"HTTP {e.code}: {body}") from None
        raise RuntimeError("rate limited")

    def chat(self, models: list[str], messages: list[dict], tools: list | None = None,
             temperature: float = 0.0) -> LLMResponse:
        """Try each model in order (the fallback chain). Bifrost can do this itself;
        doing it here too keeps the benchmark framework-independent."""
        errors = []
        for model in models:
            payload = {"model": model, "messages": messages, "temperature": temperature}
            if tools:
                payload["tools"] = tools
            try:
                data = self._post(payload)
            except Exception as e:  # noqa: BLE001 - recorded in the trace
                errors.append(f"{model}: {e}")
                continue
            msg = data["choices"][0]["message"]
            usage = data.get("usage") or {}
            return LLMResponse(
                content=msg.get("content") or "",
                tool_calls=msg.get("tool_calls") or [],
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                model=data.get("model", model),
                raw={"fallback_errors": errors},
            )
        raise ModelUnavailable("; ".join(errors))
