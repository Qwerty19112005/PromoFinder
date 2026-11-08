"""One client for any OpenAI-compatible endpoint (Groq, Gemini, OpenRouter, OpenAI, Ollama, ...).

What it adds on top of the openai package:
- JSON replies checked against a pydantic model. It asks for a strict JSON schema first (Groq
  constrains decoding to the schema for gpt-oss and Qwen); if the provider rejects that, it falls
  back to plain JSON mode, then to JSON described in the prompt. One repair retry on a bad reply.
- A disk cache keyed by a hash of the whole request, so a rerun costs nothing and gives the same
  answer. Prompts put fixed instructions first and show text last, which also lets the provider's
  own prefix cache work (Groq: cached tokens do not count toward rate limits).
- Rate-limit handling: waits on 429 using retry-after (or Gemini's retryDelay in the error body),
  paces requests using the x-ratelimit-remaining-tokens / reset headers so it rarely hits 429 at
  all, and can also keep to a fixed requests-per-minute cap for providers that send no headers.
  A daily quota that resets at a fixed time (Gemini's free tier) stops further calls at once.
- A log of every request and reply (raw outputs for the submission) and token totals per model.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field

from pydantic import BaseModel, ValidationError

from .config import ModelConfig

STRIP_KEYS = {"title", "default", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
              "minLength", "maxLength", "pattern", "format", "minItems", "maxItems", "examples"}


def strict_schema(model: type[BaseModel]) -> dict:
    """Pydantic schema -> the subset strict mode accepts: refs inlined, every object closed and fully required."""
    raw = model.model_json_schema()
    defs = raw.pop("$defs", {})

    def fix(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return fix(copy.deepcopy(defs[node["$ref"].split("/")[-1]]))
            node = {k: fix(v) for k, v in node.items() if k not in STRIP_KEYS}
            if node.get("type") == "object" and "properties" in node:
                node["required"] = list(node["properties"])
                node["additionalProperties"] = False
            return node
        if isinstance(node, list):
            return [fix(x) for x in node]
        return node

    return fix(raw)


def _retry_hint(text: str) -> float:
    """Gemini puts the wait in the error body: "retryDelay": "33s" or "Please retry in 33.5s"."""
    m = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?([\d.]+)s", text) or re.search(r"retry in ([\d.]+)\s*s", text)
    return float(m.group(1)) if m else 0.0


def _parse_duration(v: str | None) -> float:
    """Groq reset headers look like '1m26.4s', '7.66s' or '120ms'."""
    if not v:
        return 0.0
    total = 0.0
    for num, unit in re.findall(r"([\d.]+)\s*(ms|h|m|s)", v):
        total += float(num) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit]
    return total


@dataclass
class Usage:
    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    provider_cached_tokens: int = 0
    seconds: float = 0.0
    retries: int = 0
    rate_limit_waits: float = 0.0
    repairs: int = 0
    failures: int = 0


@dataclass
class LLM:
    cfg: ModelConfig
    cache_dir: str = "cache"
    log_dir: str | None = None
    temperature: float = 0.2
    usage: Usage = field(default_factory=Usage)
    json_mode: str = "schema"  # schema -> object -> prompt, downgraded automatically on rejection
    on_event: object = None  # optional callback(str) for progress messages

    def __post_init__(self):
        from openai import OpenAI
        if not (self.cfg.base_url and self.cfg.model):
            raise ValueError(f"{self.cfg.stage}: set LLM_BASE_URL and LLM_MODEL (or {self.cfg.stage}_*)")
        self.client = OpenAI(base_url=self.cfg.base_url, api_key=self.cfg.api_key or "none", max_retries=0, timeout=120)
        os.makedirs(self.cache_dir, exist_ok=True)
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
        self.temperature = getattr(self.cfg, "temperature", self.temperature)
        self._lock = threading.Lock()
        self._n = 0
        self._min_gap = 60.0 / self.cfg.rpm if getattr(self.cfg, "rpm", None) else 0.0
        self._last_call = 0.0
        self.exhausted = False  # set when a fixed daily quota runs out; later calls return None at once
        self._max_key = "max_tokens"  # OpenAI's reasoning models want max_completion_tokens instead
        self._schema_misses = 0  # replies in schema mode that failed validation in a row
        self._tokens_left: float | None = None
        self._tokens_limit = 0.0
        self._tokens_reset_at = 0.0

    # ---------- public ----------
    def json(self, messages: list[dict], model: type[BaseModel], tag: str, max_tokens: int = 2000) -> BaseModel | None:
        schema = strict_schema(model)
        key = hashlib.sha256(json.dumps(
            {"m": self.cfg.model, "u": self.cfg.base_url, "msg": messages, "schema": schema, "t": self.temperature,
             "r": self.cfg.reasoning_effort, "mt": max_tokens}, sort_keys=True).encode()).hexdigest()
        path = os.path.join(self.cache_dir, key + ".json")
        if os.path.exists(path):
            rec = json.load(open(path, encoding="utf-8"))
            try:
                obj = model.model_validate(rec["parsed"])
                self.usage.cache_hits += 1
                self._log(tag, messages, rec, cached=True)
                return obj
            except ValidationError:
                pass
        msgs = list(messages)
        for attempt in range(2):
            text, meta = self._call(msgs, schema, model.__name__, max_tokens)
            if text is None:
                break
            try:
                data = json.loads(_extract_json(text))
                obj = model.model_validate(data)
                rec = {"model": self.cfg.model, "mode": meta["mode"], "text": text, "parsed": obj.model_dump(),
                       "usage": meta["usage"], "seconds": meta["seconds"], "attempt": attempt}
                json.dump(rec, open(path, "w", encoding="utf-8"), ensure_ascii=False)
                self._log(tag, msgs, rec, cached=False)
                if attempt == 0 and meta["mode"] == "schema":
                    self._schema_misses = 0
                return obj
            except (ValueError, ValidationError) as exc:
                self.usage.repairs += 1
                self._log(tag, msgs, {"model": self.cfg.model, "text": text, "error": str(exc)[:500]}, cached=False)
                if meta["mode"] == "schema":
                    # a provider that silently ignores response_format never told the model the schema
                    self._schema_misses += 1
                    if self._schema_misses >= 2:
                        self.json_mode = "prompt"
                        self._emit(f"{self.cfg.model}: replies keep missing the schema; putting it in the prompt from now on")
                msgs = msgs + [{"role": "assistant", "content": text[:4000]},
                               {"role": "user", "content": f"That reply was not valid ({str(exc)[:300]}). "
                                                           "Reply again with one JSON object only, matching this JSON schema:\n"
                                                           + json.dumps(schema)}]
        self.usage.failures += 1
        return None

    # ---------- internals ----------
    def _emit(self, msg: str):
        if callable(self.on_event):
            self.on_event(msg)

    def _pace(self, estimate: int):
        """Wait if the provider said fewer tokens remain this minute than this request needs."""
        if self._tokens_left is not None and self._tokens_left < estimate:
            # the bucket refills continuously (limit per minute), so wait only for the shortfall
            full = max(0.0, self._tokens_reset_at - time.time())
            refill = (estimate - self._tokens_left) / max(1.0, self._tokens_limit / 60.0) + 1.0 if self._tokens_limit else full
            wait = min(full, refill)
            if wait > 0:
                self._emit(f"{self.cfg.model}: pacing {wait:.0f}s for the per-minute token limit")
                self.usage.rate_limit_waits += wait
                time.sleep(wait)
            self._tokens_left = None

    def _request_kwargs(self, msgs, schema, name, max_tokens, mode):
        kw = {"model": self.cfg.model, "messages": msgs, self._max_key: max_tokens}
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        if self.cfg.reasoning_effort:
            kw["extra_body"] = {"reasoning_effort": self.cfg.reasoning_effort}
        if mode == "schema":
            kw["response_format"] = {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}
        elif mode == "object":
            kw["response_format"] = {"type": "json_object"}
        if mode in ("object", "prompt"):
            kw["messages"] = msgs[:-1] + [{"role": msgs[-1]["role"], "content": msgs[-1]["content"] +
                                           "\n\nReply with one JSON object only, matching this JSON schema:\n" +
                                           json.dumps(schema)}]
        return kw

    def _call(self, msgs, schema, name, max_tokens):
        import openai
        # replies are usually far below max_tokens; over-estimating only makes the pacer wait longer
        estimate = sum(len(m["content"]) for m in msgs) // 4 + int(max_tokens * 0.3)
        delay = 2.0
        if self.exhausted:
            return None, None
        for attempt in range(8):
            self._pace(estimate)
            gap = self._last_call + self._min_gap - time.time()
            if gap > 0:
                time.sleep(gap)
            self._last_call = time.time()
            kw = self._request_kwargs(msgs, schema, name, max_tokens, self.json_mode)
            t0 = time.time()
            try:
                raw = self.client.chat.completions.with_raw_response.create(**kw)
                resp = raw.parse()
            except openai.RateLimitError as exc:
                msg = str(exc).lower()
                wait = float(exc.response.headers.get("retry-after") or 0) or _parse_duration(
                    exc.response.headers.get("x-ratelimit-reset-tokens")) or _retry_hint(msg) or delay
                if "requestsperday" in msg or "perdayperproject" in msg:
                    # Gemini's daily request quota: resets at midnight Pacific, so waiting will not help
                    self._emit(f"{self.cfg.model}: daily request quota used up; skipping remaining calls")
                    self.exhausted = True
                    return None, None
                if "per day" in msg or "(tpd)" in msg or "(rpd)" in msg:
                    self._emit(f"{self.cfg.model}: daily limit reached: {str(exc)[:200]}")
                    if wait > 120:
                        return None, None
                self._emit(f"{self.cfg.model}: rate limited, waiting {wait:.0f}s")
                self.usage.retries += 1
                self.usage.rate_limit_waits += wait
                time.sleep(min(wait + 0.5, 120))
                continue
            except openai.BadRequestError as exc:
                text = str(exc).lower()
                if "max_completion_tokens" in text and self._max_key == "max_tokens":
                    self._emit(f"{self.cfg.model}: provider wants max_completion_tokens; switching")
                    self._max_key = "max_completion_tokens"
                    continue
                if "failed_generation" in text or "failed to validate" in text or "failed to generate" in text:
                    # Groq rejects a generation that broke the schema; treat it like a bad reply and retry
                    self.usage.repairs += 1
                    self._emit(f"{self.cfg.model}: provider could not produce valid JSON, retrying ({attempt + 1})")
                    if attempt >= 2:
                        return None, None
                    continue
                if self.json_mode != "prompt" and any(k in text for k in ("response_format", "json_schema", "schema", "json mode", "strict")):
                    self.json_mode = {"schema": "object", "object": "prompt"}[self.json_mode]
                    self._emit(f"{self.cfg.model}: provider rejected the JSON setting, falling back to '{self.json_mode}'")
                    continue
                if "reasoning" in text and self.cfg.reasoning_effort:
                    self._emit(f"{self.cfg.model}: provider rejected reasoning_effort, sending without it")
                    self.cfg.reasoning_effort = None
                    continue
                if "temperature" in text and self.temperature is not None:
                    self._emit(f"{self.cfg.model}: provider rejected temperature, sending without it")
                    self.temperature = None
                    continue
                raise
            except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError) as exc:
                self._emit(f"{self.cfg.model}: {type(exc).__name__}, retrying in {delay:.0f}s")
                self.usage.retries += 1
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            secs = time.time() - t0
            h = raw.headers
            if h.get("x-ratelimit-remaining-tokens"):
                self._tokens_left = float(h["x-ratelimit-remaining-tokens"])
                self._tokens_limit = float(h.get("x-ratelimit-limit-tokens") or 0)
                self._tokens_reset_at = time.time() + _parse_duration(h.get("x-ratelimit-reset-tokens")) + 0.5
            u = resp.usage
            usage = u.model_dump() if u else {}
            with self._lock:
                self.usage.calls += 1
                self.usage.seconds += secs
                if u:
                    self.usage.prompt_tokens += u.prompt_tokens or 0
                    self.usage.completion_tokens += u.completion_tokens or 0
                    det = getattr(u, "completion_tokens_details", None)
                    self.usage.reasoning_tokens += (getattr(det, "reasoning_tokens", 0) or 0) if det else 0
                    pdet = getattr(u, "prompt_tokens_details", None)
                    self.usage.provider_cached_tokens += (getattr(pdet, "cached_tokens", 0) or 0) if pdet else 0
            text = resp.choices[0].message.content or ""
            if resp.choices[0].finish_reason == "length":
                self._emit(f"{self.cfg.model}: reply hit max_tokens ({max_tokens}); consider raising it")
            return text, {"mode": self.json_mode, "usage": usage, "seconds": round(secs, 2)}
        return None, None

    def _log(self, tag, msgs, rec, cached):
        if not self.log_dir:
            return
        with self._lock:
            self._n += 1
            n = self._n
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag)[:60]
        out = {"tag": tag, "stage": self.cfg.stage, "model": self.cfg.model, "cached": cached,
               "messages": msgs, **rec}
        json.dump(out, open(os.path.join(self.log_dir, f"{self.cfg.stage.lower()}_{n:04d}_{safe}.json"), "w",
                            encoding="utf-8"), ensure_ascii=False, indent=1)


def _extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("{") or text.startswith("["):
        return text
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in reply")
    return m.group(0)
