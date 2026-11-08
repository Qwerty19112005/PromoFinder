"""Probe a Gemini key: which models it sees, and which way of passing the key the OpenAI endpoint accepts."""
import json
import os
import sys
import time
import urllib.error
import urllib.request

from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
KEY = os.environ["GEMINI_API_KEY"].strip()
BASE = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/").rstrip("/") + "/"
NATIVE = "https://generativelanguage.googleapis.com/v1beta/"


def http(url, body=None, headers=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json", **(headers or {})},
                                 method="POST" if body is not None else "GET")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return r.status, json.loads(r.read().decode()), dict(r.headers), time.time() - t0
    except urllib.error.HTTPError as e:
        txt = e.read().decode(errors="replace")
        try:
            return e.code, json.loads(txt), dict(e.headers), time.time() - t0
        except ValueError:
            return e.code, txt[:600], dict(e.headers), time.time() - t0


# 1. models visible on the native endpoint
code, data, _, _ = http(NATIVE + "models?pageSize=200", headers={"x-goog-api-key": KEY})
print("native list models:", code)
models = []
if code == 200:
    for m in data.get("models", []):
        if "generateContent" in m.get("supportedGenerationMethods", []):
            models.append(m["name"].split("/")[-1])
    print(f"{len(models)} models support generateContent:")
    print("  " + ", ".join(models))
else:
    print(json.dumps(data)[:600])

model = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
body = {"model": model, "messages": [{"role": "user", "content": "Reply with the single word: ready"}], "max_tokens": 50}

# 2. the OpenAI-compatible endpoint, three ways of passing the key
for label, url, headers in [
    ("Bearer header", BASE + "chat/completions", {"Authorization": "Bearer " + KEY}),
    ("x-goog-api-key header", BASE + "chat/completions", {"x-goog-api-key": KEY}),
    ("?key= query", BASE + "chat/completions?key=" + KEY, {}),
]:
    code, data, hdrs, secs = http(url, body, headers)
    if code == 200:
        msg = data["choices"][0]["message"].get("content")
        print(f"\nopenai-compat, {label}: 200 in {secs:.1f}s -> {msg!r}; usage {data.get('usage')}")
    else:
        print(f"\nopenai-compat, {label}: {code} -> {json.dumps(data)[:500]}")
    rl = {k: v for k, v in hdrs.items() if "ratelimit" in k.lower() or "quota" in k.lower()}
    if rl:
        print("  rate-limit headers:", rl)
