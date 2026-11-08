"""Check the model settings in .env: list models, send one tiny request, print the rate-limit headers."""
import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

base, key = os.getenv("LLM_BASE_URL"), os.getenv("LLM_API_KEY")
models = [m for m in (os.getenv("LLM_MODEL"), os.getenv("JUDGE_MODEL")) if m]
if not (base and key and models):
    sys.exit("Set LLM_BASE_URL, LLM_API_KEY and LLM_MODEL in .env first.")

client = OpenAI(base_url=base, api_key=key)
print("endpoint:", base)
try:
    ids = sorted(m.id for m in client.models.list())
    print(f"models visible to this key ({len(ids)}):", ", ".join(ids))
except Exception as exc:  # some providers do not list models
    print("could not list models:", exc)

for model in dict.fromkeys(models):
    t0 = time.time()
    raw = client.chat.completions.with_raw_response.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with the single word: ready"}],
        max_tokens=200,
    )
    resp = raw.parse()
    print(f"\n{model}: {resp.choices[0].message.content!r} in {time.time() - t0:.1f} s, usage {resp.usage}")
    for h, v in raw.headers.items():
        if "ratelimit" in h.lower():
            print(f"  {h}: {v}")
