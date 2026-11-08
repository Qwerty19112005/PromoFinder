"""Robustness for the interview run: a script written in a slightly different template, and a model
provider that behaves differently from Groq. No API key and no confidential script needed."""
import os
import sys

import openai

try:  # openai 3.x ships its HTTP layer as httpx2; older releases use httpx
    import httpx2 as httpx
except ImportError:
    import httpx
import pytest
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from promofinder.config import ModelConfig  # noqa: E402
from promofinder.llm import LLM  # noqa: E402
from promofinder.parse import parse_episode_text  # noqa: E402

HEAD = "EPISODE NUMBER: 3\nEPISODE LOGLINE: A maid is accused at a wedding.\n\nSCENE 1\n"
VARIANTS = {
    "pack": "**ACTION:**\nStart: (00:01.20) | End: (00:03.00)\nShe slaps him.\n\n"
            "**MAID (Angry, Shaking):**\nStart: (00:03.00) | End: (00:05.50)\nI never stole anything!\n",
    "no bold": "ACTION:\nStart: (00:01.20) | End: (00:03.00)\nShe slaps him.\n\n"
               "MAID (Angry, Shaking):\nStart: (00:03.00) | End: (00:05.50)\nI never stole anything!\n",
    "hyphen": "**ACTION:**\nStart Time: 00:01.20 - End Time: 00:03.00\nShe slaps him.\n\n"
              "**MAID (Angry, Shaking):**\nStart Time: 00:03.00 - End Time: 00:05.50\nI never stole anything!\n",
    "inline": "**ACTION** [00:01.20 - 00:03.00]\nShe slaps him.\n\n"
              "**MAID (Angry, Shaking)** [00:03.00 - 00:05.50]\nI never stole anything!\n",
}


@pytest.mark.parametrize("name", list(VARIANTS))
def test_template_variants(name):
    ep = parse_episode_text(HEAD + VARIANTS[name], "episode-003.txt")
    got = [(m.kind, m.speaker, m.emotion, m.start, m.end, m.text) for m in ep.moments]
    assert got == [("ACTION", "", "", 1.2, 3.0, "She slaps him."),
                   ("DIALOGUE", "MAID", "Angry, Shaking", 3.0, 5.5, "I never stole anything!")], got
    assert ep.ep == 3
    looser = [p for p in ep.problems if "looser template" in p]
    assert bool(looser) == (name != "pack")  # the pack's own format never takes the fallback path
    assert not [p for p in ep.problems if "timed blocks in the file" in p]  # nothing lost


class Reply(BaseModel):
    verdict: str
    ok: bool


class FakeProvider:
    """Answers like a provider that differs from Groq; records every request."""

    def __init__(self, script):
        self.script, self.requests = list(script), []
        self.chat = self
        self.completions = self
        self.with_raw_response = self

    def create(self, **kw):
        self.requests.append(kw)
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        outer = self

        class Raw:
            headers = {}

            def parse(self_inner):
                class Msg:
                    content = step

                class Choice:
                    message = Msg()
                    finish_reason = "stop"

                class Resp:
                    choices = [Choice()]
                    usage = None
                return Resp()
        return Raw()


def bad_request(msg):
    return openai.BadRequestError(msg, response=httpx.Response(400, request=httpx.Request("POST", "http://x")), body=None)


def make_llm(tmp_path, script):
    cfg = ModelConfig("JUDGE", "http://x", "k", "some-reasoning-model", "low")
    llm = LLM(cfg, cache_dir=str(tmp_path))
    llm.client = FakeProvider(script)
    return llm


def test_openai_reasoning_model_parameters(tmp_path):
    llm = make_llm(tmp_path, [
        bad_request("Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead."),
        bad_request("Unsupported value: 'temperature' does not support 0.2 with this model. Only the default (1) value is supported."),
        '{"verdict": "fine", "ok": true}',
    ])
    out = llm.json([{"role": "user", "content": "judge this"}], Reply, tag="t")
    assert out == Reply(verdict="fine", ok=True)
    last = llm.client.requests[-1]
    assert "max_completion_tokens" in last and "max_tokens" not in last and "temperature" not in last


def test_provider_that_ignores_response_format(tmp_path):
    llm = make_llm(tmp_path, [
        "The clip works because the slap lands early.",  # schema ignored: prose instead of JSON
        '{"verdict": "fine", "ok": true}',
        "Still prose.",
        '{"verdict": "second", "ok": false}',
        '{"verdict": "third", "ok": true}',
    ])
    msgs = lambda n: [{"role": "user", "content": f"judge clip {n}"}]
    assert llm.json(msgs(1), Reply, tag="a").verdict == "fine"
    repair = llm.client.requests[1]["messages"][-1]["content"]
    assert '"verdict"' in repair and '"ok"' in repair  # the repair request carries the schema itself
    assert llm.json(msgs(2), Reply, tag="b").verdict == "second"
    assert llm.json_mode == "prompt"  # two misses in a row: schema goes into the prompt from now on
    assert llm.json(msgs(3), Reply, tag="c").verdict == "third"
    assert '"verdict"' in llm.client.requests[-1]["messages"][-1]["content"]


def test_gemini_daily_quota_stops_calls(tmp_path):
    err = openai.RateLimitError(
        "Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, "
        "quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier, retryDelay: '20s'",
        response=httpx.Response(429, request=httpx.Request("POST", "http://x")), body=None)
    llm = make_llm(tmp_path, [err])
    assert llm.json([{"role": "user", "content": "x"}], Reply, tag="q") is None
    assert llm.exhausted and len(llm.client.requests) == 1
    assert llm.json([{"role": "user", "content": "y"}], Reply, tag="r") is None  # no further request sent
    assert len(llm.client.requests) == 1
