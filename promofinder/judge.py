"""Pass 2: a cold viewer judges each shortlisted clip from its text alone with yes/no checks
(CheckEval, arXiv 2403.18771); code turns the answers into a score with fixed weights."""
from __future__ import annotations

from pydantic import BaseModel

from .candidates import Candidate
from .parse import Show

SYSTEM = """You are testing a promo for a microdrama app. Imagine you are scrolling Instagram and have never seen this show or heard of these characters. You get the script of one 20-60 second clip, line by line, with the time in seconds from the start of the clip. ACTION lines describe what is seen; SOUND FX what is heard. [cut] marks a jump to another moment.

Answer from the clip alone. Do not use any knowledge of the rest of the show.
- premise: one sentence on what is going on, as a stranger would understand it. If you cannot tell, say so.
- first_seconds: what happens in the first 3 seconds.
- open_question: the question left unanswered at the very end, or "" if none.
Then answer each check true or false:
- hook_in_first_3s: something in the first 3 seconds would make a scroller stop (a slap, a shocking line, a reveal, an open conflict).
- premise_clear: a stranger understands who is against whom and why it matters.
- needs_earlier_episodes: important parts only make sense with knowledge from earlier in the show.
- names_confusing: names, titles or references would confuse a stranger.
- ends_unresolved: the clip cuts while something important is still unresolved.
- resolved_before_cut: the clip's main question is answered before it ends, so the cliffhanger is spent.
- spoils_major_twist: the clip gives away a big secret or twist the show seems to save for later.
- why_it_works: one line a social media manager would write on why this promo works, or "" if it does not.
- weakness: one line on its main weakness."""

WEIGHTS = {"hook_in_first_3s": 2.0, "premise_clear": 2.0, "ends_unresolved": 2.0, "needs_earlier_episodes": -1.0,
           "resolved_before_cut": -1.0, "names_confusing": -0.5, "spoils_major_twist": -1.0}
BASE = 2.5  # so a clip that passes every check scores 8.5 and one that fails every check scores 0


class Verdict(BaseModel):
    premise: str
    first_seconds: str
    open_question: str
    hook_in_first_3s: bool
    premise_clear: bool
    needs_earlier_episodes: bool
    names_confusing: bool
    ends_unresolved: bool
    resolved_before_cut: bool
    spoils_major_twist: bool
    why_it_works: str
    weakness: str


def clip_text(show: Show, cand: Candidate) -> str:
    lines, offset = [], 0.0
    for i, seg in enumerate(cand.segments):
        if i:
            lines.append("[cut]")
        for m in show.by_ep[seg.ep].moments:
            if m.end > seg.start + 0.05 and m.start < seg.end - 0.05:
                lines.append(f"[{offset + max(0.0, m.start - seg.start):4.1f}s] {m.who()}: {m.text}")
        offset += seg.dur
    return "\n".join(lines)


def score(v: Verdict) -> float:
    return BASE + sum(w * float(getattr(v, k)) for k, w in WEIGHTS.items())


def judge_all(show: Show, cands: list[Candidate], llm, progress=None) -> dict[str, dict]:
    out = {}
    for i, c in enumerate(cands):
        if progress:
            progress(f"judge: clip {i + 1}/{len(cands)}")
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Clip ({c.dur:.1f} s):\n{clip_text(show, c)}"}]
        v = llm.json(msgs, Verdict, tag=f"judge_{c.cid}", max_tokens=1200)
        if v is not None:
            out[c.cid] = {**v.model_dump(), "judge_score": round(score(v), 2)}
    return out
