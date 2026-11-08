"""Pass 1: a cheap model reads the whole show a few episodes at a time and scores the moments that matter.

The model refers to moments only by their number; code maps numbers back to times, so it cannot invent a
timestamp. Numbers that do not exist are dropped and counted (a quality signal).

The highlight tags and their points follow the rules that ByteDance researchers distilled from
professional short-drama ad editors (HIVE, arXiv 2507.02790, tables 6 and 5: female- and male-audience
highlights plus general highlights), adapted to this genre. hook / cliff / context are this project's
own 0-3 scales; cliff asks how open the next event is (suspense as uncertainty about what comes next,
Wilmot and Keller, ACL 2020).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from .parse import Episode, Moment, Show

TAG_POINTS = {
    "slap_or_violence": 3, "confrontation": 3, "heroine_suffering": 3, "heroine_fights_back": 3,
    "rescue_by_lead": 3, "romantic_tension": 3, "first_meeting_of_leads": 3, "breakup_or_rejection": 3,
    "pregnancy": 3, "accident_or_peril": 3, "identity_reveal_or_secret": 2, "villain_scheme": 2,
    "male_lead_protects": 2, "sweet_moment": 2, "comedy": 2, "turning_point": 2, "cliffhanger": 2,
    "emotional_resonance": 1,
}
Tag = Literal[tuple(TAG_POINTS)]  # type: ignore[valid-type]

SYSTEM = """You are a promo editor at a microdrama app. Promos are 20-60 second clips cut from a show and posted on Instagram, TikTok and YouTube to pull in new viewers. A good promo opens on a hook in its first 3 seconds, ends on a cliffhanger (cut while something important is unresolved), and makes sense to a stranger who has never seen the show.

You will read episodes of one show. Each line is one numbered moment:
<number> | WHO (emotion tags): text
WHO is a character, ACTION (what is seen) or SOUND FX (what is heard). Camera directions are removed.

For each episode, list only the moments that matter for promos: at most 10 per episode, fewer is fine. For each moment give:
- m: the moment number exactly as shown. Never invent a number.
- tags: the highlight patterns it shows, from this list:
  slap_or_violence: a slap, a blow, a fight, someone dragged, thrown or seized
  confrontation: an argument or face-off
  heroine_suffering: the heroine humiliated, mocked, wronged or abandoned
  heroine_fights_back: the heroine strikes back at those who wronged her
  rescue_by_lead: the heroine in trouble and the male lead steps in
  romantic_tension: flirting, a charged look, a near-kiss or a kiss
  first_meeting_of_leads: the leads meet for the first time
  breakup_or_rejection: a jilting, divorce, broken engagement or disowning
  pregnancy: a pregnancy or a child's parentage
  accident_or_peril: an accident, illness, poison, fire or mortal danger
  identity_reveal_or_secret: a hidden identity, status or secret comes out
  villain_scheme: a villain appears or plots
  male_lead_protects: the male lead defends or supports the heroine
  sweet_moment: tenderness between the leads
  comedy: a funny twist
  turning_point: a major plot shift
  cliffhanger: the episode or scene ends on an open question
  emotional_resonance: strong sadness, joy, shock or fear
- hook: 0-3. How strongly this moment would grab a stranger if a promo OPENED on it. 3 = instantly gripping on its own (a slap, a shocking line, a reveal); 2 = strong; 1 = mild; 0 = not an opener. Lines that need earlier context score low.
- cliff: 0-3. How badly a viewer would need to know what happens next if a promo CUT right after this moment. 3 = a big question is wide open (a threat just made, a door opens on someone, a secret about to come out); 2 = a clear open question; 1 = slight; 0 = the moment resolves something.
- context: 0-3. How much a stranger must know from earlier episodes to understand this moment. 0 = nothing.
- note: at most 12 words on what happens.
Judge only from the text."""


class Beat(BaseModel):
    m: int
    tags: list[Tag]
    hook: int
    cliff: int
    context: int
    note: str


class EpisodeBeats(BaseModel):
    ep: int
    moments: list[Beat]


class Pass1Reply(BaseModel):
    episodes: list[EpisodeBeats]


def moment_line(m: Moment) -> str:
    return f"{m.idx} | {m.who()}: {m.text}"


def episode_block(e: Episode) -> str:
    head = f"### Episode {e.ep}" + (f"\nLogline: {e.logline}" if e.logline else "")
    return head + "\n" + "\n".join(moment_line(m) for m in e.moments)


def batches(show: Show, budget_tokens: int = 2800) -> list[list[Episode]]:
    out, cur, used = [], [], 0
    for e in show.episodes:
        t = len(episode_block(e)) // 4 + 10
        if cur and used + t > budget_tokens:
            out.append(cur)
            cur, used = [], 0
        cur.append(e)
        used += t
    if cur:
        out.append(cur)
    return out


def clamp(v: int) -> int:
    return max(0, min(3, int(v)))


def score_show(show: Show, llm, characters: list[str], budget_tokens: int = 2800, progress=None) -> dict:
    """Returns {"beats": {moment_id: {...}}, "invalid": [...], "batches": n, "failed_batches": [...]}."""
    beats, invalid, failed = {}, [], []
    groups = batches(show, budget_tokens)
    for i, group in enumerate(groups):
        if progress:
            progress(f"pass 1: episodes {group[0].ep}-{group[-1].ep} ({i + 1}/{len(groups)})")
        user = (f"Show: {show.title}\nMain characters: {', '.join(characters)}\n\n"
                + "\n\n".join(episode_block(e) for e in group))
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        reply = llm.json(msgs, Pass1Reply, tag=f"pass1_E{group[0].ep}-E{group[-1].ep}",
                         max_tokens=900 * len(group) + 600)
        if reply is None:
            failed.append([e.ep for e in group])
            continue
        valid_eps = {e.ep: e for e in group}
        for eb in reply.episodes:
            e = valid_eps.get(eb.ep)
            for b in eb.moments:
                if e is None or not 1 <= b.m <= len(e.moments):
                    invalid.append(f"E{eb.ep}.M{b.m}")
                    continue
                mid = e.moments[b.m - 1].id
                tags = sorted(set(b.tags))
                beats[mid] = {"tags": tags, "points": sum(TAG_POINTS[t] for t in tags), "hook": clamp(b.hook),
                              "cliff": clamp(b.cliff), "context": clamp(b.context), "note": b.note[:120]}
    return {"beats": beats, "invalid": invalid, "batches": len(groups), "failed_batches": failed}
