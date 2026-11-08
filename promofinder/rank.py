"""Final ranking by pairwise comparison (which promo to post first).

Method from Pairwise Ranking Prompting (Qin et al., 2023, arXiv 2306.17563), all-pairs variant:
every pair is asked twice with the order swapped; a pick that survives the swap is a win (1 point),
an inconsistent pair is a tie (half a point each). The share of inconsistent pairs is reported as a
direct measure of the model's position bias. All-pairs costs n(n-1) calls, so it runs on a short list
(8 by default = 56 calls) chosen by the judge's score with near-duplicates and same-episode repeats removed.
"""
from __future__ import annotations

import itertools
from typing import Literal

from pydantic import BaseModel

from .candidates import Candidate
from .judge import clip_text
from .parse import Show

SYSTEM = """You decide which of two promos a microdrama app should post first on Instagram, TikTok and YouTube to win new viewers. Each promo is the script of a 20-60 second clip, with times in seconds from the start of the clip; ACTION lines are what is seen, SOUND FX what is heard, [cut] a jump to another moment. The audience has never seen the show.
Pick the promo more likely to make a stranger stop scrolling and then tap through to watch the show. Weigh: how strongly the first 3 seconds grab; whether a stranger understands the situation; how badly the ending makes them need to know what happens next. The order in which the promos are shown does not matter.
Reply with a one-sentence reason, then better: "A" or "B"."""


class Preference(BaseModel):
    reason: str
    better: Literal["A", "B"]


def rank_pairwise(show: Show, cands: list[Candidate], llm, progress=None) -> dict:
    wins = {c.cid: 0.0 for c in cands}
    reasons = {c.cid: [] for c in cands}
    pairs, consistent, missing = [], 0, 0
    combos = list(itertools.combinations(cands, 2))
    for k, (x, y) in enumerate(combos):
        if progress:
            progress(f"rank: pair {k + 1}/{len(combos)}")
        picks = []
        for a, b in ((x, y), (y, x)):
            user = f"Promo A ({a.dur:.0f} s):\n{clip_text(show, a)}\n\nPromo B ({b.dur:.0f} s):\n{clip_text(show, b)}"
            p = llm.json([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], Preference,
                         tag=f"rank_{a.cid}_vs_{b.cid}", max_tokens=600)
            picks.append(None if p is None else ((a if p.better == "A" else b).cid, p.reason))
        if None in picks:
            missing += 1
            wins[x.cid] += 0.5
            wins[y.cid] += 0.5
            pairs.append({"a": x.cid, "b": y.cid, "result": "no answer"})
            continue
        (w1, r1), (w2, r2) = picks
        if w1 == w2:
            consistent += 1
            wins[w1] += 1
            reasons[w1].append(r1)
            result = w1
        else:
            wins[x.cid] += 0.5
            wins[y.cid] += 0.5
            result = "tie (order flipped the answer)"
        pairs.append({"a": x.cid, "b": y.cid, "pick_ab": w1, "pick_ba": w2, "result": result, "reasons": [r1, r2]})
    n = len(combos)
    return {"wins": wins, "reasons": reasons, "pairs": pairs, "n_pairs": n,
            "consistency": round(consistent / max(1, n - missing), 3), "missing": missing}
