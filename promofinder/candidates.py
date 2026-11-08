"""Build candidate promo clips from scored moments, using code only.

A clip is one or more segments (episode, start, end) in play order. Every candidate obeys the
brief's rules by construction, and `check` re-verifies them:
- total length 20 to 60 s
- the hook moment starts within the first 3 s
- no segment boundary cuts through a spoken line
Shapes:
- single:    one stretch, hook -> ... -> cliffhanger moment
- stitch:    the end of episode N run straight into the start of episode N+1, ending on a cliffhanger there
- cold_open: a short striking moment from later in the episode first (a flash-forward), then the run-up
             that leads towards it, cut before it happens
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .parse import Episode, Moment, Show, mmss

MIN_LEN, MAX_LEN, HOOK_WINDOW = 20.0, 60.0, 3.0
LEAD, TAIL, EPS = 0.4, 0.3, 0.05


@dataclass
class Segment:
    ep: int
    start: float
    end: float

    @property
    def dur(self) -> float:
        return self.end - self.start

    def label(self) -> str:
        return f"E{self.ep:02d} {mmss(self.start)}-{mmss(self.end)}"


@dataclass
class Candidate:
    shape: str
    segments: list[Segment]
    hook_id: str
    cliff_id: str
    scores: dict = field(default_factory=dict)
    cid: str = ""

    @property
    def dur(self) -> float:
        return sum(s.dur for s in self.segments)

    @property
    def ep(self) -> int:
        return self.segments[0].ep

    def label(self) -> str:
        return "; ".join(s.label() for s in self.segments)

    def moments(self, show: Show) -> list[tuple[float, Moment]]:
        """(time from clip start, moment) in play order."""
        out, offset = [], 0.0
        for seg in self.segments:
            for m in show.by_ep[seg.ep].moments:
                if m.end > seg.start + EPS and m.start < seg.end - EPS:
                    out.append((offset + max(0.0, m.start - seg.start), m))
            offset += seg.dur
        return out


def _cut_line(ep: Episode, t: float, slack: float = EPS):
    for m in ep.moments:
        if m.kind == "DIALOGUE" and m.start + slack < t < m.end - slack:
            return m
    return None


def snap_start(ep: Episode, hook: Moment) -> float | None:
    s = max(0.0, hook.start - LEAD)
    line = _cut_line(ep, s)
    if line is not None and line is not hook:
        s = line.end if line.end <= hook.start + EPS else line.start
    if hook.start - s > HOOK_WINDOW or s > hook.start + EPS:
        return None
    return s


def snap_end(ep: Episode, t: float) -> float:
    e = min(ep.runtime, t + TAIL)
    for _ in range(4):
        line = _cut_line(ep, e)
        if line is None:
            break
        e = max(t, line.start - EPS) if line.start >= t - EPS else line.end
    return min(ep.runtime, e)


def check(show: Show, cand: Candidate, slack: float = 0.3) -> list[str]:
    """Rule violations for a candidate (empty list = passes)."""
    out = []
    if not MIN_LEN - 1e-6 <= cand.dur <= MAX_LEN + 1e-6:
        out.append(f"length {cand.dur:.1f}s outside {MIN_LEN:.0f}-{MAX_LEN:.0f}s")
    for seg in cand.segments:
        if seg.ep not in show.by_ep:
            out.append(f"no episode {seg.ep}")
            continue
        ep = show.by_ep[seg.ep]
        if seg.end <= seg.start or seg.start < -EPS or seg.end > ep.runtime + 0.5:
            out.append(f"bad segment {seg.label()}")
        for t in (seg.start, seg.end):
            if _cut_line(ep, t, slack):
                out.append(f"cuts a line at E{seg.ep:02d} {mmss(t)}")
    if cand.hook_id:
        hook_at = next((t for t, m in cand.moments(show) if m.id == cand.hook_id), None)
        if hook_at is None or hook_at > HOOK_WINDOW + 1e-6:
            out.append("hook not within the first 3 s")
    return out


def overlap(a: Candidate, b: Candidate) -> float:
    """Shared screen time as a share of the shorter clip."""
    inter = 0.0
    for sa in a.segments:
        for sb in b.segments:
            if sa.ep == sb.ep:
                inter += max(0.0, min(sa.end, sb.end) - max(sa.start, sb.start))
    return inter / max(1e-6, min(a.dur, b.dur))


def generate(show: Show, hook_of, cliff_of, hook_min: float = 2, cliff_min: float = 2,
             per_hook: int = 3) -> list[Candidate]:
    """hook_of(moment) and cliff_of(moment) return scores on a 0-3 scale."""
    cands: list[Candidate] = []
    for ep in show.episodes:
        ms = ep.moments
        if not ms:
            continue
        last = ms[-1]
        cliffs = [m for m in ms if cliff_of(m) >= cliff_min or m is last]
        hooks = [m for m in ms if hook_of(m) >= hook_min]
        nxt = show.by_ep.get(ep.ep + 1)
        for h in hooks:
            s = snap_start(ep, h)
            if s is None:
                continue
            # single stretch
            singles = []
            for c in cliffs:
                if c.end <= h.end - EPS:
                    continue
                e = snap_end(ep, c.end)
                if MIN_LEN <= e - s <= MAX_LEN:
                    singles.append(Candidate("single", [Segment(ep.ep, s, e)], h.id, c.id))
            singles.sort(key=lambda x: (-cliff_of(show.moment(x.cliff_id)), x.dur))
            cands += singles[:per_hook]
            # stitch into the next episode
            tail = ep.runtime - s
            if nxt and nxt.moments and tail < MAX_LEN - 5:
                b0 = max(0.0, nxt.moments[0].start - LEAD)
                stitches = []
                for c in nxt.moments:
                    if not (cliff_of(c) >= cliff_min or c is nxt.moments[-1]):
                        continue
                    e = snap_end(nxt, c.end)
                    if MIN_LEN <= tail + (e - b0) <= MAX_LEN:
                        stitches.append(Candidate("stitch", [Segment(ep.ep, s, ep.runtime), Segment(nxt.ep, b0, e)], h.id, c.id))
                stitches.sort(key=lambda x: (-cliff_of(show.moment(x.cliff_id)), x.dur))
                cands += stitches[:max(1, per_hook - 1)]
            # cold open: this short hook first, then the earlier run-up that leads towards it
            if hook_of(h) >= 3 and h.end - h.start <= 6:
                e_a = snap_end(ep, h.end)
                a = Segment(ep.ep, s, e_a)
                if a.dur > 8:
                    continue
                opens = []
                for c in cliffs:
                    if c.end > s - 1.0 or c is h:
                        continue
                    e_b = snap_end(ep, c.end)
                    if e_b > s - EPS:
                        continue
                    room_hi, room_lo = MAX_LEN - a.dur, MIN_LEN - a.dur
                    starts = [o for o in ms if e_b - room_hi <= o.start <= e_b - room_lo and o.start < c.start]
                    if not starts:
                        continue
                    o = max(starts, key=lambda m: (hook_of(m), -m.start))
                    b_s = max(0.0, o.start - LEAD)
                    line = _cut_line(ep, b_s)
                    if line is not None:
                        b_s = line.start
                    if MIN_LEN <= a.dur + (e_b - b_s) <= MAX_LEN:
                        opens.append(Candidate("cold_open", [a, Segment(ep.ep, b_s, e_b)], h.id, c.id))
                opens.sort(key=lambda x: (-cliff_of(show.moment(x.cliff_id)), x.dur))
                cands += opens[:max(1, per_hook - 2)]
    out, seen = [], set()
    for c in cands:
        key = tuple((s.ep, round(s.start, 1), round(s.end, 1)) for s in c.segments)
        if key not in seen and not check(show, c):
            seen.add(key)
            out.append(c)
    for i, c in enumerate(out):
        c.cid = f"C{i + 1:04d}"
    return out


def suppress(cands: list[Candidate], key, limit: int, max_overlap: float = 0.5,
             per_episode: int | None = None, seed: list[Candidate] | None = None) -> list[Candidate]:
    """Best-first selection that drops near-duplicates (non-maximum suppression) and caps clips per episode.
    `seed` clips are kept first and count toward the limits."""
    kept, per_ep = list(seed or []), {}
    for k in kept:
        per_ep[k.ep] = per_ep.get(k.ep, 0) + 1
    for c in sorted(cands, key=key, reverse=True):
        if len(kept) >= limit:
            break
        if c in kept:
            continue
        if any(overlap(c, k) >= max_overlap or {c.hook_id, c.cliff_id} & {k.hook_id, k.cliff_id} for k in kept):
            continue  # near-duplicate: mostly the same footage, or built on the same opening or closing moment
        if per_episode is not None and per_ep.get(c.ep, 0) >= per_episode:
            continue
        kept.append(c)
        per_ep[c.ep] = per_ep.get(c.ep, 0) + 1
        if len(kept) >= limit:
            break
    return kept
