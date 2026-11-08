"""Run the whole pipeline on one show and write every output to runs/<timestamp>/."""
from __future__ import annotations

import csv
import json
import os
import time

from . import candidates as C
from . import judge as J
from . import names, rank, rules, score
from .config import load_configs
from .llm import LLM
from .parse import Show, load_show, mmss


def _pct(values, p):
    v = sorted(values)
    return v[int(p * (len(v) - 1))] if v else 0.0


def rule_scores(show: Show):
    last = {e.ep: e.moments[-1].id for e in show.episodes if e.moments}
    H = {m.id: rules.hook(m) for e in show.episodes for m in e.moments}
    K = {m.id: rules.cliff(m, m.id == last.get(m.ep)) for e in show.episodes for m in e.moments}
    return H, K


def baseline(show: Show, final_n: int):
    H, K = rule_scores(show)
    hook_min = max(0.5, _pct(H.values(), 0.90))
    cliff_min = max(0.5, _pct(K.values(), 0.85))
    cands = C.generate(show, lambda m: H[m.id], lambda m: K[m.id], hook_min=hook_min, cliff_min=cliff_min)
    for c in cands:
        ms = [m for _, m in c.moments(show)]
        c.scores["rules"] = round(2 * H[c.hook_id] + 2 * K[c.cliff_id] + sum(rules.heat(m) for m in ms) / max(1, len(ms)), 3)
    top = C.suppress(cands, key=lambda c: c.scores["rules"], limit=final_n, per_episode=2)
    return cands, top, {"hook_min": hook_min, "cliff_min": cliff_min}


def pre_score(show: Show, c: C.Candidate, beats: dict) -> float:
    hb = beats.get(c.hook_id, {})
    cb = beats.get(c.cliff_id, {})
    cliff_m = show.moment(c.cliff_id)
    is_end = show.by_ep[cliff_m.ep].moments[-1].id == cliff_m.id
    cliff = max(cb.get("cliff", 0), 1.5 if is_end else 0)
    inside = c.moments(show)
    points = sum(beats.get(m.id, {}).get("points", 0) for _, m in inside)
    heat = min(3.0, points / max(1.0, c.dur / 30) / 3)
    context = max([beats.get(m.id, {}).get("context", 0) for t, m in inside if t <= C.HOOK_WINDOW] or [0])
    early = 0.5 if c.ep <= 10 else 0.0
    return round(2 * hb.get("hook", 0) + 2 * cliff + heat - context + early, 3)


def make_shortlist(cands: list[C.Candidate], judge_n: int, early_eps: int = 10, early_slots: int = 8) -> list[C.Candidate]:
    """Clips sent to the judge. Scores tie a lot near the top, so the best clip of each early episode gets
    a place first (new viewers start at episode 1), then the rest is filled by score."""
    key = lambda c: c.scores["pre"]
    early = []
    for e in sorted({c.ep for c in cands if c.ep <= early_eps}):
        best = C.suppress([c for c in cands if c.ep == e], key=key, limit=1)
        early += best
    early = sorted(early, key=key, reverse=True)[:early_slots]
    return C.suppress(cands, key=key, limit=judge_n, per_episode=3, seed=early)


def flags_for(show: Show, c: C.Candidate, v: dict | None) -> list[str]:
    f = []
    if v:
        if v.get("needs_earlier_episodes"):
            f.append("judge: needs earlier episodes")
        if not v.get("premise_clear"):
            f.append("judge: premise unclear to a stranger")
        if v.get("resolved_before_cut") or not v.get("ends_unresolved"):
            f.append("judge: cliffhanger weak or resolved before the cut")
        if v.get("spoils_major_twist"):
            f.append("judge: may spoil a twist")
        if v.get("names_confusing"):
            f.append("judge: names may confuse")
        if not v.get("hook_in_first_3s"):
            f.append("judge: no hook in first 3 s")
    if show.moment(c.hook_id).kind != "DIALOGUE":
        f.append("hook is visual or sound only: check the footage delivers it")
    if c.ep > 10:
        f.append(f"from episode {c.ep}: a viewer starting at episode 1 waits a long time for this")
    if c.shape == "stitch":
        f.append("joins two episodes: check the join plays smoothly")
    if c.shape == "cold_open":
        f.append("flash-forward first: check it reads as a teaser, not a jump")
    return f


def cliff_display(show: Show, c: C.Candidate, beats: dict):
    """The line to show as 'the cliffhanger': the most suspenseful moment in the last 6 s of the clip,
    preferring dialogue, so a trailing stage direction (rain, music) is not reported as the cliffhanger."""
    tail = [(t, m) for t, m in c.moments(show) if t >= c.dur - 6.0]
    if not tail:
        return show.moment(c.cliff_id)
    val = (lambda m: beats.get(m.id, {}).get("cliff", 0)) if beats else (lambda m: rules.cliff(m, False))
    return max(tail, key=lambda tm: (val(tm[1]), tm[1].kind == "DIALOGUE", tm[0]))[1]


def promo_record(show, c, rank_no, verdict, beats, rank_info, pool_ids):
    hm, cm = show.moment(c.hook_id), cliff_display(show, c, beats)
    v = verdict or {}
    if c.cid in pool_ids and rank_info:
        w = rank_info["wins"][c.cid]
        rs = rank_info["reasons"][c.cid]
        why_rank = f"Won {w:g} of {len(pool_ids) - 1} head-to-head comparisons" + (f". {rs[0]}" if rs else "")
    else:
        why_rank = f"Outside the head-to-head round; ordered by the judge's checklist score ({v.get('judge_score', 0):g} of 8.5)"
    return {
        "rank": rank_no, "id": c.cid, "shape": c.shape, "duration_s": round(c.dur, 1),
        "segments": [{"episode": s.ep, "start": mmss(s.start), "end": mmss(s.end),
                      "start_s": round(s.start, 2), "end_s": round(s.end, 2)} for s in c.segments],
        "segments_text": c.label(),
        "hook": {"moment": hm.id, "line": f"{hm.who()}: {hm.text}"},
        "cliffhanger": {"moment": cm.id, "line": f"{cm.who()}: {cm.text}"},
        "why_it_works": v.get("why_it_works") or beats.get(c.hook_id, {}).get("note", ""),
        "why_this_rank": why_rank,
        "premise_as_seen_by_judge": v.get("premise", ""),
        "open_question": v.get("open_question", ""),
        "weakness": v.get("weakness", ""),
        "checks": {k: v.get(k) for k in J.WEIGHTS} if v else {},
        "scores": {k: c.scores.get(k) for k in ("pairwise_wins", "judge", "pre", "rules") if k in c.scores},
        "flags": flags_for(show, c, verdict),
        "script": J.clip_text(show, c).split("\n"),
    }


def _write_csv(path, rows, cols):
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def run(source, out_root: str = "runs", cache_dir: str = "cache", env: dict | None = None, progress=print,
        judge_n: int = 30, rank_n: int = 8, final_n: int = 20, use_llm: bool = True, title: str | None = None,
        episodes: tuple[int, int] | None = None) -> dict:
    """episodes=(first, last) scans only those episodes, e.g. (1, 10) for a quick demo."""
    t_start = time.time()
    timings = {}
    show = load_show(source, title)
    if episodes:
        lo, hi = episodes
        show = Show(show.title, [e for e in show.episodes if lo <= e.ep <= hi])
        if not show.episodes:
            raise ValueError(f"No episodes numbered {lo} to {hi} in this upload.")
    if not any(e.moments for e in show.episodes):
        raise ValueError(
            "No timed script lines found. Each line should look like a head line (for example **ACTION:** or "
            "**NAME (Emotion):**) followed by 'Start: (mm:ss.ss) | End: (mm:ss.ss)' and the text. "
            + ("Problems: " + "; ".join(show.problems[:5]) if show.problems else f"Episode files read: {len(show.episodes)}."))
    run_dir = os.path.join(out_root, time.strftime("%Y%m%d-%H%M%S"))
    raw_dir = os.path.join(run_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    progress(f"parsed {len(show.episodes)} episodes, {sum(len(e.moments) for e in show.episodes)} moments; output in {run_dir}")
    if show.problems:
        progress("format problems: " + "; ".join(show.problems[:5]))

    cfgs = load_configs(env)
    if use_llm and not all(cfgs[s].base_url and cfgs[s].model and cfgs[s].api_key for s in cfgs):
        progress("no complete model settings found: running the rules-only baseline")
        use_llm = False
    llms = {}
    if use_llm:
        for s in cfgs:
            llms[s] = LLM(cfgs[s], cache_dir=cache_dir, log_dir=raw_dir, on_event=progress)

    t = time.time()
    name_info = names.normalise(show, llms.get("JUDGE"))
    timings["names"] = time.time() - t
    progress(f"names: {name_info['source']}")

    t = time.time()
    base_all, base_top, base_params = baseline(show, final_n)
    timings["baseline"] = time.time() - t
    progress(f"baseline: {len(base_all)} rule candidates, top {len(base_top)} kept")

    beats, p1 = {}, {"invalid": [], "failed_batches": [], "batches": 0}
    verdicts, rank_info, pool, final = {}, None, [], []
    if use_llm:
        t = time.time()
        p1 = score.score_show(show, llms["SCORE"], names.main_characters(show), progress=progress)
        beats = p1["beats"]
        timings["pass1"] = time.time() - t
        progress(f"pass 1: {len(beats)} moments scored, {len(p1['invalid'])} invalid references dropped")

        if beats:
            hook_of = lambda m: beats.get(m.id, {}).get("hook", 0)
            cliff_of = lambda m: beats.get(m.id, {}).get("cliff", 0)
            cands = C.generate(show, hook_of, cliff_of, hook_min=2, cliff_min=2)
            for c in cands:
                c.scores["pre"] = pre_score(show, c, beats)
        else:
            progress("pass 1 returned nothing: judging the rules-only candidates instead")
            cands = base_all
            for c in cands:
                c.scores["pre"] = c.scores.get("rules", 0)
        Hr, Kr = rule_scores(show)
        for c in cands:
            ms = [m for _, m in c.moments(show)]
            c.scores.setdefault("rules", round(2 * Hr[c.hook_id] + 2 * Kr[c.cliff_id] + sum(rules.heat(m) for m in ms) / max(1, len(ms)), 3))
        shortlist = make_shortlist(cands, judge_n)
        progress(f"candidates: {len(cands)} built, {len(shortlist)} shortlisted for the judge")

        t = time.time()
        verdicts = J.judge_all(show, shortlist, llms["JUDGE"], progress=progress)
        timings["judge"] = time.time() - t
        for c in shortlist:
            if c.cid in verdicts:
                c.scores["judge"] = verdicts[c.cid]["judge_score"]
        judged = [c for c in shortlist if c.cid in verdicts]
        jkey = lambda c: (c.scores.get("judge", 0), c.scores["pre"])
        pool = C.suppress(judged, key=jkey, limit=rank_n, per_episode=2)

        t = time.time()
        if len(pool) >= 2:
            rank_info = rank.rank_pairwise(show, pool, llms["RANK"], progress=progress)
            for c in pool:
                c.scores["pairwise_wins"] = rank_info["wins"][c.cid]
        timings["rank"] = time.time() - t
        pool_ids = {c.cid for c in pool}
        ranked = sorted(pool, key=lambda c: (c.scores.get("pairwise_wins", 0),) + jkey(c), reverse=True)
        rest = sorted([c for c in judged if c.cid not in pool_ids], key=jkey, reverse=True)
        final = (ranked + rest)[:final_n]
    else:
        cands, shortlist, pool_ids = [], [], set()
        final = base_top

    promos = [promo_record(show, c, i + 1, verdicts.get(c.cid), beats, rank_info, pool_ids if use_llm else set())
              for i, c in enumerate(final)]
    base_recs = [promo_record(show, c, i + 1, None, {}, None, set()) for i, c in enumerate(base_top)]
    for r in promos + base_recs:
        r["hook_line"], r["cliffhanger_line"] = r["hook"]["line"], r["cliffhanger"]["line"]
        r["flags_text"] = "; ".join(r["flags"])
    cols = ["rank", "segments_text", "duration_s", "shape", "hook_line", "cliffhanger_line", "why_it_works",
            "why_this_rank", "weakness", "flags_text", "id"]
    json.dump(promos, open(os.path.join(run_dir, "promos.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    _write_csv(os.path.join(run_dir, "promos.csv"), promos, cols)
    json.dump(base_recs, open(os.path.join(run_dir, "baseline.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    _write_csv(os.path.join(run_dir, "baseline.csv"), base_recs, cols)
    json.dump(name_info, open(os.path.join(run_dir, "names.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if use_llm:
        beat_rows = {mid: {**b, "line": f"{show.moment(mid).who()}: {show.moment(mid).text}",
                           "start": mmss(show.moment(mid).start)} for mid, b in sorted(beats.items())}
        json.dump({"beats": beat_rows, "invalid_references": p1["invalid"], "failed_batches": p1["failed_batches"]},
                  open(os.path.join(run_dir, "pass1_beats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        _write_csv(os.path.join(run_dir, "candidates.csv"),
                   [{"id": c.cid, "segments": c.label(), "shape": c.shape, "duration_s": round(c.dur, 1),
                     "pre": c.scores.get("pre"), "rules": c.scores.get("rules"), "judge": c.scores.get("judge"),
                     "shortlisted": c in shortlist} for c in cands],
                   ["id", "segments", "shape", "duration_s", "pre", "rules", "judge", "shortlisted"])
        json.dump({cid: {**v, "segments": next(c.label() for c in shortlist if c.cid == cid)} for cid, v in verdicts.items()},
                  open(os.path.join(run_dir, "judge_verdicts.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        if rank_info:
            json.dump(rank_info, open(os.path.join(run_dir, "pairwise.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    usage = {s: {**vars(l.usage), "model": l.cfg.model} for s, l in llms.items()}
    summary = {
        "show": show.title, "episodes": len(show.episodes), "minutes": round(sum(e.runtime for e in show.episodes) / 60, 1),
        "moments": sum(len(e.moments) for e in show.episodes), "format_problems": show.problems,
        "used_models": use_llm, "models": {s: c.public() for s, c in cfgs.items()} if use_llm else {},
        "params": {"judge_n": judge_n, "rank_n": rank_n, "final_n": final_n, "baseline": base_params,
                   "episodes": list(episodes) if episodes else "all"},
        "pass1": {"moments_scored": len(beats), "invalid_references": len(p1["invalid"]),
                  "failed_batches": p1["failed_batches"], "batches": p1["batches"]},
        "candidates": len(cands), "shortlisted": len(shortlist), "judged": len(verdicts),
        "pairwise": {k: rank_info[k] for k in ("n_pairs", "consistency", "missing")} if rank_info else None,
        "rule_violations_in_final": sum(bool(C.check(show, c)) for c in final),
        "seconds": {k: round(v, 1) for k, v in timings.items()} | {"total": round(time.time() - t_start, 1)},
        "usage": usage, "run_dir": run_dir,
    }
    json.dump(summary, open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    from .report import write_report
    write_report(run_dir, show, summary, promos, base_recs, final, base_top)
    progress(f"done in {summary['seconds']['total']:.0f}s: {len(promos)} promos written to {run_dir}")
    return summary
