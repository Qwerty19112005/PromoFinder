"""Write report.md for a run: ranked promos, quality checks, the baseline and each promo's script."""
from __future__ import annotations

import os

from . import candidates as C


def _cell(s: str, n: int = 90) -> str:
    s = (s or "").replace("|", "/").replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "..."


def _rate(verdicts: list[dict], key: str) -> str:
    if not verdicts:
        return "n/a"
    k = sum(bool(v.get(key)) for v in verdicts)
    return f"{k}/{len(verdicts)} ({100 * k / len(verdicts):.0f}%)"


def write_report(run_dir, show, summary, promos, base_recs, final, base_top):
    L = []
    L.append(f"# Promo finder run: {show.title}\n")
    L.append(f"- Run folder: `{run_dir}`")
    L.append(f"- Show: {summary['episodes']} episodes, {summary['minutes']} min, {summary['moments']} moments "
             f"(dialogue, action and sound lines; camera lines dropped)")
    if summary["used_models"]:
        for s, m in summary["models"].items():
            u = summary["usage"].get(s, {})
            L.append(f"- {s}: `{m['model']}` at {m['base_url']}, reasoning {m['reasoning_effort'] or 'default'}; "
                     f"{u.get('calls', 0)} calls, {u.get('cache_hits', 0)} from cache, "
                     f"{u.get('prompt_tokens', 0) + u.get('completion_tokens', 0):,} tokens")
    else:
        L.append("- No model used: rules-only baseline")
    L.append(f"- Time: {summary['seconds']}")
    if summary["format_problems"]:
        L.append(f"- Format problems: {'; '.join(summary['format_problems'])}")

    L.append("\n## Ranked promos\n")
    L.append("| # | Segments | Length | Hook (first 3 s) | Cliffhanger (cut) | Why it works |")
    L.append("|---|---|---|---|---|---|")
    for p in promos:
        L.append(f"| {p['rank']} | {p['segments_text']} | {p['duration_s']} s | {_cell(p['hook']['line'], 70)} | "
                 f"{_cell(p['cliffhanger']['line'], 70)} | {_cell(p['why_it_works'], 90)} |")

    L.append("\n## Quality checks (automatic)\n")
    L.append(f"- Brief's rules (20-60 s, hook within 3 s, no cut inside a spoken line): "
             f"{len(final) - summary['rule_violations_in_final']} of {len(final)} promos pass.")
    if summary["used_models"]:
        p1 = summary["pass1"]
        L.append(f"- Pass 1 scored {p1['moments_scored']} moments in {p1['batches']} batches; "
                 f"{p1['invalid_references']} references to moments that do not exist were dropped; "
                 f"failed batches: {p1['failed_batches'] or 'none'}.")
        vs = []
        try:
            import json
            vs = list(json.load(open(os.path.join(run_dir, "judge_verdicts.json"), encoding="utf-8")).values())
        except Exception:
            pass
        L.append(f"- Cold-viewer judge on {len(vs)} shortlisted clips: hook in first 3 s {_rate(vs, 'hook_in_first_3s')}; "
                 f"premise clear {_rate(vs, 'premise_clear')}; ends unresolved {_rate(vs, 'ends_unresolved')}; "
                 f"needs earlier episodes {_rate(vs, 'needs_earlier_episodes')}; resolved before the cut "
                 f"{_rate(vs, 'resolved_before_cut')}; may spoil a twist {_rate(vs, 'spoils_major_twist')}.")
        pw = summary.get("pairwise")
        if pw:
            L.append(f"- Pairwise ranking: {pw['n_pairs']} pairs, each asked in both orders. The same winner both ways in "
                     f"{100 * pw['consistency']:.0f}% of pairs; the rest flipped with the order (position bias) and count as ties.")
        top_m, top_b = final[:10], base_top[:10]
        shared = sum(any(C.overlap(a, b) >= 0.5 for b in top_b) for a in top_m)
        L.append(f"- Against the rules-only baseline: {shared} of the top {len(top_m)} promos overlap a baseline top-10 pick by at least half.")
    flagged = [p for p in promos if p["flags"]]
    L.append(f"\n### Where it may pick badly ({len(flagged)} of {len(promos)} promos flagged)\n")
    for p in flagged:
        L.append(f"- #{p['rank']} {p['segments_text']}: {'; '.join(p['flags'])}")

    L.append("\n## Each promo\n")
    for p in promos:
        L.append(f"### #{p['rank']}: {p['segments_text']} ({p['duration_s']} s, {p['shape']})\n")
        L.append(f"- Hook: {p['hook']['line']}")
        L.append(f"- Cliffhanger: {p['cliffhanger']['line']}")
        L.append(f"- Why it works: {p['why_it_works']}")
        L.append(f"- Why this rank: {p['why_this_rank']}")
        if p["premise_as_seen_by_judge"]:
            L.append(f"- What a stranger sees (judge): {p['premise_as_seen_by_judge']}")
            L.append(f"- Question left open: {p['open_question']}")
            L.append(f"- Weakness: {p['weakness']}")
        if p["flags"]:
            L.append(f"- Flags: {'; '.join(p['flags'])}")
        L.append("\n```\n" + "\n".join(p["script"]) + "\n```\n")

    L.append("\n## Rules-only baseline (no model)\n")
    L.append("| # | Segments | Length | Hook | Cliffhanger |")
    L.append("|---|---|---|---|---|")
    for p in base_recs:
        L.append(f"| {p['rank']} | {p['segments_text']} | {p['duration_s']} s | {_cell(p['hook']['line'], 70)} | "
                 f"{_cell(p['cliffhanger']['line'], 70)} |")
    open(os.path.join(run_dir, "report.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
