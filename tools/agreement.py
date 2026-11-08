"""Judge a run's shortlist with a second model and report agreement with the main judge.
Usage: tools/agreement.py <run folder> [model] [--provider PREFIX] [--limit N]"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from promofinder import candidates as C  # noqa: E402
from promofinder import judge as J  # noqa: E402
from promofinder import names, score  # noqa: E402
from promofinder.config import load_configs, load_provider  # noqa: E402
from promofinder.llm import LLM  # noqa: E402
from promofinder.parse import load_show  # noqa: E402
from promofinder.pipeline import make_shortlist, pre_score  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.join(os.path.dirname(__file__), "..")


def spearman(a, b):
    def ranks(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (va * vb) if va and vb else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("model", nargs="?")
    ap.add_argument("--provider")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--pack", help="the show zip (default: the first .zip in source/)")
    args = ap.parse_args()
    run_dir = args.run_dir
    summary = json.load(open(os.path.join(run_dir, "summary.json"), encoding="utf-8"))
    first = json.load(open(os.path.join(run_dir, "judge_verdicts.json"), encoding="utf-8"))
    cfgs = load_configs()
    show = load_show(args.pack or next(os.path.join(ROOT, "source", f) for f in sorted(os.listdir(os.path.join(ROOT, "source")))
                                       if f.lower().endswith(".zip")))
    names.normalise(show, LLM(cfgs["JUDGE"], cache_dir=os.path.join(ROOT, "cache")))
    p1 = score.score_show(show, LLM(cfgs["SCORE"], cache_dir=os.path.join(ROOT, "cache")), names.main_characters(show))
    beats = p1["beats"]
    cands = C.generate(show, lambda m: beats.get(m.id, {}).get("hook", 0), lambda m: beats.get(m.id, {}).get("cliff", 0))
    for c in cands:
        c.scores["pre"] = pre_score(show, c, beats)
    shortlist = make_shortlist(cands, summary["params"]["judge_n"])
    missing = [c.cid for c in shortlist if c.cid not in first]
    if missing:
        print(f"warning: {len(missing)} rebuilt clips are not in the run's verdicts; the run may have used other settings")
    second_cfg = load_provider(args.provider) if args.provider else cfgs["SCORE"]
    if args.model:
        second_cfg.model = args.model
    if args.limit:
        shortlist = shortlist[:args.limit]
    safe = re.sub(r"[^A-Za-z0-9.-]+", "-", second_cfg.model)
    llm2 = LLM(second_cfg, cache_dir=os.path.join(ROOT, "cache"),
               log_dir=os.path.join(run_dir, f"raw_agreement_{safe}"), on_event=print)
    second = J.judge_all(show, shortlist, llm2, progress=print)
    both = [c.cid for c in shortlist if c.cid in first and c.cid in second]
    lines = [f"# Two-model agreement on {len(both)} of {len(shortlist)} shortlisted clips\n",
             f"Judge A: `{summary['models']['JUDGE']['model']}` (the run). Judge B: `{second_cfg.model}` "
             f"at `{second_cfg.base_url}`, reasoning `{second_cfg.reasoning_effort}`, temperature `{second_cfg.temperature}`.\n",
             "| Check | Same answer | Yes rate A / B | Cohen's kappa |", "|---|---|---|---|"]
    for k in J.WEIGHTS:
        n = max(1, len(both))
        same = sum(first[c][k] == second[c][k] for c in both)
        pa, pb = sum(bool(first[c][k]) for c in both) / n, sum(bool(second[c][k]) for c in both) / n
        pe = pa * pb + (1 - pa) * (1 - pb)  # agreement expected by chance from the two yes rates
        kappa = (same / n - pe) / (1 - pe) if pe < 1 else float("nan")
        lines.append(f"| {k} | {same}/{len(both)} ({100 * same / n:.0f}%) | {100 * pa:.0f}% / {100 * pb:.0f}% | {kappa:.2f} |")
    sa, sb = [first[c]["judge_score"] for c in both], [second[c]["judge_score"] for c in both]
    top = lambda d: set(sorted(both, key=lambda c: d[c]["judge_score"], reverse=True)[:10])
    lines.append(f"\nSpearman correlation of the two judges' scores: {spearman(sa, sb):.2f}.")
    lines.append(f"Top-10 overlap: {len(top(first) & top(second))} of 10 clips appear in both judges' top 10.")
    u = llm2.usage
    lines.append(f"Judge B tokens: {u.prompt_tokens + u.completion_tokens:,} in {u.calls} new calls "
                 f"({u.reasoning_tokens:,} reasoning; {u.cache_hits} from cache; {u.repairs} repairs; {u.failures} failures).")
    if llm2.exhausted:
        lines.append("Judge B's daily quota ran out part-way; the table covers only the clips it judged.")
    text = "\n".join(lines) + "\n"
    print(text)
    open(os.path.join(run_dir, f"agreement_{safe}.md"), "w", encoding="utf-8").write(text)


if __name__ == "__main__":
    main()
