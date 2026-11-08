"""Compare the tool's promos with hand-made gold picks.

Usage:  .venv\\Scripts\\python tools/evaluate.py <run folder> [gold csv] [--include-example]

A gold promo counts as found when one of the tool's top-k promos covers at least half of its screen time.
Reports recall in the top 10 and top 20 for the full pipeline and the rules-only baseline, the start and
end error of each match, and a chance level: the same recall for random clips that obey the same rules
(average of 300 draws). Writes evaluation.md into the run folder.
Rows whose notes start with "EXAMPLE FROM CLAUDE" are left out unless --include-example is given.
"""
import csv
import glob
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from promofinder.candidates import _cut_line  # noqa: E402
from promofinder.parse import load_show  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.join(os.path.dirname(__file__), "..")
SEG = re.compile(r"E\s*(\d+)\s+([\d:.]+)\s*-\s*([\d:.]+)", re.I)


def to_secs(t):
    parts = t.split(":")
    return float(parts[-1]) + (int(parts[-2]) * 60 if len(parts) > 1 else 0)


def segs_from_text(s):
    return [(int(e), to_secs(a), to_secs(b)) for e, a, b in SEG.findall(s)]


def dur(segs):
    return sum(b - a for _, a, b in segs)


def coverage(gold, pred):
    inter = sum(max(0.0, min(gb, pb) - max(ga, pa)) for ge, ga, gb in gold for pe, pa, pb in pred if ge == pe)
    return inter / max(1e-6, dur(gold))


def evaluate(gold, preds, k):
    found, errors = [], []
    for g in gold:
        best = max(((coverage(g["segs"], p), i) for i, p in enumerate(preds[:k])), default=(0, -1))
        if best[0] >= 0.5:
            p = preds[best[1]]
            errors.append((abs(g["segs"][0][1] - p[0][1]), abs(g["segs"][-1][2] - p[-1][2])))
            found.append((g["id"], best[1] + 1, round(best[0], 2)))
        else:
            found.append((g["id"], None, round(best[0], 2)))
    return found, errors


def random_windows(show, n, rng):
    out = []
    eps = [e for e in show.episodes if e.runtime > 25]
    while len(out) < n:
        e = rng.choice(eps)
        length = rng.uniform(20, min(60, e.runtime))
        a = rng.uniform(0, e.runtime - length)
        b = a + length
        if _cut_line(e, a, 0.3) or _cut_line(e, b, 0.3):
            continue
        out.append([(e.ep, a, b)])
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    run_dir = args[0]
    gold_path = args[1] if len(args) > 1 else os.path.join(ROOT, "gold", "gold_promos.csv")
    include_example = "--include-example" in sys.argv
    with open(gold_path, encoding="utf-8-sig", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if (r.get("segments") or "").strip()]
    gold = [{"id": r["promo_id"].strip(), "segs": segs_from_text(r["segments"])} for r in rows
            if include_example or not (r.get("notes") or "").strip().upper().startswith("EXAMPLE FROM CLAUDE")]
    gold = [g for g in gold if g["segs"]]
    if not gold:
        sys.exit("no gold promos to compare against")
    systems = {}
    for name, fname in (("pipeline", "promos.json"), ("rules baseline", "baseline.json")):
        p = os.path.join(run_dir, fname)
        if os.path.exists(p):
            recs = json.load(open(p, encoding="utf-8"))
            systems[name] = [[(s["episode"], s["start_s"], s["end_s"]) for s in r["segments"]] for r in recs]
    pack = glob.glob(os.path.join(ROOT, "source", "screenplays", "*"))[0]
    show = load_show(pack)
    rng = random.Random(7)
    lines = [f"# Evaluation against {len(gold)} gold promos\n", f"Run: `{run_dir}`; gold: `{gold_path}`\n",
             "| System | Found in top 10 | Found in top 20 | Median start error | Median end error |", "|---|---|---|---|---|"]
    detail = []
    for name, preds in systems.items():
        f10, _ = evaluate(gold, preds, 10)
        f20, err = evaluate(gold, preds, 20)
        h10, h20 = sum(r is not None for _, r, _ in f10), sum(r is not None for _, r, _ in f20)
        med = lambda xs: sorted(xs)[len(xs) // 2] if xs else float("nan")
        lines.append(f"| {name} | {h10}/{len(gold)} | {h20}/{len(gold)} | {med([e[0] for e in err]):.1f} s | {med([e[1] for e in err]):.1f} s |")
        detail.append(f"\n**{name}**: " + "; ".join(f"{gid} -> {'#' + str(r) if r else 'not found'} (cover {c})" for gid, r, c in f20))
    trials = [evaluate(gold, random_windows(show, 20, rng), k) for _ in range(300) for k in (10,)]
    r10 = sum(sum(r is not None for _, r, _ in f) for f, _ in trials) / len(trials)
    trials20 = [evaluate(gold, random_windows(show, 20, rng), 20) for _ in range(300)]
    r20 = sum(sum(r is not None for _, r, _ in f) for f, _ in trials20) / len(trials20)
    lines.append(f"| random valid clips (chance, mean of 300 draws) | {r10:.2f}/{len(gold)} | {r20:.2f}/{len(gold)} | | |")
    lines += detail
    lines.append("\nOne annotator made the gold picks, so these numbers measure agreement with one person's taste, not truth.")
    text = "\n".join(lines) + "\n"
    print(text)
    open(os.path.join(run_dir, "evaluation.md"), "w", encoding="utf-8").write(text)


if __name__ == "__main__":
    main()
