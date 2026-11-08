"""Blind viewer test: does a viewer rate the tool's promos above simpler alternatives?

  .venv\\Scripts\\python tools/blind_check.py make  <run folder> [n_tool n_baseline n_random]   (default 12 6 6)
      writes blind/sheet.md (the clips to read) and blind/sheet.csv; the answer key goes to blind/key.json
  .venv\\Scripts\\python tools/blind_check.py rate  <run folder> <clip> <hook y/n> <ending y/n> <sense y/n> <overall 1-5> [comment]
  .venv\\Scripts\\python tools/blind_check.py score <run folder>      reveals the sources and writes blind/result.md

Why this design (decided 8 Nov): a hand-made "gold" list of promos is a noisy yardstick, because a show
has many good promos and two people rarely pick the same ones. Rating clips instead asks the question
the brief cares about (would a viewer stop, follow it, and want more?), and mixing three sources blind
gives the ratings something to be compared against:
  tool      the pipeline's top promos, in rank order
  baseline  the rules-only baseline (word lists, no model) built with the same clip rules
  random    random clips that obey the same rules (20-60 s, no cut inside a spoken line)
The sheet shows only the script text with times from the clip's start: no episode, rank or source.
For every clip the rater answers the brief's three tests (hook in the first 3 s, ends unresolved,
makes sense to a stranger) and gives an overall 1-5. The score also compares the rater's three
answers with the model judge's answers on the tool's clips, which is a direct check on the judge.
"""
import csv
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from evaluate import random_windows  # noqa: E402
from promofinder.candidates import Candidate, Segment, overlap  # noqa: E402
from promofinder.judge import clip_text  # noqa: E402
from promofinder.parse import load_show  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.join(os.path.dirname(__file__), "..")
COLS = ["clip", "length_s", "hook_yn", "ending_yn", "sense_yn", "overall_1_to_5", "comment"]
CHECKS = [("hook_yn", "hook_in_first_3s", "hook in first 3 s"), ("ending_yn", "ends_unresolved", "ends unresolved"),
          ("sense_yn", "premise_clear", "makes sense to a stranger")]


def pack_path():
    src = os.path.join(ROOT, "source")
    return next(os.path.join(src, f) for f in sorted(os.listdir(src)) if f.lower().endswith(".zip"))


def segs_of(p):
    return [Segment(s["episode"], s["start_s"], s["end_s"]) for s in p["segments"]]


def make(run_dir, n_tool=12, n_base=6, n_rand=6):
    show = load_show(pack_path())
    tool = json.load(open(os.path.join(run_dir, "promos.json"), encoding="utf-8"))[:n_tool]
    base = json.load(open(os.path.join(run_dir, "baseline.json"), encoding="utf-8"))
    items = [{"source": "tool", "ref": p["rank"], "id": p["id"], "segs": segs_of(p)} for p in tool]
    as_cand = lambda segs: Candidate("x", segs, "", "")
    clash = lambda segs: any(overlap(as_cand(segs), as_cand(i["segs"])) >= 0.5 for i in items)
    for p in base:  # baseline clips that are mostly the same footage as a tool clip would be rated twice
        if sum(i["source"] == "baseline" for i in items) >= n_base:
            break
        if not clash(segs_of(p)):
            items.append({"source": "baseline", "ref": p["rank"], "id": p["id"], "segs": segs_of(p)})
    rng = random.Random(20261108)
    while sum(i["source"] == "random" for i in items) < n_rand:
        w = random_windows(show, 1, rng)[0]
        segs = [Segment(e, a, b) for e, a, b in w]
        if not clash(segs):
            items.append({"source": "random", "ref": sum(i["source"] == "random" for i in items) + 1, "id": "", "segs": segs})
    rng.shuffle(items)
    out = os.path.join(run_dir, "blind")
    os.makedirs(out, exist_ok=True)
    md = ["# Blind viewer test",
          "",
          "Read each clip as if it came up on your feed and you have never seen this show. Times count from the clip's start; "
          "ACTION is what you see, SOUND FX what you hear, [cut] a jump to another moment.",
          "",
          "For each clip answer four things, e.g. `7 y y n 3`:",
          "1. Hook: does something in the first 3 s make you stop scrolling? (y/n)",
          "2. Ending: does it cut while something is still unresolved, so you want to know what happens? (y/n)",
          "3. Sense: do you understand who is against whom and why it matters, without the rest of the show? (y/n)",
          "4. Overall: how likely would you be to tap through and watch the show? (1 = scroll past, 5 = watching now)",
          ""]
    key, rows = [], []
    for n, it in enumerate(items, 1):
        dur = sum(s.dur for s in it["segs"])
        key.append({"clip": n, "source": it["source"], "ref": it["ref"], "id": it["id"],
                    "segments": [[s.ep, round(s.start, 2), round(s.end, 2)] for s in it["segs"]]})
        rows.append({"clip": n, "length_s": round(dur, 1)})
        md += [f"## Clip {n} ({dur:.0f} s)", "", "```", clip_text(show, as_cand(it["segs"])), "```", ""]
    json.dump(key, open(os.path.join(out, "key.json"), "w", encoding="utf-8"), indent=1)
    with open(os.path.join(out, "sheet.csv"), "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    open(os.path.join(out, "sheet.md"), "w", encoding="utf-8").write("\n".join(md))
    counts = {s: sum(i["source"] == s for i in items) for s in ("tool", "baseline", "random")}
    print(f"wrote {len(items)} clips to {out} ({counts}); read sheet.md, never open key.json before scoring")


def rate(run_dir, clip, hook, ending, sense, overall, comment=""):
    path = os.path.join(run_dir, "blind", "sheet.csv")
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig", newline="")))
    yn = lambda v: {"y": "y", "yes": "y", "n": "n", "no": "n"}[v.strip().lower()]
    assert 1 <= float(overall) <= 5, "overall must be 1-5"
    hit = [r for r in rows if r["clip"] == str(clip)]
    assert hit, f"no clip {clip}"
    hit[0].update({"hook_yn": yn(hook), "ending_yn": yn(ending), "sense_yn": yn(sense), "overall_1_to_5": overall,
                   "comment": comment})
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    done = sum(bool(r["overall_1_to_5"]) for r in rows)
    print(f"clip {clip} saved ({done} of {len(rows)} rated)")


def spearman(a, b):
    def ranks(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r, i = [0.0] * len(x), 0
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
    va, vb = sum((x - ma) ** 2 for x in ra) ** 0.5, sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (va * vb) if va and vb else float("nan")


def score(run_dir):
    out = os.path.join(run_dir, "blind")
    key = {k["clip"]: k for k in json.load(open(os.path.join(out, "key.json"), encoding="utf-8"))}
    rows = [r for r in csv.DictReader(open(os.path.join(out, "sheet.csv"), encoding="utf-8-sig", newline=""))
            if r["overall_1_to_5"].strip()]
    promos = {p["id"]: p for p in json.load(open(os.path.join(run_dir, "promos.json"), encoding="utf-8"))}
    by = {s: [r for r in rows if key[int(r["clip"])]["source"] == s] for s in ("tool", "baseline", "random")}
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    ov = lambda rs: [float(r["overall_1_to_5"]) for r in rs]
    lines = [f"# Blind viewer test result\n", f"Rated {len(rows)} of {len(key)} clips. One rater, reading script text, not watching video.\n",
             "| Source | Clips | Mean overall (1-5) | Hook yes | Ending yes | Makes sense yes |", "|---|---|---|---|---|---|"]
    for s, rs in by.items():
        yes = lambda col: f"{sum(r[col] == 'y' for r in rs)}/{len(rs)}"
        lines.append(f"| {s} | {len(rs)} | {mean(ov(rs)):.2f} | {yes('hook_yn')} | {yes('ending_yn')} | {yes('sense_yn')} |")
    for other in ("baseline", "random"):
        a, b = ov(by["tool"]), ov(by[other])
        wins = sum((x > y) + 0.5 * (x == y) for x in a for y in b)
        lines.append(f"\nA tool clip outrates a {other} clip in {wins:g} of {len(a) * len(b)} pairings "
                     f"({100 * wins / max(1, len(a) * len(b)):.0f}%; 50% = no better).")
    t = [(key[int(r["clip"])]["ref"], float(r["overall_1_to_5"])) for r in by["tool"]]
    if len(t) >= 3:
        lines.append(f"Within the tool's clips, Spearman correlation between the tool's rank and the rating: "
                     f"{spearman([-x for x, _ in t], [y for _, y in t]):.2f} (1 = the tool ordered them exactly as the rater would).")
    lines += ["\nRater vs the model judge on the tool's clips:\n", "| Test | Same answer | Rater yes | Judge yes |", "|---|---|---|---|"]
    for col, check, label in CHECKS:
        pairs = [(r[col] == "y", bool(promos[key[int(r["clip"])]["id"]]["checks"].get(check))) for r in by["tool"]]
        if pairs:
            lines.append(f"| {label} | {sum(a == b for a, b in pairs)}/{len(pairs)} | {sum(a for a, _ in pairs)} | {sum(b for _, b in pairs)} |")
    lines.append("\nPer clip (revealed):\n\n| Clip | Source | Ref | Overall | Hook | Ending | Sense | Comment |\n|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: int(r["clip"])):
        k = key[int(r["clip"])]
        lines.append(f"| {r['clip']} | {k['source']} | {k['ref']} | {r['overall_1_to_5']} | {r['hook_yn']} | {r['ending_yn']} | {r['sense_yn']} | {r['comment']} |")
    text = "\n".join(lines) + "\n"
    print(text)
    open(os.path.join(out, "result.md"), "w", encoding="utf-8").write(text)


if __name__ == "__main__":
    cmd, run_dir = sys.argv[1], sys.argv[2]
    if cmd == "make":
        make(run_dir, *[int(x) for x in sys.argv[3:6]])
    elif cmd == "rate":
        rate(run_dir, *sys.argv[3:])
    else:
        score(run_dir)
