"""Check a hand-made promo list against the promo rules, and print each clip's script.

Usage:  .venv\\Scripts\\python tools/check_gold.py [gold csv] [pack dir]
Default csv: gold/gold_promos.csv. Columns: promo_id, segments, hook, cliffhanger, rating_1_to_5, notes.
`segments` is one or more  E<ep> mm:ss.s-mm:ss.s  separated by ;  for example
    E03 00:58.0-01:36.3; E03 02:13.2-02:20.5

Checks: episode exists, times inside the episode, total length 20-60 s, the start and end do not
cut through the middle of a spoken line, ids unique, rating 1-5. It also prints what is on screen in
the first 3 s (is the hook really there?) and the full clip text (does it make sense on its own?).
"""
import csv
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from inspect_pack import parse_episode  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.join(os.path.dirname(__file__), "..")
SEG = re.compile(r"E\s*(\d+)\s+([\d:.]+)\s*-\s*([\d:.]+)", re.I)
SLACK = 0.3  # seconds of tolerance when deciding whether a cut lands inside a line


def to_secs(t):
    parts = t.split(":")
    return float(parts[-1]) + (int(parts[-2]) * 60 if len(parts) > 1 else 0)


def mmss(t):
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def label(b):
    return f"{b['speaker']} ({b['emotion']})" if b["kind"] == "DIALOGUE" else b["kind"]


def check_row(row, eps):
    pid = row.get("promo_id", "?").strip()
    problems, notes = [], []
    segs = SEG.findall(row.get("segments", ""))
    if not segs:
        return pid, [f"could not read segments {row.get('segments')!r}; write e.g. E03 00:58.0-01:36.3"], [], []
    total, clip = 0.0, []
    for i, (ep_s, a_s, b_s) in enumerate(segs):
        ep, a, b = int(ep_s), to_secs(a_s), to_secs(b_s)
        if ep not in eps:
            problems.append(f"segment {i + 1}: there is no episode {ep}")
            continue
        blocks = [x for x in eps[ep]["blocks"] if x["kind"] != "CAMERA"]
        runtime = max(x["end"] for x in eps[ep]["blocks"])
        if b <= a:
            problems.append(f"segment {i + 1}: ends before it starts")
            continue
        if a < 0 or b > runtime + 0.5:
            problems.append(f"segment {i + 1}: outside episode {ep} (runtime {mmss(runtime)})")
        total += b - a
        for x in blocks:
            if x["kind"] == "DIALOGUE" and x["start"] + SLACK < a < x["end"] - SLACK:
                problems.append(f"segment {i + 1} starts mid-line at {mmss(a)}: line runs "
                                f"{mmss(x['start'])}-{mmss(x['end'])} \"{x['text'][:50]}\"")
            if x["kind"] == "DIALOGUE" and x["start"] + SLACK < b < x["end"] - SLACK:
                problems.append(f"segment {i + 1} ends mid-line at {mmss(b)}: line runs "
                                f"{mmss(x['start'])}-{mmss(x['end'])} \"{x['text'][:50]}\"")
        inside = [x for x in blocks if x["end"] > a + 0.05 and x["start"] < b - 0.05]
        clip.append((ep, a, b, inside))
    if clip and not 20 <= total <= 60:
        problems.append(f"total length {total:.1f} s; a promo must be 20-60 s")
    rating = row.get("rating_1_to_5", "").strip()
    if rating and rating not in {"1", "2", "3", "4", "5"}:
        problems.append(f"rating {rating!r} should be 1-5")
    for col in ("hook", "cliffhanger"):
        if not row.get(col, "").strip():
            notes.append(f"{col} is empty")
    return pid, problems, notes, (total, clip)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "gold", "gold_promos.csv")
    pack = sys.argv[2] if len(sys.argv) > 2 else glob.glob(os.path.join(ROOT, "source", "screenplays", "*"))[0]
    eps = {e["ep"]: e for e in map(parse_episode, sorted(glob.glob(os.path.join(pack, "episode-*.txt"))))}
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if any(v and v.strip() for v in r.values())]
    seen, bad = set(), 0
    for row in rows:
        pid, problems, notes, info = check_row(row, eps)
        if pid in seen:
            problems.append(f"promo_id {pid} used twice")
        seen.add(pid)
        bad += bool(problems)
        print("=" * 78)
        total = info[0] if info else 0
        print(f"{pid}   {row.get('segments', '').strip()}   total {total:.1f} s   "
              f"{'PROBLEMS' if problems else 'rules OK'}")
        for p in problems:
            print("  ! " + p)
        for n in notes:
            print("  - " + n)
        if not info:
            continue
        first, elapsed = [], 0.0
        for ep, a, b, inside in info[1]:
            for x in inside:
                if elapsed + max(0.0, x["start"] - a) < 3.0:
                    first.append(f"{label(x)}: {x['text'][:70]}")
            elapsed += b - a
        print("  first 3 s : " + (" | ".join(first) if first else "nothing starts in the first 3 s"))
        for ep, a, b, inside in info[1]:
            print(f"  --- E{ep} {mmss(a)}-{mmss(b)} ({b - a:.1f} s)")
            for x in inside:
                print(f"    [{mmss(x['start'])}-{mmss(x['end'])}] {label(x)}: {x['text']}")
    print("=" * 78)
    print(f"{len(rows)} promos checked, {bad} with problems")


if __name__ == "__main__":
    main()
