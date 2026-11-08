"""Command line: python -m promofinder.run <pack folder or .zip> [--quick] [--episodes 5-20] [--rules-only]"""
from __future__ import annotations

import argparse
import sys

from .pipeline import run


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Find 20-60 s promos in a microdrama screenplay pack.")
    ap.add_argument("source", help="folder or .zip of episode .txt files")
    ap.add_argument("--rules-only", action="store_true", help="no model calls; rules baseline only")
    ap.add_argument("--judge-n", type=int, default=30, help="clips sent to the cold-viewer judge")
    ap.add_argument("--rank-n", type=int, default=8, help="clips in the all-pairs ranking (n*(n-1) calls)")
    ap.add_argument("--final-n", type=int, default=20, help="promos in the final list")
    ap.add_argument("--episodes", default=None, help="scan only these episodes, e.g. 1-10 (quick demo)")
    ap.add_argument("--quick", action="store_true", help="quick demo: episodes 1-10, 12 clips judged, 4 ranked")
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", default="runs")
    a = ap.parse_args(argv)
    eps = tuple(int(x) for x in a.episodes.split("-")) if a.episodes else None
    if a.quick:
        eps, a.judge_n, a.rank_n, a.final_n = eps or (1, 10), 12, 4, 10
    s = run(a.source, out_root=a.out, use_llm=not a.rules_only, judge_n=a.judge_n, rank_n=a.rank_n,
            final_n=a.final_n, title=a.title, progress=lambda m: print(m, flush=True), episodes=eps)
    print(f"report: {s['run_dir']}/report.md")


if __name__ == "__main__":
    main()
