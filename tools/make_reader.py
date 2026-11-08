"""Write an easy-to-read copy of every episode ([start-end] WHO (emotion): text) for picking promos by hand."""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from inspect_pack import parse_episode  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.join(os.path.dirname(__file__), "..")


def mmss(t):
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def episode_text(e):
    end = max((b["end"] for b in e["blocks"]), default=0)
    out = [f"===== EPISODE {e['ep']}   (runtime {mmss(end)})", f"Logline: {e['logline']}", ""]
    for b in e["blocks"]:
        if b["kind"] == "CAMERA":
            continue
        who = f"{b['speaker']} ({b['emotion']})" if b["kind"] == "DIALOGUE" else b["kind"]
        out.append(f"[{mmss(b['start'])}-{mmss(b['end'])}] {who}: {b['text']}")
    return "\n".join(out) + "\n"


def main():
    pack = sys.argv[1] if len(sys.argv) > 1 else glob.glob(os.path.join(ROOT, "source", "screenplays", "*"))[0]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "source", "readable")
    os.makedirs(out_dir, exist_ok=True)
    every = []
    for f in sorted(glob.glob(os.path.join(pack, "episode-*.txt"))):
        e = parse_episode(f)
        text = episode_text(e)
        every.append(text)
        with open(os.path.join(out_dir, f"E{e['ep']:02d}.txt"), "w", encoding="utf-8") as fh:
            fh.write(text)
    with open(os.path.join(out_dir, "ALL_EPISODES.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n\n".join(every))
    print(f"wrote {len(every)} episodes to {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
