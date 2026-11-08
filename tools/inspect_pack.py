"""Parse a microdrama screenplay pack and report what is in it.

Every block in the pack looks like
    **HEAD:**
    Start: (mm:ss.ss) | End: (mm:ss.ss)
    text...
where HEAD is CAMERA, ACTION, SOUND FX, or a speaker with emotion tags such as "LADY MARGARET (Cruel, Savoring)".

Usage:
    python tools/inspect_pack.py <pack dir>              statistics and format checks
    python tools/inspect_pack.py <pack dir> 3 12 41      also print the dialogue and action of those episodes
"""
import collections
import glob
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BLOCK = re.compile(
    r"\*\*(?P<head>[^*]+?):\*\*\s*\n\s*Start:\s*\((?P<s>[\d:.]+)\)\s*\|\s*End:\s*\((?P<e>[\d:.]+)\)\s*\n"
    r"(?P<body>.*?)(?=\n\s*\*\*|\n--- SCRIPT END|\nSCENE \d+|\Z)",
    re.S,
)
KINDS = ("CAMERA", "ACTION", "SOUND FX")


def secs(t):
    m, s = t.split(":")
    return int(m) * 60 + float(s)


def parse_episode(path):
    raw = open(path, encoding="utf-8", errors="replace").read()
    ep = int(re.search(r"episode-(\d+)", os.path.basename(path)).group(1))
    blocks = []
    for m in BLOCK.finditer(raw):
        head = m.group("head").strip()
        kind = head if head in KINDS else "DIALOGUE"
        speaker, emotion = "", ""
        if kind == "DIALOGUE":
            mm = re.match(r"(.*?)\s*\((.*)\)\s*$", head)
            speaker, emotion = (mm.group(1), mm.group(2)) if mm else (head, "")
        blocks.append(dict(ep=ep, kind=kind, speaker=speaker, emotion=emotion,
                           start=secs(m.group("s")), end=secs(m.group("e")),
                           text=" ".join(m.group("body").split())))
    logline = re.search(r"EPISODE LOGLINE:\s*(.*)", raw)
    return dict(ep=ep, raw=raw, blocks=blocks, logline=logline.group(1).strip() if logline else "")


def main():
    pack = sys.argv[1]
    want = {int(x) for x in sys.argv[2:]}
    files = sorted(glob.glob(os.path.join(pack, "episode-*.txt")))
    total = collections.Counter()
    speakers = collections.Counter()
    problems = []
    for f in files:
        e = parse_episode(f)
        expected = e["raw"].count("Start: (")
        if expected != len(e["blocks"]):
            problems.append(f"E{e['ep']}: {expected} 'Start:' lines but {len(e['blocks'])} blocks parsed")
        for b in e["blocks"]:
            if b["end"] < b["start"]:
                problems.append(f"E{e['ep']}: block ends before it starts at {b['start']:.2f}")
            total[b["kind"]] += 1
            total["chars_" + b["kind"]] += len(b["text"])
            if b["kind"] == "DIALOGUE":
                speakers[b["speaker"]] += 1
        total["seconds"] += max((b["end"] for b in e["blocks"]), default=0)
        if e["ep"] in want:
            print(f"===== E{e['ep']}: {e['logline']}")
            for b in e["blocks"]:
                if b["kind"] != "CAMERA":
                    who = f"{b['speaker']} ({b['emotion']})" if b["kind"] == "DIALOGUE" else b["kind"]
                    print(f"{b['start']:7.2f}-{b['end']:7.2f} {who}: {b['text']}")
    print(f"episodes {len(files)}, runtime {total['seconds'] / 60:.1f} min")
    for k in ("DIALOGUE", "ACTION", "SOUND FX", "CAMERA"):
        print(f"  {k:9} {total[k]:5} blocks, {total['chars_' + k]:7} chars (~{total['chars_' + k] // 4} tokens)")
    no_cam = sum(total["chars_" + k] for k in ("DIALOGUE", "ACTION", "SOUND FX"))
    print(f"  text without CAMERA: ~{no_cam // 4} tokens; with CAMERA: ~{(no_cam + total['chars_CAMERA']) // 4} tokens")
    print("speaker labels:", ", ".join(f"{s} {n}" for s, n in speakers.most_common(25)))
    print("format problems:", problems or "none")


if __name__ == "__main__":
    main()
