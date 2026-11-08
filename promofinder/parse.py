"""Parse a microdrama screenplay pack into episodes and numbered moments.

A pack is a folder or a zip of text files, one per episode. Each timed block looks like
    **HEAD:**
    Start: (mm:ss.ss) | End: (mm:ss.ss)
    text
where HEAD is CAMERA, ACTION, SOUND FX, or a speaker with emotion tags, e.g. "LADY MARGARET (Cruel, Savoring)".

A *moment* is every block except CAMERA (camera lines describe framing, cost two thirds of the
tokens and carry little story). Moments are numbered per episode in time order, so a moment id
like E12.M031 is stable and the language model can refer to it without ever writing a timestamp.
"""
from __future__ import annotations

import io
import os
import re
import zipfile
from dataclasses import dataclass, field

TIME = r"(?:\d+:)?\d{1,2}:\d{1,2}(?:\.\d+)?"
BLOCK = re.compile(
    r"\*\*(?P<head>[^*\n]+?)\s*:?\s*\*\*\s*:?\s*\n"
    r"\s*[Ss]tart\s*:\s*\(?\s*(?P<s>" + TIME + r")\s*\)?\s*\|\s*[Ee]nd\s*:\s*\(?\s*(?P<e>" + TIME + r")\s*\)?[^\n]*\n"
    r"(?P<body>.*?)(?=\n\s*\*\*|\n-{2,}\s*SCRIPT END|\n\s*SCENE\b|\Z)",
    re.S,
)
# Looser templates, tried only when BLOCK finds fewer blocks than the file has time lines, so the
# pack's own format always takes the original path. They cover a new show written slightly differently:
# (W = spaces or tabs only, so no part of a pattern can run on into the next line)
W = r"[ \t]*"
SEP = W + r"(?:\||-|–|—|to|,)" + W
ALT_BLOCK = re.compile(  # head line without bold, or "Start Time 00:01 - End 00:03" style time lines
    r"^" + W + r"(?:\*\*)?(?P<head>[^*\n]+?)" + W + r":?" + W + r"(?:\*\*)?" + W + r":?" + W + r"\n"
    + W + r"(?:[Ss]tart(?:" + W + r"[Tt]ime)?" + W + r"[:\-]?" + W + r")?\(?" + W + r"(?P<s>" + TIME + r")" + W + r"\)?" + SEP +
    r"(?:[Ee]nd(?:" + W + r"[Tt]ime)?" + W + r"[:\-]?" + W + r")?\(?" + W + r"(?P<e>" + TIME + r")" + W + r"\)?[^\n]*\n"
    r"(?P<body>.*?)(?=\n[^\n]*\n" + W + r"(?:[Ss]tart[^\n]*?)?\(?" + W + TIME + r"|\n\s*\*\*|\n-{2,}\s*SCRIPT END|\n\s*SCENE\b|\Z)",
    re.S | re.M,
)
INLINE_BLOCK = re.compile(  # times on the head line: "**MAID (Angry)** [00:03.0 - 00:05.5]"
    r"^" + W + r"(?:\*\*)?(?P<head>[^*\n\[]+?)" + W + r"(?:\*\*)?" + W + r":?" + W + r"[\[(]" + W
    + r"(?:[Ss]tart" + W + r":?" + W + r")?(?P<s>" + TIME + r")" + SEP +
    r"(?:[Ee]nd" + W + r":?" + W + r")?(?P<e>" + TIME + r")" + W + r"[\])]" + W + r"(?:\*\*)?" + W + r":?" + W + r"\n"
    r"(?P<body>.*?)(?=\n" + W + r"(?:\*\*)?[^\n\[]*[\[(]" + W + r"(?:[Ss]tart" + W + r":?" + W + r")?" + TIME
    + r"|\n-{2,}\s*SCRIPT END|\n\s*SCENE\b|\Z)",
    re.S | re.M,
)
KIND_ALIASES = {
    "CAMERA": "CAMERA", "SHOT": "CAMERA",
    "ACTION": "ACTION", "ACTIONS": "ACTION",
    "SOUND FX": "SOUND FX", "SOUND": "SOUND FX", "SFX": "SOUND FX", "SOUND EFFECT": "SOUND FX",
    "SOUND EFFECTS": "SOUND FX", "MUSIC": "SOUND FX",
}
# Parentheticals that are delivery notes rather than emotions: (O.S.), (V.O.), (CONT'D)
TECH = {"OS", "VO", "OC", "CONTD", "CONT", "OFF", "OFFSCREEN", "ONSCREEN", "INTO PHONE", "ON PHONE"}


def secs(t: str) -> float:
    total = 0.0
    for part in t.split(":"):
        total = total * 60 + float(part)
    return total


def mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


@dataclass
class Moment:
    ep: int
    idx: int  # 1-based position in the episode
    kind: str  # DIALOGUE, ACTION, SOUND FX
    speaker: str  # label as written in the script ("" for non-dialogue)
    emotion: str
    start: float
    end: float
    text: str
    name: str = ""  # normalised character name, filled by names.py

    @property
    def id(self) -> str:
        return f"E{self.ep:02d}.M{self.idx:03d}"

    def who(self, use_names: bool = True) -> str:
        if self.kind != "DIALOGUE":
            return self.kind
        n = (self.name or self.speaker) if use_names else self.speaker
        return f"{n} ({self.emotion})" if self.emotion else n


@dataclass
class Episode:
    ep: int
    title: str
    logline: str
    notes: str
    moments: list[Moment]
    camera: list[tuple[float, float, str]]
    runtime: float
    source: str
    problems: list[str] = field(default_factory=list)

    def lines(self) -> list[Moment]:
        return [m for m in self.moments if m.kind == "DIALOGUE"]


@dataclass
class Show:
    title: str
    episodes: list[Episode]

    def episode(self, ep: int) -> Episode:
        return self.by_ep[ep]

    @property
    def by_ep(self) -> dict[int, Episode]:
        return {e.ep: e for e in self.episodes}

    def moment(self, mid: str) -> Moment:
        ep, idx = mid[1:].split(".M")
        return self.by_ep[int(ep)].moments[int(idx) - 1]

    @property
    def problems(self) -> list[str]:
        return [p for e in self.episodes for p in e.problems]


def split_head(head: str) -> tuple[str, str, str]:
    """Return (kind, speaker, emotion) for a block header."""
    h = " ".join(head.split())
    key = re.sub(r"\(.*?\)", "", h).strip().upper()
    if key in KIND_ALIASES:
        return KIND_ALIASES[key], "", ""
    parens = re.findall(r"\(([^()]*)\)", h)
    name = re.sub(r"\([^()]*\)", "", h).strip(" :-")
    emo = [p.strip() for p in parens if re.sub(r"[^A-Z ]", "", p.upper()).strip() not in TECH]
    return "DIALOGUE", name.upper(), ", ".join(e for e in emo if e)


def parse_episode_text(raw: str, source: str = "", fallback_ep: int = 0) -> Episode:
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    m = re.search(r"EPISODE NUMBER\s*:\s*(?:Episode\s*)?(\d+)", raw, re.I)
    if m:
        ep = int(m.group(1))
    else:
        m = re.search(r"episode[-_ ]?(\d+)", os.path.basename(source), re.I) or re.search(r"(\d+)", os.path.basename(source))
        ep = int(m.group(1)) if m else fallback_ep
    logline = re.search(r"EPISODE LOGLINE\s*:\s*(.*)", raw, re.I)
    notes = re.search(r"CONTINUITY NOTES\s*:\s*(.*)", raw, re.I)
    def read(pattern):
        blocks, camera, problems = [], [], []
        for b in pattern.finditer(raw):
            kind, speaker, emotion = split_head(b.group("head"))
            start, end = secs(b.group("s")), secs(b.group("e"))
            text = " ".join(b.group("body").split())
            if end < start:
                problems.append(f"E{ep}: block at {mmss(start)} ends before it starts")
                start, end = end, start
            if kind == "CAMERA":
                camera.append((start, end, text))
            else:
                blocks.append((start, len(blocks), kind, speaker, emotion, end, text))
        return blocks, camera, problems

    blocks, camera, problems = read(BLOCK)
    expected = len(re.findall(r"^\s*[Ss]tart\s*:\s*\(?\s*" + TIME, raw, re.M))
    if not blocks or expected != len(blocks) + len(camera):
        # the file does not follow the pack's template exactly: try the looser ones and keep the best
        expected = max(expected, len(re.findall(r"^[^\n]*" + TIME + r"[^\n]*" + TIME, raw, re.M)))
        best = max((read(p) for p in (ALT_BLOCK, INLINE_BLOCK)), key=lambda r: len(r[0]) + len(r[1]))
        if len(best[0]) + len(best[1]) > len(blocks) + len(camera):
            blocks, camera, problems = best
            problems.append(f"E{ep}: read with a looser template than the pack's; check the parsed lines")
    if expected != len(blocks) + len(camera):
        problems.append(f"E{ep}: {expected} timed blocks in the file but {len(blocks) + len(camera)} parsed")
    if not blocks:
        problems.append(f"E{ep}: no dialogue or action blocks found")
    blocks.sort(key=lambda x: (x[0], x[1]))
    moments = [Moment(ep, i + 1, k, sp, em, s, e, t) for i, (s, _, k, sp, em, e, t) in enumerate(blocks)]
    runtime = max([x.end for x in moments] + [c[1] for c in camera] + [0.0])
    return Episode(ep, f"Episode {ep}", logline.group(1).strip() if logline else "",
                   notes.group(1).strip() if notes else "", moments, camera, runtime, source, problems)


def _read_files(path_or_bytes) -> list[tuple[str, str]]:
    """Return (name, text) for every .txt episode file in a folder, a zip path, or zip bytes."""
    out = []
    if isinstance(path_or_bytes, (bytes, bytearray)) or (isinstance(path_or_bytes, str) and path_or_bytes.lower().endswith(".zip")):
        data = path_or_bytes if isinstance(path_or_bytes, (bytes, bytearray)) else open(path_or_bytes, "rb").read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for n in z.namelist():
                if n.lower().endswith(".txt") and not os.path.basename(n).startswith("."):
                    out.append((n, z.read(n).decode("utf-8", errors="replace")))
    else:
        for root, _, files in os.walk(path_or_bytes):
            for f in files:
                if f.lower().endswith(".txt"):
                    p = os.path.join(root, f)
                    out.append((p, open(p, encoding="utf-8", errors="replace").read()))
    return out


def load_show(path_or_bytes, title: str | None = None) -> Show:
    files = _read_files(path_or_bytes)
    files = [(n, t) for n, t in files if re.search(TIME + SEP + r"(?:[Ee]nd\s*:?\s*)?\(?\s*" + TIME, t)
             or "Start:" in t or "start:" in t]
    files.sort(key=lambda nt: nt[0])
    eps = [parse_episode_text(t, n, i + 1) for i, (n, t) in enumerate(files)]
    eps.sort(key=lambda e: e.ep)
    seen = set()
    for e in eps:
        if e.ep in seen:
            e.problems.append(f"episode number {e.ep} appears twice")
        seen.add(e.ep)
    if title is None:
        parent = os.path.basename(os.path.dirname(files[0][0].replace("\\", "/"))) if files else ""
        if parent:
            title = parent
        elif isinstance(path_or_bytes, str):
            title = os.path.splitext(os.path.basename(os.path.normpath(path_or_bytes)))[0]
        title = re.sub(r"[-_ ]script$", "", title or "show", flags=re.I)
    return Show(title or "show", eps)
