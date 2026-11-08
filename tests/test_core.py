"""Tests that need no API key and no confidential script: a tiny synthetic show in the same format."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from promofinder import candidates as C  # noqa: E402
from promofinder.llm import strict_schema  # noqa: E402
from promofinder.parse import load_show, parse_episode_text, split_head  # noqa: E402
from promofinder.score import Pass1Reply  # noqa: E402


def block(head, s, e, text):
    return f"**{head}:**\nStart: ({s}) | End: ({e})\n{text}\n\n"


def make_episode(n):
    out = f"EPISODE NUMBER: Episode {n}\nEPISODE LOGLINE: Test logline {n}.\nCONTINUITY NOTES: none\n\n--- SCRIPT START ---\n\nSCENE 01\n"
    t = 0.0
    for i in range(30):
        s, e = t, t + 2.5
        out += block("CAMERA", f"00:{s:05.2f}", f"00:{e:05.2f}", "Wide shot.") if i % 5 == 0 else ""
        head = "ACTION" if i % 4 == 0 else f"HERO (Furious)" if i % 2 else "VILLAIN (O.S.) (Cold)"
        mm, ss = divmod(s, 60)
        me, se = divmod(e, 60)
        out += block(head, f"{int(mm):02d}:{ss:05.2f}", f"{int(me):02d}:{se:05.2f}", f'"Line {i}?"' if head != "ACTION" else "He slaps the table.")
        t = e + 0.5
    return out + "--- SCRIPT END ---\n"


def test_split_head():
    assert split_head("MATRIARCH (O.S.) (Firm)") == ("DIALOGUE", "MATRIARCH", "Firm")
    assert split_head("LADY MARGARET (Cruel, Savoring)") == ("DIALOGUE", "LADY MARGARET", "Cruel, Savoring")
    assert split_head("SOUND FX")[0] == "SOUND FX"
    assert split_head("Camera")[0] == "CAMERA"


def test_parse_counts(tmp_path):
    for n in (1, 2):
        (tmp_path / f"episode-00{n}.txt").write_text(make_episode(n), encoding="utf-8")
    show = load_show(str(tmp_path))
    assert [e.ep for e in show.episodes] == [1, 2]
    assert len(show.episodes[0].moments) == 30
    assert show.problems == []
    m = show.episodes[0].moments[2]  # i=2 in make_episode: the VILLAIN (O.S.) (Cold) line
    assert m.id == "E01.M003" and m.speaker == "VILLAIN" and m.emotion == "Cold"


def test_candidates_obey_rules(tmp_path):
    for n in (1, 2):
        (tmp_path / f"episode-00{n}.txt").write_text(make_episode(n), encoding="utf-8")
    show = load_show(str(tmp_path))
    cands = C.generate(show, lambda m: 3 if m.idx % 7 == 0 else 0, lambda m: 3 if m.idx % 5 == 0 else 0)
    assert cands, "expected some candidates"
    for c in cands:
        assert C.check(show, c) == [], (c.label(), C.check(show, c))
        assert 20 <= c.dur <= 60


def test_check_catches_mid_line_cut():
    ep = parse_episode_text(make_episode(1), "episode-001.txt")
    from promofinder.parse import Show
    show = Show("t", [ep])
    line = ep.lines()[3]
    bad = C.Candidate("single", [C.Segment(1, (line.start + line.end) / 2, (line.start + line.end) / 2 + 30)], "", "")
    assert any("cuts a line" in p for p in C.check(show, bad))


def test_strict_schema_closed_and_required():
    sch = strict_schema(Pass1Reply)

    def walk(n):
        if isinstance(n, dict):
            if n.get("type") == "object" and "properties" in n:
                assert n["additionalProperties"] is False
                assert set(n["required"]) == set(n["properties"])
            assert "$ref" not in n
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)
    walk(sch)
