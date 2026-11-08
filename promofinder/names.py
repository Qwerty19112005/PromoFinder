"""Normalise speaker labels per show (MARGARET / LADY MARGARET, or a 'MYSTERIOUS MAN' who is named later).

One cheap model call per show sees every label with its line count, the episodes it appears in and its
first line, plus the episode loglines, and maps each label to one character name. If no model is
available, labels are kept as written (title case). The map is saved with the run so it can be checked.
"""
from __future__ import annotations

import collections

from pydantic import BaseModel

from .parse import Show

SYSTEM = """You normalise speaker labels in a screenplay. The same character is sometimes written under different labels: with or without a title (MARGARET, LADY MARGARET), or as a description such as MYSTERIOUS MAN before the name is revealed.
Using the episode loglines and the labels, map every label to one character name.
- Map a descriptive label (THE EX-FIANCE, WOMAN IN RED, MYSTERIOUS MAN) to a named character when the loglines make the identity clear, for example when the logline of the episode where the label appears names that person in that role. Otherwise keep the label itself. Never invent a name that is not in the loglines or labels.
- Generic labels such as LORD, MAID or GUARD that could be different people stay as they are.
- Use short names in title case: a first name, or a title plus surname (Margaret, Lady Whitmore).
- reason: at most 10 words. Include every label exactly once, spelled exactly as given."""


class LabelMap(BaseModel):
    label: str
    name: str
    reason: str


class NameMap(BaseModel):
    characters: list[LabelMap]


def label_table(show: Show) -> list[dict]:
    count, eps, first = collections.Counter(), collections.defaultdict(list), {}
    for e in show.episodes:
        for m in e.lines():
            count[m.speaker] += 1
            if e.ep not in eps[m.speaker]:
                eps[m.speaker].append(e.ep)
            first.setdefault(m.speaker, m.text[:70])
    rows = []
    for label, n in count.most_common():
        es = eps[label]
        rows.append({"label": label, "lines": n, "episodes": f"E{es[0]}-E{es[-1]}" if len(es) > 1 else f"E{es[0]}",
                     "first": first[label]})
    return rows


def build_messages(show: Show) -> list[dict]:
    logl = "\n".join(f"E{e.ep}: {e.logline}" for e in show.episodes if e.logline)
    labels = "\n".join(f"{r['label']} | {r['lines']} lines | {r['episodes']} | {r['first']}" for r in label_table(show))
    user = f"Show: {show.title}\n\nEpisode loglines:\n{logl}\n\nLabels (label | lines | episodes | first line):\n{labels}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def apply(show: Show, mapping: dict[str, str]) -> None:
    for e in show.episodes:
        for m in e.moments:
            if m.kind == "DIALOGUE":
                m.name = mapping.get(m.speaker) or m.speaker.title()


def normalise(show: Show, llm=None) -> dict:
    labels = [r["label"] for r in label_table(show)]
    mapping = {lab: lab.title() for lab in labels}
    source = "kept as written (no model)"
    if llm is not None:
        reply = llm.json(build_messages(show), NameMap, tag="names", max_tokens=4000)
        if reply is not None:
            got = {c.label.strip().upper(): c.name.strip() for c in reply.characters if c.name.strip()}
            mapping.update({lab: got[lab] for lab in labels if lab in got})
            source = f"model ({llm.cfg.model}); {sum(lab in got for lab in labels)} of {len(labels)} labels mapped"
    apply(show, mapping)
    return {"source": source, "map": mapping}


def main_characters(show: Show, k: int = 12) -> list[str]:
    c = collections.Counter(m.name or m.speaker for e in show.episodes for m in e.lines())
    return [n for n, _ in c.most_common(k)]
