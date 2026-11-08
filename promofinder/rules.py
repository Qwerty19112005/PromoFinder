"""Rules-only moment scores (emotion tags, punctuation, keywords, actions, sounds), no model.
Used for the baseline and as the fallback when no API key is set."""
from __future__ import annotations

import re

from .parse import Moment

AROUSAL = r"furious|enraged|livid|rage|screaming|shrieking|shouting|yelling|shocked|stunned|horrified|terrified|" \
          r"panicked|desperate|sobbing|crying|weeping|gasping|devastated|outraged|venomous|menacing|threatening|" \
          r"cruel|gloating|smug|mocking|vicious|hysterical|frantic|seething|defiant|commanding|explosive|accusing|" \
          r"heartbroken|betrayed|broken|pleading|begging|disgusted|contemptuous|icy|cold|triumphant|savage"
STAKES = r"slap|strike|hit|kill|killed|die|dying|dead|death|blood|poison|knife|sword|dagger|gun|arrest|guards?|" \
         r"prison|hang|divorce|pregnant|baby|child|marry|married|wedding|duke|duchess|secret|liar|lie|betray|" \
         r"never|dare|disgrace|ruin|destroy|traitor|murder|thief|heir|inherit|throne|burn|fire"
ACTIONS = r"slaps?|strikes?|punch(?:es)?|grabs?|shoves?|pushes|collapses?|kneels?|kiss(?:es)?|faints?|bleeds?|" \
          r"draws?|seizes?|throws?|hurls?|chokes?|stabs?|falls?|drags?|tears?|rips?|smashes|shatters?|storms?|" \
          r"bursts?|lunges?|pins?|restrains?|reveals?|unmasks?|removes? (?:his|her) (?:hood|mask|veil)"
SFX = r"slap|crack|thud|gasp|scream|crash|shatter|sting|cliffhanger|drone|swell|ominous|gunshot|thunder|boom|" \
      r"heartbeat|silence|freeze"
CLIFF_SFX = r"cliffhanger|ominous|swell|freeze|freezes|sting|suspense|cuts? to black"


def _count(pattern: str, text: str) -> int:
    return len(re.findall(r"\b(?:" + pattern + r")\b", text, re.I))


def heat(m: Moment) -> float:
    """General intensity of a moment, 0 to 3."""
    s = 0.0
    s += min(1.5, 0.75 * _count(AROUSAL, m.emotion))
    s += min(1.0, 0.5 * _count(STAKES, m.text))
    if m.kind == "DIALOGUE":
        s += min(0.75, 0.375 * m.text.count("!"))
    if m.kind == "ACTION":
        s += min(1.5, 0.75 * _count(ACTIONS, m.text))
    if m.kind == "SOUND FX":
        s += min(1.0, 0.5 * _count(SFX, m.text))
    return min(3.0, s)


def hook(m: Moment) -> float:
    """How well this moment could open a promo, 0 to 3: intense, and short enough to land fast."""
    s = heat(m)
    if m.kind == "DIALOGUE" and (m.end - m.start) > 6:
        s -= 0.5
    return max(0.0, min(3.0, s))


def cliff(m: Moment, is_episode_end: bool) -> float:
    """How open the story is if the clip cuts right after this moment, 0 to 3."""
    s = 0.0
    t = m.text.strip().strip('"').strip()
    if m.kind == "DIALOGUE":
        if t.endswith("?"):
            s += 1.0
        if t.endswith("...") or t.endswith("--") or t.endswith("\u2014"):
            s += 1.0
        s += 0.5 * min(2, _count(STAKES, t))
    if m.kind == "SOUND FX" and _count(CLIFF_SFX, m.text):
        s += 1.5
    if m.kind == "ACTION" and re.search(r"freez|frozen|door (?:opens|swings)|reveal|silhouette|shadow", m.text, re.I):
        s += 1.0
    if is_episode_end:
        s += 1.5  # every episode in these scripts is written to end on a cliffhanger
    return min(3.0, s)
