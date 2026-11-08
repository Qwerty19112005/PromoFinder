# Promo finder for microdrama scripts

Reads a show's episode scripts and returns ranked 20 to 60 second promos: the episode and start and end time of every segment, the hook (first 3 seconds), the cliffhanger (the cut), one line on why it works, why it is ranked where it is, and automatic flags for where it may be picking badly.

Code does everything that must be exact (parsing, timestamps, length and cut rules, de-duplication). Models do what needs judgement (which moments grab, which leave a question open, whether a clip makes sense to a stranger, which of two promos to post first). The model never writes a timestamp: it refers to numbered moments and code turns numbers into times.

## Run it

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt       (Windows; use .venv/bin/pip elsewhere)
copy .env.example .env                              then put your API key in .env
.venv\Scripts\python -m promofinder.run path\to\show.zip
.venv\Scripts\streamlit run app\streamlit_app.py    (web app: upload a zip, see and download results)
```

`--quick` scans episodes 1-10 with a smaller judge and ranking round (a few minutes on a free tier; a full 50-episode show takes about 20 minutes there). `--episodes 5-20` scans a range. `--rules-only` runs without any model (the baseline). `--judge-n`, `--rank-n`, `--final-n` set the run size. The web app has the same settings in its sidebar, and its promo table has a Decision column where an editor accepts or rejects each promo and downloads the decisions. Each run writes `runs/<timestamp>/`: `promos.csv`, `promos.json`, `report.md`, the baseline, every candidate with its scores, the judge's verdicts, every pairwise decision, token use, and every raw model request and reply in `raw/`.

## Swapping models

Any OpenAI-compatible endpoint works (Groq, Gemini, OpenRouter, OpenAI, a local Ollama). Set `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` in `.env` (or in the app's sidebar); override one stage with `SCORE_*`, `JUDGE_*` or `RANK_*` (for example a stronger in-house model for the judge only). Run on Groq (gpt-oss-20b and gpt-oss-120b) and on Gemini 3.8 Flash. The client adapts to the provider: strict JSON schema, then plain JSON mode, then the schema in the prompt; it drops parameters a provider rejects (`temperature`, `reasoning_effort`) and switches to `max_completion_tokens` for models that need it; it paces itself from rate-limit headers or a fixed requests-per-minute cap. See `.env.example`.

## How it works

| Step | Who | What |
|---|---|---|
| Parse | code | Every dialogue, action and sound line becomes a numbered moment (E12.M031). Camera lines are dropped: two thirds of the text, little story. Format problems are reported, not hidden. |
| Name map | model, 1 call | Merges label drift (MARGARET / LADY MARGARET; a "mysterious man" who is named later). |
| Pass 1 | cheap model | Reads the show a few episodes per call. For notable moments: highlight tags with points (from rules ad editors use for short drama, see sources), hook 0-3, cliffhanger 0-3, context needed 0-3. Moment numbers that do not exist are dropped and counted. |
| Candidates | code | Windows that start at most 3 s before a hook, end on a cliffhanger moment, last 20 to 60 s and never cut inside a spoken line. Three shapes: one stretch; end of one episode into the next; a short flash-forward first, then the run-up. |
| Shortlist | code | Score, then drop near-duplicates (half the footage shared, or the same opening or closing moment) and cap clips per episode. |
| Judge | stronger model | A "cold viewer" sees only the clip, states its premise and the open question, and answers yes/no checks (hook in 3 s, premise clear, needs earlier episodes, ends unresolved, resolved before the cut, spoils a twist). Code turns the answers into a score with fixed weights. |
| Rank | model | All pairs of the top 8, each asked in both orders; a pick that survives the swap wins, an inconsistent pair is a tie. The share of inconsistent pairs is reported as position bias. |
| Baseline | code | Same candidate rules, scored by emotion tags, punctuation and keywords only. |

## How we know it is good

Automatic, every run (in `report.md`): rule pass rate, invalid moment references, the judge's check rates, pairwise consistency, overlap with the rules-only baseline, per-promo flags.

With a person: `tools/blind_check.py` is a blind viewer test. It mixes the tool's top promos with the rules-only baseline's and with random clips that obey the same rules, hides which is which, and asks the rater the brief's three questions (hook in 3 s, ends unresolved, makes sense to a stranger) plus an overall 1-5. It then reports the mean rating by source, how often a tool clip outrates each alternative, whether the tool's order matches the rater's, and how often the rater agrees with the model judge. `tools/agreement.py` has a second model judge the same clips. `tools/evaluate.py` and `tools/check_gold.py` compare a run with hand-picked promos, if you have them.

## Limits

It reads scripts, not footage: a hook that is only visual depends on the shoot, and timing is the script's. The judge and ranker are language models with their own taste and position bias (measured, not removed). Free-tier rate limits make a full 50-episode show take about 20 minutes; the code paces itself and caches every reply, so a rerun is free. Scripts in a slightly different template (no bold markers, other time separators, times on the speaker line) are read with a warning; anything else stops with a message saying what format was expected.

## Sources for the design

- HIVE: human-inspired editing of short-drama ad clips, with editor-derived highlight rules and separate opening and ending selection. Wang et al., arXiv 2507.02790.
- Pairwise ranking prompting, both orders, ties when inconsistent. Qin et al., arXiv 2306.17563.
- Checklist (yes/no) judging for consistency. CheckEval, Lee et al., arXiv 2403.18771.
- Suspense as uncertainty about what comes next. Wilmot and Keller, ACL 2020.
- Long inputs used poorly in the middle, so the show is read in small batches. Liu et al., arXiv 2307.03172.
- Trailer moments from screenplays with a person choosing the final cut. Papalampidi, Keller and Lapata, arXiv 2111.08774.
