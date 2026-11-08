"""Promo finder web app.  Run locally:  .venv\\Scripts\\streamlit run app/streamlit_app.py

Upload a show (a .zip of episode .txt files, or the .txt files themselves), choose models, and get ranked
20-60 s promos with timestamps, hook, cliffhanger, why each works, and automatic quality flags.
An editor then accepts or rejects each promo in the table, and downloads those decisions.
No script is stored in this app; uploads are processed in a temporary folder.
API keys come from Streamlit secrets or the sidebar, never from the code. A key kept in secrets is
never put into a page field, so it cannot be revealed from the browser.
"""
import io
import json
import os
import sys
import tempfile
import zipfile

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from promofinder.pipeline import run  # noqa: E402

st.set_page_config(page_title="Promo finder", layout="wide")


def secret(name, default=""):
    try:
        return st.secrets.get(name, os.environ.get(name, default))
    except Exception:
        return os.environ.get(name, default)


st.title("Promo finder for microdrama scripts")
st.caption("Finds 20-60 second promos that open on a hook in the first 3 seconds, end on a cliffhanger and "
           "make sense to a stranger. Code enforces timing; models score moments, judge clips and rank them; "
           "an editor makes the final call in the table.")
st.info("A full 50-episode show takes about 20 minutes on Groq's free tier (8,000 tokens a minute per model). "
        "Tick **Quick demo** in the sidebar for a run of a few minutes on episodes 1-10. To use another model, "
        "paste its OpenAI-compatible base URL, key and model names in the sidebar. Uploaded scripts are only "
        "kept for this session and are sent to the model provider you set.")

with st.sidebar:
    st.header("Models")
    st.caption("Any OpenAI-compatible endpoint. Defaults: Groq free tier.")
    base = st.text_input("Base URL", secret("LLM_BASE_URL", "https://api.groq.com/openai/v1"))
    app_key = secret("LLM_API_KEY")
    typed_key = st.text_input("API key", "", type="password",
                              placeholder="leave empty to use the app's own key" if app_key else "paste a key",
                              help="Leave empty to use the app's key (if it has one); with no key at all, "
                                   "only the rules-only baseline runs.")
    key = typed_key.strip() or app_key
    score_model = st.text_input("Pass 1 model (cheap, reads the whole show)", secret("LLM_MODEL", "openai/gpt-oss-20b"))
    judge_model = st.text_input("Judge and name-map model (stronger)", secret("JUDGE_MODEL", "openai/gpt-oss-120b"))
    rank_model = st.text_input("Ranking model (empty = judge model)", secret("RANK_MODEL", ""))
    effort = st.selectbox("Reasoning effort", ["low", "medium", "high", "(not sent)"], index=0,
                          help="gpt-oss and Gemini 3.8 accept low/medium/high; choose (not sent) for models without it.")
    st.header("Run size")
    quick = st.checkbox("Quick demo (episodes 1-10, 12 clips judged, 4 ranked)", value=False)
    eps_text = st.text_input("Episodes to scan", "1-10" if quick else "", disabled=quick,
                             placeholder="all, or a range such as 5-20")
    judge_n = st.slider("Clips sent to the judge", 6, 40, 30, disabled=quick)
    rank_n = st.slider("Clips in the head-to-head ranking", 3, 10, 8, help="n x (n-1) model calls", disabled=quick)
    final_n = st.slider("Promos in the final list", 5, 30, 20, disabled=quick)
    rules_only = st.checkbox("Rules only (no model calls)", value=not key)
    if quick:
        eps_text, judge_n, rank_n, final_n = "1-10", 12, 4, 10

episodes = None
if eps_text.strip():
    try:
        lo, hi = (int(x) for x in eps_text.replace(" ", "").split("-"))
        episodes = (lo, hi)
    except ValueError:
        st.sidebar.error("Episodes must look like 5-20.")

files = st.file_uploader("Episode scripts: one .zip, or several .txt files", type=["zip", "txt"],
                         accept_multiple_files=True)

if st.button("Find promos", type="primary", disabled=not files):
    tmp = tempfile.mkdtemp(prefix="promo_")
    src = os.path.join(tmp, "show")
    os.makedirs(src, exist_ok=True)
    for f in files:
        data = f.getvalue()
        if f.name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for n in z.namelist():
                    if n.lower().endswith(".txt") and not os.path.basename(n).startswith("."):
                        sub = os.path.join(src, os.path.basename(os.path.dirname(n)) or "show")
                        os.makedirs(sub, exist_ok=True)
                        open(os.path.join(sub, os.path.basename(n)), "wb").write(z.read(n))
        else:
            open(os.path.join(src, f.name), "wb").write(data)
    env = {"LLM_BASE_URL": base, "LLM_API_KEY": key, "LLM_MODEL": score_model, "JUDGE_MODEL": judge_model,
           "RANK_MODEL": rank_model}
    if effort != "(not sent)":
        env["LLM_REASONING_EFFORT"] = effort
    with st.status("Running...", expanded=True) as status:
        try:
            summary = run(src, out_root=os.path.join(tmp, "runs"), cache_dir=os.path.join(tempfile.gettempdir(), "promo_cache"),
                          env=env, progress=status.write, judge_n=judge_n, rank_n=rank_n, final_n=final_n,
                          use_llm=not rules_only, episodes=episodes)
            status.update(label=f"Done in {summary['seconds']['total']:.0f} s", state="complete")
            st.session_state["run_dir"] = summary["run_dir"]
        except Exception as exc:
            status.update(label="Failed", state="error")
            st.exception(exc)

run_dir = st.session_state.get("run_dir")
if run_dir and os.path.exists(os.path.join(run_dir, "promos.json")):
    promos = json.load(open(os.path.join(run_dir, "promos.json"), encoding="utf-8"))
    base_promos = json.load(open(os.path.join(run_dir, "baseline.json"), encoding="utf-8"))
    summary = json.load(open(os.path.join(run_dir, "summary.json"), encoding="utf-8"))
    st.subheader(f"{summary['show']}: {summary['episodes']} episodes, {summary['minutes']} min")
    if summary.get("format_problems"):
        st.warning("Format problems: " + "; ".join(summary["format_problems"]))

    def table(recs):
        return pd.DataFrame([{"#": r["rank"], "Segments": r["segments_text"], "Length (s)": r["duration_s"],
                              "Hook": r["hook"]["line"], "Cliffhanger": r["cliffhanger"]["line"],
                              "Why it works": r["why_it_works"], "Flags": len(r["flags"])} for r in recs])

    tab1, tab2, tab3, tab4 = st.tabs(["Ranked promos", "Each promo", "Quality checks", "Rules-only baseline"])
    with tab1:
        st.caption("Editor's call: set Decision for each promo (and a note if useful), then download the decisions. "
                   "The share accepted is the simplest measure of whether the tool saves an editor's time.")
        df = table(promos)
        df.insert(1, "Decision", "")
        df.insert(2, "Editor note", "")
        edited = st.data_editor(
            df, hide_index=True, width="stretch", key=f"decisions_{run_dir}",
            column_config={"Decision": st.column_config.SelectboxColumn("Decision", options=["", "accept", "maybe", "reject"]),
                           "Editor note": st.column_config.TextColumn("Editor note")},
            disabled=[c for c in df.columns if c not in ("Decision", "Editor note")])
        decided = edited[edited["Decision"] != ""]
        n_acc = int((edited["Decision"] == "accept").sum())
        st.write(f"Decided {len(decided)} of {len(edited)}; accepted {n_acc}"
                 + (f" ({100 * n_acc / len(decided):.0f}% of those decided)." if len(decided) else "."))
        st.download_button("Download editor decisions (csv)", edited.to_csv(index=False).encode("utf-8"),
                           "editor_decisions.csv", disabled=len(decided) == 0)
    with tab2:
        for r in promos:
            with st.expander(f"#{r['rank']}  {r['segments_text']}  ({r['duration_s']} s, {r['shape']})"):
                st.markdown(f"**Hook:** {r['hook']['line']}  \n**Cliffhanger:** {r['cliffhanger']['line']}  \n"
                            f"**Why it works:** {r['why_it_works']}  \n**Why this rank:** {r['why_this_rank']}")
                if r["premise_as_seen_by_judge"]:
                    st.markdown(f"**What a stranger sees:** {r['premise_as_seen_by_judge']}  \n"
                                f"**Left open:** {r['open_question']}  \n**Weakness:** {r['weakness']}")
                if r["flags"]:
                    st.warning("; ".join(r["flags"]))
                st.code("\n".join(r["script"]), language=None)
    with tab3:
        report = open(os.path.join(run_dir, "report.md"), encoding="utf-8").read()
        qc = report.split("## Quality checks (automatic)")[1].split("## Each promo")[0] if "## Quality checks" in report else ""
        st.markdown(qc)
        st.json({k: summary[k] for k in ("pass1", "pairwise", "seconds", "usage") if k in summary}, expanded=False)
    with tab4:
        st.caption("Same candidate rules, scored by keyword and emotion-tag rules only, for comparison.")
        st.dataframe(table(base_promos), hide_index=True, width="stretch")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, fs in os.walk(run_dir):
            for f in fs:
                p = os.path.join(root, f)
                z.write(p, os.path.relpath(p, run_dir))
    c1, c2, c3, c4 = st.columns(4)
    c1.download_button("promos.csv", open(os.path.join(run_dir, "promos.csv"), "rb").read(), "promos.csv")
    c2.download_button("promos.json", open(os.path.join(run_dir, "promos.json"), "rb").read(), "promos.json")
    c3.download_button("report.md", open(os.path.join(run_dir, "report.md"), "rb").read(), "report.md")
    c4.download_button("Everything (zip, with raw model outputs)", buf.getvalue(), "promo_run.zip")
