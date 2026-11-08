"""Smoke tests for the Streamlit app (no model calls). The run-loaded test needs a local run folder."""
import glob
import os

import pytest
from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(__file__), "..", "app", "streamlit_app.py")
RUNS = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "runs", "2*", "promos.json")))


def test_key_from_secrets_never_reaches_the_page(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "gsk_SECRET_FOR_TEST")
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    key_box = [t for t in at.text_input if t.label == "API key"][0]
    assert key_box.value == ""  # the app's key is used behind the scenes, never shown in the field
    assert "SECRET_FOR_TEST" not in str([t.value for t in at.text_input])
    assert not at.checkbox[-1].value  # "Rules only" is off because the app has a key


def test_quick_demo_sets_episode_range():
    at = AppTest.from_file(APP, default_timeout=60).run()
    quick = [c for c in at.checkbox if c.label.startswith("Quick demo")][0]
    quick.check().run()
    assert not at.exception
    eps = [t for t in at.text_input if t.label == "Episodes to scan"][0]
    assert eps.value == "1-10" and eps.disabled


@pytest.mark.skipif(not RUNS, reason="no local run folder")
def test_run_loaded_shows_decision_table():
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["run_dir"] = os.path.dirname(RUNS[-1])
    at.run()
    assert not at.exception
    assert any("Decided 0 of" in m.value for m in at.markdown)
    assert len(at.tabs) == 4
