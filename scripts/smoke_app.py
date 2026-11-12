"""Headless check that the UI renders (no uploads)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from streamlit.testing.v1 import AppTest

at = AppTest.from_file("app.py", default_timeout=120).run()
assert not at.exception, at.exception
print("title:", at.title[0].value, "| info:", at.info[0].value[:60], "| errors:", [e.value for e in at.error])
