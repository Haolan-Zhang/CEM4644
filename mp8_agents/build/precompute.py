"""Instructor-side: runs every default setting of the notebook once with a Gemini key, so the stored outputs in
data/cache/ let the class run those settings without a key and without the free tier's rate limits.

    GEMINI_API_KEY=... python build/precompute.py

Parts 1 (one prompt and the workflow, 5 drawings), 2 (all 8 combinations of the three tools, 6 questions) and 3
(with and without the skill). Outputs that are already stored are not requested again. Rerun after changing a default
prompt, tool description or the skill in aec_agent/texts.py (the old outputs stay in the cache under their own keys).
"""
import itertools
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import IPython.display as D  # noqa: E402


def _quiet(*objs, **kw):
    class _H:
        def update(self, *a, **k):
            pass
    return _H() if kw.get("display_id") else None


D.display = _quiet
from aec_agent import lab, ui  # noqa: E402

ui.display = _quiet

if __name__ == "__main__":
    lab.setup(os.environ.get("GEMINI_API_KEY", ""))
    lab.one_prompt(); lab.workflow()
    for combo in itertools.product([False, True], repeat=3):
        lab.agent_tools(*combo)
    lab.agent_skill(False); lab.agent_skill(True)
    print("cache:", len(list((REPO / "data" / "cache").glob("*.json"))), "outputs")
