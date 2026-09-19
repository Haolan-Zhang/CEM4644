"""Instructor-side: run SAM 3 once on every workshop drawing x phrase (main wording and every alternative) and
store the masks under data/masks, so the notebook is instant and works without a GPU.

    python build/precompute_masks.py                 # every set that declares a mask folder (= the workshop)
    python build/precompute_masks.py --sets workshop --threshold 0.1 --redo

Masks are stored with a LOW score threshold; the notebook applies the student's threshold when reading them.
The homework needs no precomputed masks: it only draws boxes.
"""
import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from aec_seg.config import SETS              # noqa: E402
from aec_seg.data import MaskStore, Sheets   # noqa: E402
from aec_seg.engine import Sam3Engine        # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="*", default=[k for k, s in SETS.items() if s.masks])
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--only", nargs="*", default=None, help="sheet ids")
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()
    eng = Sam3Engine()
    for key in a.sets:
        spec = SETS[key]
        if not spec.masks:
            print(f"== {key}: no mask folder in the set config, skipped"); continue
        sheets = Sheets(REPO / spec.folder, key)
        store = MaskStore(REPO / spec.masks)
        t0 = time.time()
        prompts = [t.prompt for t in spec.things] + [alt for t in spec.things for alt in t.alternatives]
        for sh in sheets:
            if a.only and sh.id not in a.only:
                continue
            img = sh.load()
            line = []
            for prompt in prompts:
                if store.has(sh.id, prompt) and not a.redo:
                    continue
                res = eng.segment(img, prompt, threshold=a.threshold)
                store.save(sh.id, res)
                line.append(f"{prompt}:{int((res.scores >= 0.3).sum())}")
            print(f"{sh.id}: " + "  ".join(line), flush=True)
        print(f"== {key}: {len(prompts)} phrases x {len(sheets)} drawings in {time.time() - t0:.0f} s", flush=True)
