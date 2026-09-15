"""Instructor-side: run SAM 3 once on every plan x thing phrase (and every alternative wording) and store the
masks under data/masks so the notebooks are instant even without a GPU.
    python build/precompute_masks.py --sets homes_a homes_b --threshold 0.1
Masks are stored with a LOW score threshold; the notebook applies the student's threshold when reading them.
"""
import argparse, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from aec_seg.config import SETS            # noqa: E402
from aec_seg.data import MaskStore, PlanSet  # noqa: E402
from aec_seg.engine import Sam3Engine       # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="*", default=list(SETS))
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--only", nargs="*", default=None, help="plan ids")
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()
    eng = Sam3Engine()
    for key in a.sets:
        spec = SETS[key]
        plans = PlanSet(REPO / spec.folder, key)
        store = MaskStore(REPO / spec.masks)
        t0 = time.time()
        for p in plans.plans:
            if a.only and p.id not in a.only:
                continue
            img = p.load()
            prompts = [t.prompt for t in spec.things] + [alt for t in spec.things for alt in t.alternatives]
            line = []
            for prompt in prompts:
                if store.has(p.id, prompt) and not a.redo:
                    continue
                res = eng.segment(img, prompt, threshold=a.threshold)
                store.save(p.id, res)
                line.append(f"{prompt}:{int((res.scores >= 0.4).sum())}")
            print(f"{p.id}: " + "  ".join(line), flush=True)
        print(f"== {key}: done in {time.time() - t0:.0f} s", flush=True)
