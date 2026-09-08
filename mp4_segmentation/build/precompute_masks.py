"""Instructor-side: run SAM 3 once on every photo x material phrase (and on alternative phrasings for the
'phrase lab' photos) and store the masks under data/masks so the notebooks are instant even without a GPU.
    python build/precompute_masks.py --sets site interior --threshold 0.2
Masks are stored with a LOW score threshold (0.2); the notebook applies the student's threshold when reading them.
"""
import argparse, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from aec_seg.config import SETS            # noqa: E402
from aec_seg.data import MaskStore, PhotoSet  # noqa: E402
from aec_seg.engine import Sam3Engine       # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="*", default=["site", "interior"])
    ap.add_argument("--threshold", type=float, default=0.2)
    ap.add_argument("--only", nargs="*", default=None, help="photo ids")
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()
    eng = Sam3Engine()
    for key in a.sets:
        spec = SETS[key]
        photos = PhotoSet(REPO / spec.folder, key)
        store = MaskStore(REPO / spec.masks)
        t0 = time.time()
        for p in photos.photos:
            if a.only and p.id not in a.only:
                continue
            img = p.load()
            prompts = [(m.prompt, m.key) for m in spec.materials]
            if p.id in spec.phrase_lab_photos:
                prompts += [(alt, m.key) for m in spec.materials for alt in m.alternatives]
            line = []
            for prompt, mkey in prompts:
                if store.has(p.id, prompt) and not a.redo:
                    continue
                res = eng.segment(img, prompt, threshold=a.threshold)
                store.save(p.id, res)
                line.append(f"{prompt}:{len(res)}/{res.area_pct(0.5):.0f}%")
            print(f"{p.id}: " + "  ".join(line), flush=True)
        print(f"== {key}: done in {time.time() - t0:.0f} s", flush=True)
