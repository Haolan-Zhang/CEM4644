"""Instructor-side: assemble the tricky galleries under ../data/tricky/<dataset>/ for the detection lab."""
import argparse, json, shutil, sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from aec_det.config import DETSETS          # noqa: E402
from aec_det.data import DetSet             # noqa: E402
from aec_det.evaluate import evaluate_set   # noqa: E402
from aec_det.models import Detector         # noqa: E402
from aec_det.tricky import perturb          # noqa: E402
from aec_det import ui                      # noqa: E402


def out_of_scope_images():
    from skimage import data
    ims = [("person", "an astronaut (a person, with a helmet of a very different kind)", Image.fromarray(data.astronaut())),
           ("cat", "a cat", Image.fromarray(data.chelsea())),
           ("coffee", "a cup of coffee", Image.fromarray(data.coffee()))]
    plan = Image.new("RGB", (640, 640), "white")
    d = ImageDraw.Draw(plan)
    d.rectangle([50, 50, 590, 590], outline="black", width=6); d.line([320, 50, 320, 590], fill="black", width=4); d.line([50, 350, 590, 350], fill="black", width=4)
    d.text((100, 170), "ROOM 101", fill="black"); d.text((380, 450), "ROOM 102", fill="black")
    ims.append(("plan", "a floor-plan drawing", plan))
    ims.append(("grey", "a blank grey image", Image.new("RGB", (640, 640), (128, 128, 128))))
    return ims


class Gallery:
    def __init__(self, out: Path):
        self.out = out
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        self.entries = []

    def add(self, img: Image.Image, caption: str, group: str, expected: str, source: str):
        i = len(self.entries) + 1
        fn = f"{i:02d}_{group}.jpg"
        img = img.convert("RGB")
        if max(img.size) > 640:
            s = 640 / max(img.size)
            img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
        img.save(self.out / fn, "JPEG", quality=85)
        self.entries.append({"id": i, "file": fn, "caption": caption, "group": group, "expected": expected, "source": source})

    def save(self):
        (self.out / "tricky.json").write_text(json.dumps(self.entries, indent=2))
        print(f"  {len(self.entries)} tricky photos -> {self.out}")


def build(key: str, work: Path):
    spec = DETSETS[key]
    print("==", key)
    test = DetSet.from_folder(work / key / "test", spec.classes, spec.display)
    det = Detector(REPO / spec.course_model, name="course", classes=test.pretty_classes)
    res = evaluate_set(det, test, progress=False)
    errs = res.image_errors(0.5)
    g = Gallery(REPO / spec.tricky_dir)
    truth = lambda it: ui.gt_summary(it, test.pretty_classes)  # noqa: E731
    # (a) real photos with most misses / most false alarms
    for i, n_fn, n_fp, n_tp, _, _ in sorted(errs, key=lambda r: -r[1])[:3]:
        g.add(test[i].load(), f"a real test photo where the model misses {n_fn} object(s)", "hard_real", truth(test[i]), test[i].path.name)
    for i, n_fn, n_fp, n_tp, _, _ in sorted(errs, key=lambda r: -r[2])[:2]:
        g.add(test[i].load(), f"a real test photo with {n_fp} false alarm(s)", "hard_real", truth(test[i]), test[i].path.name)
    # (b) crowded: most labelled objects
    crowded = sorted(range(len(test)), key=lambda i: -len(test[i].cls))[:2]
    for i in crowded:
        g.add(test[i].load(), f"a crowded photo with {len(test[i].cls)} labelled objects", "crowded", truth(test[i]), test[i].path.name)
    # (c) synthetic edits of a photo the model gets fully right
    clean = [i for i, n_fn, n_fp, n_tp, _, _ in errs if n_fn == 0 and n_fp == 0 and 2 <= n_tp <= 6]
    base_i = clean[0] if clean else 0
    im, tr = test[base_i].load(), truth(test[base_i])
    g.add(perturb(im, shrink=4), "the same photo 'moved away' (everything 4x smaller)", "synthetic", tr, test[base_i].path.name)
    g.add(perturb(im, brightness=0.25), "the same photo, very dark", "synthetic", tr, test[base_i].path.name)
    g.add(perturb(im, blur=5), "the same photo, blurred (shaky camera)", "synthetic", tr, test[base_i].path.name)
    g.add(perturb(im, rotate=90), "the same photo rotated 90°", "synthetic", tr, test[base_i].path.name)
    g.add(perturb(im, shadow=0.9), "the same photo with a strong shadow", "synthetic", tr, test[base_i].path.name)
    g.add(perturb(im, grayscale=True), "the same photo in black & white", "synthetic", tr, test[base_i].path.name)
    g.add(im, "the original photo (the model gets this one right)", "synthetic", tr, test[base_i].path.name)
    # (d) other domain
    if spec.other_domain and (work / spec.other_domain / "test").exists():
        o = DETSETS[spec.other_domain]
        otest = DetSet.from_folder(work / o.key / "test", o.classes, o.display)
        rng = np.random.default_rng(5)
        for i in rng.choice(len(otest), size=3, replace=False):
            it = otest[int(i)]
            g.add(it.load(), f"a photo from another dataset ({o.title.split(':')[0]})", "other_domain",
                  f"(other dataset's labels) {ui.gt_summary(it, otest.pretty_classes)}", f"{o.key}:{it.path.name}")
    # (e) out of scope
    for k, cap, im in out_of_scope_images():
        g.add(im, cap, "out_of_scope", "", f"skimage:{k}" if k in ("person", "cat", "coffee") else "synthetic")
    g.save()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--only", nargs="*", default=["construction_safety", "excavators"])
    a = ap.parse_args()
    for key in a.only:
        build(key, Path(a.work))
    print("done")
