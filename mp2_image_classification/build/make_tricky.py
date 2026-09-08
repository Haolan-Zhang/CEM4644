"""Instructor-side: assemble the 'tricky photos' galleries under ../data/tricky/<dataset>/.

Each gallery mixes (a) real test photos the course model gets wrong, (b) borderline photos,
(c) photos from a different domain, (d) out-of-scope images, (e) synthetic edits of real photos.
Every entry carries an 'expected' answer per task: a label, '?' (debatable) or null (no right answer).
"""
import argparse, json, shutil, sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from aec_lab.config import DATASETS          # noqa: E402
from aec_lab.data import ImageSet            # noqa: E402
from aec_lab.models import Classifier        # noqa: E402
from aec_lab.tricky import perturb           # noqa: E402


def out_of_scope_images():
    from skimage import data
    ims = [("cat", "a cat", Image.fromarray(data.chelsea())),
           ("person", "a person", Image.fromarray(data.astronaut())),
           ("coffee", "a cup of coffee", Image.fromarray(data.coffee()))]
    for key, fn, cap in [("brick", data.brick, "brick texture (skimage sample)"),
                         ("gravel", data.gravel, "gravel texture (skimage sample)"),
                         ("grass", data.grass, "grass texture (skimage sample)")]:
        try:
            ims.append((key, cap, Image.fromarray(fn()).convert("RGB")))
        except Exception as e:  # needs network for pooch download
            print("  skipping", key, e)
    ims.append(("grey", "a blank grey image", Image.new("RGB", (256, 256), (128, 128, 128))))
    rng = np.random.default_rng(0)
    ims.append(("noise", "pure random noise", Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))))
    plan = Image.new("RGB", (256, 256), "white")
    d = ImageDraw.Draw(plan)
    d.rectangle([20, 20, 236, 236], outline="black", width=4)
    d.line([128, 20, 128, 236], fill="black", width=3); d.line([20, 140, 236, 140], fill="black", width=3)
    d.rectangle([60, 20, 96, 24], fill="white"); d.rectangle([150, 232, 190, 236], fill="white")
    d.text((40, 70), "ROOM 101", fill="black"); d.text((150, 180), "ROOM 102", fill="black")
    ims.append(("plan", "a floor-plan drawing", plan))
    return ims


class Gallery:
    def __init__(self, out: Path):
        self.out = out
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        self.entries = []

    def add(self, img: Image.Image, caption: str, group: str, expected_binary, expected_multi, source: str):
        i = len(self.entries) + 1
        fn = f"{i:02d}_{group}.jpg"
        img = img.convert("RGB")
        if max(img.size) > 320:
            s = 320 / max(img.size)
            img = img.resize((max(64, round(img.width * s)), max(64, round(img.height * s))), Image.LANCZOS)
        img.save(self.out / fn, "JPEG", quality=88)
        self.entries.append({"id": i, "file": fn, "caption": caption, "group": group,
                             "expected": {"binary": expected_binary, "multiclass": expected_multi}, "source": source})

    def save(self):
        (self.out / "tricky.json").write_text(json.dumps(self.entries, indent=2))
        print(f"  {len(self.entries)} tricky photos -> {self.out}")


def build(key: str, work: Path, has_tasks):
    spec = DATASETS[key]
    print("==", key)
    test = ImageSet.from_folder(work / key / "test", spec.classes, spec.display)
    g = Gallery(REPO / spec.tricky_dir)
    multi_name = lambda i: spec.pretty(spec.classes[test[i][1]])  # noqa: E731
    has_multi = spec.multiclass_model is not None and "multiclass" in has_tasks

    if spec.binary and spec.binary_model:
        clf = Classifier.load(REPO / spec.binary_model)
        btest = test.relabel(spec.binary.mapping, spec.binary.classes)
        y = np.array(btest.labels())
        P = clf.predict_paths(btest.paths())
        pred, conf = P.argmax(1), P.max(1)
        pos, neg = spec.binary.positive, spec.binary.negative
        # (a) confident mistakes
        miss = [i for i in np.argsort(-conf) if y[i] == 1 and pred[i] == 0][:3]
        alarm = [i for i in np.argsort(-conf) if y[i] == 0 and pred[i] == 1][:3]
        for i in miss:
            g.add(test.load(int(i)), f"a real '{multi_name(i)}' the model misses", "hard_real", pos,
                  multi_name(i) if has_multi else None, test[i][0].name)
        for i in alarm:
            g.add(test.load(int(i)), f"a real '{multi_name(i)}' the model flags", "hard_real", neg,
                  multi_name(i) if has_multi else None, test[i][0].name)
        # (b) borderline: correct but least confident positives
        border = [i for i in np.argsort(conf) if y[i] == 1 and pred[i] == 1][:2]
        for i in border:
            g.add(test.load(int(i)), f"borderline '{multi_name(i)}' (model barely sure)", "borderline", pos,
                  multi_name(i) if has_multi else None, test[i][0].name)
        # (e) synthetic edits of a clear negative and a clear positive
        clear_neg = [i for i in np.argsort(-conf) if y[i] == 0 and pred[i] == 0][0]
        clear_pos = [i for i in np.argsort(-conf) if y[i] == 1 and pred[i] == 1][0]
        neg_img, pos_img = test.load(int(clear_neg)), test.load(int(clear_pos))
        neg_multi = multi_name(clear_neg) if has_multi else None
        pos_multi = multi_name(clear_pos) if has_multi else None
        g.add(perturb(neg_img, line=5, line_angle=60), f"a dark line DRAWN on a '{neg_multi or neg}' photo", "synthetic", neg, neg_multi, test[clear_neg][0].name)
        g.add(perturb(neg_img, shadow=0.9), f"a strong shadow added to a '{neg_multi or neg}' photo", "synthetic", neg, neg_multi, test[clear_neg][0].name)
        g.add(perturb(pos_img, blur=6), f"a real '{pos_multi or pos}', blurred (shaky camera)", "synthetic", pos, pos_multi, test[clear_pos][0].name)
        g.add(perturb(pos_img, brightness=0.25), f"a real '{pos_multi or pos}', very dark", "synthetic", pos, pos_multi, test[clear_pos][0].name)
        g.add(perturb(pos_img, rotate=90), f"a real '{pos_multi or pos}', rotated 90°", "synthetic", pos, pos_multi, test[clear_pos][0].name)
        canvas = neg_img.copy(); small = pos_img.resize((neg_img.width // 3, neg_img.height // 3))
        canvas.paste(small, (neg_img.width // 3, neg_img.height // 3))
        g.add(canvas, f"a '{pos_multi or pos}' photo shrunk into the middle of a '{neg_multi or neg}' photo", "synthetic", pos, "?", test[clear_pos][0].name)
        g.add(perturb(pos_img, zoom=4), f"a real '{pos_multi or pos}', zoomed in 4x", "synthetic", "?", "?", test[clear_pos][0].name)
        # (c) other domain
        if spec.other_domain and (work / spec.other_domain / "test").exists():
            o = DATASETS[spec.other_domain]
            otest = ImageSet.from_folder(work / o.key / "test", o.classes, o.display)
            rng = np.random.default_rng(3)
            for cls in o.classes[:4]:
                sub = otest.only(cls)
                i = int(rng.integers(len(sub)))
                exp_b = o.binary.mapping[cls] if o.binary else "?"
                # translate the other dataset's binary words into this dataset's words
                exp_b = pos if (o.binary and exp_b == o.binary.positive) else (neg if o.binary else "?")
                g.add(sub.load(i), f"'{o.pretty(cls)}' photo from another dataset ({o.title.split('/')[0].strip()})",
                      "other_domain", exp_b, "?" if has_multi else None, f"{o.key}:{sub[i][0].name}")
    elif spec.multiclass_model:
        clf = Classifier.load(REPO / spec.multiclass_model)
        y = np.array(test.labels()); P = clf.predict_paths(test.paths()); pred, conf = P.argmax(1), P.max(1)
        for i in [i for i in np.argsort(-conf) if pred[i] != y[i]][:6]:
            g.add(test.load(int(i)), f"a real '{multi_name(i)}' the model gets wrong", "hard_real", None, multi_name(i), test[i][0].name)
        for i in [i for i in np.argsort(conf) if pred[i] == y[i]][:2]:
            g.add(test.load(int(i)), f"borderline '{multi_name(i)}' (model barely sure)", "borderline", None, multi_name(i), test[i][0].name)
        clear, seen = [], set()
        for i in np.argsort(-conf):
            if pred[i] == y[i] and y[i] not in seen:
                clear.append(int(i)); seen.add(y[i])
            if len(clear) == 3:
                break
        for i in clear:
            im, nm = test.load(i), multi_name(i)
            g.add(perturb(im, blur=6), f"a real '{nm}', blurred", "synthetic", None, nm, test[i][0].name)
            g.add(perturb(im, grayscale=True), f"a real '{nm}', black & white", "synthetic", None, nm, test[i][0].name)
            g.add(perturb(im, zoom=3), f"a real '{nm}', zoomed in 3x", "synthetic", None, "?", test[i][0].name)
        if spec.other_domain and (work / spec.other_domain / "test").exists():
            o = DATASETS[spec.other_domain]
            otest = ImageSet.from_folder(work / o.key / "test", o.classes, o.display)
            rng = np.random.default_rng(3)
            for cls in o.classes[:3]:
                sub = otest.only(cls); i = int(rng.integers(len(sub)))
                g.add(sub.load(i), f"'{o.pretty(cls)}' photo from another dataset", "other_domain", None, None, f"{o.key}:{sub[i][0].name}")
    # (d) out of scope
    for k, cap, im in out_of_scope_images():
        g.add(im, cap, "out_of_scope", None, None, f"skimage:{k}" if k in ("cat", "person", "coffee", "brick", "gravel", "grass") else "synthetic")
    g.save()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--only", nargs="*", default=["facade_defects", "concrete_cracks", "facade_styles"])
    a = ap.parse_args()
    for key in a.only:
        build(key, Path(a.work), has_tasks=("binary", "multiclass"))
    print("done")
