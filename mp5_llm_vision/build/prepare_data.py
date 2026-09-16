"""Instructor-side: pick the MP5 examples from the earlier labs (with their answer keys) and record what the earlier labs'
specialist models say about them, for the generalist-vs-specialist comparison.

    python build/prepare_data.py [--variants workshop homework]

Writes data/<variant>/photos (MP2 test photos, 2 per class), data/<variant>/sites (MP3 test photos with every box),
data/<variant>/plans (MP4 plans with answer keys), each with an index.json, plus credits.json.
"""
import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
sys.path.insert(0, str(REPO))
from aec_llm.config import SPECS  # noqa: E402

MP2, MP3, MP4 = ROOT / "mp2_image_classification", ROOT / "mp3_object_detection", ROOT / "mp4_segmentation"
SCRATCH = REPO / "_unzipped"


def unzip(zip_path: Path) -> Path:
    target = SCRATCH / zip_path.stem
    if not target.exists():
        SCRATCH.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(SCRATCH)
    return target


def photos(spec, out: Path, per_class: int = 2):
    sys.path.insert(0, str(MP2))
    from aec_lab.config import DATASETS
    from aec_lab.models import Classifier
    ds = DATASETS[spec.photos.mp2_key]
    names = [ds.display[c] for c in ds.classes]
    assert names == spec.photos.classes, (names, spec.photos.classes)
    test = unzip(MP2 / "data" / ds.zip_name) / "test"
    d = out / "photos"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    items = []
    for folder, name in zip(ds.classes, names):
        files = sorted((test / folder).glob("*.jpg"))
        step = max(1, len(files) // per_class)
        for i in range(per_class):
            src = files[i * step]
            dst = d / f"{folder}_{i + 1}.jpg"
            shutil.copy(src, dst)
            items.append({"file": dst.name, "truth": name})
    clf = Classifier.load(MP2 / spec.photos.mp2_model)
    assert list(clf.classes) == names, (clf.classes, names)
    proba = clf.predict_paths([d / it["file"] for it in items])
    for it, p in zip(items, proba):
        it["specialist"] = {"label": clf.classes[int(np.argmax(p))], "confidence": round(float(p.max()), 3)}
    (d / "index.json").write_text(json.dumps({"classes": names, "specialist_name": spec.photos.specialist, "items": items}, indent=1, ensure_ascii=False))
    acc = np.mean([it["specialist"]["label"] == it["truth"] for it in items])
    print(f"{spec.key}/photos: {len(items)} photos, specialist accuracy {acc * 100:.0f} %")


def sites(spec, out: Path, n_pick: int = 6):
    sys.path.insert(0, str(MP3))
    from aec_det.config import DETSETS
    from aec_det.models import Detector
    ds = DETSETS[spec.sites.mp3_key]
    test = unzip(MP3 / "data" / ds.zip_name) / "test"
    cands = []
    for p in sorted((test / "images").glob("*.jpg")):
        lab = test / "labels" / (p.stem + ".txt")
        rows = [l.split() for l in lab.read_text().splitlines() if l.strip()] if lab.exists() else []
        labels = [spec.sites.mp3_map[ds.classes[int(r[0])]] for r in rows]
        n = len(rows)
        ok = (4 <= n <= 12 and any(l in ("no-helmet", "no-vest") for l in labels)) if spec.key == "workshop" else (1 <= n <= 5)
        if ok:
            cands.append((p, rows, labels))
    idx = np.linspace(0, len(cands) - 1, n_pick).round().astype(int)
    d = out / "sites"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    det = Detector(MP3 / spec.sites.mp3_model, classes=ds.classes)
    items = []
    for k, i in enumerate(idx, 1):
        p, rows, labels = cands[i]
        im = Image.open(p).convert("RGB")
        W, H = im.size
        dst = d / f"site_{k}.jpg"
        shutil.copy(p, dst)
        truth = []
        for (c, x, y, w, h), l in zip(rows, labels):
            x, y, w, h = float(x), float(y), float(w), float(h)
            truth.append({"label": l, "box": [round((x - w / 2) * W, 1), round((y - h / 2) * H, 1), round((x + w / 2) * W, 1), round((y + h / 2) * H, 1)]})
        r = det.predict(im, conf=0.25)
        sp = [{"label": spec.sites.mp3_map[ds.classes[int(c)]], "box": [round(float(v), 1) for v in b], "conf": round(float(cf), 3)}
              for b, cf, c in zip(r.xyxy, r.conf, r.cls)]
        items.append({"file": dst.name, "size": [W, H], "truth": truth, "specialist": sp})
    (d / "index.json").write_text(json.dumps({"classes": spec.sites.classes, "specialist_name": spec.sites.specialist, "items": items}, indent=1))
    print(f"{spec.key}/sites: {len(items)} photos, {sum(len(it['truth']) for it in items)} boxes in the answer key, "
          f"{sum(len(it['specialist']) for it in items)} specialist boxes at 0.25")


def plans(spec, out: Path):
    sys.path.insert(0, str(MP4))
    from aec_seg.config import SETS
    from aec_seg.data import MaskStore, PlanSet
    from aec_seg.ui import box_iou
    s = SETS[spec.plans.mp4_set]
    ps = PlanSet(MP4 / s.folder)
    store = MaskStore(MP4 / s.masks)
    d = out / "plans"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    items = []
    for pid in spec.plans.ids:
        p = ps[pid]
        shutil.copy(p.path, d / p.path.name)
        shutil.copy(MP4 / s.folder / f"{pid}.key.json", d / f"{pid}.key.json")
        res = store.load(pid, "room")
        keep = np.where(res.scores >= 0.3)[0] if res is not None else []
        rooms = []
        for r in p.rooms(indoor_only=True):
            best, bi = 0.0, -1
            for i in keep:
                v = box_iou(r["box"], list(map(float, res.boxes[i])))
                if v > best:
                    best, bi = v, int(i)
            rooms.append({"label": r["label"], "type": r["type"], "truth_m2": r["area_m2"],
                          "sam_m2": round(p.area_m2(res.masks[bi].sum()), 2) if best >= 0.3 else None})
        items.append({"id": pid, "file": p.path.name, "key": f"{pid}.key.json", "title": p.title,
                      "specialist": {"name": "MP4: SAM 3 asked for 'room' (confidence 0.3)", "rooms": rooms}})
        found = sum(r["sam_m2"] is not None for r in rooms)
        print(f"{spec.key}/plans: {pid} {p.title}: SAM 3 'room' found {found}/{len(rooms)} rooms")
    (d / "index.json").write_text(json.dumps({"items": items, "credit": ps.credits_text()}, indent=1, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=list(SPECS))
    a = ap.parse_args()
    for key in a.variants:
        spec = SPECS[key]
        out = REPO / spec.folder
        photos(spec, out)
        sites(spec, out)
        plans(spec, out)
        (out / "credits.json").write_text(json.dumps({
            "photos": spec.photos.source, "sites": spec.sites.source,
            "plans": "CubiCasa5K (CC BY-NC-SA 4.0), prepared for MP4; see mp4_segmentation/README.md",
        }, indent=1, ensure_ascii=False))
    print("done")


if __name__ == "__main__":
    main()
