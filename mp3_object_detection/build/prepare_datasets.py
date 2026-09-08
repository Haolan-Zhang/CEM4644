"""Instructor-side: package the YOLO-format detection sets that ship with the lab.

  python build/prepare_datasets.py --key construction_safety --src <folder with train/valid/test> --work <work>
Produces work/<key>/{train_full,valid,test} for course-model training and data/<key>.zip
(<key>/{train,test}/{images,labels} + about.json) for the students.
"""
import argparse, json, random, shutil, sys, zipfile
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from aec_det.config import DETSETS  # noqa: E402

SEED = 4644
ABOUT = {
    "construction_safety": {
        "title": "Workers and PPE on construction sites",
        "source": "https://huggingface.co/datasets/LibreYOLO/construction-safety-gsnvb (Roboflow 100 'construction-safety')",
        "license": "CC-BY-4.0",
        "citation": "Ciaglia et al., Roboflow 100: A Rich, Multi-Domain Object Detection Benchmark (2022), arXiv:2211.13523",
    },
    "excavators": {
        "title": "Construction machinery: excavators, dump trucks, wheel loaders",
        "source": "https://huggingface.co/datasets/LibreYOLO/excavators-czvg9 (Roboflow 100 'excavators')",
        "license": "CC-BY-4.0",
        "citation": "Ciaglia et al., Roboflow 100: A Rich, Multi-Domain Object Detection Benchmark (2022), arXiv:2211.13523",
    },
}


def copy_pair(img: Path, lab_dir: Path, out_split: Path, new_stem: str, quality: int = 82, max_side: int = 640):
    (out_split / "images").mkdir(parents=True, exist_ok=True); (out_split / "labels").mkdir(parents=True, exist_ok=True)
    im = Image.open(img).convert("RGB")
    if max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    im.save(out_split / "images" / f"{new_stem}.jpg", "JPEG", quality=quality, optimize=True)
    lab = lab_dir / (img.stem + ".txt")
    (out_split / "labels" / f"{new_stem}.txt").write_text(lab.read_text() if lab.exists() else "")


def list_images(split_dir: Path):
    return sorted(p for p in (split_dir / "images").iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


def build(key: str, src: Path, work: Path, data_dir: Path, pool_n: int, test_n: int, val_n: int, test_from_train: int = 0):
    spec = DETSETS[key]
    rng = random.Random(SEED)
    root = work / key
    if root.exists():
        shutil.rmtree(root)
    train = list_images(src / "train"); valid = list_images(src / "valid"); test = list_images(src / "test")
    rng.shuffle(train); rng.shuffle(valid)
    # course-model training: train minus a validation slice
    course_val, held, course_train = train[:val_n], train[val_n:val_n + test_from_train], train[val_n + test_from_train:]
    for i, p in enumerate(course_train):
        copy_pair(p, src / "train" / "labels", root / "train_full", f"tr{i:04d}")
    for i, p in enumerate(course_val):
        copy_pair(p, src / "train" / "labels", root / "valid", f"va{i:04d}")
    # unseen test for students and for the course-model report: official test + part of official valid
    test_items = [(p, src / "test" / "labels") for p in test] + [(p, src / "train" / "labels") for p in held] + \
                 [(p, src / "valid" / "labels") for p in valid]
    test_items = test_items[:test_n]
    for i, (p, ld) in enumerate(test_items):
        copy_pair(p, ld, root / "test", f"te{i:04d}")
    # student training pool: part of the course training images
    pool = course_train[:pool_n]
    for i, p in enumerate(pool):
        copy_pair(p, src / "train" / "labels", root / "train", f"tr{i:04d}")
    (root / "data.yaml").write_text(f"path: {root.resolve()}\ntrain: train_full/images\nval: valid/images\nnames:\n" +
                                    "".join(f"  {i}: '{c}'\n" for i, c in enumerate(spec.classes)))
    about = dict(ABOUT[key], name=key, classes=spec.classes,
                 counts={"train_pool": len(pool), "test": len(test_items), "course_train": len(course_train), "course_val": len(course_val)})
    (root / "about.json").write_text(json.dumps(about, indent=2))
    zip_path = data_dir / spec.zip_name
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{key}/about.json", json.dumps(about, indent=2))
        for split in ("train", "test"):
            for p in sorted((root / split).rglob("*")):
                if p.is_file():
                    zf.write(p, f"{key}/{p.relative_to(root)}")
    print(f"{key}: course_train {len(course_train)}, course_val {len(course_val)}, pool {len(pool)}, test {len(test_items)} -> "
          f"{zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--data-dir", default=str(REPO / "data"))
    ap.add_argument("--pool", type=int, default=300)
    ap.add_argument("--test", type=int, default=250)
    ap.add_argument("--val", type=int, default=100)
    ap.add_argument("--test-from-train", type=int, default=0, help="hold out this many training images into the test set")
    a = ap.parse_args()
    build(a.key, Path(a.src), Path(a.work), Path(a.data_dir), a.pool, a.test, a.val, a.test_from_train)
