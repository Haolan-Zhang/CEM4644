"""
Instructor-side script: builds the small, license-clean image sets that ship
with the MP2 lab (as zip files under ../data) plus the larger "full" training
sets used to train the course models (kept outside the repo).

Students never run this. Run once on the instructor machine:

    python build/prepare_datasets.py --hf-cache /path/to/hf_cache --work /path/to/work

Sources (all downloaded from the Hugging Face Hub):
  * facade_defects  <- chandrabhuma/building_defect_vqa  (BD3 building defect dataset, CC-BY-4.0)
  * concrete_cracks <- mohammadnajeeb/concrete_crack_images (CC-BY-4.0)
  * pavement_surfaces <- kauevestena/deep_pavements_surface_patches (MIT)
"""
import argparse, glob, io, json, os, random, shutil, sys, zipfile
from pathlib import Path

from PIL import Image

SEED = 4644
JPEG_QUALITY = 85


def save_jpeg(img: Image.Image, path: Path, size: int | None):
    img = img.convert("RGB")
    if size is not None and img.size != (size, size):
        w, h = img.size
        s = size / min(w, h)
        img = img.resize((max(size, round(w * s)), max(size, round(h * s))), Image.LANCZOS)
        w, h = img.size
        left, top = (w - size) // 2, (h - size) // 2
        img = img.crop((left, top, left + size, top + size))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "JPEG", quality=JPEG_QUALITY, optimize=True)


def write_split(items, root: Path, split: str, size: int | None):
    """items: list of (class_name, PIL image or bytes or path). Writes root/split/class/class_0001.jpg"""
    counters = {}
    for cls, src in items:
        counters[cls] = counters.get(cls, 0) + 1
        if isinstance(src, (bytes, bytearray)):
            img = Image.open(io.BytesIO(src))
        elif isinstance(src, (str, Path)):
            img = Image.open(src)
        else:
            img = src
        save_jpeg(img, root / split / cls / f"{cls}_{counters[cls]:04d}.jpg", size)
    return counters


def stratified_take(items, per_class, rng):
    by_cls = {}
    for cls, src in items:
        by_cls.setdefault(cls, []).append((cls, src))
    out = []
    for cls in sorted(by_cls):
        lst = by_cls[cls]
        rng.shuffle(lst)
        out.extend(lst[:per_class])
    rng.shuffle(out)
    return out


def zip_dir(src_root: Path, splits, zip_path: Path, about: dict):
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    name = src_root.name
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:  # jpeg does not compress further
        zf.writestr(f"{name}/about.json", json.dumps(about, indent=2))
        for split in splits:
            for p in sorted((src_root / split).rglob("*.jpg")):
                zf.write(p, f"{name}/{p.relative_to(src_root)}")
    print(f"  wrote {zip_path} ({zip_path.stat().st_size/1e6:.1f} MB)")


# ----------------------------------------------------------------------------- facade
def build_facade(hf_cache: Path, work: Path, data_dir: Path):
    import pyarrow.parquet as pq

    print("== facade_defects (BD3)")
    root = work / "facade_defects"
    if root.exists():
        shutil.rmtree(root)
    snap = glob.glob(str(hf_cache / "datasets--chandrabhuma--building_defect_vqa/snapshots/*/data"))[0]
    rng = random.Random(SEED)

    def read(pattern):
        items = []
        for f in sorted(glob.glob(os.path.join(snap, pattern))):
            t = pq.read_table(f, columns=["image", "answer"])
            for img, ans in zip(t.column("image").to_pylist(), t.column("answer").to_pylist()):
                items.append((ans.strip().lower().replace(" ", "_"), img["bytes"]))
        return items

    train_items = read("train-*.parquet")
    test_items = read("test-*.parquet")
    rng.shuffle(train_items)
    print("  train_full:", write_split(train_items, root, "train_full", 256))
    print("  test:", write_split(test_items, root, "test", 256))
    pool = stratified_take(train_items, 200, rng)
    print("  train (student pool):", write_split(pool, root, "train", 256))
    about = {
        "name": "facade_defects",
        "title": "Building facade / wall surface defects (BD3)",
        "source": "https://huggingface.co/datasets/chandrabhuma/building_defect_vqa (repackaged from https://github.com/Praveenkottari/BD3-Dataset)",
        "license": "CC-BY-4.0",
        "citation": "Kottari, P. and Arjunan, P. BD3: Building Defect Dataset. https://github.com/Praveenkottari/BD3-Dataset",
        "notes": "Photos of concrete/stone walls of 50+ buildings taken ~1 m from the wall with a smartphone. Resized to 256x256 JPEG for the lab.",
    }
    zip_dir(root, ["train", "test"], data_dir / "facade_defects.zip", about)


# ----------------------------------------------------------------------------- concrete
def build_concrete(hf_cache: Path, work: Path, data_dir: Path):
    print("== concrete_cracks")
    root = work / "concrete_cracks"
    if root.exists():
        shutil.rmtree(root)
    snap = Path(glob.glob(str(hf_cache / "datasets--mohammadnajeeb--concrete_crack_images/snapshots/*/data"))[0])
    rng = random.Random(SEED)
    name_map = {"Positive": "crack", "Negative": "no_crack"}

    def read(zip_name):
        items = []
        with zipfile.ZipFile(snap / zip_name) as zf:
            for n in zf.namelist():
                if not n.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                cls = name_map[n.split("/")[-2]]
                items.append((cls, zf.read(n)))
        return items

    train_items = read("train.zip")
    test_items = read("test.zip")
    print("  train_full:", write_split(stratified_take(train_items, 4000, rng), root, "train_full", None))
    print("  test:", write_split(stratified_take(test_items, 500, rng), root, "test", None))
    print("  train (student pool):", write_split(stratified_take(train_items, 500, rng), root, "train", None))
    about = {
        "name": "concrete_cracks",
        "title": "Concrete surface cracks",
        "source": "https://huggingface.co/datasets/mohammadnajeeb/concrete_crack_images",
        "license": "CC-BY-4.0",
        "citation": "Ozgenel, C.F. (2019). Concrete Crack Images for Classification. Mendeley Data. https://data.mendeley.com/datasets/5y9wdsg2zt",
        "notes": "227x227 photos of concrete surfaces (METU campus buildings). Subset used in the lab.",
    }
    zip_dir(root, ["train", "test"], data_dir / "concrete_cracks.zip", about)


# ----------------------------------------------------------------------------- pavement
def build_pavement(work: Path, data_dir: Path, hf_saved: Path, min_side=96):
    from datasets import load_from_disk

    print("== pavement_surfaces")
    root = work / "pavement_surfaces"
    if root.exists():
        shutil.rmtree(root)
    ds = load_from_disk(str(hf_saved))
    names = ds.features["label"].names
    rng = random.Random(SEED)
    items = []
    for ex in ds:
        if min(ex["image"].size) >= min_side:
            items.append((names[ex["label"]].replace(" ", "_"), ex["image"].convert("RGB")))
    by_cls = {}
    for it in items:
        by_cls.setdefault(it[0], []).append(it)
    train, test = [], []
    for cls, lst in sorted(by_cls.items()):
        rng.shuffle(lst)
        n_test = max(8, round(len(lst) * 0.25))
        test.extend(lst[:n_test])
        train.extend(lst[n_test:])
    rng.shuffle(train)
    print("  train_full:", write_split(train, root, "train_full", 256))
    print("  test:", write_split(test, root, "test", 256))
    print("  train (student pool):", write_split(train, root, "train", 256))
    about = {
        "name": "pavement_surfaces",
        "title": "Pavement / ground surface materials",
        "source": "https://huggingface.co/datasets/kauevestena/deep_pavements_surface_patches",
        "license": "MIT",
        "citation": "Vestena, K. Deep Pavements dataset. https://github.com/kauevestena/deep_pavements_dataset",
        "notes": f"Street-level surface patches; images with a side shorter than {min_side}px were dropped. Resized to 256x256 JPEG.",
    }
    zip_dir(root, ["train", "test"], data_dir / "pavement_surfaces.zip", about)


# ----------------------------------------------------------------------------- facade styles
def build_styles(work: Path, data_dir: Path, plates_dir: Path, manifest_path: Path, n_test: int = 12):
    import re
    import pandas as pd
    from aec_lab.config import DATASETS  # noqa

    print("== facade_styles")
    spec = DATASETS["facade_styles"]
    root = work / "facade_styles"
    if root.exists():
        shutil.rmtree(root)
    m = pd.read_parquet(manifest_path)
    norm = lambda s: re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")  # noqa: E731
    m["cls"] = m["style_name"].map(norm)
    files = {p.stem: p for p in Path(plates_dir).rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")}
    rng = random.Random(SEED)
    train, test = [], []
    for cls in spec.classes:
        ids = sorted(m.loc[m["cls"] == cls, "plate_id"].tolist())
        ids = [i for i in ids if i in files]
        assert ids, f"no plates found for {cls}"
        rng.shuffle(ids)
        test.extend((cls, files[i]) for i in ids[:n_test])
        train.extend((cls, files[i]) for i in ids[n_test:])
    rng.shuffle(train)
    print("  train_full:", write_split(train, root, "train_full", 256))
    print("  test:", write_split(test, root, "test", 256))
    print("  train (student pool):", write_split(train, root, "train", 256))
    about = {
        "name": "facade_styles",
        "title": "Architectural styles of building façades (synthetic)",
        "source": "https://huggingface.co/datasets/Jonathandav/facade-styles",
        "license": "MIT",
        "citation": "Jonathandav, Facade: synthetic architectural style corpus (2025), Hugging Face.",
        "notes": "1000 SDXL-Turbo generated plates, 20 styles; 10 styles used here, 50 plates each, resized to 256x256 JPEG.",
    }
    zip_dir(root, ["train", "test"], data_dir / "facade_styles.zip", about)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-cache", required=True)
    ap.add_argument("--work", required=True, help="folder for full (unshipped) training sets")
    ap.add_argument("--data-dir", default=str(Path(__file__).resolve().parents[1] / "data"))
    ap.add_argument("--pavement-saved", default=None, help="datasets.save_to_disk folder for the pavement set")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--styles-plates", default=None, help="folder with the facade-styles plates")
    ap.add_argument("--styles-manifest", default=None, help="plate_manifest.parquet of facade-styles")
    a = ap.parse_args()
    hf_cache, work, data_dir = Path(a.hf_cache), Path(a.work), Path(a.data_dir)
    work.mkdir(parents=True, exist_ok=True)
    todo = a.only or ["facade", "concrete", "styles"]
    if "facade" in todo:
        build_facade(hf_cache, work, data_dir)
    if "concrete" in todo:
        build_concrete(hf_cache, work, data_dir)
    if "pavement" in todo and a.pavement_saved:
        build_pavement(work, data_dir, Path(a.pavement_saved))
    if "styles" in todo and a.styles_plates:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        build_styles(work, data_dir, Path(a.styles_plates), Path(a.styles_manifest))
    print("done")
