"""Instructor-side, after curate_photos.py: drop weak photos and auto-crop the 1937 album scans
(beige mount, black frame and the caption strip) so that area percentages refer to the photo itself."""
import argparse, json, sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
DROP = {"site": ["site_17"], "interior": ["int_02", "int_03", "int_04", "int_05", "int_08", "int_14"]}
ALBUM = {"site": [f"site_{i}" for i in range(23, 33)], "interior": ["int_11", "int_12", "int_13"]}


def autocrop_album(im: Image.Image, left=0.10, right=0.035, top=0.05, bottom=0.05, caption=0.12) -> Image.Image:
    """The album scans share one layout: a tan mount (punched on the left) around a print whose bottom strip
    carries a typed caption. Margins were calibrated on the scans; a slight over-crop is harmless."""
    W, H = im.size
    x1, x2 = int(W * left), int(W * (1 - right))
    y1, y2 = int(H * top), int(H * (1 - bottom))
    y2 = y2 - int((y2 - y1) * caption)
    return im.crop((x1, y1, x2, y2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", default=str(REPO / "data" / "photos"))
    a = ap.parse_args()
    root = Path(a.photos)
    for group in ("site", "interior"):
        cred_p = root / group / "credits.json"
        cred = json.loads(cred_p.read_text())
        keep = []
        for c in cred:
            f = root / group / c["file"]
            if c["id"] in DROP.get(group, []):
                if f.exists():
                    f.unlink()
                print("dropped", c["id"]); continue
            if c["id"] in ALBUM.get(group, []) and not c.get("cropped"):
                im = Image.open(f).convert("RGB")
                out = autocrop_album(im)
                out.save(f, "JPEG", quality=88, optimize=True)
                c["cropped"] = True; c["size"] = list(out.size)
                print(f"cropped {c['id']}: {im.size} -> {out.size}")
            keep.append(c)
        cred_p.write_text(json.dumps(keep, indent=2, ensure_ascii=False))
        print(group, "->", len(keep), "photos")
