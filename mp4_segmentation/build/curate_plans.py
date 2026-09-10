"""Downloads the four public-domain foundation plans used in Part 5 from Wikimedia Commons and cuts out
the plan-view region of each sheet (the full sheets are too dense: SAM 3 only works on the plan itself).

Run:  python build/curate_plans.py            (writes data/photos/plans/*.jpg + credits.json)

Sources: Library of Congress HABS/HAER measured drawings and US Army Corps of Engineers standard
drawings, all public domain, as mirrored on Wikimedia Commons (file pages in credits.json).
"""
import json
import re
import sys
import time
from pathlib import Path

import requests
from PIL import Image, ImageOps

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "photos" / "plans"
API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "CEM4644-course-material/1.0 (contact: course instructor)"}

# id, Commons file title, crop (x1, y1, x2, y2 as fractions of the sheet), max width, title, author, licence
PLANS = [
    ("plan_dormitory",
     "File:0021-01-78 sheet 09 - USACE-p16021coll10-14314.jpeg",
     (0.13, 0.035, 0.52, 0.38), 3600,
     "Airmen's dormitory, three-storey concrete frame: foundation plan and details (US Army standard drawing 21-01-78, sheet 09, 1956)",
     "U.S. Army Corps of Engineers (Wikimedia Commons file '0021-01-78 sheet 09 - USACE-p16021coll10-14314.jpeg')",
     "Public domain (work of the US federal government)",
     "left part of the foundation plan (column lines 1 to 9) at the scan's native resolution; contrast stretched"),
    ("plan_mess_hall",
     "File:0021-01-30 sheet 040 - USACE-p16021coll10-13955.jpeg",
     (0.22, 0.30, 0.74, 0.90), 2000,
     "Barracks for 450 enlisted men, kitchen and mess hall: foundation plan (US Army standard drawing 21-01-30, sheet 040, 1956)",
     "U.S. Army Corps of Engineers (Wikimedia Commons file '0021-01-30 sheet 040 - USACE-p16021coll10-13955.jpeg')",
     "Public domain (work of the US federal government)",
     "plan-view region cropped from the full sheet; contrast stretched"),
    ("plan_whittier",
     "File:Foundation Plan - Foundation Plan and Details of Footing (drawing S1) - Whittier State School, Hospital and Receiving Building, 11850 East Whittier Boulevard, Whittier, Los HABS CAL,19-WHIT,3-34.tif",
     (0.17, 0.07, 0.80, 0.47), 2000,
     "Whittier State School, Hospital and Receiving Building: foundation plan and details of footings (drawing S1, 1930s)",
     "Historic American Buildings Survey (HABS), Library of Congress, copy of original drawing (via Wikimedia Commons)",
     "Public domain",
     "plan-view region cropped from the full sheet; contrast stretched"),
    ("plan_mill",
     "File:Foundation Plan - Shenandoah-Dives Mill, 135 County Road 2, Silverton, San Juan County, CO HAER CO-91 (sheet 7 of 27).png",
     (0.10, 0.06, 0.82, 0.80), 2000,
     "Shenandoah-Dives Mill, Silverton, Colorado: foundation plan (HAER CO-91, measured drawing, 1990s)",
     "Historic American Engineering Record (HAER), Library of Congress, delineated for HAER (via Wikimedia Commons)",
     "Public domain",
     "plan-view region cropped from the full sheet; contrast stretched"),
]


def imageinfo(title: str, width: int = 3600) -> dict:
    r = requests.post(API, data={"action": "query", "prop": "imageinfo", "iiprop": "url|size|extmetadata",
                                 "iiurlwidth": width, "titles": title, "format": "json"}, headers=UA, timeout=90).json()
    pages = r["query"]["pages"]
    page = next(iter(pages.values()))
    if "imageinfo" not in page:
        raise SystemExit(f"not found on Commons: {title}")
    return page["imageinfo"][0]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    credits = []
    for pid, title, (x1, y1, x2, y2), max_w, ptitle, author, lic, note in PLANS:
        ii = imageinfo(title)
        raw = OUT / f"_{pid}_sheet.jpg"
        if not raw.exists():
            print("downloading", pid)
            for attempt, wait in enumerate((0, 30, 60, 120)):
                time.sleep(wait)
                rr = requests.get(ii["thumburl"], headers=UA, timeout=180)
                if rr.status_code == 429:          # Commons rate limit on the originals endpoint: back off and retry
                    print("  rate-limited, waiting...")
                    continue
                rr.raise_for_status()
                break
            else:
                raise SystemExit("Commons keeps rate-limiting; try again later")
            raw.write_bytes(rr.content)
            time.sleep(3.0)
        im = Image.open(raw).convert("RGB")
        W, H = im.size
        c = im.crop((round(x1 * W), round(y1 * H), round(x2 * W), round(y2 * H)))
        if c.width > max_w:
            s = max_w / c.width
            c = c.resize((max_w, round(c.height * s)), Image.LANCZOS)
        c = ImageOps.autocontrast(c, cutoff=0.5)
        c.save(OUT / f"{pid}.jpg", quality=92)
        raw.unlink()
        credits.append({"id": pid, "file": f"{pid}.jpg", "title": ptitle, "author": author, "license": lic,
                        "source": ii.get("descriptionurl", "https://commons.wikimedia.org/wiki/" + title.replace(" ", "_")),
                        "note": note, "size": list(c.size)})
        print(f"{pid}: {c.size}")
    (OUT / "credits.json").write_text(json.dumps(credits, indent=2, ensure_ascii=False))
    print("wrote", OUT / "credits.json")


if __name__ == "__main__":
    main()
