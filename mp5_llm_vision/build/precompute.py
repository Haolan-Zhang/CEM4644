"""Instructor-side: run Gemini once on every built-in example x prompt the notebooks use and store the replies in
data/cache/<variant>, so the notebooks are instant and work without a key. Needs GEMINI_API_KEY in the environment.

    python build/precompute.py [--variants workshop homework] [--model gemini-3.8-flash]

Only missing replies are requested (the cache is keyed by model, prompt, schema, image and run).
"""
import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from aec_llm import config as C  # noqa: E402
from aec_llm import tasks  # noqa: E402
from aec_llm.client import GeminiClient  # noqa: E402
from aec_llm.data import Examples  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=list(C.SPECS))
    ap.add_argument("--model", default=C.DEFAULT_MODEL)
    ap.add_argument("--repeats", type=int, default=3, help="runs of the JSON lab")
    a = ap.parse_args()
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("set GEMINI_API_KEY")
    for v in a.variants:
        spec = C.SPECS[v]
        ex = Examples(REPO, spec)
        client = GeminiClient(key, model=a.model, cache_dir=REPO / "data" / "cache" / v, log=print)
        t0 = time.time(); n0 = len(client.cache)
        log = lambda s: print("   " + s, flush=True)   # noqa: E731
        print(f"== {v}: {n0} replies cached before", flush=True)
        # the most useful answers first, in case a daily quota ends the run
        print("classify basic + schema"); tasks.classify(client, ex.photos, C.fill("classify_basic", spec), ex.photo_classes, True, log=log)
        print("detect schema"); tasks.detect(client, ex.sites, C.fill("detect", spec), ex.site_classes, True, log=log)
        for p in ex.plans:
            print(f"rooms {p.id}")
            tasks.segment_rooms(client, p, C.fill("rooms_masks", spec), "llm", True, log=log)
            tasks.segment_rooms(client, p, C.fill("rooms_boxes", spec), "llm+sam", True, sam=None, log=log)   # same prompt and schema as Step 4b; SAM 3 runs live there
        print("classify basic, plain"); tasks.classify(client, ex.photos, C.fill("classify_basic", spec), ex.photo_classes, False, log=log)
        print("detect plain"); tasks.detect(client, ex.sites, C.fill("detect", spec), ex.site_classes, False, log=log)
        for p in ex.plans:
            print(f"rooms {p.id} masks, plain"); tasks.segment_rooms(client, p, C.fill("rooms_masks", spec), "llm", False, log=log)
        for key_ in ("classify_described", "classify_careful"):
            print(key_ + " + schema"); tasks.classify(client, ex.photos, C.fill(key_, spec), ex.photo_classes, True, log=log)
        print("count schema + plain")
        for s in ex.sites:
            tasks.count(client, s, C.fill("count", spec), spec.sites.count_field, True)
        tasks.count(client, ex.sites[0], C.fill("count", spec), spec.sites.count_field, False)
        print("describe + JSON lab")
        ph = ex.photos[0]
        for q in C.DESCRIBE_QUESTIONS:
            client.ask(ph.load(), C.fill("describe", spec, question=q), image_key=ph.cache_key)
        for run in range(a.repeats):
            client.ask(ph.load(), C.fill("classify_basic", spec), run=run, image_key=ph.cache_key)
            client.ask(ph.load(), C.fill("classify_basic", spec), schema=C.classify_schema(ex.photo_classes), run=run, image_key=ph.cache_key)
        print(f"== {v}: {len(client.cache) - n0} new replies, {client.calls} live calls, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
