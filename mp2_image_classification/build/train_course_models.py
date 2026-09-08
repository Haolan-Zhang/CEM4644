"""Instructor-side: train the course models on the full training sets produced by
prepare_datasets.py and save them (fp16 safetensors) under ../models."""
import argparse, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aec_lab.config import DATASETS            # noqa: E402
from aec_lab.data import ImageSet              # noqa: E402
from aec_lab.evaluate import evaluate_set, summary_text  # noqa: E402
from aec_lab.train import quick_train          # noqa: E402

JOBS = {
    "facade_multiclass": dict(dataset="facade_defects", task="multiclass", epochs=12, lr=3e-4, batch=64, ls=0.1),
    "facade_binary": dict(dataset="facade_defects", task="binary", epochs=8, lr=3e-4, batch=64, ls=0.05),
    "concrete_binary": dict(dataset="concrete_cracks", task="binary", epochs=3, lr=3e-4, batch=64, ls=0.0),
    "styles_multiclass": dict(dataset="facade_styles", task="multiclass", epochs=20, lr=3e-4, batch=32, ls=0.1),
}


def load_sets(spec, work, task):
    tr = ImageSet.from_folder(work / spec.key / "train_full", spec.classes, spec.display, "train_full")
    te = ImageSet.from_folder(work / spec.key / "test", spec.classes, spec.display, "test")
    if task == "binary":
        tr, te = tr.relabel(spec.binary.mapping, spec.binary.classes), te.relabel(spec.binary.mapping, spec.binary.classes)
    return tr, te


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--only", nargs="*", default=list(JOBS))
    a = ap.parse_args()
    work = Path(a.work)
    base = REPO / "models" / "base_convnextv2_femto"
    for job in a.only:
        j = JOBS[job]
        spec = DATASETS[j["dataset"]]
        tr, te = load_sets(spec, work, j["task"])
        print(f"\n=== {job}: {len(tr)} train / {len(te)} test, classes {tr.pretty_classes}", flush=True)
        t0 = time.time()
        clf, hist = quick_train(tr, te, base, epochs=j["epochs"], lr=j["lr"], batch_size=j["batch"], pretrained=True,
                                max_test=None, name=job, progress=False, num_workers=8, label_smoothing=j["ls"],
                                log=lambda s: print(s, flush=True))
        res = evaluate_set(clf, te, progress=False)
        print(summary_text(res), flush=True)
        out = REPO / (spec.binary_model if j["task"] == "binary" else spec.multiclass_model)
        clf.save(out)
        (out / "training_log.json").write_text(json.dumps({
            "job": job, "dataset": spec.key, "task": j["task"], "base_model": "facebook/convnextv2-femto-1k-224",
            "n_train": len(tr), "n_test": len(te), "epochs": j["epochs"], "lr": j["lr"], "batch": j["batch"],
            "label_smoothing": j["ls"], "history": hist, "test_accuracy": res.accuracy,
            "confusion": res.confusion().tolist(), "classes": res.classes, "seconds": round(time.time() - t0)}, indent=2))
        print(f"saved -> {out}  ({sum(f.stat().st_size for f in out.iterdir()) / 1e6:.1f} MB)", flush=True)
    print("all done")
