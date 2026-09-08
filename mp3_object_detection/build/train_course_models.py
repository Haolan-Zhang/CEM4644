"""Instructor-side: fine-tune YOLO11n on the full training set of a dataset and save it under ../models."""
import argparse, json, os, shutil, sys, time
os.environ.setdefault("YOLO_VERBOSE", "False")
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from aec_det.config import DETSETS          # noqa: E402
from aec_det.data import DetSet             # noqa: E402
from aec_det.evaluate import evaluate_set, summary_text  # noqa: E402
from aec_det.models import Detector         # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--weights", default=None, help="skip training: evaluate and save this best.pt")
    a = ap.parse_args()
    from ultralytics import YOLO
    spec = DETSETS[a.key]
    work = Path(a.work) / a.key
    base = REPO / "models" / "yolo11n.pt"
    t0 = time.time()
    work = work.resolve()
    if a.weights:
        best = Path(a.weights)
    else:
        model = YOLO(str(base))
        def cb(trainer):
            m = trainer.metrics or {}
            print(f"  epoch {trainer.epoch + 1}: mAP50 {m.get('metrics/mAP50(B)', 0) * 100:.1f}  P {m.get('metrics/precision(B)', 0) * 100:.0f}  R {m.get('metrics/recall(B)', 0) * 100:.0f}", flush=True)
        model.add_callback("on_fit_epoch_end", cb)
        model.train(data=str(work / "data.yaml"), epochs=a.epochs, imgsz=640, batch=a.batch, device="0", project=str(work / "runs"),
                    name="course", exist_ok=True, verbose=False, plots=False, workers=8, seed=0, patience=60, deterministic=False)
        best = Path(model.trainer.save_dir) / "weights" / "best.pt"
        print("weights at", best, flush=True)
    out = REPO / spec.course_model
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out)
    det = Detector(out, name="course", classes=[spec.pretty(c) for c in spec.classes])
    test = DetSet.from_folder(work / "test", spec.classes, spec.display)
    res = evaluate_set(det, test, progress=False)
    print(summary_text(res, 0.5))
    (out.parent / f"{a.key}_training_log.json").write_text(json.dumps({
        "key": a.key, "base": "yolo11n.pt (COCO)", "epochs": a.epochs, "n_train": len(list((work / 'train_full' / 'images').iterdir())),
        "n_test": len(test), "map50": res.map50(), "counts_at_0.5": res.counts_at(0.5), "seconds": round(time.time() - t0)}, indent=2))
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f} s")
